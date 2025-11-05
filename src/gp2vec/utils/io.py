"""
I/O Utilities for S3 Access and File Operations

This module provides robust S3 access utilities with retry logic,
exponential backoff, and support for both anonymous and authenticated access.
"""

import io
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import boto3
import s3fs
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Exponential backoff factor
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class S3Client:
    """
    Robust S3 client with retry logic and multiple authentication methods.
    
    Supports both anonymous access and various credential sources including
    EarthScope SDK tokens, IAM roles, and AWS credentials.
    """
    
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        region_name: str = "us-west-2",
        anon: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize S3 client.
        
        Args:
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key  
            aws_session_token: AWS session token
            region_name: AWS region name
            anon: Use anonymous access
            max_retries: Maximum retry attempts
        """
        self.region_name = region_name
        self.anon = anon
        self.max_retries = max_retries
        
        # Initialize clients
        self._init_clients(
            aws_access_key_id,
            aws_secret_access_key, 
            aws_session_token
        )
    
    def _init_clients(
        self,
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_session_token: Optional[str],
    ):
        """Initialize boto3 and s3fs clients."""
        try:
            if self.anon:
                # Anonymous access
                self.boto3_client = boto3.client(
                    's3',
                    region_name=self.region_name,
                    config=boto3.session.Config(
                        signature_version='UNSIGNED',
                        retries={'max_attempts': self.max_retries}
                    )
                )
                self.s3fs_client = s3fs.S3FileSystem(anon=True)
                
            else:
                # Authenticated access
                session_kwargs = {'region_name': self.region_name}
                
                if aws_access_key_id and aws_secret_access_key:
                    session_kwargs.update({
                        'aws_access_key_id': aws_access_key_id,
                        'aws_secret_access_key': aws_secret_access_key,
                    })
                    if aws_session_token:
                        session_kwargs['aws_session_token'] = aws_session_token
                
                session = boto3.Session(**session_kwargs)
                self.boto3_client = session.client(
                    's3',
                    config=boto3.session.Config(
                        retries={'max_attempts': self.max_retries}
                    )
                )
                
                # s3fs with credentials
                self.s3fs_client = s3fs.S3FileSystem(
                    key=aws_access_key_id,
                    secret=aws_secret_access_key,
                    token=aws_session_token,
                )
            
            logger.info(f"Initialized S3 client (anonymous={self.anon})")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 clients: {e}")
            raise
    
    @retry_with_backoff(exceptions=(ClientError, ConnectionError))
    def list_objects(
        self, 
        bucket: str, 
        prefix: str = "",
        max_keys: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket with retry logic.
        
        Args:
            bucket: S3 bucket name
            prefix: Object prefix filter
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object metadata dictionaries
        """
        try:
            paginator = self.boto3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=bucket,
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_keys} if max_keys else {}
            )
            
            objects = []
            for page in page_iterator:
                if 'Contents' in page:
                    objects.extend(page['Contents'])
            
            logger.debug(f"Listed {len(objects)} objects from s3://{bucket}/{prefix}")
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list objects in s3://{bucket}/{prefix}: {e}")
            raise
    
    @retry_with_backoff(exceptions=(ClientError, ConnectionError, IOError))
    def read_object(
        self, 
        bucket: str, 
        key: str,
        byte_range: Optional[tuple] = None,
    ) -> bytes:
        """
        Read object from S3 with retry logic.
        
        Args:
            bucket: S3 bucket name
            key: Object key
            byte_range: Optional (start, end) byte range
            
        Returns:
            Object content as bytes
        """
        try:
            kwargs = {'Bucket': bucket, 'Key': key}
            
            if byte_range:
                start, end = byte_range
                kwargs['Range'] = f'bytes={start}-{end}'
            
            response = self.boto3_client.get_object(**kwargs)
            content = response['Body'].read()
            
            logger.debug(f"Read {len(content)} bytes from s3://{bucket}/{key}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to read s3://{bucket}/{key}: {e}")
            raise
    
    @retry_with_backoff(exceptions=(ClientError, ConnectionError))
    def object_exists(self, bucket: str, key: str) -> bool:
        """
        Check if object exists in S3.
        
        Args:
            bucket: S3 bucket name
            key: Object key
            
        Returns:
            True if object exists, False otherwise
        """
        try:
            self.boto3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Error checking s3://{bucket}/{key}: {e}")
                raise
    
    def get_object_info(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get object metadata.
        
        Args:
            bucket: S3 bucket name
            key: Object key
            
        Returns:
            Object metadata dictionary or None if not found
        """
        try:
            response = self.boto3_client.head_object(Bucket=bucket, Key=key)
            return {
                'size': response.get('ContentLength', 0),
                'last_modified': response.get('LastModified'),
                'etag': response.get('ETag', '').strip('"'),
                'content_type': response.get('ContentType', ''),
            }
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            else:
                logger.error(f"Error getting info for s3://{bucket}/{key}: {e}")
                raise
    
    def open_file(self, s3_path: str, mode: str = 'rb') -> io.IOBase:
        """
        Open S3 file using s3fs.
        
        Args:
            s3_path: Full S3 path (s3://bucket/key)
            mode: File open mode
            
        Returns:
            File-like object
        """
        try:
            return self.s3fs_client.open(s3_path, mode)
        except Exception as e:
            logger.error(f"Failed to open {s3_path}: {e}")
            raise
    
    def download_file(
        self, 
        bucket: str, 
        key: str, 
        local_path: Union[str, Path]
    ):
        """
        Download file from S3 to local path.
        
        Args:
            bucket: S3 bucket name
            key: Object key
            local_path: Local file path
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.boto3_client.download_file(bucket, key, str(local_path))
            logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download s3://{bucket}/{key}: {e}")
            raise
    
    def upload_file(
        self, 
        local_path: Union[str, Path], 
        bucket: str, 
        key: str
    ):
        """
        Upload file from local path to S3.
        
        Args:
            local_path: Local file path
            bucket: S3 bucket name
            key: Object key
        """
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        try:
            self.boto3_client.upload_file(str(local_path), bucket, key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
            raise


class LocalCache:
    """
    Simple local cache for S3 objects.
    
    Provides LRU-style caching with size limits and TTL support.
    """
    
    def __init__(
        self, 
        cache_dir: Union[str, Path],
        max_size_gb: float = 10.0,
        default_ttl_hours: int = 24,
    ):
        """
        Initialize local cache.
        
        Args:
            cache_dir: Cache directory path
            max_size_gb: Maximum cache size in GB
            default_ttl_hours: Default TTL in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.default_ttl_seconds = default_ttl_hours * 3600
        
        logger.info(f"Initialized cache at {self.cache_dir} (max size: {max_size_gb:.1f} GB)")
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash of key to avoid filesystem issues
        import hashlib
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _is_valid(self, cache_path: Path) -> bool:
        """Check if cached file is still valid."""
        if not cache_path.exists():
            return False
        
        # Check TTL
        mtime = cache_path.stat().st_mtime
        age_seconds = time.time() - mtime
        
        return age_seconds < self.default_ttl_seconds
    
    def get(self, key: str) -> Optional[bytes]:
        """Get cached content for key."""
        cache_path = self._get_cache_path(key)
        
        if self._is_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    content = f.read()
                logger.debug(f"Cache hit for {key}")
                return content
            except Exception as e:
                logger.warning(f"Failed to read cache for {key}: {e}")
        
        return None
    
    def put(self, key: str, content: bytes):
        """Store content in cache."""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                f.write(content)
            logger.debug(f"Cached {len(content)} bytes for {key}")
            
            # Clean up old files if needed
            self._cleanup_if_needed()
            
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")
    
    def _cleanup_if_needed(self):
        """Clean up old cache files if size limit exceeded."""
        total_size = sum(
            f.stat().st_size 
            for f in self.cache_dir.glob('*.cache') 
            if f.is_file()
        )
        
        if total_size > self.max_size_bytes:
            # Remove oldest files first
            cache_files = [
                (f.stat().st_mtime, f) 
                for f in self.cache_dir.glob('*.cache') 
                if f.is_file()
            ]
            cache_files.sort()  # Sort by mtime (oldest first)
            
            removed_size = 0
            target_size = self.max_size_bytes * 0.8  # Remove to 80% of limit
            
            for mtime, cache_file in cache_files:
                if total_size - removed_size <= target_size:
                    break
                
                try:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    removed_size += file_size
                    logger.debug(f"Removed cached file {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """
    Parse S3 path into bucket and key.
    
    Args:
        s3_path: S3 path (s3://bucket/key or just bucket/key)
        
    Returns:
        Tuple of (bucket, key)
    """
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]  # Remove s3:// prefix
    
    parts = s3_path.split('/', 1)
    if len(parts) == 1:
        return parts[0], ''
    else:
        return parts[0], parts[1]


def get_s3_credentials_from_earthscope() -> Optional[Dict[str, str]]:
    """
    Get temporary S3 credentials from EarthScope SDK.
    
    Returns:
        Dictionary with AWS credentials or None if not available
    """
    try:
        # This would integrate with EarthScope SDK
        # Placeholder implementation
        logger.info("EarthScope SDK integration not implemented")
        return None
    except ImportError:
        logger.info("EarthScope SDK not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get EarthScope credentials: {e}")
        return None


# Global S3 client instance
_default_s3_client = None

def get_s3_client(**kwargs) -> S3Client:
    """Get or create default S3 client."""
    global _default_s3_client
    
    if _default_s3_client is None:
        _default_s3_client = S3Client(**kwargs)
    
    return _default_s3_client