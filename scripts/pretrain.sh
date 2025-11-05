#!/bin/bash

# GP2Vec Pretraining Script
# 
# This script handles the complete pretraining pipeline for GP2Vec:
# 1. Environment setup and dependency checking
# 2. Data preparation (manifest building, metadata fetching)  
# 3. Model pretraining with configurable parameters
# 4. Checkpoint management and monitoring
#
# Usage: ./pretrain.sh [config] [options]
# Example: ./pretrain.sh configs/experiment/debug.yaml --data.batch_size=16

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="configs/config.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to display usage
usage() {
    cat << EOF
GP2Vec Pretraining Script

Usage: $0 [CONFIG] [OPTIONS]

Arguments:
  CONFIG              Path to Hydra config file (default: ${DEFAULT_CONFIG})

Options:
  -h, --help         Show this help message
  -n, --dry-run      Show commands that would be executed without running them
  -s, --skip-data    Skip data preparation steps
  -c, --continue     Continue from existing checkpoint
  -f, --force        Force overwrite existing outputs
  -v, --verbose      Verbose output
  -q, --quiet        Minimal output
  
Environment Variables:
  GP2VEC_DATA_ROOT   Root directory for data (default: ./data)
  GP2VEC_OUTPUT_DIR  Output directory for experiments (default: ./outputs)
  GP2VEC_CACHE_DIR   Cache directory (default: ./cache)
  CUDA_VISIBLE_DEVICES  GPU devices to use
  
Examples:
  # Basic training with default config
  $0
  
  # Debug training with small model and data
  $0 configs/experiment/debug.yaml
  
  # Override config parameters
  $0 --data.batch_size=32 --model.embed_dim=512
  
  # Distributed training on multiple GPUs
  CUDA_VISIBLE_DEVICES=0,1,2,3 $0 --train.devices=4
  
  # Continue from checkpoint
  $0 --continue --train.resume.ckpt_path=outputs/last.ckpt
  
EOF
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not found"
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: ${python_version}"
    
    # Check if we're in a virtual environment
    if [[ -z "${VIRTUAL_ENV}" ]] && [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
        log_warn "No virtual environment detected. Consider using conda/venv."
    else
        log_info "Environment: ${VIRTUAL_ENV:-$CONDA_DEFAULT_ENV}"
    fi
    
    # Check if gp2vec package is installed
    if ! python3 -c "import gp2vec" &> /dev/null; then
        log_warn "GP2Vec package not found. Installing in development mode..."
        cd "${PROJECT_ROOT}"
        pip install -e .
    fi
    
    # Check for CUDA availability
    local cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [[ "${cuda_available}" == "True" ]]; then
        local gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        log_info "CUDA available with ${gpu_count} GPU(s)"
    else
        log_warn "CUDA not available, will use CPU"
    fi
    
    log_success "Prerequisites check completed"
}

# Function to prepare data
prepare_data() {
    log_info "Preparing data..."
    
    local manifest_file="${GP2VEC_CACHE_DIR}/manifest.parquet"
    local metadata_file="${GP2VEC_CACHE_DIR}/metadata/stations_iris.parquet"
    
    # Create directories
    mkdir -p "${GP2VEC_CACHE_DIR}/metadata"
    mkdir -p "${GP2VEC_DATA_ROOT}"
    
    # Build manifest if it doesn't exist or if forced
    if [[ ! -f "${manifest_file}" ]] || [[ "${FORCE}" == "true" ]]; then
        log_info "Building S3 manifest..."
        
        local manifest_cmd="python3 ${SCRIPT_DIR}/make_manifest.py \
            --output ${manifest_file} \
            --cache-dir ${GP2VEC_CACHE_DIR} \
            --workers 4"
        
        if [[ "${VERBOSE}" == "true" ]]; then
            manifest_cmd="${manifest_cmd} --verbose"
        elif [[ "${QUIET}" == "true" ]]; then
            manifest_cmd="${manifest_cmd} --quiet"
        fi
        
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "Would run: ${manifest_cmd}"
        else
            eval "${manifest_cmd}"
        fi
    else
        log_info "Using existing manifest: ${manifest_file}"
    fi
    
    # Fetch metadata if it doesn't exist or if forced
    if [[ ! -f "${metadata_file}" ]] || [[ "${FORCE}" == "true" ]]; then
        log_info "Fetching station metadata..."
        
        local metadata_cmd="python3 ${SCRIPT_DIR}/fetch_metadata.py \
            --output ${metadata_file} \
            --cache-dir ${GP2VEC_CACHE_DIR}/metadata \
            --extract-features"
        
        if [[ "${VERBOSE}" == "true" ]]; then
            metadata_cmd="${metadata_cmd} --verbose"
        elif [[ "${QUIET}" == "true" ]]; then
            metadata_cmd="${metadata_cmd} --quiet"
        fi
        
        if [[ "${DRY_RUN}" == "true" ]]; then
            log_info "Would run: ${metadata_cmd}"
        else
            eval "${metadata_cmd}"
        fi
    else
        log_info "Using existing metadata: ${metadata_file}"
    fi
    
    log_success "Data preparation completed"
}

# Function to run training
run_training() {
    log_info "Starting GP2Vec pretraining..."
    
    # Build training command
    local train_cmd="python3 -m gp2vec.train.train"
    
    # Add config file
    if [[ -n "${CONFIG_FILE}" ]]; then
        train_cmd="${train_cmd} --config-path=$(dirname ${CONFIG_FILE}) --config-name=$(basename ${CONFIG_FILE} .yaml)"
    fi
    
    # Add overrides
    for override in "${CONFIG_OVERRIDES[@]}"; do
        train_cmd="${train_cmd} ${override}"
    done
    
    # Add environment-based overrides
    if [[ -n "${GP2VEC_DATA_ROOT}" ]]; then
        train_cmd="${train_cmd} data_root=${GP2VEC_DATA_ROOT}"
    fi
    
    if [[ -n "${GP2VEC_OUTPUT_DIR}" ]]; then
        train_cmd="${train_cmd} output_dir=${GP2VEC_OUTPUT_DIR}"
    fi
    
    if [[ -n "${GP2VEC_CACHE_DIR}" ]]; then
        train_cmd="${train_cmd} cache_dir=${GP2VEC_CACHE_DIR}"
    fi
    
    # Log command
    log_info "Training command: ${train_cmd}"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_info "Would run: ${train_cmd}"
        return 0
    fi
    
    # Create output directory
    mkdir -p "${GP2VEC_OUTPUT_DIR}"
    
    # Run training
    cd "${PROJECT_ROOT}"
    
    if [[ "${VERBOSE}" == "true" ]]; then
        eval "${train_cmd}"
    else
        eval "${train_cmd}" 2>&1 | tee "${GP2VEC_OUTPUT_DIR}/training.log"
    fi
    
    local exit_code=$?
    
    if [[ ${exit_code} -eq 0 ]]; then
        log_success "Training completed successfully"
    else
        log_error "Training failed with exit code ${exit_code}"
        return ${exit_code}
    fi
}

# Function to cleanup on exit
cleanup() {
    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        log_error "Script failed with exit code ${exit_code}"
    fi
    exit ${exit_code}
}

# Main function
main() {
    # Set default values
    CONFIG_FILE=""
    CONFIG_OVERRIDES=()
    DRY_RUN="false"
    SKIP_DATA="false"
    CONTINUE="false"
    FORCE="false"
    VERBOSE="false"
    QUIET="false"
    
    # Set default environment variables
    export GP2VEC_DATA_ROOT="${GP2VEC_DATA_ROOT:-${PROJECT_ROOT}/data}"
    export GP2VEC_OUTPUT_DIR="${GP2VEC_OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
    export GP2VEC_CACHE_DIR="${GP2VEC_CACHE_DIR:-${PROJECT_ROOT}/cache}"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -n|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -s|--skip-data)
                SKIP_DATA="true"
                shift
                ;;
            -c|--continue)
                CONTINUE="true"
                shift
                ;;
            -f|--force)
                FORCE="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -q|--quiet)
                QUIET="true"
                shift
                ;;
            --*)
                # Config override
                CONFIG_OVERRIDES+=("$1")
                shift
                ;;
            *.yaml|*.yml)
                # Config file
                CONFIG_FILE="$1"
                shift
                ;;
            *)
                log_error "Unknown argument: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Use default config if none specified
    if [[ -z "${CONFIG_FILE}" ]]; then
        CONFIG_FILE="${PROJECT_ROOT}/${DEFAULT_CONFIG}"
    fi
    
    # Validate config file exists
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        log_error "Config file not found: ${CONFIG_FILE}"
        exit 1
    fi
    
    # Print configuration
    log_info "GP2Vec Pretraining Configuration:"
    log_info "  Config file: ${CONFIG_FILE}"
    log_info "  Data root: ${GP2VEC_DATA_ROOT}"
    log_info "  Output dir: ${GP2VEC_OUTPUT_DIR}"
    log_info "  Cache dir: ${GP2VEC_CACHE_DIR}"
    
    if [[ ${#CONFIG_OVERRIDES[@]} -gt 0 ]]; then
        log_info "  Overrides: ${CONFIG_OVERRIDES[*]}"
    fi
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log_warn "DRY RUN MODE - No actual execution"
    fi
    
    # Set up signal handling
    trap cleanup EXIT INT TERM
    
    # Execute steps
    check_prerequisites
    
    if [[ "${SKIP_DATA}" == "false" ]]; then
        prepare_data
    else
        log_info "Skipping data preparation"
    fi
    
    run_training
    
    log_success "GP2Vec pretraining pipeline completed successfully!"
}

# Run main function with all arguments
main "$@"