# GP2Vec: Self-Supervised Learning for Seismic Data

GP2Vec is a PyTorch implementation of Wav2Vec2-style self-supervised learning adapted for seismic waveform data. It learns robust representations from continuous seismic data hosted on S3, with optional conditioning on station metadata.

## 🌟 Key Features

- **Wav2Vec2-inspired architecture** adapted for 3-component seismic data (Z, N, E)
- **S3-native data pipeline** for scalable access to EarthScope/SCEDC data
- **Station metadata conditioning** using FDSN web services
- **WebDataset streaming** for efficient training on large datasets
- **PyTorch Lightning** integration with distributed training support
- **Hydra configuration** for flexible experiment management
- **Production-ready** with comprehensive logging, monitoring, and checkpointing

## 🏗️ Architecture

GP2Vec follows the Wav2Vec2 architecture with adaptations for seismic data:

1. **Feature Encoder**: 1D CNN that processes raw waveforms into latent representations
2. **Vector Quantizer**: Learns discrete codebook representations (Gumbel or EMA-based)
3. **Context Encoder**: Transformer that models temporal dependencies
4. **Metadata Fusion**: Optional conditioning on station coordinates and instrument metadata
5. **Contrastive Learning**: InfoNCE loss for self-supervised pretraining

```
Waveform (3, 3000) → CNN → Features (768, T) → VQ → Quantized → Transformer → Contextual Features
                                    ↓                              ↑
                            Station Metadata ――――――――――――――――――――――┘
                                    ↓
                              Contrastive Loss
```

## 📦 Installation

### Prerequisites

- Python ≥ 3.9
- PyTorch ≥ 2.4
- CUDA (optional, for GPU training)

### From Source

```bash
git clone https://github.com/your-org/gp2vec.git
cd gp2vec
pip install -e .
```

### Dependencies

Core dependencies are automatically installed:
- `torch`, `torchaudio` - Deep learning framework
- `pytorch-lightning` - Training framework
- `hydra-core` - Configuration management
- `obspy` - Seismic data processing
- `s3fs`, `boto3` - S3 data access
- `webdataset` - Streaming datasets
- `pandas`, `pyarrow` - Data manipulation

## 🚀 Quick Start

### 1. Basic Training

```bash
# Train with default configuration
python -m gp2vec.train.train

# Debug training (small model, limited data)
python -m gp2vec.train.train --config-name=experiment/debug

# Production training (large model, full dataset)
python -m gp2vec.train.train --config-name=experiment/production
```

### 2. Using the Pretraining Script

```bash
# Full pipeline: data preparation + training
./scripts/pretrain.sh

# With custom configuration
./scripts/pretrain.sh configs/experiment/debug.yaml

# Override specific parameters
./scripts/pretrain.sh --data.batch_size=32 --model.embed_dim=512
```

### 3. Data Pipeline

```python
from gp2vec.data.datapipes import SeismicDataPipeline
from gp2vec.data.metadata import StationMetadataManager

# Set up data pipeline
pipeline = SeismicDataPipeline(
    manifest_path="cache/manifest.parquet",
    metadata_manager=StationMetadataManager(),
    target_sampling_rate=100.0,
    window_length=30.0,
)

# Create streaming dataset
dataset = pipeline.create_webdataset(shard_size=10000)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### 4. Model Usage

```python
import torch
from gp2vec.models.gp2vec import GP2Vec

# Create model
model = GP2Vec()

# Example waveform data (batch_size=4, channels=3, time=3000)
waveforms = torch.randn(4, 3, 3000)

# Station metadata
metadata = {
    'latitude': torch.tensor([34.0, 34.1, 34.2, 34.3]),
    'longitude': torch.tensor([-118.2, -118.3, -118.4, -118.5]),
    'elevation': torch.tensor([100.0, 150.0, 200.0, 250.0]),
}

# Extract features
model.eval()
with torch.no_grad():
    features = model.encode(waveforms, metadata)  # (4, T, 768)
```

## 📊 Data

### Supported Data Sources

- **EarthScope/SCEDC**: Primary data source via S3 (`s3://scedc-pds/continuous_waveforms/`)
- **Custom S3 buckets**: Any S3-compatible storage with miniSEED files
- **Local files**: Direct file system access for smaller datasets

### Data Preparation

1. **Build Manifest**:
```bash
python scripts/make_manifest.py \
    --bucket scedc-pds \
    --prefix continuous_waveforms/ \
    --output cache/manifest.parquet \
    --networks CI AZ US TA
```

2. **Fetch Metadata**:
```bash
python scripts/fetch_metadata.py \
    --client IRIS \
    --output cache/metadata/stations_iris.parquet \
    --extract-features
```

### Data Processing

- **Preprocessing**: Detrending, demeaning, normalization, quality control
- **Windowing**: Configurable window length with overlap
- **Augmentation**: Time shifts, amplitude scaling, noise injection, filtering
- **Streaming**: WebDataset-based pipeline for scalable training

## ⚙️ Configuration

GP2Vec uses Hydra for configuration management. Configurations are organized as:

```
configs/
├── config.yaml              # Main config file
├── data/
│   ├── default.yaml         # Default data settings
│   └── small.yaml           # Small dataset for testing
├── model/
│   ├── default.yaml         # Default model architecture
│   └── small.yaml           # Small model for debugging
├── train/
│   ├── default.yaml         # Default training settings
│   └── debug.yaml           # Debug training settings
└── experiment/
    ├── debug.yaml           # Complete debug experiment
    └── production.yaml      # Production experiment
```

### Key Configuration Options

#### Data Configuration
```yaml
# data/default.yaml
processing:
  target_sampling_rate: 100.0
  window_length: 30.0
  overlap: 0.5

augmentation:
  enabled: true
  probability: 0.8
  time_shift:
    max_shift: 0.1
  
metadata:
  enabled: true
  fdsn_client: IRIS
```

#### Model Configuration
```yaml
# model/default.yaml
feature_encoder:
  conv_layers:
    - {channels: 64, kernel_size: 10, stride: 5}
    - {channels: 128, kernel_size: 8, stride: 4}
    # ...

context_encoder:
  embed_dim: 768
  num_heads: 12
  num_layers: 12

quantizer:
  type: gumbel
  codebook_size: 320
  num_codebooks: 2
```

#### Training Configuration
```yaml
# train/default.yaml
optimizer:
  name: adamw
  lr: 1e-4
  weight_decay: 0.01

lr_scheduler:
  name: cosine_with_warmup
  warmup_steps: 10000

trainer:
  max_epochs: 100
  devices: auto
  strategy: ddp
  precision: 16-mixed
```

### Configuration Overrides

```bash
# Override from command line
python -m gp2vec.train.train \
    data.batch_size=64 \
    model.embed_dim=1024 \
    train.max_epochs=200

# Use different configs
python -m gp2vec.train.train \
    --config-path=configs \
    --config-name=experiment/production
```

## 🔬 Downstream Evaluation

GP2Vec includes tools for evaluating learned representations on downstream tasks:

```python
from gp2vec.train.evaluate_downstream import DownstreamEvaluator

# Load pretrained model
evaluator = DownstreamEvaluator(model)

# Evaluate on phase picking
results = evaluator.evaluate_phase_picking(
    waveforms, pick_labels, metadata
)

# Evaluate on tremor detection  
results = evaluator.evaluate_tremor_detection(
    waveforms, tremor_labels, metadata
)
```

### Supported Tasks

- **Phase Picking**: Binary classification of P/S wave arrivals
- **Tremor Detection**: Binary classification of tremor vs. normal signals
- **Magnitude Estimation**: Multi-class classification of earthquake magnitude bins

### Evaluation Protocols

- **Linear Probing**: Freeze backbone, train linear classifier
- **Sklearn Evaluation**: Extract features, train sklearn models
- **Fine-tuning**: End-to-end fine-tuning (planned)

## 🖥️ Distributed Training

GP2Vec supports various distributed training strategies:

```yaml
# Multi-GPU training
train:
  trainer:
    devices: 4
    strategy: ddp
    
# Multi-node training
train:
  trainer:
    devices: 8
    num_nodes: 4
    strategy: ddp
```

### Supported Strategies

- **DDP** (DistributedDataParallel): Standard multi-GPU training
- **FSDP** (FullyShardedDataParallel): Memory-efficient training for large models
- **DeepSpeed**: Advanced optimization strategies

## 📈 Monitoring and Logging

### Weights & Biases Integration

```yaml
wandb:
  enabled: true
  project: gp2vec
  entity: your-team
  tags: [self-supervised, seismic]
```

### TensorBoard Support

```yaml
loggers:
  tensorboard:
    enabled: true
    log_graph: true
```

### Rich Progress Bars

Interactive progress bars with training metrics, GPU utilization, and throughput monitoring.

## 🧪 Examples

See the `examples/` directory for complete usage examples:

- `basic_training.py`: Simple training script
- `data_pipeline.py`: Data loading and processing
- `model_usage.py`: Model inference and feature extraction

## 🛠️ Development

### Project Structure

```
gp2vec/
├── src/gp2vec/
│   ├── data/              # Data loading and processing
│   ├── models/            # Model architectures  
│   ├── train/             # Training and evaluation
│   └── utils/             # Utilities
├── configs/               # Hydra configurations
├── scripts/               # Operational scripts
├── examples/              # Usage examples
└── tests/                 # Unit tests (planned)
```

### Code Quality

The project uses modern Python tooling:

- **Black**: Code formatting
- **Ruff**: Fast linting and formatting
- **MyPy**: Static type checking
- **Pytest**: Unit testing framework

```bash
# Format code
black src/ examples/ scripts/

# Lint code  
ruff check src/ examples/ scripts/

# Type checking
mypy src/gp2vec
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run quality checks
5. Submit a pull request

## 📋 Requirements

### Minimum System Requirements

- **RAM**: 16GB+ (32GB+ recommended for large datasets)
- **Storage**: 100GB+ for data caching
- **GPU**: 8GB+ VRAM (16GB+ recommended)

### Cloud Deployment

GP2Vec is designed for cloud deployment with:

- **AWS S3** integration for data access
- **Kubernetes** deployment support
- **Docker** containerization
- **Multi-node** distributed training

## 🤝 Citation

If you use GP2Vec in your research, please cite:

```bibtex
@software{gp2vec2024,
  title={GP2Vec: Self-Supervised Learning for Seismic Waveform Representation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/gp2vec}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full documentation](https://gp2vec.readthedocs.io) (planned)
- **Issues**: [GitHub Issues](https://github.com/your-org/gp2vec/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/gp2vec/discussions)

## 🗺️ Roadmap

### Current Features ✅
- [x] Wav2Vec2-style architecture for seismic data
- [x] S3-native data pipeline with WebDataset streaming
- [x] Station metadata conditioning
- [x] PyTorch Lightning training framework
- [x] Hydra configuration system
- [x] Distributed training support
- [x] Downstream evaluation tools

### Planned Features 🚧
- [ ] Additional downstream tasks (magnitude estimation, source characterization)
- [ ] Pre-trained model zoo
- [ ] Advanced augmentation strategies
- [ ] Real-time inference capabilities
- [ ] Integration with seismological workflows
- [ ] Comprehensive documentation and tutorials

### Future Research 🔮
- [ ] Multi-modal learning (waveforms + spectrograms + metadata)
- [ ] Federated learning across seismic networks
- [ ] Integration with physics-informed neural networks
- [ ] Uncertainty quantification in representations

---

**GP2Vec** - Advancing seismology through self-supervised representation learning 🌍📈

This document outlines an end-to-end plan to build a Wav2Vec 2.0–style self-supervised foundation model for seismology. It covers data access from EarthScope S3, station metadata integration, model architecture, training strategy, and repository scaffolding suitable for hand-off to an engineering team or code-generation system.

## 1. Core Ideas from Johnson et al. (2025)

- Adopt a Wav2Vec 2.0 objective on continuous seismic waveforms: mask latent spans, predict quantized targets, and reuse the encoder for downstream tasks such as phase picking, tremor characterization, magnitude estimation, and slow slip proxies.[^johnson2025]
- Train on long, contiguous segments, masking contiguous time spans and sampling negatives from other temporal windows or channels; support multi-component inputs (Z/N/E) and targeted volcano or fault scenarios as in the reference study.[^johnson2025]

## 2. Streaming miniSEED from EarthScope S3

Two supported approaches:

### A. EarthScope SDK → Temporary AWS Credentials → boto3

- Exchange EarthScope credentials for temporary AWS keys using the SDK, then use `boto3` to list and fetch miniSEED objects.[^earthscope-sdk]

### B. Direct Access Tutorials (s3fs / boto3)

- Follow the `earthscope-s3-direct-access` examples to read data with `s3fs` or `boto3`, respecting the documented bucket and prefix structure.[^earthscope-repo]

**Python sketch:**

```python
import io
import s3fs
from obspy import read

fs = s3fs.S3FileSystem(anon=True)  # or provide credentials from EarthScope SDK / boto3
with fs.open("s3://<earthscope-bucket>/<prefix>/<file>.mseed", "rb") as f:
    st = read(io.BytesIO(f.read()))  # ObsPy accepts file-like objects
```

Adjust bucket and prefix paths per EarthScope documentation.[^earthscope-sdk]

## 3. Station Metadata (FDSN → ObsPy)

- Retrieve StationXML via `obspy.clients.fdsn.Client.get_stations`, requesting `level="response"` to obtain instrument metadata.[^obspy-inventory]
- Extract key fields (lat/lon/elev, instrument/datalogger descriptions, azimuth, dip, sampling rate, response gain) using ObsPy `Inventory` APIs; optionally leverage wrappers like `fdsn_station_info` for CSV exports.[^fdsn-station-info]

**Python sketch:**

```python
from obspy.clients.fdsn import Client

inv = Client("IRIS").get_stations(
    network="IU", station="ANMO", level="response", starttime=t0, endtime=t1
)
net = inv.networks[0]
sta = net.stations[0]
ch = sta.channels[0]
meta = {
    "network": net.code,
    "station": sta.code,
    "loc": ch.location_code,
    "chan": ch.code,
    "latitude": sta.latitude,
    "longitude": sta.longitude,
    "elevation": sta.elevation,
    "azimuth": ch.azimuth,
    "dip": ch.dip,
    "sample_rate": ch.sample_rate,
    "response_gain": (
        ch.response.instrument_sensitivity.value if ch.response else None
    ),
    "sensor": ch.sensor.description if ch.sensor else None,
    "datalogger": ch.data_logger.description if ch.data_logger else None,
}
```

### Metadata Embeddings

- **Categorical variables** (network, station, location, channel, sensor, datalogger): map to learnable embeddings (16–64 dimensions each).
- **Continuous variables** (latitude, longitude, elevation, azimuth, dip, sample rate, gain): normalize and feed through a small MLP to produce dense features.
- **Response curves**: summarize via scalar features or sample log-amplitude response over fixed frequencies, then encode with a 1D convolution.
- Concatenate and project the metadata embedding to the model dimension, fusing with waveform tokens via additive conditioning, cross-attention, or FiLM-style scaling.

## 4. Wav2Vec-Style Model (PyTorch/Lightning)

- **Feature encoder:** 1D convolution stack over single or tri-axial channels with 10–25 ms kernels; downsample to ~50–100 Hz latent frame rate.
- **Vector quantizer:** Gumbel-softmax or EMA k-means codebook to produce discrete targets with diversity regularization.
- **Context network:** Transformer encoder (8–24 layers, 256–768 hidden units) with random masking of contiguous latent spans (covering roughly 30–60% of steps).
- **Loss:** Contrastive InfoNCE objective between masked positions and quantized targets plus a codebook diversity loss.
- **Augmentations:** Random bandpass filtering, amplitude jitter, Gaussian noise, time shifts, channel dropout, and optional instrument response removal using metadata inventories.
- **Optional multitask heads:** Predict sampling-rate bins, channel families, or station regions from masked contexts to encourage invariance.

## 5. Data Pipeline (S3 → Shards → Training)

1. **Catalog:** List S3 keys (organized by network/station/year/day) via `s3fs` or `boto3`; store a Parquet manifest containing object key, time bounds, sampling rate, and byte size.[^earthscope-repo]
2. **Sharding:** Assemble WebDataset or torchdata shards (1–2 GB) comprising 20–60 second windows and context negatives.
3. **Streaming:** Use torchdata/DataPipes or WebDataset to stream from S3, decode miniSEED with ObsPy, resample, optionally remove instrument response, and cache as needed.
4. **Metadata join:** Merge station/channel IDs with pre-fetched StationXML features and attach metadata embeddings to each sample.[^obspy-inventory]
5. **Distributed training:** Train via PyTorch Lightning with FSDP or DeepSpeed, mixed precision, EMA, and periodic checkpoints written back to S3.

## 6. Minimal Code Blocks

**List and stream from EarthScope S3:**

```python
import io
import random

import obspy
import s3fs

fs = s3fs.S3FileSystem(anon=True)  # or authenticate via key/secret/token
keys = fs.ls("s3://<earthscope-bucket>/<miniSEED-prefix>/")
key = random.choice([k for k in keys if k.endswith(".mseed")])
with fs.open(key, "rb") as f:
    st = obspy.read(io.BytesIO(f.read()))

st.merge(fill_value="interpolate")
st.detrend("linear")
st.taper(0.05)
```

Match the bucket and prefix naming per EarthScope documentation.[^earthscope-sdk]

**Instrument correction (optional):**

```python
from obspy.signal import filter

st.remove_response(inventory=inv, output="VEL")  # requires matching Inventory metadata
st.filter("bandpass", freqmin=0.1, freqmax=20.0, corners=4, zerophase=True)
```

## 7. Codex Prompt for Repository Scaffolding

Copy-paste the block below into a code-generation tool to scaffold the project:

````text
You are generating a production-ready Python repo called `gp2vec` that trains a Wav2Vec2-style self-supervised model on S3-hosted miniSEED seismic data with station-metadata conditioning.

## Requirements
- Python 3.11; PyTorch >=2.4; PyTorch Lightning >=2.4; torchaudio; obspy; s3fs; boto3; pandas; pyarrow; hydra-core; webdataset (or torchdata); rich; typer; pydantic; tqdm.
- Optional: deepspeed or fsdp.shim; wandb or lightning-fabric logger.

## Repository layout
- README.md: overview, quickstart, data access notes (EarthScope S3), Johnson et al. (2025) context, citations.
- LICENSE: MIT
- pyproject.toml + `src/gp2vec` package
- `src/gp2vec/data/`
  - `s3_manifest.py`: list S3 prefixes, build a Parquet manifest of miniSEED files (key, t0, t1, sr, nbytes).
  - `decoder.py`: read miniSEED from file-like bytes via ObsPy; merge; basic QC.
  - `metadata.py`: FDSN client fetch; parse ObsPy Inventory → dict of numeric/categorical features; caching to Parquet.
  - `datapipes.py`: WebDataset/torchdata pipelines to stream windows from S3, join metadata, apply augmentations.
- `src/gp2vec/models/`
  - `feature_encoder.py`: 1D CNN stack turning waveform into latent features; supports 1 or 3 channels.
  - `vq.py`: Gumbel/EMA codebook quantizer with diversity loss.
  - `transformer.py`: context encoder (TransformerEncoder).
  - `losses.py`: contrastive + codebook diversity; masking utilities.
  - `gp2vec.py`: full model that fuses waveform tokens with station-metadata embeddings (add/cross-attn/FiLM).
- `src/gp2vec/train/`
  - `module.py`: LightningModule (optimizer, sched, EMA, loggers).
  - `train.py`: Hydra-driven CLI (`python -m gp2vec.train.train ...`).
  - `evaluate_downstream.py`: linear probe for phase picking / tremor detection.
- `src/gp2vec/utils/`
  - `aug.py`: bandpass, amp jitter, noise, time shift, channel dropout.
  - `geo.py`: lat/lon normalization; region bucketing.
  - `io.py`: S3 helpers using s3fs/boto3; retry/backoff.
- `configs/`
  - `data.yaml`: S3 bucket, prefixes, window length/stride, sample rate, resample settings.
  - `model.yaml`: CNN/Transformer sizes, codebook size, mask probs.
  - `train.yaml`: batch size, lr, optimizer, precision, distributed strategy.
- `scripts/`
  - `make_manifest.py`: crawl S3, build `manifest.parquet`.
  - `fetch_metadata.py`: query FDSN for networks/stations/channels used, write `metadata.parquet`.
  - `pretrain.sh`: example launch (DDP/FSDP/DeepSpeed).
- `examples/`
  - `earthscope_s3_demo.ipynb`: list + read miniSEED from S3, visualize with ObsPy.
  - `metadata_demo.ipynb`: fetch StationXML and build embeddings.
- `tests/`: unit tests for data decode, metadata parse, model forward, masking, quantizer.
- `.github/workflows/ci.yml`: run unit tests + style checks.

## Implementation specifics
- **S3 access**: Implement both anonymous and credentialed flows. Prefer `s3fs` for streaming; fall back to `boto3` for signed URLs. Provide exponential backoff.
- **Windowing**: From each file, create overlapping windows (e.g., 30 s with 75% overlap). Balance positives/negatives within a batch.
- **Sampling rate**: Resample to a small set (e.g., 50/100 Hz). Store original sr in metadata and optionally predict it as an auxiliary task.
- **Metadata embeddings**:
  - Categorical: learned embeddings (dims: 32 each) for network/station/location/channel/sensor/datalogger.
  - Numeric: lat/lon/elev/azimuth/dip/sr/gain standardized → 2-layer MLP → 128-d.
  - Optional response curve: 64 log-spaced freqs → 1D conv → 64-d.
  - Concatenate and project to `d_model`, then fuse with waveform tokens via (configurable) add/cross-attention/FiLM.
- **Masking**: Span mask ~50% of time steps; span length ~10–20 latent steps; apply channel-drop at 10–20%.
- **Quantizer**: Codebook size 2^13–2^15; Gumbel-softmax with temperature schedule or EMA-kmeans.
- **Loss**: Contrastive InfoNCE on masked positions; diversity loss to use all codewords; temperature-scaling on logits.
- **Logging**: Learning curves, codebook usage histograms, example reconstructions; save checkpoints & cfgs to S3.
- **Downstream**: Provide a linear probe script on labeled picks/tremor windows to validate representation quality.
- **Repro**: Set seeds, deterministic flags where possible; document non-determinism from cudnn.

## Quickstart snippets (put in README)
### Read miniSEED from S3
```python
from gp2vec.data.decoder import read_mseed_s3
st = read_mseed_s3("s3://<bucket>/<prefix>/<file>.mseed")  # returns ObsPy Stream
```

### Fetch & embed metadata

```python
from gp2vec.data.metadata import build_metadata_table
df = build_metadata_table(networks=["IU"], starttime="2018-05-01", endtime="2018-08-31")
```

### Pretrain

```bash
python -m gp2vec.train.train data=configs/data.yaml model=configs/model.yaml train=configs/train.yaml
```

## Quality bar

* Type-checked (mypy), linted (ruff/black); 90% unit test coverage on utils/data; CI runs on pushes/PRs.
* Docstrings and sphinx-friendly docs generation target.
````

## 8. Citations for README

- Johnson, B. et al. (2025). *Automatic speech recognition predicts contemporaneous earthquake fault slip and tremor.* Nature Communications.[^johnson2025]
- EarthScope SDK: *Direct S3 Access to SAGE miniSEED data repository.*[^earthscope-sdk]
- `earthscope-s3-direct-access` tutorial repository.[^earthscope-repo]
- ObsPy Inventory and FDSN documentation.[^obspy-inventory]
- `fdsn_station_info` ObsPy wrapper.[^fdsn-station-info]

---

[^johnson2025]: Johnson, B., et al. (2025). *Automatic speech recognition predicts contemporaneous earthquake fault slip and tremor.* Nature Communications. https://www.nature.com/articles/s41467-025-55994-9.pdf
[^earthscope-sdk]: EarthScope SDK Documentation. *Direct S3 Access to SAGE miniSEED data repository.* https://docs.earthscope.org/projects/SDK/en/stable/content/s3_direct_access_tutorial.html
[^earthscope-repo]: Niyiyu, Y. *earthscope-s3-direct-access* (GitHub repository). https://github.com/niyiyu/earthscope-s3-direct-access
[^obspy-inventory]: ObsPy Documentation. *obspy.core.inventory.* https://docs.obspy.org/packages/autogen/obspy.core.inventory.html
[^fdsn-station-info]: Flyrok. *fdsn_station_info* (GitHub repository). https://github.com/flyrok/fdsn_station_info
