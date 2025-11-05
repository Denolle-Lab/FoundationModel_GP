
## 7. Codex Prompt for Repository Scaffolding

Copy-paste the block below into a code-generation tool to scaffold the project:

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
