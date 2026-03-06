# Trilobyte — Main Implementation

This directory contains the **main training code** for **Trilobyte**: lossless neural audio compression. It includes the GPT-2 byte-level training script, dataset metadata, and experiment configs.

For the official encode/decode codec, see [trilobyte-lossless-codec](https://github.com/pnlong/trilobyte-lossless-codec).

---

## Layout

```
trilobyte/
  train_gpt2.py             # GPT-2 byte-level audio training
  environment.yml           # Conda environment
  configs/
    runs/                   # Training run/experiment configs
      mega_run_data.json
      scale_run_data.json
    dataset_info/           # Per-dataset file metadata (46 JSONs)
      *_info.json           # filename -> {length, sample_rate, bits_per_sample, n_channels}
      *_lengths*.json       # MusDB18 length manifests
      new_train_files.json  # Multi-dataset train split
      new_val_files.json    # Multi-dataset val split
```

---

## `train_gpt2.py`

**GPT-2** training for **byte-level audio**: loads audio as byte tokens, trains autoregressive next-token prediction. Used for models that predict LSBs from MSBs or full byte streams for compression. Includes mu-law encode, `load_audio_raw` / `load_audio_mulaw`, and `split_to_bytes` (PCM -> interleaved byte tokens).

---

## `configs/`

### `configs/runs/`

Experiment configs driving multi-dataset or multi-run training without hardcoding paths in Python. Each JSON has a `datasets` key mapping dataset name -> `{train_data_dir, val_data_dir, train_metadata_path, val_metadata_path, sample_rate, stereo_interleave, ...}`.

| File | Description |
|---|---|
| `mega_run_data.json` | Full multi-dataset training run config |
| `scale_run_data.json` | Scaling experiment run config |

### `configs/dataset_info/`

Dataset manifests used by `train_gpt2.py` (`AudioByteDataset`) to discover audio files and read correct format. Datasets covered: beethoven, birdvox, epidemic, librispeech, ljspeech, musdb18stereo, sc09, torrent, trilobyte, vctk, youtube_mix.

- **`*_train_info.json` / `*_val_info.json`** — per-dataset train/val split metadata: `filename -> {length, sample_rate, bits_per_sample, n_channels}`
- **`new_train_files.json` / `new_val_files.json`** — multi-dataset split lists spanning torr, ljspeech, vctk, birdvox, epidemic, sc09, beethoven, youtube_mix
- **`musdbstereo_lengths*.json`** — MusDB18 stereo clip lengths for chunking
- **`musdbstereo_valid_mix_info.json`** — MusDB18 valid-mix file metadata

Paths in configs point at machine-specific dirs (e.g. `graft*`, `arrakis`); adjust for your environment or generate new manifests for your data layout.

---

## Quick start

- **Train GPT-2 byte-level:** `python train_gpt2.py --metadata_path configs/dataset_info/<dataset>_train_info.json`
- **Encode/decode audio:** See the [trilobyte-lossless-codec](https://github.com/pnlong/trilobyte-lossless-codec) repo.
