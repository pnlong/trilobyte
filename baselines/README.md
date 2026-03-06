# baselines

Baseline experiments and evaluation pipelines for the Trilobyte lossless audio compression project. Sourced from [pnlong/lossless_nac](https://github.com/pnlong/lossless_nac).

---

## Layout

| Directory | Description |
|---|---|
| `in_context_eval/` | Language-Modeling-Is-Compression (LMIC) in-context evaluation (was `lmic/`) |
| `flac_eval/` | FLAC compression evaluation across datasets |
| `nac/` | Neural audio codec (NAC) implementations and tests |

### Root-level shared utilities

| File | Description |
|---|---|
| `utils.py` | Shared utilities used across subsystems |
| `rice.py` | Rice coding implementation |
| `logging_for_zach.py` | Logging helpers |
| `preprocess_musdb18.py` | Convert MusDB18 to WAV; writes `mixes.csv` |
| `process_musdb18_wav.py` | Convert preprocessed MusDB18 NPY data to WAV |
| `torrent_data_overview.py` | Summary stats for Torrent dataset subsets |

---

## Quick start

- **FLAC baseline:** Run `flac_eval/flac_eval.sh` → inspect `flac_eval_results.csv`; plot with `flac_eval/flac_eval_plot.py`.
- **LMIC (Trilobyte / Llama / FLAC):** Run `in_context_eval/language_modeling_is_compression/compress_audio.py`; batch runner `in_context_eval/lmic_eval.sh`; plot with `in_context_eval/lmic_eval_plot.py`.
- **Lossless neural codecs:** See `nac/lossless_compressors/`; tests in `nac/test_lossless_compressors/`.

Paths in scripts often assume specific machines (e.g. `yggdrasil`, `pando`) and data roots; adjust constants or CLI args for your environment.

---

## Datasets (referenced in code)

MusDB18 (mono/stereo, mixes/stems), LibriSpeech, LJSpeech, Epidemic Sound, VCTK, Torrent (16/24-bit, pro/amateur/freeload), Birdvox, Beethoven piano sonatas, YouTube Mix, SC09. See `flac_eval/flac_eval.py` and `in_context_eval/language_modeling_is_compression/data_loaders_audio.py` for dataset names and default paths.
