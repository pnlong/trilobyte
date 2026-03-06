# Trilobyte

Research repository for **Trilobyte** — a language-model-based lossless audio codec. This umbrella repo ties together the codec itself, baseline comparisons, training experiments, and paper figures.

- **Original Trilobyte experiments:** [https://github.com/ZacharyNovack/lnac](https://github.com/ZacharyNovack/lnac)
- **Original baselines home:** [https://github.com/pnlong/lossless_nac](https://github.com/pnlong/lossless_nac)

---

## Repository layout

| Directory | Description | Repo |
|---|---|---|
| `trilobyte_lossless_codec/` | Official Trilobyte codec (encode/decode CLI) | https://github.com/pnlong/trilobyte-lossless-codec |
| `baselines/` | Baseline experiments, eval pipelines, NAC codecs | https://github.com/pnlong/lossless_nac |
| `trilobyte/` | Training code, configs, model experiments | — |
| `paper_figures/` | Scripts to reproduce paper tables and figures | — |

---

## Quick start

- **Encode/decode audio:** See `trilobyte_lossless_codec/` for the CLI.
- **Run FLAC baseline:** See `baselines/flac_eval/`.
- **Run LMIC (in-context) eval:** See `baselines/in_context_eval/`.
- **NAC codec experiments:** See `baselines/nac/`.
- **Reproduce paper figures:** See `paper_figures/`.
