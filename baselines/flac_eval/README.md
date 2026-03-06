# flac_eval

FLAC compression evaluation across audio datasets.

---

## Files

| File | Description |
|---|---|
| `flac_eval.py` | Evaluates FLAC compression across all datasets; outputs CSV with sizes, compression rate, and duration |
| `flac_eval.sh` | Batch runner: iterates over FLAC levels and dataset/machine combinations |
| `flac_eval_plot.py` | Plots compression rate vs FLAC level, broken down by dataset |
| `flac_analysis.py` | Additional analysis on FLAC eval results |
| `flac_eval_results.csv` | Output CSV from `flac_eval.py` (generated at runtime) |

---

## How to run

**Single eval:**
```bash
python flac_eval.py --level 5 --dataset musdb18_mono --output flac_eval_results.csv
```

**Batch eval (all levels and datasets):**
```bash
bash flac_eval.sh
```

**Plot results:**
```bash
python flac_eval_plot.py --input flac_eval_results.csv
```

---

## Datasets supported

MusDB18 (mono/stereo), LibriSpeech, LJSpeech, Epidemic Sound, VCTK, Torrent (16/24-bit), Birdvox, Beethoven, YouTube Mix, SC09.
