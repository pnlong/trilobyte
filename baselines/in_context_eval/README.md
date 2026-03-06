# in_context_eval

Language-Modeling-Is-Compression (LMIC) in-context evaluation for audio compression. Evaluates Trilobyte, Llama-2, and FLAC as compressors by measuring bits-per-byte on audio chunks.

---

## Layout

```
in_context_eval/
  language_modeling_is_compression/   # Core eval library
    compress_audio.py                 # Entry point: run compressor on a dataset
    compressors_audio/                # Compressor implementations
      compressor.py                   # Interface and registry
      trilobyte.py                    # Trilobyte with arithmetic coding
      llama.py                        # Llama-2
      flac.py                         # FLAC
      language_model.py               # Generic LM compressor
    data_loaders_audio.py             # Dataset iterators and byte extraction
    constants_audio.py                # Sample rate, bit depth, paths
    utils_audio.py                    # Audio helpers
  lmic_eval.sh                        # Batch runner for compress_audio.py
  lmic_eval_plot.py                   # Plotting for LMIC eval results
  dataset_channels.py                 # Channel/waveform iteration for datasets
  notes.md                            # Miscellaneous notes
```

---

## How to run

**Single evaluation:**
```bash
python language_modeling_is_compression/compress_audio.py \
    --compressor trilobyte \
    --dataset musdb18_mono \
    --output results.csv
```

**Batch evaluation:**
```bash
bash lmic_eval.sh
```

**Plot results:**
```bash
python lmic_eval_plot.py --input results.csv
```

---

## Compressors available

| Compressor | Description |
|---|---|
| `trilobyte` | Trilobyte neural codec with arithmetic coding |
| `llama` | Llama-2 language model |
| `flac` | Standard FLAC baseline |
