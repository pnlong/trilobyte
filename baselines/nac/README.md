# nac

Neural audio codec (NAC) implementations, mix-and-match entropy coder experiments, and tests.

---

## Layout

| Path | Description |
|---|---|
| `lossless_compressors/` | LDAC, LEC, LNAC, and FLAC variant codec implementations |
| `m&m/` | Mix-and-match entropy coder experiments |
| `test_lossless_compressors/` | pytest test suite for lossless codecs |
| `compare_lpc_dac_residuals_distribution.py` | Residual distribution comparison: LPC vs DAC codecs |

---

## `lossless_compressors/`

| File | Description |
|---|---|
| `ldac.py` | Lossless DAC (Descript Audio Codec): codes + Rice-coded residuals for bit-exact decode |
| `lec.py` | Lossless EnCodec (Meta): same pipeline with EnCodec |
| `lnac.py` | Lossless custom NAC: DAC-style model with optional mid/side stereo decorrelation |
| `flacb.py` | FLAC-based codec variant |
| `iflac.py` | Interleaved FLAC variant |
| `nflac.py` | NAC-FLAC hybrid variant |

---

## `test_lossless_compressors/`

pytest suite covering `flac`, `flacb`, `iflac`, `ldac`, `lec`, `lnac`, and `nflac`.

```bash
pytest nac/test_lossless_compressors/
```

---

## `compare_lpc_dac_residuals_distribution.py`

Compares residual distributions of LPC-based (FLAC) vs DAC-based codecs (LDAC, LEC, LNAC). Plots magnitude and log-density of residuals for codec comparison analysis.
