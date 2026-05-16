# Sign-Magnitude Asymmetry in Pre-Norm Transformers

Code accompanying the paper:

> **A Geometric Analysis of Sign-Magnitude Asymmetry in Pre-Norm Transformers under Ternary Quantization**
>
> Dong Lei, 2026
>
> arXiv:XXXX.XXXXX

## Overview

This repository provides experiment scripts to reproduce all key results in the paper. The paper establishes that:

1. In a two-layer ReLU + RMSNorm model, sign-flip perturbations produce **pi/(pi-2) ~ 2.75x** more transverse output energy than magnitude perturbations (Theorem 3)
2. Ternary quantization error is a pure magnitude-type perturbation with cos^2 -> 2/pi (Theorem 4)
3. Multi-layer compounding is **falsified**; outlier features are the real amplification mechanism (Exp D)

## Quick Reproduction

```bash
# Install dependencies
pip install -r requirements.txt

# Reproduce Table 2 (Section 9.3): Theorem 3 verification (V1 architecture)
# Runs in ~5 min on CPU, ~1 min on GPU
python experiments/verify_theorem3.py --mode V1 --output results/theorem3_v1.json

# Reproduce Table 5 (Section 9.5): Outlier sign-flip (Exp D)
# Requires GPU + ~16GB VRAM for TinyLlama
python experiments/dn_outlier.py \
    --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --device cuda --dtype float16 --output results/dn_outlier.json

# Reproduce Table 3 (Section 9.4): Multi-layer falsification (V2, V3)
python experiments/exp5_multilayer_toy.py --n-seeds 20 --output results/multilayer.json

# Reproduce Exp B: Trained weights multi-layer test
python experiments/exp5_trained_weights.py --device cuda --dtype float16 \
    --output results/exp_b_trained.json

# Reproduce Table 6 (Section 9.6): Theorem 4 synthetic verification
python experiments/verify_theorem4.py --mode synthetic --device cuda \
    --output results/theorem4_synthetic.json

# Reproduce Table 7 (Section 9.7): Proposition 1 verification
python experiments/verify_proposition1.py --mode synthetic --n-seeds 20 \
    --output results/proposition1.json
```

## Experiment-to-Paper Mapping

| Paper Section | Table/Result | Script | Hardware | Time |
|---|---|---|---|---|
| Section 9.2 (Fact 1) | Table 1: Cross-model cos^2 | `verify_theorem4.py --mode real` | GPU 16GB | ~10 min/model |
| Section 9.3 (Theorem 3) | Table 2: V1 c(p) values | `verify_theorem3.py --mode V1` | CPU or GPU | ~5 min |
| Section 9.4 (Multi-layer) | Table 3: V2/V3/Exp B | `exp5_multilayer_toy.py` | CPU | ~15 min |
| Section 9.4 (Exp B) | Trained weights | `exp5_trained_weights.py` | GPU 16GB | ~20 min |
| Section 9.5 (Exp D) | Table 5: Outlier sign-flip | `dn_outlier.py` | GPU 16GB | ~30 min |
| Section 9.6 (Theorem 4) | Table 6: Synthetic cos^2 | `verify_theorem4.py --mode synthetic` | CPU or GPU | ~2 min |
| Section 9.7 (Prop 1) | Table 7: Energy equality | `verify_proposition1.py` | CPU | ~2 min |

## Pre-computed Results

The `results/` directory contains pre-computed outputs from our experiments (20 seeds where applicable). These can be used to verify the numbers reported in the paper without re-running.

| File | Corresponds to |
|---|---|
| `theorem4_verification.json` | Tables 1, 6 (Fact 1 + Theorem 4) |
| `exp5_multilayer_toy.json` | Table 3, Appendix D.1 (V2 multi-layer) |
| `exp5_multilayer_toy_residual.json` | Appendix D.2 (V3 with residual) |
| `exp5_trained_weights.json` | Exp B (trained TinyLlama layers) |
| `dn_outlier.json` | Table 5, Appendix D.3 (outlier sign-flip) |
| `proposition1_verification.json` | Table 7 (Proposition 1) |

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy
- transformers (for real-model experiments only)
- datasets (for perplexity evaluation only)

## Key Theoretical Predictions Verified

| Prediction | Formula | Measured | Script |
|---|---|---|---|
| Bussgang constant | cos^2(w, sign(w)) -> 2/pi = 0.637 | 0.618-0.633 | `verify_theorem4.py` |
| Theorem 3 (p=0.01) | c(0.01) = 2.610 | 2.639 +/- 0.012 | `verify_theorem3.py` |
| Theorem 4 (synthetic) | cos^2(Wx, W_T x) -> 2/pi | 0.6367-0.6372 | `verify_theorem4.py` |
| Multi-layer decline | C_6 < C_1 | 3.066 < 3.587 | `exp5_multilayer_toy.py` |
| Outlier leverage | outlier/non-outlier ratio | 2,180x | `dn_outlier.py` |

## Citation

```bibtex
@article{lei2026sign,
  title={A Geometric Analysis of Sign-Magnitude Asymmetry in Pre-Norm Transformers under Ternary Quantization},
  author={Lei, Dong},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
