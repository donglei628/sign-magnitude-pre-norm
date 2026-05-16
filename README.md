# Sign-Magnitude Asymmetry in Pre-Norm Transformers

Code accompanying the paper:

> **A Geometric Analysis of Sign-Magnitude Asymmetry in Pre-Norm Transformers under Ternary Quantization**
>
> Dong Lei, 2026
>
> *arXiv link will be added after submission.*

## Overview

This repository provides experiment scripts to reproduce all key results in the paper. The paper establishes that:

1. In a two-layer ReLU + RMSNorm model, sign-flip perturbations produce **pi/(pi-2) ~ 2.75x** more transverse output energy than magnitude perturbations (Theorem 3)
2. Ternary quantization error is a pure magnitude-type perturbation with cos^2 -> 2/pi (Theorem 4)
3. Multi-layer compounding is **falsified**; outlier features are the real amplification mechanism (Exp D)

## Hardware

All experiments were run on:
- **CPU**: Any modern x86-64 CPU (synthetic experiments)
- **GPU**: Single NVIDIA GPU with >= 16 GB VRAM (for real-model experiments; tested on A100/RTX 4090)
- **RAM**: 32 GB

Synthetic experiments (`verify_theorem3.py`, `verify_proposition1.py`, `exp5_multilayer_toy.py`) run on CPU in minutes. Real-model experiments (`dn_outlier.py`, `exp5_trained_weights.py`, `verify_theorem4.py --mode real`) require a GPU.

## Quick Reproduction

```bash
# Install dependencies
pip install -r requirements.txt

# All experiments use 20 seeds by default: [42, 123, 456, ..., 9000].
# To reproduce exact paper numbers, use the default seeds (no --seeds flag needed).

# Section 9.3: Theorem 3 verification (V1 architecture, core result)
python experiments/verify_theorem3.py --mode V1 --output results/theorem3_v1.json

# Section 9.5: Outlier sign-flip (Exp D, requires GPU)
python experiments/dn_outlier.py \
    --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --device cuda --dtype float16 --output results/dn_outlier.json

# Section 9.4: Multi-layer falsification (V2, V3)
python experiments/exp5_multilayer_toy.py --n-seeds 20 --output results/multilayer.json

# Section 9.4: Exp B - Trained weights multi-layer test
python experiments/exp5_trained_weights.py --device cuda --dtype float16 \
    --output results/exp_b_trained.json

# Section 9.6: Theorem 4 synthetic verification
python experiments/verify_theorem4.py --mode synthetic --device cuda \
    --output results/theorem4_synthetic.json

# Section 9.7: Proposition 1 verification
python experiments/verify_proposition1.py --mode synthetic --n-seeds 20 \
    --output results/proposition1.json
```

## Experiment-to-Paper Mapping

| Paper Section | Result | Script | Hardware | Time |
|---|---|---|---|---|
| Section 9.2 (Fact 1) | Cross-model cos^2 | `verify_theorem4.py --mode real` | GPU 16GB | ~10 min/model |
| Section 9.3 (Theorem 3) | V1 c(p) across p={0.01, 0.02, 0.05, 0.10} | `verify_theorem3.py --mode V1` | CPU or GPU | ~5 min |
| Section 9.4 (Multi-layer) | V2/V3 C_L vs L | `exp5_multilayer_toy.py` | CPU | ~15 min |
| Section 9.4 (Exp B) | Trained TinyLlama weights | `exp5_trained_weights.py` | GPU 16GB | ~20 min |
| Section 9.5 (Exp D) | Outlier sign-flip PPL | `dn_outlier.py` | GPU 16GB | ~30 min |
| Section 9.6 (Theorem 4) | Synthetic cos^2 | `verify_theorem4.py --mode synthetic` | CPU or GPU | ~2 min |
| Section 9.7 (Prop 1) | Energy equality + radial fraction | `verify_proposition1.py` | CPU | ~2 min |

## Pre-computed Results

The `results/` directory contains pre-computed outputs from our experiments (20 seeds where applicable). These can be used to verify the numbers reported in the paper without re-running.

| File | Corresponds to |
|---|---|
| `theorem4_verification.json` | Sections 9.2, 9.6 (Fact 1 + Theorem 4) |
| `exp5_multilayer_toy.json` | Section 9.4, Appendix D.1 (V2 multi-layer) |
| `exp5_multilayer_toy_residual.json` | Appendix D.2 (V3 with residual) |
| `exp5_trained_weights.json` | Section 9.4, Exp B (trained TinyLlama layers) |
| `dn_outlier.json` | Section 9.5, Appendix D.3 (outlier sign-flip) |
| `proposition1_verification.json` | Section 9.7 (Proposition 1) |

## Key Theoretical Predictions Verified

| Prediction | Formula | Measured | Deviation | Script |
|---|---|---|---|---|
| Bussgang constant | cos^2(w, sign(w)) -> 2/pi = 0.637 | 0.606-0.633 (5 models) | < 5% | `verify_theorem4.py` |
| Theorem 3 (p=0.01) | c(0.01) = 2.610 | 2.639 +/- 0.012 | +1.1% | `verify_theorem3.py` |
| Theorem 3 (p=0.02) | c(0.02) = 2.527 | 2.583 +/- 0.010 | +2.2% | `verify_theorem3.py` |
| Theorem 3 (p=0.05) | c(0.05) = 2.378 | 2.465 +/- 0.010 | +3.7% | `verify_theorem3.py` |
| Theorem 3 (p=0.10) | c(0.10) = 2.178 | 2.299 +/- 0.008 | +5.6% | `verify_theorem3.py` |
| Theorem 4 (synthetic) | cos^2(Wx, W_T x) -> 2/pi | 0.6367-0.6372 | < 0.1% | `verify_theorem4.py` |
| Multi-layer decline | C_6 < C_1 | 3.066 < 3.587 | -14.5% | `exp5_multilayer_toy.py` |
| Outlier leverage | outlier/non-outlier ratio | 2,180x | -- | `dn_outlier.py` |

## Data and Model Sources

This code uses pre-trained models from public sources:
- **TinyLlama-1.1B**: Apache 2.0 license ([HuggingFace](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T))
- **Qwen2.5 family (0.5B-3B)**: Apache 2.0 license ([HuggingFace](https://huggingface.co/Qwen))
- **WikiText-2**: Creative Commons CC BY-SA 3.0

No model weights are redistributed in this repository. They are downloaded on demand via the `transformers` library.

## Citation

```bibtex
@article{lei2026sign,
  title={A Geometric Analysis of Sign-Magnitude Asymmetry in Pre-Norm Transformers under Ternary Quantization},
  author={Lei, Dong},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
