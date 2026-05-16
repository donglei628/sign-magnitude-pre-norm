"""
Theorem 4 Direct Verification: Ternary Quantization Angular Error
=================================================================
Verifies two predictions of Theorem 4:

  (a) Row-level: cos2(w_i, sign(w_i)) → 2/π  for each row of W
  (b) Vector-level: cos2(Wx, W_T x) → 2/π  for uniform sphere inputs x_hat

where W_T = s·sign(W), s = ||W||_1 / n (ternary quantization).

The row-level prediction is already partially tested in Exp 2; this script
focuses on the vector-level prediction which is the key untested claim.

Usage:
  # Single model
  python utils/verify_theorem4.py --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
      --device cuda --dtype float16

  # Multiple models (specify one at a time, results accumulate)
  python utils/verify_theorem4.py --model Qwen/Qwen2.5-1.5B \
      --device cuda --dtype float16 --output docs/theorem4_verification.json

  # Synthetic verification (no model needed)
  python utils/verify_theorem4.py --mode synthetic --device cuda
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Any, List

import torch
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

TWO_OVER_PI = 2.0 / math.pi  # 0.63662...

SEED_LIST = [
    42, 123, 456, 789, 1000,
    1234, 2000, 2345, 3000, 3456,
    4000, 4567, 5000, 5678, 6000,
    6789, 7000, 7890, 8000, 9000,
]


def sample_sphere(n: int, k: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Sample k vectors uniformly on S^{n-1}."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(k, n, generator=g)
    x = x / x.norm(dim=1, keepdim=True)
    return x.to(device)


@torch.no_grad()
def measure_row_cos2(W: torch.Tensor) -> Dict[str, float]:
    """Row-level cos2(w_i, sign(w_i)) for each row of W."""
    # W_T = s * sign(W), s = ||w_i||_1 / n per row
    n = W.shape[1]
    s = W.abs().sum(dim=1, keepdim=True) / n  # [m, 1]
    W_T = s * torch.sign(W)  # [m, n]

    # cos2(w_i, W_T_i) for each row
    dot = (W * W_T).sum(dim=1)  # [m]
    norm_w = W.norm(dim=1)  # [m]
    norm_wt = W_T.norm(dim=1)  # [m]

    cos2 = (dot / (norm_w * norm_wt + 1e-30)) ** 2  # [m]

    return {
        "mean": cos2.mean().item(),
        "std": cos2.std().item(),
        "min": cos2.min().item(),
        "max": cos2.max().item(),
        "theory": TWO_OVER_PI,
        "rel_error": abs(cos2.mean().item() - TWO_OVER_PI) / TWO_OVER_PI,
    }


@torch.no_grad()
def measure_vector_cos2(W: torch.Tensor, n_sphere: int, seed: int,
                        device: str, batch_size: int = 2000) -> Dict[str, float]:
    """Vector-level cos2(Wx, W_T x) for uniform sphere inputs."""
    m, n = W.shape
    W_dev = W.to(device)

    # Ternary quantization: W_T = s * sign(W), s = ||W||_1 / (m*n)
    # Per-row scale: s_i = ||w_i||_1 / n
    s = W_dev.abs().sum(dim=1, keepdim=True) / n  # [m, 1]
    W_T = s * torch.sign(W_dev)  # [m, n]

    cos2_all = []

    for b_start in range(0, n_sphere, batch_size):
        k = min(batch_size, n_sphere - b_start)
        X = sample_sphere(n, k, seed * 100000 + b_start, device)  # [k, n]

        Y = W_dev @ X.T     # [m, k]
        Y_T = W_T @ X.T     # [m, k]

        # cos2(Y[:,j], Y_T[:,j]) for each sphere sample j
        dot = (Y * Y_T).sum(dim=0)  # [k]
        norm_y = Y.norm(dim=0)      # [k]
        norm_yt = Y_T.norm(dim=0)   # [k]

        cos2 = (dot / (norm_y * norm_yt + 1e-30)) ** 2  # [k]
        cos2_all.append(cos2.cpu())

    cos2_all = torch.cat(cos2_all)
    return {
        "mean": cos2_all.mean().item(),
        "std": cos2_all.std().item(),
        "min": cos2_all.min().item(),
        "max": cos2_all.max().item(),
        "theory": TWO_OVER_PI,
        "rel_error": abs(cos2_all.mean().item() - TWO_OVER_PI) / TWO_OVER_PI,
        "n_samples": cos2_all.numel(),
    }


def run_synthetic(dims: list, n_seeds: int, n_sphere: int,
                  device: str) -> Dict[str, Any]:
    """Verify Theorem 4 on random Gaussian matrices."""
    results = {}

    for m, n in dims:
        key = f"{m}x{n}"
        print(f"\n  Synthetic {key}...")
        row_cos2_list = []
        vec_cos2_list = []

        for si, seed in enumerate(SEED_LIST[:n_seeds]):
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            W = torch.randn(m, n, generator=g) / math.sqrt(n)

            row_result = measure_row_cos2(W.to(device))
            vec_result = measure_vector_cos2(W, n_sphere, seed, device)

            row_cos2_list.append(row_result["mean"])
            vec_cos2_list.append(vec_result["mean"])

            print(f"    seed={seed}: row cos2={row_result['mean']:.6f}, "
                  f"vec cos2={vec_result['mean']:.6f}  "
                  f"(theory={TWO_OVER_PI:.6f})")

        results[key] = {
            "m": m, "n": n,
            "row_cos2_mean": float(np.mean(row_cos2_list)),
            "row_cos2_std": float(np.std(row_cos2_list)),
            "vec_cos2_mean": float(np.mean(vec_cos2_list)),
            "vec_cos2_std": float(np.std(vec_cos2_list)),
            "theory": TWO_OVER_PI,
            "row_rel_error": abs(np.mean(row_cos2_list) - TWO_OVER_PI) / TWO_OVER_PI,
            "vec_rel_error": abs(np.mean(vec_cos2_list) - TWO_OVER_PI) / TWO_OVER_PI,
        }

    return results


def run_real_model(model_name: str, device: str, dtype: str,
                   n_sphere: int, max_layers: int = 20) -> Dict[str, Any]:
    """Verify Theorem 4 on real model weights."""
    from transformers import AutoModelForCausalLM

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)

    print(f"\n  Loading model: {model_name} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Extract linear layers (excluding embeddings/lm_head)
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "embed" in name.lower() or "lm_head" in name.lower():
                continue
            layers.append((name, module))

    if max_layers and len(layers) > max_layers:
        # Sample evenly across layers
        indices = np.linspace(0, len(layers) - 1, max_layers, dtype=int)
        layers = [layers[i] for i in indices]

    print(f"  Measuring {len(layers)} layers...")

    layer_results = []
    row_cos2_global = []
    vec_cos2_global = []

    for li, (name, module) in enumerate(layers):
        W = module.weight.data.float()  # always measure in float32
        m, n = W.shape

        row_result = measure_row_cos2(W.to(device))
        vec_result = measure_vector_cos2(W, n_sphere, 42, device)

        row_cos2_global.append(row_result["mean"])
        vec_cos2_global.append(vec_result["mean"])

        layer_results.append({
            "name": name,
            "shape": [m, n],
            "row_cos2": row_result,
            "vec_cos2": vec_result,
        })

        print(f"    [{li+1}/{len(layers)}] {name} ({m}x{n}): "
              f"row={row_result['mean']:.6f}, vec={vec_result['mean']:.6f} "
              f"(2/pi={TWO_OVER_PI:.6f})")

    # Clean up
    del model
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "n_layers_measured": len(layers),
        "layers": layer_results,
        "summary": {
            "row_cos2_mean": float(np.mean(row_cos2_global)),
            "row_cos2_std": float(np.std(row_cos2_global)),
            "vec_cos2_mean": float(np.mean(vec_cos2_global)),
            "vec_cos2_std": float(np.std(vec_cos2_global)),
            "theory": TWO_OVER_PI,
            "row_rel_error": abs(np.mean(row_cos2_global) - TWO_OVER_PI) / TWO_OVER_PI,
            "vec_rel_error": abs(np.mean(vec_cos2_global) - TWO_OVER_PI) / TWO_OVER_PI,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Theorem 4 verification: ternary quantization cos2 → 2/π")
    parser.add_argument("--mode", choices=["synthetic", "real", "both"],
                        default="both")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID (required for real/both mode)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds for synthetic experiments")
    parser.add_argument("--n-sphere", type=int, default=10000,
                        help="Number of sphere samples per measurement")
    parser.add_argument("--max-layers", type=int, default=20,
                        help="Max layers to measure per model")
    parser.add_argument("--output", type=str,
                        default="docs/theorem4_verification.json")
    args = parser.parse_args()

    print("=" * 70)
    print("Theorem 4 Verification: cos2(Wx, W_T x) → 2/π")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  Theory: 2/π = {TWO_OVER_PI:.8f}")
    print("=" * 70)

    data = {}

    # Load existing results if present (for accumulating multiple models)
    if os.path.exists(args.output):
        with open(args.output) as f:
            data = json.load(f)
        print(f"\nLoaded existing results from {args.output}")

    if args.mode in ("synthetic", "both"):
        print("\n--- Synthetic Verification ---")
        dims = [(256, 512), (512, 1024), (1024, 2048), (2048, 4096)]
        data["synthetic"] = run_synthetic(dims, args.n_seeds, args.n_sphere,
                                          args.device)

    if args.mode in ("real", "both"):
        if args.model is None:
            print("ERROR: --model required for real/both mode")
            sys.exit(1)
        print(f"\n--- Real Model Verification: {args.model} ---")
        model_key = args.model.replace("/", "_")
        if "real_models" not in data:
            data["real_models"] = {}
        data["real_models"][model_key] = run_real_model(
            args.model, args.device, args.dtype,
            args.n_sphere, args.max_layers)

    # Save
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Source':<30} {'Row cos2':>12} {'Vec cos2':>12} {'Theory':>10} {'Vec Err%':>10}")
    print("-" * 74)

    if "synthetic" in data:
        for key, v in data["synthetic"].items():
            print(f"  Synthetic {key:<19} {v['row_cos2_mean']:>12.6f} "
                  f"{v['vec_cos2_mean']:>12.6f} {TWO_OVER_PI:>10.6f} "
                  f"{v['vec_rel_error']*100:>9.2f}%")

    if "real_models" in data:
        for model_key, v in data["real_models"].items():
            s = v["summary"]
            print(f"  {v['model']:<28} {s['row_cos2_mean']:>12.6f} "
                  f"{s['vec_cos2_mean']:>12.6f} {TWO_OVER_PI:>10.6f} "
                  f"{s['vec_rel_error']*100:>9.2f}%")

    print(f"\nTheory prediction: 2/π = {TWO_OVER_PI:.8f}")
    print("Done.")


if __name__ == "__main__":
    main()
