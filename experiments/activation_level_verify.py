"""Candidate #2b: Activation-level single-flip verification.

Instead of measuring dNLL (precision-limited to ~1e-4), directly compute
the perturbation energy ||Δy||² = 4w²x_j² at the layer output.
This is exact (no measurement noise) and tests the α² scaling of R(α,n).

Key quantity: for flip W[i,j] → -W[i,j], the output perturbation is
  Δy = -2 · W[i,j] · x_j · e_i
  ||Δy||² = 4 · W[i,j]² · x_j²

Normalize by weight: ||Δy||² / (4w²) = x_j²
Average over tokens:  E[x_j²] = E[||x||²] · α_j²

So the per-column average perturbation energy scales as α_j².
"""
import argparse
import json
import math
import sys
import os
import numpy as np
import torch

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))
from dn_outlier import get_linear_layers, collect_activation_stats, identify_outliers
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def measure_column_perturbation_energy(model, tokenizer, device, target_name,
                                        n_sequences=10, seq_len=2048):
    """Measure E[x_j²] for every input dimension j of the target layer.

    Returns:
        x_sq_mean: array of shape [n_input], E[x_j²] averaged over tokens
        x_norm_sq_mean: scalar, E[||x||²]
        alpha_sq: array of shape [n_input], E[x̂_j²] = E[x_j²] / E[||x||²]
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]

    # Storage for per-dimension x² and ||x||²
    x_sq_accum = []
    x_norm_sq_accum = []

    def hook_fn(module, inp, out):
        x = inp[0].float()  # [1, seq, hidden]
        x_sq = (x[0] ** 2).detach().cpu()  # [seq, hidden]
        x_norm_sq = x_sq.sum(dim=-1)       # [seq]
        x_sq_accum.append(x_sq)
        x_norm_sq_accum.append(x_norm_sq)

    # Register hook
    hook = None
    for name, module in model.named_modules():
        if name == target_name:
            hook = module.register_forward_hook(hook_fn)
            break

    if hook is None:
        raise RuntimeError(f"Module {target_name} not found")

    # Run forward passes
    for i in range(n_sequences):
        start = i * seq_len
        end = start + seq_len
        if end > len(input_ids):
            break
        ids = input_ids[start:end].unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=(device != "cpu")):
            model(ids)

    hook.remove()

    # Aggregate
    all_x_sq = torch.cat(x_sq_accum, dim=0)       # [total_tokens, hidden]
    all_x_norm_sq = torch.cat(x_norm_sq_accum, dim=0)  # [total_tokens]

    x_sq_mean = all_x_sq.mean(dim=0).numpy()           # [hidden]
    x_norm_sq_mean = float(all_x_norm_sq.mean())        # scalar

    # alpha² = E[x̂_j²] = E[x_j²] / E[||x||²]  (approximately)
    alpha_sq = x_sq_mean / x_norm_sq_mean

    return x_sq_mean, x_norm_sq_mean, alpha_sq


def main():
    parser = argparse.ArgumentParser(
        description="Activation-level perturbation energy verification")
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--target-layer", type=int, default=12)
    parser.add_argument("--n-sequences", type=int, default=10)
    parser.add_argument("--output", type=str,
                        default="docs/activation_level_verify.json")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype, torch.float16)

    print("=" * 70)
    print("Candidate #2b: Activation-Level Perturbation Energy Verification")
    print(f"  Model: {args.model}")
    print(f"  Target layer: {args.target_layer}")
    print(f"  Sequences: {args.n_sequences}")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch_dtype, trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()

    n_hidden = model.config.hidden_size
    print(f"  Hidden dim n = {n_hidden}")

    # Find target module (q_proj at layer N)
    target_name = None
    for name, module in model.named_modules():
        if (f"layers.{args.target_layer}." in name and
                "q_proj" in name and isinstance(module, torch.nn.Linear)):
            target_name = name
            break
    if target_name is None:
        print("ERROR: Could not find target q_proj module")
        return
    print(f"  Target module: {target_name}")

    # Get weight matrix
    target_module = dict(model.named_modules())[target_name]
    W = target_module.weight.data.float().cpu()
    m, n = W.shape
    print(f"  Weight shape: {m} x {n}")

    # Compute weight column norms
    w_col_norm_sq = (W ** 2).sum(dim=0).numpy()  # [n]
    print(f"  Weight column norm range: [{np.sqrt(w_col_norm_sq.min()):.4f}, "
          f"{np.sqrt(w_col_norm_sq.max()):.4f}]")

    # Measure activation statistics
    print(f"\nMeasuring activation statistics ({args.n_sequences} sequences)...")
    x_sq_mean, x_norm_sq_mean, alpha_sq = measure_column_perturbation_energy(
        model, tokenizer, args.device, target_name, args.n_sequences)

    alpha = np.sqrt(alpha_sq)
    print(f"  E[||x||^2] = {x_norm_sq_mean:.2f}")
    print(f"  alpha range: [{alpha.min():.6f}, {alpha.max():.6f}]")
    print(f"  alpha median: {np.median(alpha):.6f}")

    # Per-column perturbation energy (averaged over all rows):
    # E[||Delta_y||^2] per random flip in column j
    #   = E_rows[4 * w_{ij}^2] * E[x_j^2]
    #   = 4 * (||W[:,j]||^2 / m) * E[x_j^2]
    energy_per_col = 4.0 * (w_col_norm_sq / m) * x_sq_mean  # [n]

    # Theoretical prediction: R(alpha_j, n) = n * alpha_j^2 * (1 - 4*alpha_j/(3*pi))
    R_theory = n * alpha_sq * (1.0 - 4.0 * alpha / (3.0 * np.pi))

    # Identify outlier vs non-outlier dims
    sorted_idx = np.argsort(alpha)
    k = max(1, int(n * 0.05))  # top 5%
    outlier_mask = np.zeros(n, dtype=bool)
    outlier_mask[sorted_idx[-k:]] = True

    # Summary statistics
    energy_outlier = energy_per_col[outlier_mask]
    energy_non = energy_per_col[~outlier_mask]

    alpha_outlier = alpha[outlier_mask]
    alpha_non = alpha[~outlier_mask]

    w_norm_outlier = w_col_norm_sq[outlier_mask]
    w_norm_non = w_col_norm_sq[~outlier_mask]

    print(f"\n{'='*60}")
    print("Perturbation Energy Analysis")
    print(f"{'='*60}")
    print(f"  Outlier dims ({k}):  mean energy = {energy_outlier.mean():.6e}")
    print(f"  Non-outlier ({n-k}): mean energy = {energy_non.mean():.6e}")
    print(f"  Energy ratio (outlier/non): {energy_outlier.mean()/energy_non.mean():.2f}x")

    print(f"\n  Decomposition:")
    print(f"    alpha^2 ratio: {alpha_outlier.mean()**2 / alpha_non.mean()**2:.2f}x")
    print(f"    ||W_col||^2 ratio: {w_norm_outlier.mean() / w_norm_non.mean():.4f}x")
    print(f"    Combined prediction: "
          f"{(alpha_outlier.mean()**2 / alpha_non.mean()**2) * (w_norm_outlier.mean() / w_norm_non.mean()):.2f}x")

    # Rank-order test: does alpha_j predict perturbation energy ranking?
    # Spearman correlation between alpha^2 and energy
    from scipy.stats import spearmanr, pearsonr
    rho_energy_alpha, p_energy_alpha = spearmanr(alpha_sq, energy_per_col)
    r_energy_alpha, p_r = pearsonr(alpha_sq, energy_per_col)

    print(f"\n  Rank correlation (Spearman) between alpha^2 and energy: "
          f"rho = {rho_energy_alpha:.4f}, p = {p_energy_alpha:.2e}")
    print(f"  Linear correlation (Pearson): "
          f"r = {r_energy_alpha:.4f}, p = {p_r:.2e}")

    # Also test: does alpha_j^2 * ||W_col||^2 predict energy better?
    predicted_energy = alpha_sq * w_col_norm_sq
    rho_pred, p_pred = spearmanr(predicted_energy, energy_per_col)
    r_pred, p_r_pred = pearsonr(predicted_energy, energy_per_col)
    print(f"\n  Spearman(alpha^2 * ||W_col||^2, energy): "
          f"rho = {rho_pred:.4f}, p = {p_pred:.2e}")
    print(f"  Pearson(alpha^2 * ||W_col||^2, energy): "
          f"r = {r_pred:.4f}, p = {p_r_pred:.2e}")

    # Specific columns comparison (same as candidate #2)
    max_idx = int(sorted_idx[-1])
    p90_pos = int(k * 0.9)
    p90_idx = int(sorted_idx[-k:][p90_pos]) if p90_pos < k else max_idx
    med_pos = int(k * 0.5)
    med_idx = int(sorted_idx[-k:][med_pos])
    non_idx = int(sorted_idx[n // 2])

    target_columns = [
        ("max_outlier", max_idx),
        ("P90_outlier", p90_idx),
        ("median_outlier", med_idx),
        ("non_outlier_ref", non_idx),
    ]

    print(f"\n{'='*60}")
    print("Per-Column Detail (same 4 targets as Candidate #2)")
    print(f"{'='*60}")
    print(f"  {'Label':<18s} {'dim':>5s} {'alpha':>8s} {'alpha^2':>10s} "
          f"{'||W_col||^2':>12s} {'energy':>12s} {'energy_ratio':>12s}")
    print("-" * 85)

    ref_energy = energy_per_col[non_idx]
    ref_alpha_sq = alpha_sq[non_idx]

    for label, col_idx in target_columns:
        a = alpha[col_idx]
        a2 = alpha_sq[col_idx]
        wn = w_col_norm_sq[col_idx]
        e = energy_per_col[col_idx]
        ratio = e / ref_energy if ref_energy > 0 else float('inf')
        print(f"  {label:<18s} {col_idx:>5d} {a:>8.4f} {a2:>10.6f} "
              f"{wn:>12.4f} {e:>12.6e} {ratio:>11.1f}x")

    print(f"\n  Predicted ratios from alpha^2:")
    for label, col_idx in target_columns:
        a2_ratio = alpha_sq[col_idx] / ref_alpha_sq if ref_alpha_sq > 0 else float('inf')
        e_ratio = energy_per_col[col_idx] / ref_energy if ref_energy > 0 else float('inf')
        print(f"    {label:<18s}  alpha^2 ratio = {a2_ratio:>8.1f}x  "
              f"energy ratio = {e_ratio:>8.1f}x  "
              f"match = {e_ratio/a2_ratio if a2_ratio > 0 else float('inf'):>6.2f}x")

    # Quantile analysis: group dims into deciles by alpha, measure mean energy
    print(f"\n{'='*60}")
    print("Quantile Analysis: Energy vs Alpha Deciles")
    print(f"{'='*60}")
    deciles = np.array_split(sorted_idx, 10)
    print(f"  {'Decile':>8s} {'alpha_range':>20s} {'mean_energy':>14s} "
          f"{'energy_ratio':>12s} {'alpha^2_ratio':>14s}")
    print("-" * 75)

    ref_decile_energy = None
    ref_decile_alpha2 = None
    for di, d_idx in enumerate(deciles):
        d_alpha = alpha[d_idx]
        d_energy = energy_per_col[d_idx]
        d_alpha2 = alpha_sq[d_idx]
        if di == 0:
            ref_decile_energy = d_energy.mean()
            ref_decile_alpha2 = d_alpha2.mean()
        e_ratio = d_energy.mean() / ref_decile_energy
        a2_ratio = d_alpha2.mean() / ref_decile_alpha2
        print(f"  D{di:>2d}({len(d_idx):>3d}) [{d_alpha.min():.4f}, {d_alpha.max():.4f}] "
              f"{d_energy.mean():>14.6e} {e_ratio:>11.1f}x {a2_ratio:>13.1f}x")

    # Save results
    output = {
        "experiment": "activation_level_verification",
        "model": args.model,
        "hidden_dim": n_hidden,
        "target_layer": args.target_layer,
        "target_module": target_name,
        "n_sequences": args.n_sequences,
        "E_x_norm_sq": float(x_norm_sq_mean),
        "correlation": {
            "spearman_alpha2_vs_energy": {
                "rho": float(rho_energy_alpha),
                "p_value": float(p_energy_alpha),
            },
            "pearson_alpha2_vs_energy": {
                "r": float(r_energy_alpha),
                "p_value": float(p_r),
            },
            "spearman_predicted_vs_energy": {
                "rho": float(rho_pred),
                "p_value": float(p_pred),
            },
            "pearson_predicted_vs_energy": {
                "r": float(r_pred),
                "p_value": float(p_r_pred),
            },
        },
        "outlier_vs_nonoutlier": {
            "energy_ratio": float(energy_outlier.mean() / energy_non.mean()),
            "alpha_sq_ratio": float(alpha_outlier.mean()**2 / alpha_non.mean()**2),
            "w_col_norm_ratio": float(w_norm_outlier.mean() / w_norm_non.mean()),
        },
        "target_columns": {
            label: {
                "dim": int(col_idx),
                "alpha": float(alpha[col_idx]),
                "alpha_sq": float(alpha_sq[col_idx]),
                "w_col_norm_sq": float(w_col_norm_sq[col_idx]),
                "energy": float(energy_per_col[col_idx]),
                "energy_ratio": float(energy_per_col[col_idx] / ref_energy),
                "alpha_sq_ratio": float(alpha_sq[col_idx] / ref_alpha_sq),
            }
            for label, col_idx in target_columns
        },
    }
    out_path = os.path.join("c:/source/BitNet", args.output)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
