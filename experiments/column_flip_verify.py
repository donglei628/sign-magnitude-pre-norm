"""Candidate #2c: Column-flip NLL verification.

Instead of flipping one weight (dNLL ~ 1e-5, below PPL precision),
flip ALL rows in a single column simultaneously. This amplifies the
signal by ~m (number of rows), making dNLL well above the noise floor.

For column j flipped entirely: W[:,j] -> -W[:,j]
  Delta_y = -2 * W[:,j] * x_j
  ||Delta_y||^2 = 4 * ||W[:,j]||^2 * x_j^2

So dNLL should scale as alpha_j^2 * ||W[:,j]||^2.

We select ~20 columns spanning the full alpha distribution and test
whether the measured dNLL follows the predicted alpha^2 scaling.
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
from dn_outlier import evaluate_ppl
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def measure_alpha_all_dims(model, target_name, tokenizer, device,
                           n_sequences=10, seq_len=2048):
    """Measure alpha = sqrt(E[x_hat_j^2]) for every input dimension."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]

    x_sq_accum = []

    def hook_fn(module, inp, out):
        x = inp[0].float()
        x_sq = (x[0] ** 2).detach().cpu()
        x_sq_accum.append(x_sq)

    hook = None
    for name, module in model.named_modules():
        if name == target_name:
            hook = module.register_forward_hook(hook_fn)
            break
    if hook is None:
        raise RuntimeError(f"Module {target_name} not found")

    for i in range(n_sequences):
        start = i * seq_len
        end = start + seq_len
        if end > len(input_ids):
            break
        ids = input_ids[start:end].unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=(device != "cpu")):
            model(ids)

    hook.remove()

    all_x_sq = torch.cat(x_sq_accum, dim=0)
    x_sq_mean = all_x_sq.mean(dim=0).numpy()
    x_norm_sq_mean = float(all_x_sq.sum(dim=-1).mean())
    alpha_sq = x_sq_mean / x_norm_sq_mean
    alpha = np.sqrt(alpha_sq)
    return alpha, alpha_sq, x_norm_sq_mean


@torch.no_grad()
def column_flip_experiment(model, tokenizer, device, target_name, col_idx,
                           baseline_ppl, ln_base, n_samples=10):
    """Flip all rows of column col_idx and measure dNLL."""
    target_module = None
    for name, module in model.named_modules():
        if name == target_name and isinstance(module, torch.nn.Linear):
            target_module = module
            break
    if target_module is None:
        return None

    W = target_module.weight.data
    # Save original column
    original_col = W[:, col_idx].clone()
    # Flip entire column
    W[:, col_idx] = -original_col

    flipped_ppl = evaluate_ppl(model, tokenizer, device, n_samples)

    # Restore
    W[:, col_idx] = original_col

    dNLL = math.log(flipped_ppl) - ln_base
    dPPL = flipped_ppl - baseline_ppl

    return {
        "col_idx": int(col_idx),
        "flipped_ppl": float(flipped_ppl),
        "dPPL": float(dPPL),
        "dNLL": float(dNLL),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Column-flip NLL verification of alpha^2 scaling")
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--target-layer", type=int, default=12)
    parser.add_argument("--n-columns", type=int, default=20,
                        help="Number of columns to test across alpha range")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-sequences", type=int, default=10,
                        help="Sequences for alpha measurement")
    parser.add_argument("--output", type=str,
                        default="docs/column_flip_verify.json")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype, torch.float16)

    print("=" * 70)
    print("Candidate #2c: Column-Flip NLL Verification")
    print(f"  Model: {args.model}")
    print(f"  Target layer: {args.target_layer}")
    print(f"  Columns to test: {args.n_columns}")
    print(f"  PPL samples: {args.n_samples}")
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

    # Find target module
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

    # Get weight column norms
    target_module = dict(model.named_modules())[target_name]
    W = target_module.weight.data.float().cpu()
    m, n = W.shape
    w_col_norm_sq = (W ** 2).sum(dim=0).numpy()
    print(f"  Weight shape: {m} x {n}")

    # Baseline PPL
    print(f"\nEvaluating baseline PPL ({args.n_samples} samples)...")
    baseline_ppl = evaluate_ppl(model, tokenizer, args.device, args.n_samples)
    ln_base = math.log(baseline_ppl)
    print(f"  Baseline PPL: {baseline_ppl:.6f}, ln = {ln_base:.6f}")

    # Measure alpha for all dimensions
    print(f"\nMeasuring alpha ({args.n_sequences} sequences)...")
    alpha, alpha_sq, x_norm_sq = measure_alpha_all_dims(
        model, target_name, tokenizer, args.device, args.n_sequences)
    print(f"  alpha range: [{alpha.min():.6f}, {alpha.max():.6f}]")

    # Select target columns spanning the full alpha range using LOG-SPACED sampling.
    # This ensures adequate coverage of the outlier tail (alpha >> median).
    sorted_idx = np.argsort(alpha)
    n_cols = args.n_columns
    k = max(1, int(n * 0.05))  # top 5% boundary

    # Strategy: sample evenly in log(alpha) space across the FULL range
    log_alpha_min = np.log(alpha[sorted_idx[0]] + 1e-10)
    log_alpha_max = np.log(alpha[sorted_idx[-1]])
    log_targets = np.linspace(log_alpha_min, log_alpha_max, n_cols)
    alpha_targets = np.exp(log_targets)

    target_columns = []
    seen = set()
    for at in alpha_targets:
        # Find the dim with alpha closest to this target
        diffs = np.abs(alpha - at)
        # Mask already-selected dims
        for s in seen:
            diffs[s] = float('inf')
        idx = int(np.argmin(diffs))
        if idx not in seen:
            target_columns.append(idx)
            seen.add(idx)

    # Ensure max outlier is always included
    max_idx = int(sorted_idx[-1])
    if max_idx not in seen:
        target_columns.append(max_idx)
        seen.add(max_idx)

    # Also ensure min alpha reference is included
    min_idx = int(sorted_idx[0])
    if min_idx not in seen:
        target_columns.append(min_idx)
        seen.add(min_idx)

    target_columns.sort(key=lambda j: alpha[j])

    print(f"\n  Selected {len(target_columns)} target columns")
    print(f"  Alpha range of targets: [{alpha[target_columns[0]]:.6f}, "
          f"{alpha[target_columns[-1]]:.6f}]")

    # Run column-flip experiments
    print(f"\n{'='*70}")
    print("Running column-flip experiments...")
    print(f"{'='*70}")
    print(f"  {'#':>3s} {'dim':>5s} {'alpha':>8s} {'alpha^2':>10s} "
          f"{'||W_col||^2':>12s} {'dNLL':>12s} {'PPL':>10s}")
    print("-" * 70)

    results = []
    for ci, col_idx in enumerate(target_columns):
        a = alpha[col_idx]
        a2 = alpha_sq[col_idx]
        wn = w_col_norm_sq[col_idx]

        r = column_flip_experiment(
            model, tokenizer, args.device, target_name, col_idx,
            baseline_ppl, ln_base, args.n_samples)

        if r is not None:
            r["alpha"] = float(a)
            r["alpha_sq"] = float(a2)
            r["w_col_norm_sq"] = float(wn)
            r["predicted_energy"] = float(a2 * wn)
            results.append(r)

            print(f"  {ci+1:>3d} {col_idx:>5d} {a:>8.4f} {a2:>10.6f} "
                  f"{wn:>12.4f} {r['dNLL']:>12.6f} {r['flipped_ppl']:>10.4f}")

        # Save intermediate results after each column
        _save_intermediate(args, results, alpha, alpha_sq, w_col_norm_sq,
                           baseline_ppl, target_name, n_hidden, x_norm_sq,
                           target_columns)

    # Final analysis
    print(f"\n{'='*70}")
    print("Correlation Analysis")
    print(f"{'='*70}")

    dNLL_arr = np.array([r["dNLL"] for r in results])
    alpha_sq_arr = np.array([r["alpha_sq"] for r in results])
    pred_arr = np.array([r["predicted_energy"] for r in results])

    # Filter out any negative dNLL (measurement noise at very low alpha)
    valid = dNLL_arr > 0
    n_valid = valid.sum()
    n_total = len(results)
    print(f"  Valid measurements (dNLL > 0): {n_valid}/{n_total}")

    if n_valid >= 5:
        from scipy.stats import spearmanr, pearsonr

        rho_a2, p_a2 = spearmanr(alpha_sq_arr[valid], dNLL_arr[valid])
        r_a2, pr_a2 = pearsonr(alpha_sq_arr[valid], dNLL_arr[valid])

        rho_pred, p_pred = spearmanr(pred_arr[valid], dNLL_arr[valid])
        r_pred, pr_pred = pearsonr(pred_arr[valid], dNLL_arr[valid])

        print(f"\n  Spearman(alpha^2, dNLL): rho = {rho_a2:.4f}, p = {p_a2:.2e}")
        print(f"  Pearson(alpha^2, dNLL):  r = {r_a2:.4f}, p = {pr_a2:.2e}")
        print(f"\n  Spearman(alpha^2 * ||W||^2, dNLL): rho = {rho_pred:.4f}, p = {p_pred:.2e}")
        print(f"  Pearson(alpha^2 * ||W||^2, dNLL):  r = {r_pred:.4f}, p = {pr_pred:.2e}")

        # Log-log regression: log(dNLL) = a * log(alpha) + b
        # If theory is correct, slope a should be ~2 (alpha^2 scaling)
        log_alpha = np.log(alpha_sq_arr[valid])
        log_dNLL = np.log(dNLL_arr[valid])
        slope, intercept = np.polyfit(log_alpha, log_dNLL, 1)
        print(f"\n  Log-log regression: log(dNLL) = {slope:.3f} * log(alpha^2) + {intercept:.3f}")
        print(f"  Effective scaling exponent: dNLL ~ alpha^({2*slope:.2f})")
        print(f"  (Theory predicts: dNLL ~ alpha^2, i.e., slope = 1.0 for log(alpha^2))")
    else:
        print("  Too few valid measurements for correlation analysis")
        rho_a2 = p_a2 = r_a2 = pr_a2 = None
        rho_pred = p_pred = r_pred = pr_pred = None
        slope = intercept = None

    # Ratio analysis
    print(f"\n{'='*70}")
    print("Ratio Analysis (relative to lowest-alpha column)")
    print(f"{'='*70}")
    if results:
        ref = results[0]  # lowest alpha
        ref_dNLL = ref["dNLL"] if ref["dNLL"] > 0 else 1e-10
        ref_a2 = ref["alpha_sq"]

        print(f"  {'dim':>5s} {'alpha':>8s} {'dNLL':>12s} "
              f"{'dNLL_ratio':>12s} {'alpha^2_ratio':>14s} {'match':>8s}")
        print("-" * 65)
        for r in results:
            dNLL_ratio = r["dNLL"] / ref_dNLL if ref_dNLL > 0 else 0
            a2_ratio = r["alpha_sq"] / ref_a2 if ref_a2 > 0 else 0
            match = dNLL_ratio / a2_ratio if a2_ratio > 0 else 0
            print(f"  {r['col_idx']:>5d} {r['alpha']:>8.4f} {r['dNLL']:>12.6f} "
                  f"{dNLL_ratio:>11.1f}x {a2_ratio:>13.1f}x {match:>7.2f}x")

    # Save final results
    output = {
        "experiment": "column_flip_verification",
        "model": args.model,
        "hidden_dim": n_hidden,
        "target_layer": args.target_layer,
        "target_module": target_name,
        "baseline_ppl": float(baseline_ppl),
        "n_samples": args.n_samples,
        "n_sequences": args.n_sequences,
        "E_x_norm_sq": float(x_norm_sq),
        "n_columns_tested": len(results),
        "results": results,
    }
    if rho_a2 is not None:
        output["correlation"] = {
            "spearman_alpha2_vs_dNLL": {"rho": float(rho_a2), "p": float(p_a2)},
            "pearson_alpha2_vs_dNLL": {"r": float(r_a2), "p": float(pr_a2)},
            "spearman_pred_vs_dNLL": {"rho": float(rho_pred), "p": float(p_pred)},
            "pearson_pred_vs_dNLL": {"r": float(r_pred), "p": float(pr_pred)},
            "log_log_slope": float(slope),
            "effective_exponent": float(2 * slope),
        }
    out_path = os.path.join("c:/source/BitNet", args.output)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


def _save_intermediate(args, results, alpha, alpha_sq, w_col_norm_sq,
                       baseline_ppl, target_name, n_hidden, x_norm_sq,
                       target_columns):
    """Save intermediate results after each column flip."""
    output = {
        "experiment": "column_flip_verification_intermediate",
        "model": args.model,
        "hidden_dim": n_hidden,
        "target_module": target_name,
        "baseline_ppl": float(baseline_ppl),
        "n_completed": len(results),
        "n_total": len(target_columns),
        "results": results,
    }
    out_path = os.path.join("c:/source/BitNet", args.output)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
