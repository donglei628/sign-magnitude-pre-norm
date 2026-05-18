"""
Exp D: Outlier Features and Sign Sensitivity
=============================================
Tests whether sign-flips targeting outlier-feature-connected weights
are more damaging than random sign-flips.

Protocol:
  1. Run calibration data through TinyLlama, collect per-dimension
     activation magnitudes at each layer's input (post-RMSNorm)
  2. Identify top-k% "outlier dimensions" by mean absolute activation
  3. Create two perturbation sets (Frobenius-matched):
     - Outlier-targeted: sign-flip ONLY weights in columns connecting
       to outlier dimensions
     - Non-outlier: sign-flip ONLY weights in non-outlier columns
  4. Compare PPL impact: if sign sensitivity is amplified by outliers,
     outlier-targeted sign-flips should cause much more damage

Usage:
  python utils/dn_outlier.py --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
      --device cuda --dtype float16 --output docs/dn_outlier.json
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

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed")
    sys.exit(1)

SEED_LIST = [42, 123, 456, 789, 1000, 1234, 2024, 3141, 4567, 5678,
             6789, 7890, 8901, 9012, 1111, 2222, 3333, 4444, 5555, 6666]


def get_linear_layers(model):
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:
        Conv1D = None
    layers = []
    for name, module in model.named_modules():
        is_linear = isinstance(module, torch.nn.Linear)
        is_conv1d = Conv1D is not None and isinstance(module, Conv1D)
        if is_linear or is_conv1d:
            if "embed" in name.lower() or "lm_head" in name.lower():
                continue
            layers.append((name, module))
    return layers


@torch.no_grad()
def evaluate_ppl(model, tokenizer, device, n_samples=10, seq_len=2048):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]

    nlls = []
    n_tokens = 0
    for i in range(n_samples):
        start = i * seq_len
        end = start + seq_len
        if end > len(input_ids):
            break
        ids = input_ids[start:end].unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=(device != "cpu")):
            out = model(ids, labels=ids)
        n_pred = ids.shape[1] - 1
        nlls.append(out.loss.float().item() * n_pred)
        n_tokens += n_pred
    return math.exp(sum(nlls) / n_tokens) if n_tokens > 0 else float("inf")


@torch.no_grad()
def collect_activation_stats(model, tokenizer, device, n_samples=5, seq_len=2048):
    """Collect per-dimension activation magnitude statistics at each layer's input.

    Returns dict: layer_name -> activation_stats tensor [hidden_dim]
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]

    # Register hooks to collect activation statistics
    stats = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0]  # [batch, seq_len, hidden_dim]
            # Mean absolute value per dimension across batch and seq
            abs_mean = x.float().abs().mean(dim=(0, 1)).cpu()  # [hidden_dim]
            if name not in stats:
                stats[name] = {"sum": torch.zeros_like(abs_mean), "count": 0}
            stats[name]["sum"] += abs_mean
            stats[name]["count"] += 1
        return hook_fn

    # Register hooks on all linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "embed" in name.lower() or "lm_head" in name.lower():
                continue
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # Forward pass on calibration data
    for i in range(n_samples):
        start = i * seq_len
        end = start + seq_len
        if end > len(input_ids):
            break
        ids = input_ids[start:end].unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', enabled=(device != "cpu")):
            model(ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute mean activation per dimension
    result = {}
    for name, s in stats.items():
        if s["count"] > 0:
            result[name] = (s["sum"] / s["count"]).numpy()

    return result


def identify_outliers(act_stats, top_pct=0.05):
    """Identify top-k% outlier dimensions based on activation magnitude.

    Returns dict: layer_name -> set of outlier dimension indices
    """
    outliers = {}
    for name, stats in act_stats.items():
        n_dims = len(stats)
        k = max(1, int(n_dims * top_pct))
        top_indices = np.argsort(stats)[-k:]
        outliers[name] = set(top_indices.tolist())
    return outliers


def make_targeted_sign_deltas(model, outlier_dims, flip_frac, seed,
                               target="outlier"):
    """Create sign-flip perturbation targeting specific dimensions.

    target="outlier": flip only columns connected to outlier dimensions
    target="non_outlier": flip only columns connected to non-outlier dimensions
    target="random": standard random flip (baseline)
    """
    all_layers = get_linear_layers(model)
    deltas = []
    total_flipped = 0
    total_elements = 0

    for name, module in all_layers:
        W = module.weight.data
        m, n = W.shape

        if target == "random":
            torch.manual_seed(seed)
            mask = torch.rand_like(W) < flip_frac
        else:
            # Get outlier dims for this layer's input
            outlier_set = outlier_dims.get(name, set())
            mask = torch.zeros_like(W, dtype=torch.bool)

            if target == "outlier":
                # Only flip in outlier columns
                for dim in outlier_set:
                    if dim < n:
                        torch.manual_seed(seed + dim)
                        col_mask = torch.rand(m, device=W.device) < flip_frac
                        mask[:, dim] = col_mask
            elif target == "non_outlier":
                # Only flip in non-outlier columns
                non_outlier = [j for j in range(n) if j not in outlier_set]
                for dim in non_outlier:
                    torch.manual_seed(seed + dim)
                    col_mask = torch.rand(m, device=W.device) < flip_frac
                    mask[:, dim] = col_mask

        delta = -2 * W * mask.float()
        deltas.append(delta.cpu())
        total_flipped += mask.sum().item()
        total_elements += W.numel()

    return deltas, total_flipped, total_elements


def make_count_matched_deltas(model, outlier_dims, n_flips, seed,
                               target="outlier"):
    """Create sign-flip perturbation with a fixed count of flips.

    Samples exactly n_flips weights from the specified pool (outlier or
    non-outlier columns) and flips their signs. No Frobenius rescaling.
    This enables a clean per-flip leverage comparison.

    Uses per-layer proportional allocation to avoid building the full
    candidate list in memory.
    """
    all_layers = get_linear_layers(model)
    rng = np.random.RandomState(seed)

    # Step 1: Count pool size per layer
    layer_pool_sizes = []
    for name, module in all_layers:
        W = module.weight.data
        m, n = W.shape
        outlier_set = outlier_dims.get(name, set())
        if target == "outlier":
            n_cols = sum(1 for j in range(n) if j in outlier_set)
        else:
            n_cols = sum(1 for j in range(n) if j not in outlier_set)
        layer_pool_sizes.append(m * n_cols)

    total_pool = sum(layer_pool_sizes)
    actual_flips = min(n_flips, total_pool)

    # Step 2: Allocate flips to layers proportionally
    layer_flips = []
    remaining = actual_flips
    for i, size in enumerate(layer_pool_sizes):
        if i == len(layer_pool_sizes) - 1:
            layer_flips.append(remaining)
        else:
            alloc = int(round(actual_flips * size / total_pool))
            alloc = min(alloc, remaining, size)
            layer_flips.append(alloc)
            remaining -= alloc

    # Step 3: For each layer, sample positions within the pool
    deltas = []
    total_flipped = 0
    for li, (name, module) in enumerate(all_layers):
        W = module.weight.data
        m, n = W.shape
        outlier_set = outlier_dims.get(name, set())

        if target == "outlier":
            pool_cols = [j for j in range(n) if j in outlier_set]
        else:
            pool_cols = [j for j in range(n) if j not in outlier_set]

        n_to_flip = layer_flips[li]
        pool_size = m * len(pool_cols)

        delta = torch.zeros_like(W.cpu())
        if n_to_flip > 0 and pool_size > 0:
            # Sample flat indices within the pool
            flat_idx = rng.choice(pool_size, size=min(n_to_flip, pool_size),
                                  replace=False)
            n_pool_cols = len(pool_cols)
            for idx in flat_idx:
                row = idx // n_pool_cols
                col_local = idx % n_pool_cols
                col = pool_cols[col_local]
                delta[row, col] = -2.0 * W[row, col].item()
            total_flipped += len(flat_idx)

        deltas.append(delta)

    frob = math.sqrt(sum(d.pow(2).sum().item() for d in deltas))
    return deltas, total_flipped, frob


def frobenius_rescale_deltas(deltas, target_norm):
    """Rescale deltas to match target Frobenius norm."""
    current_norm = math.sqrt(sum(d.pow(2).sum().item() for d in deltas))
    if current_norm > 0:
        scale = target_norm / current_norm
        return [d * scale for d in deltas], current_norm
    return deltas, 0.0


def apply_deltas(model, deltas):
    for (_, module), delta in zip(get_linear_layers(model), deltas):
        module.weight.data.add_(delta.to(module.weight.device))


def unapply_deltas(model, deltas):
    for (_, module), delta in zip(get_linear_layers(model), deltas):
        module.weight.data.sub_(delta.to(module.weight.device))


def compute_frob_norm(deltas):
    return math.sqrt(sum(d.pow(2).sum().item() for d in deltas))


def main():
    parser = argparse.ArgumentParser(
        description="Outlier features and sign sensitivity")
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--flip-rate", type=float, default=0.05)
    parser.add_argument("--outlier-pct", type=float, default=0.05,
                        help="Top percentage of dims to consider outliers")
    parser.add_argument("--output", type=str, default="docs/dn_outlier.json")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype, torch.float16)

    print("=" * 70)
    print("Exp D: Outlier Features and Sign Sensitivity")
    print(f"  Model: {args.model}")
    print(f"  Flip rate: {args.flip_rate}")
    print(f"  Outlier top %: {args.outlier_pct*100:.0f}%")
    print(f"  Seeds: {args.n_seeds}")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {args.model} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()

    n_layers = len(get_linear_layers(model))
    print(f"  {n_layers} linear layers")

    # Baseline PPL
    print("\nEvaluating baseline PPL...")
    baseline_ppl = evaluate_ppl(model, tokenizer, args.device, args.n_samples)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Collect activation statistics
    print("\nCollecting activation statistics...")
    t0 = time.time()
    act_stats = collect_activation_stats(model, tokenizer, args.device,
                                          n_samples=5)
    print(f"  Collected stats for {len(act_stats)} layers ({time.time()-t0:.0f}s)")

    # Show outlier statistics
    print("\nOutlier analysis (top dims by activation magnitude):")
    for name in sorted(act_stats.keys())[:5]:
        stats = act_stats[name]
        top5 = np.argsort(stats)[-5:][::-1]
        top5_vals = stats[top5]
        mean_val = np.mean(stats)
        max_val = np.max(stats)
        ratio = max_val / (mean_val + 1e-10)
        print(f"  {name}: max/mean={ratio:.1f}x, "
              f"top dims={top5.tolist()}, "
              f"top vals={[f'{v:.3f}' for v in top5_vals]}")
    print(f"  ... ({len(act_stats)} layers total)")

    # Identify outlier dimensions
    outlier_dims = identify_outliers(act_stats, args.outlier_pct)

    # Run experiments for each seed
    seeds = SEED_LIST[:args.n_seeds]
    results = []

    for si, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Seed {si+1}/{len(seeds)}: {seed}")
        print(f"{'='*60}")

        # 1. Random sign-flip (baseline)
        random_deltas, n_flipped_r, n_total = make_targeted_sign_deltas(
            model, outlier_dims, args.flip_rate, seed, target="random")
        random_norm = compute_frob_norm(random_deltas)

        apply_deltas(model, random_deltas)
        random_ppl = evaluate_ppl(model, tokenizer, args.device, args.n_samples)
        unapply_deltas(model, random_deltas)
        print(f"  Random sign-flip: PPL={random_ppl:.2f}, "
              f"||dW||={random_norm:.1f}, "
              f"flipped={n_flipped_r}/{n_total}")

        # 2. Outlier-targeted sign-flip
        outlier_deltas, n_flipped_o, _ = make_targeted_sign_deltas(
            model, outlier_dims, args.flip_rate, seed, target="outlier")
        # Rescale to match random norm
        outlier_deltas, outlier_orig_norm = frobenius_rescale_deltas(
            outlier_deltas, random_norm)

        apply_deltas(model, outlier_deltas)
        outlier_ppl = evaluate_ppl(model, tokenizer, args.device, args.n_samples)
        unapply_deltas(model, outlier_deltas)
        print(f"  Outlier-targeted: PPL={outlier_ppl:.2f}, "
              f"||dW||={compute_frob_norm(outlier_deltas):.1f} "
              f"(orig {outlier_orig_norm:.1f}), "
              f"flipped={n_flipped_o}")

        # 3. Non-outlier sign-flip
        nonoutlier_deltas, n_flipped_n, _ = make_targeted_sign_deltas(
            model, outlier_dims, args.flip_rate, seed, target="non_outlier")
        nonoutlier_deltas, nonoutlier_orig_norm = frobenius_rescale_deltas(
            nonoutlier_deltas, random_norm)

        apply_deltas(model, nonoutlier_deltas)
        nonoutlier_ppl = evaluate_ppl(model, tokenizer, args.device,
                                       args.n_samples)
        unapply_deltas(model, nonoutlier_deltas)
        print(f"  Non-outlier:      PPL={nonoutlier_ppl:.2f}, "
              f"||dW||={compute_frob_norm(nonoutlier_deltas):.1f} "
              f"(orig {nonoutlier_orig_norm:.1f}), "
              f"flipped={n_flipped_n}")

        # 4. Count-matched control: same number of flips from each pool
        #    (no Frobenius rescaling - tests per-flip leverage directly)
        matched_count = n_flipped_o  # match the outlier flip count
        cm_outlier_deltas, cm_n_o, cm_norm_o = make_count_matched_deltas(
            model, outlier_dims, matched_count, seed + 10000, target="outlier")
        cm_nonoutlier_deltas, cm_n_n, cm_norm_n = make_count_matched_deltas(
            model, outlier_dims, matched_count, seed + 20000, target="non_outlier")

        apply_deltas(model, cm_outlier_deltas)
        cm_outlier_ppl = evaluate_ppl(model, tokenizer, args.device,
                                       args.n_samples)
        unapply_deltas(model, cm_outlier_deltas)

        apply_deltas(model, cm_nonoutlier_deltas)
        cm_nonoutlier_ppl = evaluate_ppl(model, tokenizer, args.device,
                                          args.n_samples)
        unapply_deltas(model, cm_nonoutlier_deltas)

        cm_ratio = (cm_outlier_ppl - baseline_ppl) / max(
            cm_nonoutlier_ppl - baseline_ppl, 0.01)
        print(f"  Count-matched ({matched_count} flips each):")
        print(f"    Outlier:     PPL={cm_outlier_ppl:.2f}, ||dW||={cm_norm_o:.1f}")
        print(f"    Non-outlier: PPL={cm_nonoutlier_ppl:.2f}, ||dW||={cm_norm_n:.1f}")
        print(f"    Per-flip leverage ratio: {cm_ratio:.2f}x")

        # Compute ratios (original Frobenius-matched experiment)
        outlier_vs_nonoutlier = (outlier_ppl - baseline_ppl) / max(
            nonoutlier_ppl - baseline_ppl, 0.01)
        outlier_vs_random = (outlier_ppl - baseline_ppl) / max(
            random_ppl - baseline_ppl, 0.01)

        print(f"  Frob-matched Outlier/Non-outlier ratio: {outlier_vs_nonoutlier:.2f}x")
        print(f"  Frob-matched Outlier/Random ratio: {outlier_vs_random:.2f}x")

        results.append({
            "seed": seed,
            "baseline_ppl": baseline_ppl,
            "random_ppl": random_ppl,
            "outlier_ppl": outlier_ppl,
            "nonoutlier_ppl": nonoutlier_ppl,
            "random_norm": random_norm,
            "outlier_orig_norm": outlier_orig_norm,
            "nonoutlier_orig_norm": nonoutlier_orig_norm,
            "n_flipped_random": n_flipped_r,
            "n_flipped_outlier": n_flipped_o,
            "n_flipped_nonoutlier": n_flipped_n,
            "outlier_vs_nonoutlier": outlier_vs_nonoutlier,
            "outlier_vs_random": outlier_vs_random,
            # Count-matched control
            "count_matched": {
                "n_flips": matched_count,
                "outlier_ppl": cm_outlier_ppl,
                "nonoutlier_ppl": cm_nonoutlier_ppl,
                "outlier_norm": cm_norm_o,
                "nonoutlier_norm": cm_norm_n,
                "leverage_ratio": cm_ratio,
            },
        })

    # Summary
    o_vs_no = [r["outlier_vs_nonoutlier"] for r in results]
    o_vs_r = [r["outlier_vs_random"] for r in results]
    o_ppls = [r["outlier_ppl"] for r in results]
    n_ppls = [r["nonoutlier_ppl"] for r in results]
    r_ppls = [r["random_ppl"] for r in results]
    cm_ratios = [r["count_matched"]["leverage_ratio"] for r in results]

    data = {
        "experiment": "exp_d_outlier_features",
        "model": args.model,
        "config": {
            "flip_rate": args.flip_rate,
            "outlier_pct": args.outlier_pct,
            "n_seeds": args.n_seeds,
            "n_samples": args.n_samples,
        },
        "baseline_ppl": baseline_ppl,
        "results": results,
        "summary": {
            "outlier_ppl": {"mean": float(np.mean(o_ppls)), "std": float(np.std(o_ppls))},
            "nonoutlier_ppl": {"mean": float(np.mean(n_ppls)), "std": float(np.std(n_ppls))},
            "random_ppl": {"mean": float(np.mean(r_ppls)), "std": float(np.std(r_ppls))},
            "outlier_vs_nonoutlier": {
                "mean": float(np.mean(o_vs_no)),
                "std": float(np.std(o_vs_no)),
                "median": float(np.median(o_vs_no)),
                "q25": float(np.percentile(o_vs_no, 25)),
                "q75": float(np.percentile(o_vs_no, 75)),
            },
            "outlier_vs_random": {"mean": float(np.mean(o_vs_r)), "std": float(np.std(o_vs_r))},
            "count_matched_leverage": {
                "mean": float(np.mean(cm_ratios)),
                "std": float(np.std(cm_ratios)),
                "median": float(np.median(cm_ratios)),
                "q25": float(np.percentile(cm_ratios, 25)),
                "q75": float(np.percentile(cm_ratios, 75)),
            },
        },
    }

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    print(f"  Seeds: {len(seeds)}")
    print(f"\n  --- Frobenius-matched experiment ---")
    print(f"  Random sign-flip PPL:      {np.mean(r_ppls):.2f} +/- {np.std(r_ppls):.2f}")
    print(f"  Outlier-targeted PPL:      {np.mean(o_ppls):.2f} +/- {np.std(o_ppls):.2f}")
    print(f"  Non-outlier PPL:           {np.mean(n_ppls):.2f} +/- {np.std(n_ppls):.2f}")
    print(f"  Outlier/Non-outlier ratio: mean={np.mean(o_vs_no):.1f}x, "
          f"median={np.median(o_vs_no):.1f}x, "
          f"IQR=[{np.percentile(o_vs_no, 25):.0f}, {np.percentile(o_vs_no, 75):.0f}]")
    print(f"\n  --- Count-matched control (equal flip count, no rescaling) ---")
    print(f"  Per-flip leverage ratio:   mean={np.mean(cm_ratios):.1f}x, "
          f"median={np.median(cm_ratios):.1f}x, "
          f"IQR=[{np.percentile(cm_ratios, 25):.0f}, {np.percentile(cm_ratios, 75):.0f}]")

    if np.mean(o_vs_no) > 1.5:
        print("\n  ==> Outlier-targeted sign-flips are significantly more damaging")
        print("      Confirms: (A2) delocalization violation amplifies sign sensitivity")
    elif np.mean(o_vs_no) > 1.1:
        print("\n  ==> Mild outlier effect detected")
    else:
        print("\n  ==> No significant outlier effect")


if __name__ == "__main__":
    main()
