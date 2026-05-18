"""Candidate #1: Noise floor sweep.

Vary flip fraction p and measure count-matched NLL leverage at each level.
Tests whether leverage changes with p (noise floor hypothesis).

Reuses infrastructure from dn_outlier.py.
"""
import argparse
import json
import math
import sys
import os
import time
import numpy as np
import torch

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)

# Import from existing infrastructure
sys.path.insert(0, os.path.dirname(__file__))
from dn_outlier import (
    get_linear_layers, evaluate_ppl, collect_activation_stats,
    identify_outliers, make_count_matched_deltas,
    apply_deltas, unapply_deltas, compute_frob_norm, SEED_LIST,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Noise floor sweep: count-matched leverage vs flip rate")
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--outlier-pct", type=float, default=0.05)
    parser.add_argument("--output", type=str,
                        default="docs/noise_floor_sweep.json")
    args = parser.parse_args()

    # Flip rates to test (original was 0.05 = 5%)
    flip_rates = [0.05, 0.005, 0.001, 0.0005, 0.0001]

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype, torch.float16)

    print("=" * 70)
    print("Candidate #1: Noise Floor Sweep")
    print(f"  Model: {args.model}")
    print(f"  Flip rates: {flip_rates}")
    print(f"  Seeds: {args.n_seeds}")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {args.model}...")
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
    ln_base = math.log(baseline_ppl)
    print(f"  Baseline PPL: {baseline_ppl:.4f}, ln(baseline) = {ln_base:.4f}")

    # Collect activation stats and identify outliers
    print("\nCollecting activation statistics...")
    act_stats = collect_activation_stats(model, tokenizer, args.device, n_samples=5)
    outlier_dims = identify_outliers(act_stats, args.outlier_pct)
    print(f"  Identified outliers for {len(outlier_dims)} layers")

    # For count-matched, we need the number of outlier weights
    # Compute from the first flip rate to get reference counts
    all_layers = get_linear_layers(model)
    total_outlier_pool = 0
    for name, module in all_layers:
        W = module.weight.data
        m, n = W.shape
        outlier_set = outlier_dims.get(name, set())
        n_cols = sum(1 for j in range(n) if j in outlier_set)
        total_outlier_pool += m * n_cols
    print(f"  Total outlier pool: {total_outlier_pool:,} weights")

    seeds = SEED_LIST[:args.n_seeds]
    all_results = {}

    for flip_rate in flip_rates:
        # Number of flips = flip_rate * total_outlier_pool
        n_flips = max(1, int(flip_rate * total_outlier_pool))
        print(f"\n{'='*60}")
        print(f"Flip rate: {flip_rate*100:.2f}% ({n_flips:,} flips)")
        print(f"{'='*60}")

        seed_results = []
        ppl_leverages = []
        nll_leverages = []

        for si, seed in enumerate(seeds):
            # Count-matched: outlier
            cm_o_deltas, cm_n_o, cm_norm_o = make_count_matched_deltas(
                model, outlier_dims, n_flips, seed + 10000, target="outlier")
            apply_deltas(model, cm_o_deltas)
            cm_o_ppl = evaluate_ppl(model, tokenizer, args.device, args.n_samples)
            unapply_deltas(model, cm_o_deltas)

            # Count-matched: non-outlier
            cm_no_deltas, cm_n_no, cm_norm_no = make_count_matched_deltas(
                model, outlier_dims, n_flips, seed + 20000, target="non_outlier")
            apply_deltas(model, cm_no_deltas)
            cm_no_ppl = evaluate_ppl(model, tokenizer, args.device, args.n_samples)
            unapply_deltas(model, cm_no_deltas)

            # PPL leverage
            ppl_lev = (cm_o_ppl - baseline_ppl) / max(cm_no_ppl - baseline_ppl, 0.001)

            # NLL leverage
            nll_o = math.log(cm_o_ppl) - ln_base if cm_o_ppl > 0 else 0
            nll_no = math.log(cm_no_ppl) - ln_base if cm_no_ppl > 0 else 0
            nll_lev = nll_o / nll_no if nll_no > 1e-6 else float('inf')

            ppl_leverages.append(ppl_lev)
            nll_leverages.append(nll_lev)

            seed_results.append({
                "seed": seed,
                "outlier_ppl": float(cm_o_ppl),
                "nonoutlier_ppl": float(cm_no_ppl),
                "ppl_leverage": float(ppl_lev),
                "nll_leverage": float(nll_lev),
                "outlier_norm": float(cm_norm_o),
                "nonoutlier_norm": float(cm_norm_no),
            })

            print(f"  Seed {si+1}/{len(seeds)}: "
                  f"outlier PPL={cm_o_ppl:.2f}, "
                  f"non-outlier PPL={cm_no_ppl:.4f}, "
                  f"PPL lev={ppl_lev:.1f}x, "
                  f"NLL lev={nll_lev:.1f}x")

        ppl_arr = np.array(ppl_leverages)
        nll_arr = np.array(nll_leverages)

        print(f"\n  Summary (p={flip_rate*100:.2f}%):")
        print(f"    PPL leverage: median={np.median(ppl_arr):.1f}x, "
              f"IQR=[{np.percentile(ppl_arr,25):.1f}, {np.percentile(ppl_arr,75):.1f}]")
        print(f"    NLL leverage: median={np.median(nll_arr):.1f}x, "
              f"IQR=[{np.percentile(nll_arr,25):.1f}, {np.percentile(nll_arr,75):.1f}]")

        all_results[str(flip_rate)] = {
            "flip_rate": flip_rate,
            "n_flips": n_flips,
            "seeds": seed_results,
            "ppl_leverage": {
                "median": float(np.median(ppl_arr)),
                "mean": float(np.mean(ppl_arr)),
                "iqr": [float(np.percentile(ppl_arr, 25)),
                        float(np.percentile(ppl_arr, 75))],
            },
            "nll_leverage": {
                "median": float(np.median(nll_arr)),
                "mean": float(np.mean(nll_arr)),
                "iqr": [float(np.percentile(nll_arr, 25)),
                        float(np.percentile(nll_arr, 75))],
            },
        }

    # Summary table
    print(f"\n{'='*60}")
    print("Summary: NLL Leverage vs Flip Rate")
    print(f"{'='*60}")
    print(f"{'p':>10s}  {'n_flips':>10s}  {'PPL_lev':>10s}  {'NLL_lev':>10s}")
    print("-" * 45)
    for fr in flip_rates:
        r = all_results[str(fr)]
        print(f"{fr*100:>9.2f}%  {r['n_flips']:>10,d}  "
              f"{r['ppl_leverage']['median']:>9.1f}x  "
              f"{r['nll_leverage']['median']:>9.1f}x")

    # Save
    output = {
        "experiment": "noise_floor_sweep",
        "model": args.model,
        "baseline_ppl": float(baseline_ppl),
        "outlier_pct": args.outlier_pct,
        "n_seeds": args.n_seeds,
        "results": all_results,
    }
    out_path = os.path.join("c:/source/BitNet", args.output)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
