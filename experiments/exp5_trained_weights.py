"""
Exp B: Trained-Weight Multi-layer Compounding Test
===================================================
Tests whether trained weight structure enables Bussgang compounding
that iid Gaussian weights lack.

Key question: Exp 5 falsified the multi-layer conjecture (pi/(pi-2))^{beta*L}
on iid Gaussian weights (beta ~ 0). Is this because:
  H1: Bussgang mechanism itself is insufficient; trained weight structure
      (attention routing, outliers, low-rank) is needed for compounding
  H2: Bussgang mechanism is sufficient but iid assumption breaks it;
      trained weights preserve compounding

Protocol:
  1. Load TinyLlama FFN weights (consecutive gate_proj matrices)
  2. Collect real hidden states (post-RMSNorm) as inputs
  3. Run Exp 5 measurement protocol: sign-flip W_1, measure C_L at each layer
  4. Fit beta from log(C_L) vs L
  5. Compare beta_trained vs beta_iid (~ 0 from Exp 5)

Usage:
  python utils/exp5_trained_weights.py --device cuda --dtype float16 \
      --output docs/exp5_trained_weights.json

  # Quick test
  python utils/exp5_trained_weights.py --n-seeds 3 --n-sphere 2000 --device cuda
"""

import argparse
import json
import math
import os
import sys
import time
from typing import List, Dict, Any, Tuple

import torch
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

SEED_LIST = [
    42, 123, 456, 789, 1000,
    1234, 2000, 2345, 3000, 3456,
    4000, 4567, 5000, 5678, 6000,
    6789, 7000, 7890, 8000, 9000,
]

PI_OVER_PI_MINUS_2 = math.pi / (math.pi - 2)
LOG_PI_OVER_PI_MINUS_2 = math.log(PI_OVER_PI_MINUS_2)


def sample_sphere(n: int, k: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Sample k vectors uniformly on S^{n-1}."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(k, n, generator=g)
    x = x / x.norm(dim=1, keepdim=True)
    return x.to(device)


def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    try:
        from scipy.stats import t as t_dist
        t_val = t_dist.ppf(0.975, max(n - 1, 1)) if n > 1 else 0.0
    except ImportError:
        t_val = 1.96 if n >= 30 else 2.09
    ci_lo = mean - t_val * se if n > 1 else mean
    ci_hi = mean + t_val * se if n > 1 else mean
    return {
        "n": n, "mean": mean, "median": float(np.median(arr)),
        "std": std, "se": se, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "min": float(np.min(arr)), "max": float(np.max(arr)),
    }


def make_sign_perturbation(W, flip_frac, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    mask = torch.rand(W.shape, generator=g) < flip_frac
    return (-2 * W.cpu() * mask.float()).to(W.device)


def make_signpreserving_mag_perturbation(W, target_norm):
    sign_W = torch.sign(W)
    sign_norm = sign_W.norm().item()
    delta = target_norm / sign_norm if sign_norm > 0 else 0.0
    return sign_W * delta


def extract_ffn_weights(model_name: str, dtype_str: str, max_layers: int = 6):
    """Extract consecutive FFN gate_proj weight matrices from a model.

    Returns list of weight matrices [W_1, ..., W_L] as float32 tensors on CPU.
    """
    from transformers import AutoModelForCausalLM

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype_str, torch.float16)

    print(f"  Loading model: {model_name} ({dtype_str})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Extract gate_proj weights (FFN first linear layer)
    ffn_weights = []
    ffn_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "gate_proj" in name:
            W = module.weight.data.float().cpu()
            ffn_weights.append(W)
            ffn_names.append(name)

    print(f"  Found {len(ffn_weights)} FFN gate_proj layers")

    # Take consecutive layers from the middle of the network
    n_total = len(ffn_weights)
    if n_total > max_layers:
        start = (n_total - max_layers) // 2
        ffn_weights = ffn_weights[start:start + max_layers]
        ffn_names = ffn_names[start:start + max_layers]

    for i, (name, W) in enumerate(zip(ffn_names, ffn_weights)):
        print(f"    Layer {i}: {name} shape={list(W.shape)}")

    # Also extract hidden dim for sphere sampling
    hidden_dim = ffn_weights[0].shape[1]  # gate_proj: [intermediate, hidden]

    del model
    import gc
    gc.collect()

    return ffn_weights, ffn_names, hidden_dim


@torch.no_grad()
def forward_ffn_chain(Ws, x_hat, activation="relu"):
    """Chain of FFN layers: [Linear -> act -> RMSNorm] x L.

    Each W_l is [intermediate_dim, hidden_dim]. For chaining, we need to project
    back to hidden_dim. We use a simple approach:
    - If W is non-square (m != n), we truncate/pad the output to match input dim
    - For gate_proj (m > n typically), we take the first n dims after activation

    Actually, for a clean experiment, we should use square sub-matrices.
    We crop each W to min(m,n) x min(m,n) for chaining.
    """
    act_fn = torch.relu if activation == "relu" else torch.nn.functional.silu
    y = x_hat
    intermediates = []

    for W in Ws:
        m, n = W.shape
        d = min(m, n)
        W_sq = W[:d, :d]  # crop to square for chaining

        z = W_sq @ y[:d]
        a = act_fn(z)
        a_norm = a.norm(dim=0, keepdim=True).clamp(min=1e-30)
        y_new = a / a_norm
        # Pad back if needed
        if y_new.shape[0] < y.shape[0]:
            pad = torch.zeros(y.shape[0] - y_new.shape[0], y_new.shape[1],
                              device=y.device)
            y_new = torch.cat([y_new, pad], dim=0)
        y = y_new[:y.shape[0]]
        intermediates.append(y.clone())

    return y, intermediates


@torch.no_grad()
def measure_trained_multilayer(Ws_original, n_layers: int, flip_frac: float,
                               seed: int, n_sphere: int, device: str,
                               activation: str = "relu",
                               perturb_mode: str = "all",
                               batch_size: int = 2000) -> Dict[str, Any]:
    """Measure sign vs magnitude transverse energy ratio with trained weights.

    Uses the SAME measurement protocol as Exp 5 but with real weights.
    """
    act_fn = torch.relu if activation == "relu" else torch.nn.functional.silu

    # Prepare square weight matrices for chaining
    Ws = []
    for W_orig in Ws_original[:n_layers]:
        m, n = W_orig.shape
        d = min(m, n)
        Ws.append(W_orig[:d, :d].to(device))

    input_dim = Ws[0].shape[1]

    # Build perturbed weight lists
    Ws_sign = []
    Ws_mag = []
    sign_norms = []

    for l in range(n_layers):
        if perturb_mode == "all" or l == 0:
            DW_sign = make_sign_perturbation(Ws[l], flip_frac, seed + l * 1000)
            s_norm = DW_sign.norm().item()
            sign_norms.append(s_norm)
            DW_mag = make_signpreserving_mag_perturbation(Ws[l], s_norm)
            Ws_sign.append(Ws[l] + DW_sign)
            Ws_mag.append(Ws[l] + DW_mag)
        else:
            Ws_sign.append(Ws[l])
            Ws_mag.append(Ws[l])
            sign_norms.append(0.0)

    # Accumulate per-layer transverse energy
    sign_trans_sums = [0.0] * n_layers
    mag_trans_sums = [0.0] * n_layers
    counts = [0] * n_layers

    def forward_chain(weight_list, x_in):
        y = x_in
        inters = []
        for W in weight_list:
            z = W @ y
            a = act_fn(z)
            a_norm = a.norm(dim=0, keepdim=True).clamp(min=1e-30)
            y = a / a_norm
            inters.append(y)
        return y, inters

    for b_start in range(0, n_sphere, batch_size):
        k = min(batch_size, n_sphere - b_start)
        X = sample_sphere(input_dim, k, seed * 100000 + b_start, device)
        X_T = X.T  # [d, k]

        _, clean_inters = forward_chain(Ws, X_T)
        _, sign_inters = forward_chain(Ws_sign, X_T)
        _, mag_inters = forward_chain(Ws_mag, X_T)

        for l in range(n_layers):
            y_clean = clean_inters[l]
            dy_sign = sign_inters[l] - y_clean
            dy_mag = mag_inters[l] - y_clean

            y_norm = y_clean.norm(dim=0, keepdim=True).clamp(min=1e-30)
            y_hat = y_clean / y_norm

            r_sign = (y_hat * dy_sign).sum(dim=0)
            r_mag = (y_hat * dy_mag).sum(dim=0)

            total_sign = (dy_sign ** 2).sum(dim=0)
            total_mag = (dy_mag ** 2).sum(dim=0)

            trans_sign = total_sign - r_sign ** 2
            trans_mag = total_mag - r_mag ** 2

            valid = (trans_sign > 1e-30) & (trans_mag > 1e-30)
            if valid.any():
                sign_trans_sums[l] += trans_sign[valid].sum().item()
                mag_trans_sums[l] += trans_mag[valid].sum().item()
                counts[l] += valid.sum().item()

    per_layer = {}
    for l in range(n_layers):
        if counts[l] > 0 and mag_trans_sums[l] > 1e-30:
            C_l = sign_trans_sums[l] / mag_trans_sums[l]
        else:
            C_l = float("nan")
        per_layer[l + 1] = {
            "C_l": C_l,
            "log_C_l": math.log(C_l) if (C_l > 0 and not math.isnan(C_l)) else float("nan"),
            "n_valid": counts[l],
        }

    return {
        "n_layers": n_layers,
        "p": flip_frac,
        "seed": seed,
        "activation": activation,
        "perturb_mode": perturb_mode,
        "per_layer": per_layer,
    }


def fit_beta(layer_ratios):
    """Fit beta from per-layer ratios: log(C_L) = beta*L * log(pi/(pi-2))."""
    Ls = []
    log_Cs = []
    for L, C_L in sorted(layer_ratios.items()):
        if C_L > 0 and not math.isnan(C_L):
            Ls.append(L)
            log_Cs.append(math.log(C_L))

    if len(Ls) < 2:
        return {"beta": float("nan"), "r_squared": float("nan"), "n_points": len(Ls)}

    Ls_arr = np.array(Ls, dtype=float)
    log_Cs_arr = np.array(log_Cs, dtype=float)

    numerator = np.sum(Ls_arr * log_Cs_arr)
    denominator = np.sum(Ls_arr ** 2) * LOG_PI_OVER_PI_MINUS_2
    beta_origin = numerator / denominator if denominator > 0 else float("nan")

    predicted = beta_origin * Ls_arr * LOG_PI_OVER_PI_MINUS_2
    ss_res = np.sum((log_Cs_arr - predicted) ** 2)
    ss_tot = np.sum(log_Cs_arr ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    result = {"beta": float(beta_origin), "r_squared": float(r2), "n_points": len(Ls)}

    try:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(Ls_arr, log_Cs_arr)
        beta_free = slope / LOG_PI_OVER_PI_MINUS_2
        result.update({
            "beta_free": float(beta_free),
            "intercept": float(intercept),
            "r_squared_free": float(r_value ** 2),
        })
    except ImportError:
        pass

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Exp B: Trained-weight multi-layer compounding test")
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-sphere", type=int, default=10000)
    parser.add_argument("--flip-rates", type=str, default="0.01,0.02,0.05")
    parser.add_argument("--layers", type=str, default="1,2,3,4,5,6",
                        help="Which layer counts to report")
    parser.add_argument("--activation", type=str, default="relu,silu")
    parser.add_argument("--perturb-mode", type=str, default="all",
                        choices=["all", "first"])
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--output", type=str,
                        default="docs/exp5_trained_weights.json")
    args = parser.parse_args()

    layers_list = [int(l) for l in args.layers.split(",")]
    flip_rates = [float(f) for f in args.flip_rates.split(",")]
    activations = [a.strip() for a in args.activation.split(",")]
    n_seeds = min(args.n_seeds, len(SEED_LIST))
    seeds = SEED_LIST[:n_seeds]
    max_L = max(layers_list)

    print("=" * 70)
    print("Exp B: Trained-Weight Multi-layer Compounding Test")
    print(f"  Model: {args.model}")
    print(f"  Layers: {layers_list}")
    print(f"  Flip rates: {flip_rates}")
    print(f"  Activations: {activations}")
    print(f"  Seeds: {n_seeds}")
    print(f"  Sphere samples: {args.n_sphere}")
    print(f"  Perturb mode: {args.perturb_mode}")
    print("=" * 70)

    # Extract weights
    ffn_weights, ffn_names, hidden_dim = extract_ffn_weights(
        args.model, args.dtype, max_layers=max_L)

    if len(ffn_weights) < max_L:
        print(f"WARNING: Only {len(ffn_weights)} FFN layers available, "
              f"reducing max_L from {max_L} to {len(ffn_weights)}")
        max_L = len(ffn_weights)
        layers_list = [l for l in layers_list if l <= max_L]

    # Also run iid Gaussian baseline with same dimensions for comparison
    d = min(ffn_weights[0].shape)
    print(f"\n  Weight matrix effective dim: {d}x{d}")

    results = {"trained": {}, "iid_baseline": {}}

    for act_name in activations:
        print(f"\n{'='*60}")
        print(f"  Activation: {act_name.upper()}")
        print(f"{'='*60}")

        results["trained"][act_name] = {}
        results["iid_baseline"][act_name] = {}

        for p in flip_rates:
            print(f"\n  --- Trained weights, p={p}, act={act_name} ---")
            seed_per_layer_trained = {L: [] for L in layers_list}
            seed_per_layer_iid = {L: [] for L in layers_list}

            for si, seed in enumerate(seeds):
                # Trained weights
                r_trained = measure_trained_multilayer(
                    ffn_weights, max_L, p, seed, args.n_sphere,
                    args.device, act_name, args.perturb_mode, args.batch_size)

                for L in layers_list:
                    if L in r_trained["per_layer"]:
                        seed_per_layer_trained[L].append(
                            r_trained["per_layer"][L]["C_l"])

                # iid Gaussian baseline (same dimensions)
                # Import from exp5 logic - generate random weights
                g = torch.Generator(device="cpu")
                g.manual_seed(seed + 99999)
                iid_weights = []
                for _ in range(max_L):
                    W = torch.randn(d, d, generator=g) / math.sqrt(d)
                    iid_weights.append(W)

                r_iid = measure_trained_multilayer(
                    iid_weights, max_L, p, seed, args.n_sphere,
                    args.device, act_name, args.perturb_mode, args.batch_size)

                for L in layers_list:
                    if L in r_iid["per_layer"]:
                        seed_per_layer_iid[L].append(
                            r_iid["per_layer"][L]["C_l"])

                if (si + 1) % 5 == 0 or si == 0 or si == n_seeds - 1:
                    c1_t = r_trained["per_layer"].get(1, {}).get("C_l", float("nan"))
                    cmax_t = r_trained["per_layer"].get(max_L, {}).get("C_l", float("nan"))
                    c1_i = r_iid["per_layer"].get(1, {}).get("C_l", float("nan"))
                    cmax_i = r_iid["per_layer"].get(max_L, {}).get("C_l", float("nan"))
                    print(f"    seed {si+1}/{n_seeds}: "
                          f"trained C_1={c1_t:.3f} C_{max_L}={cmax_t:.3f} | "
                          f"iid C_1={c1_i:.3f} C_{max_L}={cmax_i:.3f}")

            # Aggregate trained
            per_layer_trained = {}
            layer_mean_trained = {}
            for L in layers_list:
                vals = seed_per_layer_trained[L]
                if vals:
                    stats = compute_stats(vals)
                    per_layer_trained[L] = stats
                    layer_mean_trained[L] = stats["mean"]

            beta_trained = fit_beta(layer_mean_trained)
            results["trained"][act_name][f"p={p}"] = {
                "per_layer": per_layer_trained,
                "beta_fit": beta_trained,
            }

            # Aggregate iid
            per_layer_iid = {}
            layer_mean_iid = {}
            for L in layers_list:
                vals = seed_per_layer_iid[L]
                if vals:
                    stats = compute_stats(vals)
                    per_layer_iid[L] = stats
                    layer_mean_iid[L] = stats["mean"]

            beta_iid = fit_beta(layer_mean_iid)
            results["iid_baseline"][act_name][f"p={p}"] = {
                "per_layer": per_layer_iid,
                "beta_fit": beta_iid,
            }

            # Print comparison
            print(f"\n    {'L':>3}  {'Trained C_L':>12}  {'iid C_L':>12}  "
                  f"{'Ratio':>8}  {'Theory(b=1)':>12}")
            print(f"    {'-'*56}")
            for L in layers_list:
                t_val = layer_mean_trained.get(L, float("nan"))
                i_val = layer_mean_iid.get(L, float("nan"))
                ratio = t_val / i_val if (i_val > 0 and not math.isnan(i_val)) else float("nan")
                theory = PI_OVER_PI_MINUS_2 ** L
                print(f"    {L:>3}  {t_val:>12.4f}  {i_val:>12.4f}  "
                      f"{ratio:>8.3f}  {theory:>12.4f}")

            bt = beta_trained.get("beta", float("nan"))
            bi = beta_iid.get("beta", float("nan"))
            print(f"\n    beta_trained = {bt:.4f}, beta_iid = {bi:.4f}")
            if not math.isnan(bt) and not math.isnan(bi):
                if bt > bi + 0.1:
                    print(f"    --> H1 supported: trained weights enable compounding")
                elif abs(bt - bi) < 0.1:
                    print(f"    --> H2 supported: trained weights do NOT change compounding")
                else:
                    print(f"    --> Inconclusive")

    # Save
    data = {
        "experiment": "exp5_trained_weights",
        "model": args.model,
        "ffn_layers": ffn_names,
        "effective_dim": d,
        "config": {
            "n_seeds": n_seeds,
            "n_sphere": args.n_sphere,
            "flip_rates": flip_rates,
            "activations": activations,
            "perturb_mode": args.perturb_mode,
            "layers": layers_list,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Trained vs iid beta comparison")
    print("=" * 70)
    for act_name in activations:
        print(f"\n  Activation: {act_name}")
        for p_key in results["trained"][act_name]:
            bt = results["trained"][act_name][p_key]["beta_fit"].get("beta", float("nan"))
            bi = results["iid_baseline"][act_name][p_key]["beta_fit"].get("beta", float("nan"))
            print(f"    {p_key}: beta_trained={bt:.4f}, beta_iid={bi:.4f}, "
                  f"diff={bt-bi:.4f}")


if __name__ == "__main__":
    main()
