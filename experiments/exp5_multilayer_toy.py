"""
Exp 5: Multi-layer Toy Network Verification
=============================================
Verifies the multi-layer compounding conjecture from theory section 6.8:

  C_L = sign_transverse_L / mag_transverse_L ~ (pi/(pi-2))^{beta*L}

where beta in (0, 1] is the effective layer factor encoding cross-layer
correlation effects on the sign/magnitude asymmetry accumulation.

Architecture: L layers of [Linear -> Activation -> RMSNorm]
  y_hat_0 = x_hat in S^{n-1}
  For l = 1, ..., L:
    z_l = W_l @ y_hat_{l-1}
    a_l = activation(z_l)
    y_hat_l = a_l / ||a_l||   (RMSNorm)

Two perturbation modes:
  --perturb-mode all   : perturb ALL layers independently (default, matches
                         real quantization where every layer is ternary)
  --perturb-mode first : perturb W_1 only (control experiment: tests whether
                         the asymmetry compounds or just propagates)

CRITICAL: Uses sign-preserving magnitude perturbation with CONSTANT delta
(matching Theorem 3's assumption A5), NOT independent additive noise.
This creates the Bussgang alignment needed for the pi/(pi-2) ratio.

Theoretical predictions:
  L=1: C_1 = pi/(pi-2) ~ 2.75  (Theorem 3, exact)
  L>1: C_L = (pi/(pi-2))^{beta*L}  (conjecture, beta to be fitted)
  L=2, beta=1: C_2 ~ 7.57 (falls in experimental 6-24x range)

Usage:
  # Quick test
  python utils/exp5_multilayer_toy.py --n-seeds 3 --dims 512 --layers 1,2,3

  # Full production run
  python utils/exp5_multilayer_toy.py --n-seeds 20 --activation relu,silu \
      --output docs/exp5_multilayer_toy.json
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED_LIST = [
    42, 123, 456, 789, 1000,
    1234, 2000, 2345, 3000, 3456,
    4000, 4567, 5000, 5678, 6000,
    6789, 7000, 7890, 8000, 9000,
]

PI_OVER_PI_MINUS_2 = math.pi / (math.pi - 2)  # 2.7528
LOG_PI_OVER_PI_MINUS_2 = math.log(PI_OVER_PI_MINUS_2)  # 1.0124


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sample_sphere(n: int, k: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Sample k vectors uniformly on S^{n-1}."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randn(k, n, generator=g)
    x = x / x.norm(dim=1, keepdim=True)
    return x.to(device)


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics including 95% CI."""
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    try:
        from scipy.stats import t as t_dist
        t_val = t_dist.ppf(0.975, max(n - 1, 1)) if n > 1 else 0.0
    except ImportError:
        t_val = 1.96 if n >= 30 else 2.09
    if n > 1:
        ci_lo = mean - t_val * se
        ci_hi = mean + t_val * se
    else:
        ci_lo = ci_hi = mean
    return {
        "n": n, "mean": mean, "median": median, "std": std,
        "se": se, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "min": float(np.min(arr)), "max": float(np.max(arr)),
    }


def get_activation_fn(name: str):
    """Return activation function by name."""
    if name == "relu":
        return torch.relu
    elif name == "silu":
        return torch.nn.functional.silu
    else:
        raise ValueError(f"Unknown activation: {name}")


# ---------------------------------------------------------------------------
# Perturbation construction
# ---------------------------------------------------------------------------
def make_sign_perturbation(W: torch.Tensor, flip_frac: float,
                           seed: int) -> torch.Tensor:
    """Sign-flip perturbation: DW_ij = -2*W_ij with probability p."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    mask = torch.rand(W.shape, generator=g) < flip_frac
    return (-2 * W.cpu() * mask.float()).to(W.device)


def make_signpreserving_mag_perturbation(W: torch.Tensor,
                                         target_norm: float) -> torch.Tensor:
    """Sign-preserving magnitude perturbation with CONSTANT delta per entry.

    DW = sign(W) * delta, where delta is a scalar chosen so that
    ||DW||_F = target_norm.

    This is Theorem 3's assumption (A5): deterministic, uniform scaling.
    The constant delta creates Bussgang alignment after ReLU, giving
    radial fraction R -> 2/pi.

    NOTE: This is NOT the independent additive model (iid Gaussian delta)
    used in Proposition 1. The Bussgang mechanism requires constant delta.
    """
    sign_W = torch.sign(W)
    # ||sign(W) * delta||_F = |delta| * ||sign(W)||_F = |delta| * sqrt(m*n)
    sign_norm = sign_W.norm().item()
    if sign_norm > 0:
        delta = target_norm / sign_norm
    else:
        delta = 0.0
    return sign_W * delta


# ---------------------------------------------------------------------------
# Core: Multi-layer forward pass
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_multilayer(Ws: List[torch.Tensor], x_hat: torch.Tensor,
                       act_fn, residual: bool = False
                       ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """L-layer forward: [Linear -> act -> RMSNorm] x L.

    Args:
        Ws: list of weight matrices [W_1, ..., W_L], each [m, m]
        x_hat: input [m, k] (columns are samples)
        act_fn: activation function
        residual: if True, use pre-norm residual architecture:
                  y_l = RMSNorm(y_{l-1} + act(W_l @ RMSNorm(y_{l-1})))
                  This matches real Transformer pre-norm structure.

    Returns:
        y_L: final output [m, k]
        intermediates: list of per-layer outputs [y_1, ..., y_L]
    """
    y = x_hat
    intermediates = []
    for W in Ws:
        if residual:
            # Pre-norm residual: normalize -> linear -> act -> add residual
            y_norm = y.norm(dim=0, keepdim=True).clamp(min=1e-30)
            y_hat = y / y_norm
            z = W @ y_hat
            a = act_fn(z)
            y = y + a  # residual connection (no final norm needed per layer)
            intermediates.append(y.clone())
        else:
            z = W @ y
            a = act_fn(z)
            a_norm = a.norm(dim=0, keepdim=True).clamp(min=1e-30)
            y = a / a_norm
            intermediates.append(y)
    return y, intermediates


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_multilayer(m: int, n_layers: int, flip_frac: float, seed: int,
                       n_sphere: int, device: str, activation: str = "relu",
                       perturb_mode: str = "all", residual: bool = False,
                       batch_size: int = 2000) -> Dict[str, Any]:
    """Measure sign vs magnitude transverse energy ratio across L layers.

    Args:
        perturb_mode: "all" = perturb every layer independently (default)
                      "first" = perturb only W_1 (control)
        residual: if True, use pre-norm residual architecture

    Returns per-layer ratios C_l = sign_transverse / mag_transverse.
    """
    act_fn = get_activation_fn(activation)

    # Generate weight matrices W_1, ..., W_L (all m x m, iid Gaussian)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 99999)
    Ws = []
    for l in range(n_layers):
        W = torch.randn(m, m, generator=g).to(device) / math.sqrt(m)
        Ws.append(W)

    # Build perturbed weight lists
    Ws_sign = []
    Ws_mag = []
    sign_norms = []

    for l in range(n_layers):
        if perturb_mode == "all" or l == 0:
            # Generate sign-flip perturbation with layer-specific seed
            DW_sign = make_sign_perturbation(Ws[l], flip_frac,
                                             seed + l * 1000)
            s_norm = DW_sign.norm().item()
            sign_norms.append(s_norm)

            # Sign-preserving magnitude perturbation (constant delta)
            DW_mag = make_signpreserving_mag_perturbation(Ws[l], s_norm)

            Ws_sign.append(Ws[l] + DW_sign)
            Ws_mag.append(Ws[l] + DW_mag)
        else:
            # No perturbation at this layer (perturb_mode == "first")
            Ws_sign.append(Ws[l])
            Ws_mag.append(Ws[l])
            sign_norms.append(0.0)

    # Accumulate per-layer transverse energy
    sign_trans_sums = [0.0] * n_layers
    mag_trans_sums = [0.0] * n_layers
    sign_total_sums = [0.0] * n_layers
    mag_total_sums = [0.0] * n_layers
    counts = [0] * n_layers

    for b_start in range(0, n_sphere, batch_size):
        k = min(batch_size, n_sphere - b_start)
        X = sample_sphere(m, k, seed * 100000 + b_start, device)
        X_T = X.T  # [m, k]

        # Clean forward
        _, clean_inters = forward_multilayer(Ws, X_T, act_fn, residual)

        # Sign-perturbed forward
        _, sign_inters = forward_multilayer(Ws_sign, X_T, act_fn, residual)

        # Mag-perturbed forward
        _, mag_inters = forward_multilayer(Ws_mag, X_T, act_fn, residual)

        # Measure transverse energy at each layer
        for l in range(n_layers):
            y_clean = clean_inters[l]   # [m, k]
            dy_sign = sign_inters[l] - y_clean
            dy_mag = mag_inters[l] - y_clean

            # Normalize y_clean to get direction (for residual mode,
            # outputs are NOT unit-norm; for plain mode they already are)
            y_norm = y_clean.norm(dim=0, keepdim=True).clamp(min=1e-30)
            y_hat = y_clean / y_norm

            # Radial component
            r_sign = (y_hat * dy_sign).sum(dim=0)  # [k]
            r_mag = (y_hat * dy_mag).sum(dim=0)

            # Total energy
            total_sign = (dy_sign ** 2).sum(dim=0)  # [k]
            total_mag = (dy_mag ** 2).sum(dim=0)

            # Transverse energy = total - radial^2
            trans_sign = total_sign - r_sign ** 2
            trans_mag = total_mag - r_mag ** 2

            # Filter valid samples
            valid = (trans_sign > 1e-30) & (trans_mag > 1e-30)
            if valid.any():
                sign_trans_sums[l] += trans_sign[valid].sum().item()
                mag_trans_sums[l] += trans_mag[valid].sum().item()
                sign_total_sums[l] += total_sign[valid].sum().item()
                mag_total_sums[l] += total_mag[valid].sum().item()
                counts[l] += valid.sum().item()

    # Compute per-layer ratios
    per_layer = {}
    for l in range(n_layers):
        if counts[l] > 0 and mag_trans_sums[l] > 1e-30:
            C_l = sign_trans_sums[l] / mag_trans_sums[l]
        else:
            C_l = float("nan")

        # Also compute total energy ratio (before transverse projection)
        if counts[l] > 0 and mag_total_sums[l] > 1e-30:
            total_ratio = sign_total_sums[l] / mag_total_sums[l]
        else:
            total_ratio = float("nan")

        theory_l = PI_OVER_PI_MINUS_2 ** (l + 1)
        per_layer[l + 1] = {
            "C_l": C_l,
            "theory_beta1": theory_l,
            "log_C_l": math.log(C_l) if (C_l > 0 and
                                          not math.isnan(C_l)) else float("nan"),
            "total_energy_ratio": total_ratio,
            "sign_trans_total": sign_trans_sums[l],
            "mag_trans_total": mag_trans_sums[l],
            "n_valid": counts[l],
        }

    return {
        "m": m,
        "n_layers": n_layers,
        "p": flip_frac,
        "seed": seed,
        "activation": activation,
        "perturb_mode": perturb_mode,
        "sign_frob_norms": sign_norms,
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Beta fitting
# ---------------------------------------------------------------------------
def fit_beta(layer_ratios: Dict[int, float]) -> Dict[str, float]:
    """Fit beta from per-layer ratios: log(C_L) = beta*L * log(pi/(pi-2)).

    Uses linear regression on (L, log(C_L)) through origin.
    Also fits unconstrained (with intercept) for comparison.
    """
    Ls = []
    log_Cs = []
    for L, C_L in sorted(layer_ratios.items()):
        if C_L > 0 and not math.isnan(C_L):
            Ls.append(L)
            log_Cs.append(math.log(C_L))

    if len(Ls) < 2:
        return {"beta": float("nan"), "r_squared": float("nan"),
                "n_points": len(Ls)}

    Ls_arr = np.array(Ls, dtype=float)
    log_Cs_arr = np.array(log_Cs, dtype=float)

    # Method 1: Through-origin regression
    numerator = np.sum(Ls_arr * log_Cs_arr)
    denominator = np.sum(Ls_arr ** 2) * LOG_PI_OVER_PI_MINUS_2
    beta_origin = numerator / denominator if denominator > 0 else float("nan")

    predicted = beta_origin * Ls_arr * LOG_PI_OVER_PI_MINUS_2
    ss_res = np.sum((log_Cs_arr - predicted) ** 2)
    ss_tot = np.sum(log_Cs_arr ** 2)
    r2_origin = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    result = {
        "beta": float(beta_origin),
        "r_squared": float(r2_origin),
        "n_points": len(Ls),
    }

    # Method 2: With intercept (scipy linregress)
    try:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            Ls_arr, log_Cs_arr)
        beta_free = slope / LOG_PI_OVER_PI_MINUS_2
        result.update({
            "beta_free": float(beta_free),
            "intercept": float(intercept),
            "r_squared_free": float(r_value ** 2),
            "p_value": float(p_value),
            "std_err": float(std_err / LOG_PI_OVER_PI_MINUS_2),
        })
    except ImportError:
        pass

    return result


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_experiment(args) -> Dict[str, Any]:
    """Run multi-layer toy network experiment."""
    dims = [int(d) for d in args.dims.split(",")]
    layers_list = [int(l) for l in args.layers.split(",")]
    flip_rates = [float(f) for f in args.flip_rates.split(",")]
    activations = [a.strip() for a in args.activation.split(",")]
    n_seeds = min(args.n_seeds, len(SEED_LIST))
    seeds = SEED_LIST[:n_seeds]
    max_L = max(layers_list)

    results = {}

    for act_name in activations:
        print(f"\n{'='*70}")
        print(f"  Activation: {act_name.upper()}, "
              f"perturb_mode: {args.perturb_mode}")
        print(f"{'='*70}")
        results[act_name] = {}

        for m in dims:
            results[act_name][m] = {}
            for p in flip_rates:
                print(f"\n  dim={m}, p={p}, activation={act_name}")
                print(f"  {'-'*50}")

                seed_per_layer = {L: [] for L in layers_list}

                for si, seed in enumerate(seeds):
                    r = measure_multilayer(
                        m=m, n_layers=max_L, flip_frac=p, seed=seed,
                        n_sphere=args.n_sphere_samples, device=args.device,
                        activation=act_name, perturb_mode=args.perturb_mode,
                        residual=args.residual,
                        batch_size=args.batch_size)

                    for L in layers_list:
                        if L in r["per_layer"]:
                            C_l = r["per_layer"][L]["C_l"]
                            seed_per_layer[L].append(C_l)

                    if (si + 1) % 5 == 0 or si == 0 or si == n_seeds - 1:
                        c1 = r["per_layer"].get(1, {}).get("C_l", float("nan"))
                        cmax = r["per_layer"].get(
                            max_L, {}).get("C_l", float("nan"))
                        print(f"    seed {si+1}/{n_seeds}: "
                              f"C_1={c1:.3f}, C_{max_L}={cmax:.3f}")

                # Aggregate
                per_layer_stats = {}
                layer_mean_ratios = {}
                for L in layers_list:
                    vals = seed_per_layer[L]
                    if vals:
                        stats = compute_stats(vals)
                        theory = PI_OVER_PI_MINUS_2 ** L
                        stats["theory_beta1"] = theory
                        stats["deviation_from_theory_pct"] = (
                            (stats["mean"] - theory) / theory * 100
                            if theory > 0 else float("nan"))
                        per_layer_stats[L] = stats
                        layer_mean_ratios[L] = stats["mean"]

                beta_fit = fit_beta(layer_mean_ratios)

                results[act_name][m][f"p={p}"] = {
                    "per_layer": per_layer_stats,
                    "beta_fit": beta_fit,
                    "n_seeds": n_seeds,
                }

                # Print
                print(f"\n    {'L':>3}  {'C_L mean':>10}  "
                      f"{'[95%CI]':>16}  "
                      f"{'theory(b=1)':>12}  {'dev%':>8}")
                print(f"    {'-'*60}")
                for L in layers_list:
                    if L in per_layer_stats:
                        s = per_layer_stats[L]
                        ci = f"[{s['ci95_lo']:.3f},{s['ci95_hi']:.3f}]"
                        print(f"    {L:>3}  {s['mean']:>10.4f}  "
                              f"{ci:>16}  "
                              f"{s['theory_beta1']:>12.4f}  "
                              f"{s['deviation_from_theory_pct']:>7.2f}%")

                beta = beta_fit.get("beta", float("nan"))
                r2 = beta_fit.get("r_squared", float("nan"))
                print(f"\n    beta fit (origin): "
                      f"beta={beta:.4f}, R2={r2:.4f}")
                if "beta_free" in beta_fit:
                    bf = beta_fit["beta_free"]
                    r2f = beta_fit["r_squared_free"]
                    inter = beta_fit["intercept"]
                    print(f"    beta fit (free):   "
                          f"beta={bf:.4f}, R2={r2f:.4f}, "
                          f"intercept={inter:.4f}")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(results: Dict[str, Any], layers_list: List[int],
                  perturb_mode: str):
    """Print final summary tables."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Multi-layer Compounding Results "
          f"(perturb_mode={perturb_mode})")
    print(f"{'='*70}")

    all_betas_global = []

    for act_name, act_results in results.items():
        print(f"\n  Activation: {act_name.upper()}")
        print(f"  {'-'*60}")

        print(f"\n  Table: Transverse Energy Ratio "
              f"C_L = sign_trans / mag_trans")
        header = f"  {'dim':>5} {'p':>6}"
        for L in layers_list:
            header += f"  {'L='+str(L):>8}"
        header += f"  {'beta':>8}  {'R2':>6}"
        print(header)
        print(f"  {'-'*len(header)}")

        all_betas = []
        for dim_str, dim_results in sorted(act_results.items()):
            m_val = int(dim_str)
            for p_str, data in sorted(dim_results.items()):
                p_val = p_str.replace("p=", "")
                row = f"  {m_val:>5} {p_val:>6}"
                for L in layers_list:
                    if L in data["per_layer"]:
                        val = data["per_layer"][L]["mean"]
                        row += f"  {val:>8.3f}"
                    else:
                        row += f"  {'N/A':>8}"
                beta = data["beta_fit"].get("beta", float("nan"))
                r2 = data["beta_fit"].get("r_squared", float("nan"))
                row += f"  {beta:>8.4f}  {r2:>6.4f}"
                print(row)
                if not math.isnan(beta):
                    all_betas.append(beta)

        if all_betas:
            beta_stats = compute_stats(all_betas)
            print(f"\n  Overall beta ({act_name}): "
                  f"{beta_stats['mean']:.4f} +/- {beta_stats['std']:.4f} "
                  f"(range [{beta_stats['min']:.4f}, "
                  f"{beta_stats['max']:.4f}])")

            L2_pred = PI_OVER_PI_MINUS_2 ** (2 * beta_stats["mean"])
            print(f"  L=2 prediction with beta={beta_stats['mean']:.3f}: "
                  f"C_2 = {L2_pred:.2f}")
            print(f"  Experimental reference: 6-24x "
                  f"(from real Transformers)")
            all_betas_global.extend(all_betas)

    print(f"\n  {'-'*60}")
    print(f"  Interpretation:")
    if all_betas_global:
        bm = np.mean(all_betas_global)
        if bm >= 0.8:
            print(f"  STRONG: beta ~ {bm:.2f} close to 1.0")
            print(f"    Multi-layer compounding largely explains "
                  f"the 2.75 -> 6-24 gap")
        elif bm >= 0.5:
            print(f"  MODERATE: beta ~ {bm:.2f}")
            print(f"    Multi-layer effect exists but partial; "
                  f"outlier features also contribute")
        elif bm >= 0.1:
            print(f"  WEAK: beta ~ {bm:.2f}")
            print(f"    Multi-layer compounding alone insufficient; "
                  f"other mechanisms needed")
        else:
            print(f"  MINIMAL: beta ~ {bm:.2f}")
            print(f"    No significant multi-layer compounding detected")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Exp 5: Multi-layer toy network verification")
    parser.add_argument("--dims", type=str, default="512,1024,2048",
                        help="Comma-separated matrix dimensions (square)")
    parser.add_argument("--layers", type=str, default="1,2,3,4,5,6",
                        help="Comma-separated layer counts to measure")
    parser.add_argument("--flip-rates", type=str, default="0.01,0.02,0.05",
                        help="Comma-separated flip fractions")
    parser.add_argument("--activation", type=str, default="relu",
                        help="Activation(s): relu, silu, or relu,silu")
    parser.add_argument("--perturb-mode", type=str, default="all",
                        choices=["all", "first"],
                        help="'all' = perturb every layer (default), "
                        "'first' = perturb W_1 only")
    parser.add_argument("--residual", action="store_true",
                        help="Use pre-norm residual architecture "
                        "(matches real Transformers)")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-sphere-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str,
                        default="docs/exp5_multilayer_toy.json")
    args = parser.parse_args()

    layers_list = [int(l) for l in args.layers.split(",")]

    print(f"Exp 5: Multi-layer Toy Network Verification")
    print(f"  dims: {args.dims}")
    print(f"  layers: {args.layers}")
    print(f"  flip_rates: {args.flip_rates}")
    print(f"  activation: {args.activation}")
    print(f"  perturb_mode: {args.perturb_mode}")
    print(f"  residual: {args.residual}")
    print(f"  n_seeds: {args.n_seeds}")
    print(f"  n_sphere_samples: {args.n_sphere_samples}")
    print(f"  device: {args.device}")
    print(f"  theory: pi/(pi-2) = {PI_OVER_PI_MINUS_2:.6f}")
    print()

    t0 = time.time()
    results = run_experiment(args)
    elapsed = time.time() - t0

    print_summary(results, layers_list, args.perturb_mode)

    # Save to JSON
    output = {
        "experiment": "exp5_multilayer_toy",
        "config": {
            "dims": args.dims,
            "layers": args.layers,
            "flip_rates": args.flip_rates,
            "activation": args.activation,
            "perturb_mode": args.perturb_mode,
            "residual": args.residual,
            "n_seeds": args.n_seeds,
            "n_sphere_samples": args.n_sphere_samples,
            "device": args.device,
            "theory_single_layer": PI_OVER_PI_MINUS_2,
        },
        "results": {},
        "elapsed_seconds": elapsed,
    }

    for act_name, act_data in results.items():
        output["results"][act_name] = {}
        for dim, dim_data in act_data.items():
            dim_key = str(dim)
            output["results"][act_name][dim_key] = {}
            for p_str, p_data in dim_data.items():
                pl_safe = {}
                for L, stats in p_data["per_layer"].items():
                    pl_safe[str(L)] = stats
                output["results"][act_name][dim_key][p_str] = {
                    "per_layer": pl_safe,
                    "beta_fit": p_data["beta_fit"],
                    "n_seeds": p_data["n_seeds"],
                }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
