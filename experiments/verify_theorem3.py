"""
Verification of Theorem 3: Sign/Magnitude Transverse Energy Ratio
=================================================================

This script verifies Theorem 3 using the EXACT architecture from the theory:

  x_hat -> W1 (m x n) -> ReLU -> W2 (d_out x m) -> y = W2 * a

The transverse energy ratio c(p) = T_sign / T_mag should converge to:
  c(p) = pi/(pi-2) * (1-p) * (1 - 4*sqrt(p)/(3*pi)) + O(sqrt(p))
  c(0) = pi/(pi-2) ~ 2.75

CRITICAL: Previous Exp 5 used [W -> ReLU -> norm] per layer (NO W2),
which does NOT match Theorem 3's two-layer model. This script uses the
correct architecture.

Three verification modes:
  V1: Single-layer exact Theorem 3 verification (W1 -> ReLU -> W2)
  V2: Multi-layer FFN chain: [W_up -> ReLU -> W_down -> norm] x L
  V3: Multi-layer with residual: [W_up -> ReLU -> W_down + skip -> norm] x L

Usage:
  python utils/verify_theorem3.py --output docs_v002/experiments/verify_theorem3.json
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


def c_theory(p: float) -> float:
    """Theorem 3 asymptotic prediction for c(p)."""
    R_sign = p - 4 * p**1.5 / (3 * math.pi)
    R_mag = 2.0 / math.pi
    R_A = 1 - 4 * math.sqrt(p) / (3 * math.pi)
    return (1 - R_sign) / (1 - R_mag) * R_A


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
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    t_val = 1.96 if n >= 30 else 2.09
    return {
        "n": n, "mean": mean, "median": float(np.median(arr)),
        "std": std, "se": se,
        "ci95_lo": mean - t_val * se, "ci95_hi": mean + t_val * se,
        "min": float(np.min(arr)), "max": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------
def make_sign_perturbation(W: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """Sign-flip: DW_ij = -2*W_ij with probability p."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    mask = torch.rand(W.shape, generator=g) < p
    return (-2 * W.cpu() * mask.float()).to(W.device)


def make_signpreserving_mag_perturbation(W: torch.Tensor,
                                          target_norm: float) -> torch.Tensor:
    """Sign-preserving magnitude: DW = sign(W) * delta (constant delta).
    Frobenius-matched to target_norm."""
    sign_W = torch.sign(W)
    sign_norm = sign_W.norm().item()
    if sign_norm > 0:
        delta = target_norm / sign_norm
    else:
        delta = 0.0
    return sign_W * delta


# ---------------------------------------------------------------------------
# V1: Single-layer Theorem 3 verification
# ---------------------------------------------------------------------------
@torch.no_grad()
def verify_single_layer(n: int, m: int, d_out: int, p: float, seed: int,
                         n_sphere: int, device: str,
                         batch_size: int = 2000) -> Dict[str, Any]:
    """Verify Theorem 3 with correct architecture: W1 -> ReLU -> W2 -> measure.

    Model: y = W2 * ReLU(W1 * x_hat)
    Perturb W1 only. Measure transverse energy ratio on y.
    """
    # Generate W1 (m x n) and W2 (d_out x m)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 99999)
    W1 = (torch.randn(m, n, generator=g) / math.sqrt(n)).to(device)
    W2 = (torch.randn(d_out, m, generator=g) / math.sqrt(m)).to(device)

    # Perturbations on W1 only
    DW_sign = make_sign_perturbation(W1, p, seed)
    s_norm = DW_sign.norm().item()
    DW_mag = make_signpreserving_mag_perturbation(W1, s_norm)

    W1_sign = W1 + DW_sign
    W1_mag = W1 + DW_mag

    # Accumulate transverse energy
    sign_trans_sum = 0.0
    mag_trans_sum = 0.0
    sign_total_sum = 0.0
    mag_total_sum = 0.0
    sign_radial_sum = 0.0
    mag_radial_sum = 0.0
    count = 0

    for b_start in range(0, n_sphere, batch_size):
        k = min(batch_size, n_sphere - b_start)
        X = sample_sphere(n, k, seed * 100000 + b_start, device)  # [k, n]
        X_T = X.T  # [n, k]

        # Clean forward: y = W2 * ReLU(W1 * x_hat)
        a_clean = torch.relu(W1 @ X_T)          # [m, k]
        y_clean = W2 @ a_clean                    # [d_out, k]

        # Sign-perturbed
        a_sign = torch.relu(W1_sign @ X_T)
        y_sign = W2 @ a_sign

        # Mag-perturbed
        a_mag = torch.relu(W1_mag @ X_T)
        y_mag = W2 @ a_mag

        # Differences
        dy_sign = y_sign - y_clean  # [d_out, k]
        dy_mag = y_mag - y_clean

        # Direction of clean output
        y_norm = y_clean.norm(dim=0, keepdim=True).clamp(min=1e-30)
        y_hat = y_clean / y_norm

        # Radial components
        r_sign = (y_hat * dy_sign).sum(dim=0)  # [k]
        r_mag = (y_hat * dy_mag).sum(dim=0)

        # Total energy
        total_sign = (dy_sign ** 2).sum(dim=0)  # [k]
        total_mag = (dy_mag ** 2).sum(dim=0)

        # Transverse energy = total - radial^2
        trans_sign = total_sign - r_sign ** 2
        trans_mag = total_mag - r_mag ** 2

        # Filter valid
        valid = (trans_sign > 1e-30) & (trans_mag > 1e-30)
        if valid.any():
            sign_trans_sum += trans_sign[valid].sum().item()
            mag_trans_sum += trans_mag[valid].sum().item()
            sign_total_sum += total_sign[valid].sum().item()
            mag_total_sum += total_mag[valid].sum().item()
            sign_radial_sum += (r_sign[valid] ** 2).sum().item()
            mag_radial_sum += (r_mag[valid] ** 2).sum().item()
            count += valid.sum().item()

    C = sign_trans_sum / mag_trans_sum if mag_trans_sum > 1e-30 else float("nan")
    R_sign = sign_radial_sum / sign_total_sum if sign_total_sum > 1e-30 else float("nan")
    R_mag = mag_radial_sum / mag_total_sum if mag_total_sum > 1e-30 else float("nan")
    total_ratio = sign_total_sum / mag_total_sum if mag_total_sum > 1e-30 else float("nan")

    return {
        "C": C,
        "R_sign": R_sign,
        "R_mag": R_mag,
        "total_energy_ratio": total_ratio,
        "sign_frob_norm": s_norm,
        "n_valid": count,
        "theory_c": c_theory(p),
        "theory_limit": PI_OVER_PI_MINUS_2,
    }


# ---------------------------------------------------------------------------
# V2/V3: Multi-layer FFN chain
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_ffn_chain(W_ups: List[torch.Tensor], W_downs: List[torch.Tensor],
                       x_hat: torch.Tensor, residual: bool = False
                       ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Forward pass through FFN chain: [W_up -> ReLU -> W_down -> norm] x L.

    Each "FFN layer" has the same structure as real Transformer FFN:
      a = ReLU(W_up @ y_norm)
      h = W_down @ a
      if residual: y = y + h
      else: y = h
      y = y / ||y||  (normalize)

    Args:
        W_ups: list of [m_hidden, m] matrices (expand)
        W_downs: list of [m, m_hidden] matrices (contract)
        x_hat: [m, k] input (columns are samples)
        residual: if True, add residual connection
    """
    y = x_hat
    intermediates = []
    for W_up, W_down in zip(W_ups, W_downs):
        # Normalize input (pre-norm)
        y_norm_val = y.norm(dim=0, keepdim=True).clamp(min=1e-30)
        y_normed = y / y_norm_val

        # FFN: up -> ReLU -> down
        a = torch.relu(W_up @ y_normed)
        h = W_down @ a

        if residual:
            y = y + h
        else:
            y = h

        # Normalize
        out_norm = y.norm(dim=0, keepdim=True).clamp(min=1e-30)
        y = y / out_norm
        intermediates.append(y.clone())

    return y, intermediates


@torch.no_grad()
def measure_multilayer_ffn(m: int, m_hidden: int, n_layers: int,
                            p: float, seed: int, n_sphere: int,
                            device: str, residual: bool = False,
                            perturb_mode: str = "all",
                            batch_size: int = 2000) -> Dict[str, Any]:
    """Measure transverse energy ratio in multi-layer FFN chain.

    Architecture per layer: W_up (m_hidden x m) -> ReLU -> W_down (m x m_hidden) -> [+residual] -> norm

    Perturbation: sign-flip / magnitude on W_up matrices only (W_down clean).
    This matches Theorem 3's setup where W1 is perturbed and W2 is clean.
    """
    # Generate weight matrices
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 99999)
    W_ups = []
    W_downs = []
    for l in range(n_layers):
        W_up = (torch.randn(m_hidden, m, generator=g) / math.sqrt(m)).to(device)
        W_down = (torch.randn(m, m_hidden, generator=g) / math.sqrt(m_hidden)).to(device)
        W_ups.append(W_up)
        W_downs.append(W_down)

    # Build perturbed W_up lists
    W_ups_sign = []
    W_ups_mag = []
    sign_norms = []

    for l in range(n_layers):
        if perturb_mode == "all" or l == 0:
            DW_sign = make_sign_perturbation(W_ups[l], p, seed + l * 1000)
            s_norm = DW_sign.norm().item()
            sign_norms.append(s_norm)
            DW_mag = make_signpreserving_mag_perturbation(W_ups[l], s_norm)
            W_ups_sign.append(W_ups[l] + DW_sign)
            W_ups_mag.append(W_ups[l] + DW_mag)
        else:
            W_ups_sign.append(W_ups[l])
            W_ups_mag.append(W_ups[l])
            sign_norms.append(0.0)

    # W_downs are always clean (matching Theorem 3: only W1 perturbed, W2 clean)
    # Accumulate per-layer transverse energy
    sign_trans_sums = [0.0] * n_layers
    mag_trans_sums = [0.0] * n_layers
    counts = [0] * n_layers

    for b_start in range(0, n_sphere, batch_size):
        k = min(batch_size, n_sphere - b_start)
        X = sample_sphere(m, k, seed * 100000 + b_start, device)
        X_T = X.T  # [m, k]

        _, clean_inters = forward_ffn_chain(W_ups, W_downs, X_T, residual)
        _, sign_inters = forward_ffn_chain(W_ups_sign, W_downs, X_T, residual)
        _, mag_inters = forward_ffn_chain(W_ups_mag, W_downs, X_T, residual)

        for l in range(n_layers):
            y_clean = clean_inters[l]
            dy_sign = sign_inters[l] - y_clean
            dy_mag = mag_inters[l] - y_clean

            y_hat = y_clean  # already normalized in forward_ffn_chain

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
        per_layer[l + 1] = {"C_l": C_l, "n_valid": counts[l]}

    return {
        "m": m, "m_hidden": m_hidden, "n_layers": n_layers,
        "p": p, "seed": seed, "residual": residual,
        "perturb_mode": perturb_mode,
        "sign_frob_norms": sign_norms,
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Verify Theorem 3 with correct architecture")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--n-sphere-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str,
                        default="docs_v002/experiments/verify_theorem3.json")
    args = parser.parse_args()

    n_seeds = min(args.n_seeds, len(SEED_LIST))
    seeds = SEED_LIST[:n_seeds]

    all_results = {}
    t0 = time.time()

    # ==================================================================
    # V1: Single-layer Theorem 3 exact verification
    # ==================================================================
    print("=" * 70)
    print("  V1: Single-layer Theorem 3 verification")
    print("  Architecture: x_hat -> W1 -> ReLU -> W2 -> y")
    print("=" * 70)

    v1_results = {}
    dims_v1 = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024),
               (2048, 2048, 2048)]
    flip_rates = [0.01, 0.02, 0.05, 0.10]

    for n, m, d_out in dims_v1:
        dim_key = f"{n}x{m}x{d_out}"
        v1_results[dim_key] = {}
        for p in flip_rates:
            seed_Cs = []
            seed_R_signs = []
            seed_R_mags = []
            for si, seed in enumerate(seeds):
                r = verify_single_layer(n, m, d_out, p, seed,
                                        args.n_sphere_samples, args.device,
                                        args.batch_size)
                seed_Cs.append(r["C"])
                seed_R_signs.append(r["R_sign"])
                seed_R_mags.append(r["R_mag"])

            stats = compute_stats(seed_Cs)
            R_sign_stats = compute_stats(seed_R_signs)
            R_mag_stats = compute_stats(seed_R_mags)
            theory = c_theory(p)

            v1_results[dim_key][f"p={p}"] = {
                "C_stats": stats,
                "R_sign_stats": R_sign_stats,
                "R_mag_stats": R_mag_stats,
                "theory_c": theory,
                "theory_limit": PI_OVER_PI_MINUS_2,
                "deviation_pct": (stats["mean"] - theory) / theory * 100,
            }

            print(f"  dim={dim_key}, p={p:.2f}: "
                  f"C={stats['mean']:.4f} +/- {stats['se']:.4f}  "
                  f"theory={theory:.4f}  "
                  f"dev={((stats['mean']-theory)/theory*100):+.2f}%  "
                  f"R_sign={R_sign_stats['mean']:.4f}  "
                  f"R_mag={R_mag_stats['mean']:.4f} (theory 2/pi={2/math.pi:.4f})")

    all_results["V1_single_layer"] = v1_results

    # Summary table for V1
    print(f"\n  V1 Summary: C_measured vs c_theory(p)")
    print(f"  {'dim':>20} {'p':>6} {'C_mean':>8} {'theory':>8} {'dev%':>8} "
          f"{'R_sign':>8} {'R_mag':>8} {'2/pi':>8}")
    print(f"  {'-'*80}")
    for dim_key, dim_data in v1_results.items():
        for p_key, data in sorted(dim_data.items()):
            p_val = p_key.replace("p=", "")
            print(f"  {dim_key:>20} {p_val:>6} "
                  f"{data['C_stats']['mean']:>8.4f} "
                  f"{data['theory_c']:>8.4f} "
                  f"{data['deviation_pct']:>+7.2f}% "
                  f"{data['R_sign_stats']['mean']:>8.4f} "
                  f"{data['R_mag_stats']['mean']:>8.4f} "
                  f"{2/math.pi:>8.4f}")

    # ==================================================================
    # V2: Multi-layer FFN chain (no residual)
    # ==================================================================
    print(f"\n{'='*70}")
    print("  V2: Multi-layer FFN chain (NO residual)")
    print("  Architecture: [W_up -> ReLU -> W_down -> norm] x L")
    print("=" * 70)

    v2_results = {}
    m_base = 512
    m_hidden = 2048  # 4x expansion like real Transformers
    layers_list = [1, 2, 3, 4, 5, 6]
    max_L = max(layers_list)
    v2_flip_rates = [0.01, 0.05]

    for p in v2_flip_rates:
        p_key = f"p={p}"
        seed_per_layer = {L: [] for L in layers_list}

        for si, seed in enumerate(seeds):
            r = measure_multilayer_ffn(
                m=m_base, m_hidden=m_hidden, n_layers=max_L,
                p=p, seed=seed, n_sphere=args.n_sphere_samples,
                device=args.device, residual=False,
                perturb_mode="all", batch_size=args.batch_size)

            for L in layers_list:
                if L in r["per_layer"]:
                    seed_per_layer[L].append(r["per_layer"][L]["C_l"])

            if (si + 1) % 5 == 0 or si == 0:
                c1 = r["per_layer"].get(1, {}).get("C_l", float("nan"))
                cmax = r["per_layer"].get(max_L, {}).get("C_l", float("nan"))
                print(f"    seed {si+1}/{n_seeds}: C_1={c1:.3f}, C_{max_L}={cmax:.3f}")

        per_layer_stats = {}
        for L in layers_list:
            if seed_per_layer[L]:
                stats = compute_stats(seed_per_layer[L])
                per_layer_stats[str(L)] = stats

        v2_results[p_key] = {
            "m": m_base, "m_hidden": m_hidden,
            "per_layer": per_layer_stats,
        }

        print(f"\n  V2 p={p}: m={m_base}, m_hidden={m_hidden}")
        print(f"  {'L':>3}  {'C_L mean':>10}  {'CI95':>20}  {'theory(b=1)':>12}")
        print(f"  {'-'*55}")
        for L in layers_list:
            if str(L) in per_layer_stats:
                s = per_layer_stats[str(L)]
                ci = f"[{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]"
                theory = PI_OVER_PI_MINUS_2 ** L
                print(f"  {L:>3}  {s['mean']:>10.4f}  {ci:>20}  {theory:>12.4f}")

    all_results["V2_multilayer_no_residual"] = v2_results

    # ==================================================================
    # V3: Multi-layer FFN chain WITH residual
    # ==================================================================
    print(f"\n{'='*70}")
    print("  V3: Multi-layer FFN chain WITH residual")
    print("  Architecture: [W_up -> ReLU -> W_down + skip -> norm] x L")
    print("=" * 70)

    v3_results = {}
    for p in v2_flip_rates:
        p_key = f"p={p}"
        seed_per_layer = {L: [] for L in layers_list}

        for si, seed in enumerate(seeds):
            r = measure_multilayer_ffn(
                m=m_base, m_hidden=m_hidden, n_layers=max_L,
                p=p, seed=seed, n_sphere=args.n_sphere_samples,
                device=args.device, residual=True,
                perturb_mode="all", batch_size=args.batch_size)

            for L in layers_list:
                if L in r["per_layer"]:
                    seed_per_layer[L].append(r["per_layer"][L]["C_l"])

            if (si + 1) % 5 == 0 or si == 0:
                c1 = r["per_layer"].get(1, {}).get("C_l", float("nan"))
                cmax = r["per_layer"].get(max_L, {}).get("C_l", float("nan"))
                print(f"    seed {si+1}/{n_seeds}: C_1={c1:.3f}, C_{max_L}={cmax:.3f}")

        per_layer_stats = {}
        for L in layers_list:
            if seed_per_layer[L]:
                stats = compute_stats(seed_per_layer[L])
                per_layer_stats[str(L)] = stats

        v3_results[p_key] = {
            "m": m_base, "m_hidden": m_hidden,
            "per_layer": per_layer_stats,
        }

        print(f"\n  V3 (residual) p={p}: m={m_base}, m_hidden={m_hidden}")
        print(f"  {'L':>3}  {'C_L mean':>10}  {'CI95':>20}  {'theory(b=1)':>12}")
        print(f"  {'-'*55}")
        for L in layers_list:
            if str(L) in per_layer_stats:
                s = per_layer_stats[str(L)]
                ci = f"[{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]"
                theory = PI_OVER_PI_MINUS_2 ** L
                print(f"  {L:>3}  {s['mean']:>10.4f}  {ci:>20}  {theory:>12.4f}")

    all_results["V3_multilayer_with_residual"] = v3_results

    # ==================================================================
    # Also run old architecture for comparison
    # ==================================================================
    print(f"\n{'='*70}")
    print("  V0 (control): OLD Exp 5 architecture (W -> ReLU -> norm)")
    print("  This should reproduce C_1 ~ 3.6 (NOT matching Theorem 3)")
    print("=" * 70)

    v0_results = {}
    for p in [0.01, 0.05]:
        p_key = f"p={p}"
        seed_Cs = []

        for si, seed in enumerate(seeds):
            g = torch.Generator(device="cpu")
            g.manual_seed(seed + 99999)
            W = (torch.randn(512, 512, generator=g) / math.sqrt(512)).to(args.device)

            DW_sign = make_sign_perturbation(W, p, seed)
            s_norm = DW_sign.norm().item()
            DW_mag = make_signpreserving_mag_perturbation(W, s_norm)

            W_sign = W + DW_sign
            W_mag = W + DW_mag

            sign_trans_sum = 0.0
            mag_trans_sum = 0.0
            count = 0

            for b_start in range(0, args.n_sphere_samples, args.batch_size):
                k = min(args.batch_size, args.n_sphere_samples - b_start)
                X = sample_sphere(512, k, seed * 100000 + b_start, args.device)
                X_T = X.T

                # Clean: W -> ReLU -> norm (old architecture)
                a_clean = torch.relu(W @ X_T)
                a_norm = a_clean.norm(dim=0, keepdim=True).clamp(min=1e-30)
                y_clean = a_clean / a_norm

                a_sign = torch.relu(W_sign @ X_T)
                y_sign = a_sign / a_sign.norm(dim=0, keepdim=True).clamp(min=1e-30)

                a_mag = torch.relu(W_mag @ X_T)
                y_mag = a_mag / a_mag.norm(dim=0, keepdim=True).clamp(min=1e-30)

                dy_sign = y_sign - y_clean
                dy_mag = y_mag - y_clean

                r_sign = (y_clean * dy_sign).sum(dim=0)
                r_mag = (y_clean * dy_mag).sum(dim=0)

                total_sign = (dy_sign ** 2).sum(dim=0)
                total_mag = (dy_mag ** 2).sum(dim=0)

                trans_sign = total_sign - r_sign ** 2
                trans_mag = total_mag - r_mag ** 2

                valid = (trans_sign > 1e-30) & (trans_mag > 1e-30)
                if valid.any():
                    sign_trans_sum += trans_sign[valid].sum().item()
                    mag_trans_sum += trans_mag[valid].sum().item()
                    count += valid.sum().item()

            C = sign_trans_sum / mag_trans_sum if mag_trans_sum > 1e-30 else float("nan")
            seed_Cs.append(C)

        stats = compute_stats(seed_Cs)
        theory = c_theory(p)
        v0_results[p_key] = {
            "C_stats": stats,
            "theory_c": theory,
            "deviation_pct": (stats["mean"] - theory) / theory * 100,
        }
        print(f"  V0 (old arch), p={p}: C={stats['mean']:.4f} +/- {stats['se']:.4f}  "
              f"theory={theory:.4f}  dev={((stats['mean']-theory)/theory*100):+.2f}%")

    all_results["V0_old_architecture"] = v0_results

    # ==================================================================
    # Final comparison
    # ==================================================================
    print(f"\n{'='*70}")
    print("  COMPARISON: V0 (old, wrong arch) vs V1 (correct arch)")
    print("=" * 70)
    for p in [0.01, 0.05]:
        p_key = f"p={p}"
        v0_c = v0_results[p_key]["C_stats"]["mean"]
        # Find V1 at 512x512x512
        v1_c = v1_results.get("512x512x512", {}).get(p_key, {}).get("C_stats", {}).get("mean", float("nan"))
        theory = c_theory(p)
        print(f"  p={p}: V0(W->ReLU->norm)={v0_c:.4f}  "
              f"V1(W1->ReLU->W2)={v1_c:.4f}  "
              f"theory={theory:.4f}  "
              f"pi/(pi-2)={PI_OVER_PI_MINUS_2:.4f}")

    elapsed = time.time() - t0

    # Save
    output = {
        "experiment": "verify_theorem3",
        "description": "Verification of Theorem 3 with correct two-layer architecture",
        "config": {
            "n_seeds": n_seeds,
            "n_sphere_samples": args.n_sphere_samples,
            "device": args.device,
        },
        "results": all_results,
        "elapsed_seconds": elapsed,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
