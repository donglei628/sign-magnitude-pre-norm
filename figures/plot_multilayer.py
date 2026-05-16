"""
Figure 2: Multi-Layer Compounding Falsification
================================================
Plots C_L vs L for V2 (no residual), V3 (with residual), and Exp B (trained),
compared against the falsified conjecture C_L = (pi/(pi-2))^L.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# Load data from JSON results
def load_multilayer_data(filepath, activation='relu', dim='2048', p='p=0.01'):
    """Extract per-layer C_L values from experiment JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    results = data['results'][activation][dim][p]['per_layer']
    layers = sorted(results.keys(), key=int)
    means = [results[l]['mean'] for l in layers]
    ci_lo = [results[l].get('ci95_lo', results[l]['mean'] - 1.96*results[l].get('std', 0)/np.sqrt(results[l].get('n', 20))) for l in layers]
    ci_hi = [results[l].get('ci95_hi', results[l]['mean'] + 1.96*results[l].get('std', 0)/np.sqrt(results[l].get('n', 20))) for l in layers]
    return np.array([int(l) for l in layers]), np.array(means), np.array(ci_lo), np.array(ci_hi)


def load_trained_data(filepath, activation='relu', p='p=0.01'):
    """Extract per-layer C_L from trained weights experiment."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    results = data['results']['trained'][activation][p]['per_layer']
    layers = sorted(results.keys(), key=int)
    means = [results[l]['mean'] for l in layers]
    ci_lo = [results[l].get('ci95_lo', results[l]['mean'] - 1.96*results[l].get('std', 0)/np.sqrt(results[l].get('n', 10))) for l in layers]
    ci_hi = [results[l].get('ci95_hi', results[l]['mean'] + 1.96*results[l].get('std', 0)/np.sqrt(results[l].get('n', 10))) for l in layers]
    return np.array([int(l) for l in layers]), np.array(means), np.array(ci_lo), np.array(ci_hi)


# Load V2 (no residual, dim=2048, relu, p=0.01)
L_v2, C_v2, lo_v2, hi_v2 = load_multilayer_data(
    'results/exp5_multilayer_toy.json', 'relu', '2048', 'p=0.01')

# Load V3 (with residual, dim=2048, relu, p=0.01)
L_v3, C_v3, lo_v3, hi_v3 = load_multilayer_data(
    'results/exp5_multilayer_toy_residual.json', 'relu', '2048', 'p=0.01')

# Load Exp B (trained weights, relu, p=0.01)
L_tb, C_tb, lo_tb, hi_tb = load_trained_data(
    'results/exp5_trained_weights.json', 'relu', 'p=0.01')

# Falsified conjecture: C_L = (pi/(pi-2))^L
pi_ratio = np.pi / (np.pi - 2)  # 2.75
L_conj = np.arange(1, 7)
C_conj = pi_ratio ** L_conj

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

# Conjecture (dashed, gray)
ax.semilogy(L_conj, C_conj, 'k--', linewidth=1.5, alpha=0.5,
            label=r'Falsified conjecture: $(\pi/(\pi{-}2))^L$')

# V2: no residual
ax.semilogy(L_v2, C_v2, 'bs-', markersize=7, linewidth=2, label='V2 (i.i.d., no residual)')
ax.fill_between(L_v2, lo_v2, hi_v2, alpha=0.15, color='blue')

# Exp B: trained weights
ax.semilogy(L_tb, C_tb, 'r^-', markersize=7, linewidth=2, label='Exp B (trained TinyLlama, ReLU)')
ax.fill_between(L_tb, lo_tb, hi_tb, alpha=0.15, color='red')

# V3: with residual
ax.semilogy(L_v3, C_v3, 'go-', markersize=7, linewidth=2, label='V3 (i.i.d., with residual)')
ax.fill_between(L_v3, lo_v3, hi_v3, alpha=0.15, color='green')

# Annotations
ax.axhline(y=pi_ratio, color='gray', linestyle=':', alpha=0.4)
ax.text(5.5, pi_ratio * 1.05, r'$\pi/(\pi{-}2) \approx 2.75$', fontsize=10, color='gray')

ax.set_xlabel(r'Number of layers $L$', fontsize=13)
ax.set_ylabel(r'Transverse energy ratio $C_L$ (log scale)', fontsize=13)
ax.set_title(r'Multi-Layer Compounding: Falsification ($p{=}0.01$)', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.set_xticks(range(1, 7))
ax.set_xlim([0.8, 6.2])
ax.set_ylim([1, 1000])
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('figures/fig2_multilayer.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig2_multilayer.png', dpi=150, bbox_inches='tight')
print("Saved: figures/fig2_multilayer.pdf, figures/fig2_multilayer.png")
