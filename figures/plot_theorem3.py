"""
Figure 1: Theorem 3 Verification - c(p) vs p
=============================================
Plots theoretical curve c(p) = pi/(pi-2) * (1-p) * (1 - 4*sqrt(p)/(3*pi))
against measured V1 data points with error bars.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['mathtext.fontset'] = 'cm'


def c_theory(p):
    """Theoretical prediction for c(p) from Theorem 3."""
    pi = np.pi
    return (pi / (pi - 2)) * (1 - p) * (1 - 4 * np.sqrt(p) / (3 * pi))


# V1 measured data (from paper Table 2, n=2048, 20 seeds)
p_measured = np.array([0.01, 0.02, 0.05, 0.10])
c_measured = np.array([2.639, 2.583, 2.465, 2.299])
c_std = np.array([0.012, 0.010, 0.010, 0.008])

# Theoretical curve
p_curve = np.linspace(0.001, 0.15, 200)
c_curve = c_theory(p_curve)

# Plot
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# Theory curve
ax.plot(p_curve, c_curve, 'b-', linewidth=2, label=r'Theory: $c(p) = \frac{\pi}{\pi-2}(1-p)(1 - \frac{4\sqrt{p}}{3\pi})$')

# Measured points with error bars
ax.errorbar(p_measured, c_measured, yerr=2*c_std, fmt='ro', markersize=8,
            capsize=5, capthick=1.5, linewidth=1.5,
            label=r'Measured (V1, $n{=}2048$, 20 seeds)')

# Asymptotic limit
ax.axhline(y=np.pi/(np.pi-2), color='gray', linestyle='--', alpha=0.6,
           label=r'$\lim_{p\to 0} c(p) = \pi/(\pi-2) \approx 2.75$')

ax.set_xlabel(r'Flip probability $p$', fontsize=13)
ax.set_ylabel(r'Transverse energy ratio $c(p)$', fontsize=13)
ax.set_title(r'Theorem 3 Verification: Sign/Magnitude Asymmetry Ratio', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim([0, 0.12])
ax.set_ylim([2.0, 2.85])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/fig1_theorem3.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig1_theorem3.png', dpi=150, bbox_inches='tight')
print("Saved: figures/fig1_theorem3.pdf, figures/fig1_theorem3.png")
