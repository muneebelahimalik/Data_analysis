"""
positional_accuracy_figs_recolour.py
Creates two PNGs on a 10 % grey (#EDEDED) background:

1. positional_xy_accuracy.png
2. radial_error_hist.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── I/O paths ───────────────────────────────────────────────────────────────
INFILE = Path(r"C:\Users\mm17889\Data_analysis\Positional_accuracy.xlsx")
OUTDIR = INFILE.parent

# ── Load data ───────────────────────────────────────────────────────────────
df    = pd.read_excel(INFILE)
r_err = df["R_error = √(dX² + dY²+dZ²)"]

# ── Global style tweaks ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi"      : 300,
    "savefig.dpi"     : 300,
    "axes.facecolor"  : "#F7F7F7",     # very light grey
    "figure.facecolor": "#F7F7F7",
    "font.family"     : "Arial",

    # Text
    "axes.titlesize"  : 40,
    "axes.labelsize"  : 38,
    "xtick.labelsize" : 34,
    "ytick.labelsize" : 34,
    "legend.fontsize" : 34,

    # Grid
    "axes.grid"       : True,
    "grid.color"      : "#BFBFBF",
    "grid.alpha"      : 0.25,
    "grid.linewidth"  : 0.6,

    # Lines
    "lines.linewidth" : 0   # we’ll only use scatter markers
})

# Brighter palette
BLUE = "#006CFF"      # vivid blue
RED  = "#FF2B2B"      # vivid red

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – Commanded vs Observed XY
# ═══════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(18, 14))

# — Commanded (squares) —
ax.scatter(df["X_cmd"], df["Y_cmd"],
           c=BLUE, marker="s", s=350, alpha=0.9,
           edgecolors="black", linewidths=1.0,
           label="Commanded")

# — Observed (circles) —
ax.scatter(df["X_obs"], df["Y_obs"],
           c=RED, marker="o", s=350, alpha=0.9,
           edgecolors="black", linewidths=1.0,
           label="Observed")

# Optional identity line (uncomment if wanted)
# lims = np.r_[ax.get_xlim(), ax.get_ylim()].min(), np.r_[ax.get_xlim(), ax.get_ylim()].max()
# ax.plot(lims, lims, ls="--", lw=3, color="#333333", alpha=0.6, zorder=0)

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.legend(frameon=False, loc="upper left")

fig1.tight_layout()
fig1.savefig(OUTDIR / "positional_xy_accuracy.png",
             dpi=300, bbox_inches="tight")
plt.close(fig1)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 – Radial-error histogram
# ═══════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(9, 7))

bins = np.linspace(0, r_err.max(), 12)
ax.hist(r_err, bins=bins, color=BLUE, edgecolor="#FAFAFA")

mean_r = r_err.mean()
p95_r  = r_err.quantile(0.95)

ax.axvline(mean_r, color="#000000", linestyle=":", linewidth=3,
           label=f"Mean = {mean_r:.2f} mm")
ax.axvline(p95_r,  color=RED, linestyle="--", linewidth=3,
           label=f"95 % ≤ {p95_r:.2f} mm")

ax.set_xlabel("Radial error (mm)")
ax.set_ylabel("Count")
ax.set_title("Radial Error Distribution")
ax.legend(frameon=False, loc="upper right")

fig2.tight_layout()
fig2.savefig(OUTDIR / "radial_error_hist.png",
             dpi=300, bbox_inches="tight")
plt.close(fig2)
