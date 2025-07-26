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
INFILE = Path(r"M:\Research\Delta Robot\DeltaX2\Tests\Positional_accuracy.xlsx")
OUTDIR = INFILE.parent

# ── Load data ───────────────────────────────────────────────────────────────
df    = pd.read_excel(INFILE)
r_err = df["R_error = √(dX² + dY²+dZ²)"]

# ── Global style tweaks ─────────────────────────────────────────────────────
plt.rcParams.update({
    # 10 % grey canvas
    "axes.facecolor"   : "#FAFAFA",
    "figure.facecolor" : "#FAFAFA",
    "font.family"      : "Arial",  # Set font to Arial
    # Text, fonts
    "axes.titlesize"   : 34,
    "axes.labelsize"   : 36,
    "xtick.labelsize"  : 36,
    "ytick.labelsize"  : 36,
    "legend.fontsize"  : 36,
    # Grid
    "axes.grid"        : True,
    "grid.color"       : "#BFBFBF",
    "grid.alpha"       : 0.20,
    # Lines
    "lines.linewidth"  : 0             # scatter only
})

# Palette
blue = "#278BFF"   # commanded
red  = "#BA0C2F"   # observed + spec/identity

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – Commanded vs Observed XY
# ═══════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(16, 12))

ax.scatter(df["X_cmd"], df["Y_cmd"],
           c=blue, marker="s", s=100,
           edgecolors="#FAFAFA", linewidths=0.5, label="Commanded")

ax.scatter(df["X_obs"], df["Y_obs"],
           c=red,  marker="o", s=100,
           edgecolors="#FAFAFA", linewidths=0.5, label="Observed")

# Identity line
#lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
#        np.max([ax.get_xlim(), ax.get_ylim()])]
#ax.plot(lims, lims, color=red, linestyle="--", linewidth=3)

#ax.set_aspect("equal", adjustable="box")
#ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
#ax.set_title("End-Effector Positional Accuracy")
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
ax.hist(r_err, bins=bins, color=blue, edgecolor="#FAFAFA")

mean_r = r_err.mean()
p95_r  = r_err.quantile(0.95)

ax.axvline(mean_r, color="#000000", linestyle=":", linewidth=3,
           label=f"Mean = {mean_r:.2f} mm")
ax.axvline(p95_r,  color=red, linestyle="--", linewidth=3,
           label=f"95 % ≤ {p95_r:.2f} mm")

ax.set_xlabel("Radial error (mm)")
ax.set_ylabel("Count")
ax.set_title("Radial Error Distribution")
ax.legend(frameon=False, loc="upper right")

fig2.tight_layout()
fig2.savefig(OUTDIR / "radial_error_hist.png",
             dpi=300, bbox_inches="tight")
plt.close(fig2)
