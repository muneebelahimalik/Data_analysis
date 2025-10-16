"""
laser_damage_exposure_plot.py
Creates one PNG:

exposure_vs_distance.png
  • X-axis : Exposure time (s)      – independent variable
  • Y-axis : Laser-to-weed distance (mm)
  • Grey dots = individual trials
  • Blue circles + horizontal error bars = mean ± 1 SD per distance

Background is 10 % grey (#EDEDED) to match your poster canvas.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── I/O paths ──────────────────────────────────────────────────────────────
INCSV  = Path(r"M:\Research\Delta Robot\DeltaX2\Tests\laser_damage_trials_v2.csv")
OUTPNG = INCSV.with_name("exposure_vs_distance.png")

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(INCSV)

# ── Poster-scale style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize"  : (10, 7),     # prints ~10 in wide @ 300 dpi
    "axes.facecolor"  : "#FFFFFF",  # white background
    "figure.facecolor": "#FFFFFF",
    "axes.titlesize"  : 34,
    "axes.labelsize"  : 28,
    "xtick.labelsize" : 24,
    "ytick.labelsize" : 24,
    "legend.fontsize" : 22,
    "axes.grid"       : True,
    "grid.color"      : "#BFBFBF",
    "grid.alpha"      : 0.30,
    "lines.linewidth" : 0            # scatter only
})

blue = "#278BFF"   # poster blue
grey = "#9E9E9E"   # muted grey for raw points

# ── Build the figure ───────────────────────────────────────────────────────
fig, ax = plt.subplots()

# All trials (light grey)
ax.scatter(df["Exposure_s"], df["Distance_mm"],
           c=grey, s=70, alpha=0.6, label="Individual trial")

# Means & SDs by distance
grouped = df.groupby("Distance_mm")["Exposure_s"]
means   = grouped.mean()
stds    = grouped.std()

ax.errorbar(means, means.index,
            xerr=stds,
            fmt="o", color=blue, ecolor=blue,
            elinewidth=3, capsize=6, markersize=10,
            label="Mean ± 1 SD")

# Labels & title
ax.set_xlabel("Exposure time (s)")
ax.set_ylabel("Laser–weed distance (mm)")
#ax.set_title("Exposure Duration vs Stand-Off Distance in Weed Killing Trials")
ax.legend(frameon=False, loc="lower right")

fig.tight_layout()
fig.savefig(OUTPNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved graphic → {OUTPNG}")
