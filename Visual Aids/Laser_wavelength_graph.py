import numpy as np
import matplotlib.pyplot as plt

# --- Poster-sized defaults ---------------------------------------------------
plt.rcParams.update({
    "figure.figsize"   : (10, 7),     # width, height in inches – scale up if needed
    "font.family"      : "Helvetica", # Set font to Helvetica
    #"axes.titlesize"   : 38,
    "axes.labelsize"   : 36,
    "xtick.labelsize"  : 36,
    "ytick.labelsize"  : 36,
    "legend.fontsize"  : 36,
    "lines.linewidth"  : 4,
})

# --- Synthetic chlorophyll curves -------------------------------------------
λ = np.linspace(400, 700, 600)            # wavelength axis (nm)

def gaussian(x, μ, σ, amp=1):
    return amp * np.exp(-0.5 * ((x - μ) / σ) ** 2)

chl_a = gaussian(λ, 430, 12, amp=0.8)  + gaussian(λ, 662, 15, amp=1.1)
chl_b = gaussian(λ, 453, 10, amp=0.9)  + gaussian(λ, 642, 18, amp=0.8)

# --- Plot --------------------------------------------------------------------
fig, ax = plt.subplots()

ax.plot(λ, chl_a, color="royalblue", label="Chlorophyll a")
ax.plot(λ, chl_b, color="firebrick",  label="Chlorophyll b")

# vertical dashed line at ~450 nm for diode laser
ax.axvline(450, color="dimgray", linestyle="--", linewidth=3,
           label="≈450 nm Diode Laser")

# Labels & title
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Absorbance (Arbitrary Units)")
#ax.set_title("Chlorophyll Absorption vs. Wavelength$^{[5]}$")

# Clean up grid & legend
ax.grid(alpha=0.3, linewidth=1.2)
ax.legend(frameon=False, loc="center")

# Tight layout gives a bit more breathing room
fig.tight_layout()

# Save high-resolution poster version
fig.savefig("chlorophyll_absorption_poster.png", dpi=300)

plt.show()
