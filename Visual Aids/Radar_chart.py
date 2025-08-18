import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- Radar chart labels
labels = [
    "Depth Accuracy",
    "Spatial Resolution",
    "Light Robustness",
    "3D Reconstruction",
    "Weed Detection Suitability",
    "Cost"
]
num_vars = len(labels)

# --- Radar chart data (normalized from 0 to 5)
data = {
    "LiDAR": [5, 5, 5, 5, 5, 1],
    "Stereo RGB": [4, 4, 2, 4, 4, 4],
    "RGB Mono": [1, 4, 1, 1, 3, 5],
    "ToF Camera": [4, 4, 4, 4, 4, 3],
    "Thermal Camera": [1, 1, 5, 1, 2, 4],
    "Ultrasonic": [1, 1, 5, 1, 1, 5]
}

# --- Updated, high‑contrast palette (inspired by your reference figure)
colors = {
    "LiDAR":        "#0066FF",  # vivid blue
    "Stereo RGB":   "#00BFFF",  # bright cyan
    "RGB Mono":     "#1FFF5F",  # bright green
    "ToF Camera":   "#E1BF00",  # deep yellow (keeps contrast)
    "Thermal Camera":"#FF8C00", # orange
    "Ultrasonic":   "#FF0000"   # red
}

# --- Angles for radar chart
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# --- Enhanced matplotlib settings for publication quality
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,  # Base font size
    'axes.linewidth': 1.5,
    'grid.linewidth': 1,
    'lines.linewidth': 3,
    'figure.dpi': 300,  # High resolution for print
    'savefig.dpi': 600,  # Even higher for saving
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Create figure with larger size for better readability
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# --- Plot each sensor profile with thicker lines
for sensor, values in data.items():
    profile = values + values[:1]        # close the polygon
    ax.plot(angles, profile, color=colors[sensor], linewidth=3.5, label=sensor)
    ax.fill(angles, profile, color=colors[sensor], alpha=0.15)

# --- Enhanced radial labels with larger font and better positioning
for angle, label in zip(angles[:-1], labels):
    # Calculate label position further out to avoid overlap with chart lines
    label_distance = 6.8
    ax.text(angle, label_distance, label,
            ha="center", va="center",
            fontsize=18, fontweight="bold", color="black")

# --- Enhanced axis cosmetics
ax.set_xticks([])
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], 
                   fontsize=16, color="black", fontweight='bold')
ax.set_ylim(0, 5)

# --- Grid styling
ax.grid(True, alpha=0.3, linewidth=1.2)
ax.set_rgrids([1, 2, 3, 4, 5], alpha=0.7)

# --- Enhanced legend with larger font
legend_handles = [Line2D([0], [0], color=clr, lw=4, label=lab)
                  for lab, clr in colors.items()]
ax.legend(handles=legend_handles, loc='upper right',
          bbox_to_anchor=(1.35, 1.08), fontsize=16, frameon=True,
          fancybox=True, shadow=True, framealpha=0.9,
          edgecolor='black', facecolor='white')

plt.tight_layout()

# Save with high quality settings for research paper
save_path = r'M:\Research\Published Research\Papers\Review Paper\Visual Aids'
plt.savefig(f'{save_path}\sensor_comparison_radar_chart.png', dpi=600, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig(f'{save_path}\sensor_comparison_radar_chart.pdf', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none')

plt.show()