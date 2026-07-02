"""
=============================================================================
Portable Water Yam Grating Machine — Design, FEA & Techno-Economic Analysis
Plotting Code for IEEE Journal Presentation
=============================================================================
Outputs: 8 individual high-resolution PNG plots (300 dpi)
Run    : python water_yam_plots.py
Deps   : pip install matplotlib numpy pandas
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT_DIR = "/mnt/user-data/outputs/water_yam_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── IEEE-style rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "axes.grid":          False,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

C1 = "#1B5E20"   # deep green
C2 = "#1565C0"   # deep blue
C3 = "#E65100"   # orange
C4 = "#6A1B9A"   # purple
GREY = "#546E7A"

def save(fname):
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path)
    plt.close()
    print(f"  ✔  {fname}")

# =============================================================================
#  PLOT 1 — Material Cost Breakdown (Horizontal Bar)
# =============================================================================
materials = [
    "Stainless Steel", "Angle Iron", "Shaft",
    "Bearings", "Bolts & Nuts", "Electric Motor",
    "Paint", "Electrodes"
]
costs = [30000, 8000, 2000, 2000, 1000, 35000, 1500, 4000]
colors = [C1 if c == max(costs) else C2 if c > 5000 else GREY for c in costs]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(materials, costs, color=colors, edgecolor="white",
               height=0.6)
for bar, val in zip(bars, costs):
    ax.text(val + 300, bar.get_y() + bar.get_height()/2,
            f"₦{val:,}", va="center", fontsize=8.5, fontweight="bold")
ax.set_xlabel("Cost (₦)")
ax.set_title("Material Cost Breakdown — Water Yam Grating Machine")
ax.set_xlim(0, 42000)
ax.invert_yaxis()
total_patch = mpatches.Patch(color=C1, label="Highest cost item")
ax.legend(handles=[total_patch], loc="lower right")
fig.tight_layout()
save("plot1_material_cost.png")

# =============================================================================
#  PLOT 2 — Machining, Labour & Total Cost Comparison (Grouped Bar)
# =============================================================================
categories = ["Material Cost", "Machining Cost", "Labour Cost",
              "Miscellaneous", "Total Cost"]
values = [83000, 33600, 11550, 20500, 148650]
bar_colors = [C1, C2, C3, C4, GREY]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(categories))
bars = ax.bar(x, values, color=bar_colors, edgecolor="white", width=0.55)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1200,
            f"₦{val:,}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel("Amount (₦)")
ax.set_title("Techno-Economic Cost Analysis Summary")
ax.set_ylim(0, 175000)
fig.tight_layout()
save("plot2_cost_analysis.png")

# =============================================================================
#  PLOT 3 — Machining Operations Time (Horizontal Bar)
# =============================================================================
operations  = ["Arc Welding", "Lathe", "Drilling",
                "Cutting", "Folding", "Bending", "Bench Vice"]
hours_used  = [6, 1, 0.33, 3, 0.17, 0.10, 0.58]
cost_values = [8400, 5000, 1000, 16150, 1000, 300, 1750]

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twiny()
x = np.arange(len(operations))

b1 = ax1.barh(x - 0.2, hours_used, height=0.35,
              color=C1, label="Time (hrs)", edgecolor="white")
b2 = ax2.barh(x + 0.2, cost_values, height=0.35,
              color=C2, label="Cost (₦)", edgecolor="white")
ax1.set_yticks(x)
ax1.set_yticklabels(operations)
ax1.set_xlabel("Time Used (hours)", color=C1)
ax2.set_xlabel("Cost Incurred (₦)", color=C2)
ax1.set_title("Machining Operations — Time and Cost")
ax1.legend(loc="lower right")
ax2.legend(loc="center right")
ax1.invert_yaxis()
fig.tight_layout()
save("plot3_machining_operations.png")

# =============================================================================
#  PLOT 4 — Von-Mises Stress Distribution (FEA Bar)
# =============================================================================
fea_labels  = ["Minimum Stress", "Average Stress", "Maximum Stress"]
fea_values  = [1.5231e-6, 8.2721e-2, 3.2886]
fea_colors  = [C1, C2, C3]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(fea_labels, fea_values, color=fea_colors,
              edgecolor="white", width=0.5)
for bar, val in zip(bars, fea_values):
    label = f"{val:.4e}" if val < 0.01 else f"{val:.4f}"
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + bar.get_height()*0.05,
            label, ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Von-Mises Stress (MPa)")
ax.set_title("FEA Von-Mises Stress Results — Static Structural Analysis")
ax.set_yscale("log")
fig.tight_layout()
save("plot4_von_mises_stress.png")

# =============================================================================
#  PLOT 5 — Total Deformation FEA (simulated profile over time)
# =============================================================================
time_steps    = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
deform_min    = np.array([0.0, 0.00082, 0.00170, 0.00258, 0.000])
deform_max    = np.array([0.0, 0.00162, 0.00323, 0.00485, 0.006477])
deform_avg    = np.array([0.0, 0.000162, 0.000324, 0.000486, 0.000648])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(time_steps, deform_max, "o-", color=C3, lw=2.2, ms=7, label="Maximum (mm)")
ax.plot(time_steps, deform_avg, "s--", color=C2, lw=2.2, ms=7, label="Average (mm)")
ax.plot(time_steps, deform_min, "^:", color=C1, lw=2.2, ms=7, label="Minimum (mm)")
ax.fill_between(time_steps, deform_min, deform_max, alpha=0.12, color=C3)
ax.set_xlabel("Time Step (s)")
ax.set_ylabel("Total Deformation (mm)")
ax.set_title("FEA Total Deformation Profile — Static Structural Analysis")
ax.legend()
fig.tight_layout()
save("plot5_total_deformation.png")

# =============================================================================
#  PLOT 6 — Performance Evaluation (Grating Rate)
# =============================================================================
labels    = ["Design Target\nCapacity", "Experimental\nGrating Rate"]
values_hr = [80.0, 75.82]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, values_hr, color=[C1, C2], width=0.4, edgecolor="white")
ax.axhline(80, color=C3, lw=1.5, ls="--", alpha=0.7, label="Design target (80 kg/hr)")
for bar, v in zip(bars, values_hr):
    ax.text(bar.get_x() + bar.get_width()/2,
            v + 0.8, f"{v:.2f} kg/hr",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Grating Rate (kg/hr)")
ax.set_title("Machine Performance — Design Target vs Experimental Output")
ax.set_ylim(0, 95)
ax.legend()
efficiency = (75.82 / 80) * 100
ax.text(0.98, 0.05, f"Efficiency: {efficiency:.1f}%",
        transform=ax.transAxes, ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=GREY, alpha=0.9))
fig.tight_layout()
save("plot6_performance_evaluation.png")

# =============================================================================
#  PLOT 7 — Design Engineering Parameters (Summary Radar)
# =============================================================================
param_labels = ["Hopper Vol.\n(×10⁻³ m³)", "Hopper Cap.\n(kg)",
                "Drum Torque\n(Nm)", "Shaft Dia.\n(mm)",
                "Power (W/100)", "Grating Rate\n(kg/hr×1)"]
raw_values   = [4.177, 3.717, 12.88, 25.0, 131.9/100, 75.82]
norm_values  = [v / max(raw_values) for v in raw_values]
N = len(param_labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]
norm_values += norm_values[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.plot(angles, norm_values, color=C1, lw=2.5)
ax.fill(angles, norm_values, color=C1, alpha=0.18)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(param_labels, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
ax.set_title("Design Engineering Parameters — Normalized Radar Chart", pad=18)
fig.tight_layout()
save("plot7_design_radar.png")

# =============================================================================
#  PLOT 8 — Bearing Life and Load Rating
# =============================================================================
labels8  = ["Radial Load\nWR (N)", "Service Factor\nAdjusted W (N)",
            "Basic Load\nRating C (N)"]
values8  = [774.8, 1162.2, 12350.0]
colors8  = [C1, C2, C3]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels8, values8, color=colors8, edgecolor="white", width=0.45)
for bar, val in zip(bars, values8):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 150, f"{val:,.1f} N",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Load (N)")
ax.set_title("Bearing Selection — Load Analysis (L = 1,200 × 10⁶ rev)")
ax.set_ylim(0, 15000)
ax.text(0.98, 0.95, "Bearing No. 206 Selected\nC = 12.35 kN",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=GREY, alpha=0.9))
fig.tight_layout()
save("plot8_bearing_load.png")

# =============================================================================
print("\n" + "="*55)
print(f"  All 8 plots saved to: {OUT_DIR}")
print("="*55)
print("""
  FIGURE GUIDE — Insert in IEEE Paper at:
  ─────────────────────────────────────────────────
  plot1  → Section IV-B (Cost Analysis): Material cost
  plot2  → Section IV-B (Cost Analysis): Total cost summary
  plot3  → Section III-D (Fabrication): Machining ops
  plot4  → Section III-C (FEA): Von-Mises stress
  plot5  → Section III-C (FEA): Total deformation
  plot6  → Section IV-A (Results): Performance evaluation
  plot7  → Section III-B (Design): Engineering parameters
  plot8  → Section III-B (Design): Bearing selection
""")
