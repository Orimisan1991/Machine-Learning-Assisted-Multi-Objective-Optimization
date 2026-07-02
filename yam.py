import os
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# TUKEY HSD IMPORT CHECK
# ============================================================

try:
    from scipy.stats import tukey_hsd
except ImportError:
    raise ImportError(
        "Your SciPy version does not support tukey_hsd.\n"
        "Upgrade using:\n"
        "pip install --upgrade scipy"
    )

# ============================================================
# IEEE STYLE SETTINGS
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.grid": False,
    "savefig.dpi": 300
})

warnings.filterwarnings("ignore")

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)

# ============================================================
# DATA
# ============================================================

hardness = {
    0: [65.16, 64.82, 64.26],
    5: [64.51, 66.52, 65.57],
    10: [63.60, 63.73, 62.60],
    20: [62.37, 63.07, 61.98]
}

impact = {
    0: [12.40, 12.86, 13.25],
    5: [13.81, 14.66, 14.13],
    10: [13.99, 14.15, 13.96],
    20: [13.45, 13.69, 13.99]
}

tensile = {
    0: [14.19, 13.23, 11.14],
    5: [8.71, 11.58, 14.94],
    10: [11.82, 9.73, 11.97],
    20: [8.43, 5.98, 10.17]
}

flexural = {
    0: [4.74124, 4.00908, 4.12373],
    5: [4.02875, 4.30361, 4.25922],
    10: [5.05330, 6.19459, 5.34934],
    20: [5.51232, 5.93083, 6.50285]
}

# ============================================================
# PROPERTY DEFINITIONS
# ============================================================

properties = [

    (
        "Hardness",
        "Rockwell Hardness (HR)",
        hardness,
        "#1F3A5F"
    ),

    (
        "Impact Strength",
        "Impact Strength (J/mm²)",
        impact,
        "#2A7F7E"
    ),

    (
        "Tensile Strength",
        "Tensile Strength (MPa)",
        tensile,
        "#A13D2F"
    ),

    (
        "Flexural Stress",
        "Flexural Stress (MPa)",
        flexural,
        "#C97B1F"
    )
]

levels = [0, 5, 10, 20]

# ============================================================
# FIGURE 16a–d
# ANOVA MEAN PLOTS
# ============================================================

for idx, (name, ylabel, data, color) in enumerate(properties):

    arrays = [data[level] for level in levels]

    F_value, p_value = stats.f_oneway(*arrays)

    means = [np.mean(data[level]) for level in levels]

    sds = [
        np.std(data[level], ddof=1)
        for level in levels
    ]

    n = 3

    tcrit = stats.t.ppf(
        0.975,
        df=n - 1
    )

    ci95 = [
        tcrit * sd / np.sqrt(n)
        for sd in sds
    ]

    x = np.arange(len(levels))

    fig, ax = plt.subplots(
        figsize=(3.5, 2.8),
        dpi=300
    )

    ax.errorbar(
        x,
        means,
        yerr=ci95,
        fmt='o-',
        color=color,
        ecolor='black',
        linewidth=1.2,
        elinewidth=1.0,
        capsize=4,
        capthick=1.0,
        markersize=5,
        markeredgecolor='black',
        markeredgewidth=0.5,
        zorder=3
    )

    for xi, level in zip(x, levels):

        jitter = np.random.uniform(
            -0.05,
            0.05,
            len(data[level])
        )

        ax.scatter(
            xi + jitter,
            data[level],
            facecolors='none',
            edgecolors='gray',
            linewidths=0.7,
            s=18,
            zorder=2
        )

    p_text = (
        f"p = {p_value:.4f}"
        if p_value >= 0.001
        else "p < 0.001"
    )

    sig = "*" if p_value < 0.05 else "ns"

    ax.text(
        0.03,
        0.97,
        f"F = {F_value:.3f}\n{p_text}\n{sig}",
        transform=ax.transAxes,
        va="top",
        fontsize=7
    )

    ax.set_xlabel(
        "Bamboo Powder Content (wt.%)"
    )

    ax.set_ylabel(ylabel)

    ax.set_xticks(x)
    ax.set_xticklabels(levels)

    ax.grid(False)

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        length=3
    )

    plt.tight_layout()

    fname = (
        f"Figure16_{chr(97+idx)}_"
        f"{name.replace(' ', '_')}"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            fname + ".png"
        ),
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            fname + ".tiff"
        ),
        bbox_inches="tight"
    )

    plt.close(fig)

# ============================================================
# FIGURE 17a–d
# TUKEY HSD PLOTS
# ============================================================

pair_labels = [
    "0 vs 5",
    "0 vs 10",
    "0 vs 20",
    "5 vs 10",
    "5 vs 20",
    "10 vs 20"
]

pair_indices = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3)
]

for idx, (name, ylabel, data, color) in enumerate(properties):

    arrays = [data[level] for level in levels]

    result = tukey_hsd(*arrays)

    try:
        ci = result.confidence_interval(
            confidence_level=0.95
        )
    except TypeError:
        ci = result.confidence_interval()

    diffs = []
    lows = []
    highs = []
    pvals = []

    for i, j in pair_indices:

        diffs.append(
            float(result.statistic[i, j])
        )

        lows.append(
            float(ci.low[i, j])
        )

        highs.append(
            float(ci.high[i, j])
        )

        pvals.append(
            float(result.pvalue[i, j])
        )

    ypos = np.arange(
        len(pair_labels)
    )[::-1]

    fig, ax = plt.subplots(
        figsize=(3.5, 2.8),
        dpi=300
    )

    for y, diff, lo, hi, pv in zip(
        ypos,
        diffs,
        lows,
        highs,
        pvals
    ):

        point_color = (
            "#A13D2F"
            if pv < 0.05
            else "#666666"
        )

        ax.errorbar(
            diff,
            y,
            xerr=[
                [diff - lo],
                [hi - diff]
            ],
            fmt='o',
            color=point_color,
            ecolor=point_color,
            elinewidth=1.2,
            capsize=4,
            capthick=1.0,
            markersize=5,
            markeredgecolor='black',
            markeredgewidth=0.5
        )

    ax.axvline(
        0,
        color='black',
        linestyle='--',
        linewidth=0.8
    )

    ax.set_yticks(ypos)
    ax.set_yticklabels(pair_labels)

    ax.set_xlabel(
        f"Mean Difference in {name}"
    )

    ax.grid(False)

    ax.tick_params(
        direction="in",
        top=True,
        right=True,
        length=3
    )

    plt.tight_layout()

    fname = (
        f"Figure17_{chr(97+idx)}_"
        f"{name.replace(' ', '_')}"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            fname + ".png"
        ),
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            fname + ".tiff"
        ),
        bbox_inches="tight"
    )

    plt.close(fig)

print("=" * 60)
print("IEEE FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
print("Figure16_a-d : ANOVA Mean Plots")
print("Figure17_a-d : Tukey HSD Multiple Comparison Plots")
print(f"Output Folder : {OUTPUT_DIR}")
print("Formats : PNG and TIFF")
print("Total Figures : 8")
print("=" * 60)