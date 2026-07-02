"""
=============================================================================
Machine Learning-Based Prediction of Mechanical Properties of
Bamboo Powder-Reinforced PBS Green Composites
=============================================================================
Models: Random Forest | XGBoost | Artificial Neural Network (ANN)
Targets: Tensile Strength | Flexural Strength | Hardness | Impact Strength
=============================================================================
Requirements:
    pip install numpy pandas scikit-learn xgboost tensorflow matplotlib seaborn
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import os

# ─────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ─────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# Output folder
os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────
# 1.  DATASET
# ─────────────────────────────────────────────
# Three replicates per loading level
raw_data = {
    "Bamboo_pct": [0, 0, 0, 5, 5, 5, 10, 10, 10, 20, 20, 20],
    "Tensile":    [14.19108, 13.23058, 11.14138,
                    8.70904, 11.58424, 14.94237,
                   11.82356,  9.72709, 11.96903,
                    8.43114,  5.97639, 10.17000],
    "Flexural":   [3.97348, 4.58523, 4.31535,
                   4.06532, 4.22875, 4.29750,
                   4.99372, 5.49584, 6.10767,
                   5.55638, 6.45523, 5.93439],
    "Hardness":   [65.16037, 64.74724, 64.82117,
                   64.50748, 65.53171, 66.51754,
                   63.60067, 63.31122, 63.72942,
                   62.37243, 62.47250, 63.06814],
    "Impact":     [12.39612, 12.83699, 12.86082,
                   13.81485, 14.20038, 14.65807,
                   13.99403, 14.03337, 14.14561,
                   13.44853, 13.70879, 13.69164],
}

df = pd.DataFrame(raw_data)

# Mean summary (used for curve-fitting plots)
mean_df = df.groupby("Bamboo_pct").mean().reset_index()

TARGETS    = ["Tensile", "Flexural", "Hardness", "Impact"]
TARGET_LABELS = {
    "Tensile":  "Tensile Strength (MPa)",
    "Flexural": "Flexural Stress (MPa)",
    "Hardness": "Rockwell Hardness",
    "Impact":   "Impact Strength (J/mm²)",
}
COLORS = {
    "Random Forest": "#2ecc71",
    "XGBoost":       "#3498db",
    "ANN":           "#e74c3c",
}
PALETTE = ["#2d6a4f", "#52b788", "#b7e4c7", "#d8f3dc"]

print("=" * 65)
print("  BAMBOO POWDER / PBS GREEN COMPOSITES — ML ANALYSIS")
print("=" * 65)
print(f"\nDataset: {len(df)} observations | {len(TARGETS)} target properties")
print("\n── Mean Mechanical Properties by Loading Level ──")
print(mean_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ─────────────────────────────────────────────
# 2.  PRE-PROCESSING
# ─────────────────────────────────────────────
X = df[["Bamboo_pct"]].values
Y = df[TARGETS].values

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)


# ─────────────────────────────────────────────
# 3.  HELPER: METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


# ─────────────────────────────────────────────
# 4.  RANDOM FOREST
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  [1/3]  RANDOM FOREST  (Leave-One-Out CV)")
print("─" * 65)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=2,
    max_features=1.0,
    random_state=42
)

loo = LeaveOneOut()
rf_preds = np.zeros_like(Y)
for train_idx, test_idx in loo.split(X_scaled):
    rf_model.fit(X_scaled[train_idx], Y_scaled[train_idx])
    pred_scaled = rf_model.predict(X_scaled[test_idx])
    rf_preds[test_idx] = scaler_Y.inverse_transform(pred_scaled.reshape(1, -1))

rf_metrics = {}
for i, t in enumerate(TARGETS):
    r2, rmse, mae = compute_metrics(Y[:, i], rf_preds[:, i])
    rf_metrics[t] = {"R2": r2, "RMSE": rmse, "MAE": mae}
    print(f"  {t:<10}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# Fit on full data for smooth prediction curve
rf_model.fit(X_scaled, Y_scaled)
X_dense = np.linspace(0, 20, 200).reshape(-1, 1)
X_dense_scaled = scaler_X.transform(X_dense)
rf_curve = scaler_Y.inverse_transform(rf_model.predict(X_dense_scaled))


# ─────────────────────────────────────────────
# 5.  XGBOOST
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  [2/3]  XGBOOST  (Leave-One-Out CV)")
print("─" * 65)

xgb_models = []
xgb_preds  = np.zeros_like(Y)

for i, t in enumerate(TARGETS):
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    loo_preds = []
    for train_idx, test_idx in loo.split(X_scaled):
        model.fit(X_scaled[train_idx], Y_scaled[train_idx, i])
        p = model.predict(X_scaled[test_idx])
        loo_preds.append(p[0])
    xgb_preds[:, i] = scaler_Y.inverse_transform(
        np.array(loo_preds).reshape(-1, 1) *
        (scaler_Y.data_range_[i]) + scaler_Y.data_min_[i]
        if False else
        np.column_stack([
            np.zeros((len(loo_preds), i)),
            np.array(loo_preds).reshape(-1, 1),
            np.zeros((len(loo_preds), len(TARGETS) - i - 1))
        ])
    )[:, i]

    # Fix: direct inverse per-target
    y_min = Y[:, i].min()
    y_max = Y[:, i].max()
    xgb_preds[:, i] = np.array(loo_preds) * (y_max - y_min) + y_min

    model.fit(X_scaled, Y_scaled[:, i])
    xgb_models.append(model)

    r2, rmse, mae = compute_metrics(Y[:, i], xgb_preds[:, i])
    print(f"  {t:<10}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

xgb_metrics = {}
for i, t in enumerate(TARGETS):
    r2, rmse, mae = compute_metrics(Y[:, i], xgb_preds[:, i])
    xgb_metrics[t] = {"R2": r2, "RMSE": rmse, "MAE": mae}

# Smooth XGBoost curve
xgb_curve = np.zeros((200, len(TARGETS)))
for i, model in enumerate(xgb_models):
    raw = model.predict(X_dense_scaled)
    y_min = Y[:, i].min(); y_max = Y[:, i].max()
    xgb_curve[:, i] = raw * (y_max - y_min) + y_min


# ─────────────────────────────────────────────
# 6.  ARTIFICIAL NEURAL NETWORK
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  [3/3]  ANN  (Leave-One-Out CV)")
print("─" * 65)

def build_ann(input_dim=1, output_dim=4):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(output_dim, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model

ann_preds = np.zeros_like(Y_scaled)
for train_idx, test_idx in loo.split(X_scaled):
    ann = build_ann()
    es  = callbacks.EarlyStopping(monitor="val_loss", patience=80,
                                   restore_best_weights=True, verbose=0)
    ann.fit(
        X_scaled[train_idx], Y_scaled[train_idx],
        validation_split=0.2,
        epochs=1000, batch_size=4, verbose=0,
        callbacks=[es]
    )
    ann_preds[test_idx] = ann.predict(X_scaled[test_idx], verbose=0)

ann_preds_orig = scaler_Y.inverse_transform(ann_preds)

ann_metrics = {}
for i, t in enumerate(TARGETS):
    r2, rmse, mae = compute_metrics(Y[:, i], ann_preds_orig[:, i])
    ann_metrics[t] = {"R2": r2, "RMSE": rmse, "MAE": mae}
    print(f"  {t:<10}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

# Full ANN for smooth curve
ann_full = build_ann()
es_full = callbacks.EarlyStopping(monitor="val_loss", patience=100,
                                   restore_best_weights=True, verbose=0)
ann_full.fit(X_scaled, Y_scaled, validation_split=0.2,
             epochs=1500, batch_size=4, verbose=0, callbacks=[es_full])
ann_curve = scaler_Y.inverse_transform(ann_full.predict(X_dense_scaled, verbose=0))


# ─────────────────────────────────────────────
# 7.  METRICS SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PERFORMANCE METRICS SUMMARY")
print("=" * 65)

records = []
for model_name, metrics in [("Random Forest", rf_metrics),
                             ("XGBoost",       xgb_metrics),
                             ("ANN",           ann_metrics)]:
    for t in TARGETS:
        records.append({
            "Model": model_name,
            "Property": t,
            "R²":    round(metrics[t]["R2"],   4),
            "RMSE":  round(metrics[t]["RMSE"], 4),
            "MAE":   round(metrics[t]["MAE"],  4),
        })

metrics_df = pd.DataFrame(records)
print(metrics_df.to_string(index=False))
metrics_df.to_csv("figures/model_metrics.csv", index=False)


# ═══════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════

# ── Style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

LOADINGS = [0, 5, 10, 20]


# ─────────────────────────────────────────────────────────
# PLOT 1 — Experimental mechanical properties (4-panel bar)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Experimental Mechanical Properties vs Bamboo Content",
             fontsize=15, fontweight="bold", y=1.01)

for ax, t in zip(axes.flat, TARGETS):
    vals  = [df[df.Bamboo_pct == p][t].values for p in LOADINGS]
    means = [v.mean() for v in vals]
    stds  = [v.std(ddof=1) for v in vals]
    bars  = ax.bar(LOADINGS, means, width=3.5, color=PALETTE,
                   edgecolor="white", linewidth=1.2, zorder=3)
    ax.errorbar(LOADINGS, means, yerr=stds, fmt="none",
                color="#333", capsize=5, linewidth=1.5, zorder=4)
    # Individual data points
    for xi, pct in enumerate(LOADINGS):
        ys = df[df.Bamboo_pct == pct][t].values
        ax.scatter([pct]*3, ys, color="white", edgecolor="#333",
                   s=40, zorder=5)
    ax.set_xlabel("Bamboo Powder Content (%)", fontsize=10)
    ax.set_ylabel(TARGET_LABELS[t], fontsize=10)
    ax.set_title(t, fontsize=11, fontweight="bold")
    ax.set_xticks(LOADINGS)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[means.index(val)]*1.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/01_experimental_properties.png", bbox_inches="tight")
plt.close()
print("\n  ✔ Saved: figures/01_experimental_properties.png")


# ─────────────────────────────────────────────────────────
# PLOT 2 — Prediction curves for each property (3 models)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("ML Prediction Curves vs Experimental Data",
             fontsize=15, fontweight="bold")

for ax, (i, t) in zip(axes.flat, enumerate(TARGETS)):
    # Experimental scatter
    ax.scatter(df["Bamboo_pct"], df[t], color="#333", s=55,
               zorder=5, label="Experimental", marker="D", edgecolors="white")

    # Mean markers
    ax.scatter(mean_df["Bamboo_pct"], mean_df[t], color="#333",
               s=120, zorder=6, marker="*", label="Mean")

    # Model curves
    ax.plot(X_dense.flatten(), rf_curve[:, i],
            color=COLORS["Random Forest"], lw=2.2,
            linestyle="-",  label="Random Forest")
    ax.plot(X_dense.flatten(), xgb_curve[:, i],
            color=COLORS["XGBoost"], lw=2.2,
            linestyle="--", label="XGBoost")
    ax.plot(X_dense.flatten(), ann_curve[:, i],
            color=COLORS["ANN"], lw=2.2,
            linestyle="-.", label="ANN")

    ax.set_xlabel("Bamboo Powder Content (%)", fontsize=10)
    ax.set_ylabel(TARGET_LABELS[t], fontsize=10)
    ax.set_title(t, fontsize=11, fontweight="bold")
    ax.set_xticks(LOADINGS)
    ax.legend(fontsize=8, framealpha=0.7)

plt.tight_layout()
plt.savefig("figures/02_prediction_curves.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/02_prediction_curves.png")


# ─────────────────────────────────────────────────────────
# PLOT 3 — Actual vs Predicted (scatter, all models)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
fig.suptitle("Actual vs. Predicted Values  (LOOCV)",
             fontsize=14, fontweight="bold")

model_data = [
    ("Random Forest", rf_preds,        rf_metrics),
    ("XGBoost",       xgb_preds,       xgb_metrics),
    ("ANN",           ann_preds_orig,  ann_metrics),
]

for row, (mname, preds, mets) in enumerate(model_data):
    for col, t in enumerate(TARGETS):
        ax = axes[row, col]
        y_true = Y[:, col]
        y_pred = preds[:, col]
        ax.scatter(y_true, y_pred, color=COLORS[mname],
                   edgecolors="white", s=70, zorder=3)
        lo = min(y_true.min(), y_pred.min()) * 0.97
        hi = max(y_true.max(), y_pred.max()) * 1.03
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Ideal", zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        r2 = mets[t]["R2"]
        ax.text(0.05, 0.93, f"R²={r2:.4f}", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=COLORS[mname])
        if row == 0: ax.set_title(t, fontsize=10, fontweight="bold")
        if col == 0: ax.set_ylabel(f"{mname}\nPredicted", fontsize=9)
        ax.set_xlabel("Actual", fontsize=8)

plt.tight_layout()
plt.savefig("figures/03_actual_vs_predicted.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/03_actual_vs_predicted.png")


# ─────────────────────────────────────────────────────────
# PLOT 4 — R² Grouped Bar Chart
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(TARGETS))
w = 0.25
for j, (mname, mets) in enumerate([("Random Forest", rf_metrics),
                                    ("XGBoost",       xgb_metrics),
                                    ("ANN",           ann_metrics)]):
    r2_vals = [mets[t]["R2"] for t in TARGETS]
    bars = ax.bar(x + (j-1)*w, r2_vals, width=w, label=mname,
                  color=COLORS[mname], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002, f"{val:.4f}",
                ha="center", va="bottom", fontsize=7.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], fontsize=10)
ax.set_ylabel("R² Score", fontsize=11)
ax.set_title("Model Comparison — R² by Mechanical Property", fontsize=13, fontweight="bold")
ax.set_ylim(0.92, 1.005)
ax.legend(fontsize=10)
ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")
plt.tight_layout()
plt.savefig("figures/04_r2_comparison.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/04_r2_comparison.png")


# ─────────────────────────────────────────────────────────
# PLOT 5 — RMSE & MAE Side-by-Side
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, metric_key, metric_label in [
        (axes[0], "RMSE", "RMSE"),
        (axes[1], "MAE",  "MAE")]:
    for j, (mname, mets) in enumerate([("Random Forest", rf_metrics),
                                        ("XGBoost",       xgb_metrics),
                                        ("ANN",           ann_metrics)]):
        vals = [mets[t][metric_key] for t in TARGETS]
        bars = ax.bar(x + (j-1)*w, vals, width=w, label=mname,
                      color=COLORS[mname], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([t for t in TARGETS], fontsize=10)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"Model Comparison — {metric_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("figures/05_rmse_mae_comparison.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/05_rmse_mae_comparison.png")


# ─────────────────────────────────────────────────────────
# PLOT 6 — Heatmap: Pearson Correlation
# ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[["Bamboo_pct"] + TARGETS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn",
            linewidths=0.6, ax=ax, vmin=-1, vmax=1,
            annot_kws={"size": 11, "weight": "bold"})
ax.set_title("Pearson Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/06_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/06_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────
# PLOT 7 — Residuals Plot (all models, all targets)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
fig.suptitle("Residuals (Actual − Predicted)  |  LOOCV",
             fontsize=14, fontweight="bold")

for row, (mname, preds) in enumerate([("Random Forest", rf_preds),
                                       ("XGBoost",       xgb_preds),
                                       ("ANN",           ann_preds_orig)]):
    for col, t in enumerate(TARGETS):
        ax = axes[row, col]
        residuals = Y[:, col] - preds[:, col]
        ax.stem(range(len(residuals)), residuals,
                linefmt=COLORS[mname], markerfmt="o", basefmt="k-")
        ax.axhline(0, color="black", lw=1.0)
        ax.set_xlabel("Sample Index", fontsize=8)
        if col == 0: ax.set_ylabel(f"{mname}\nResidual", fontsize=9)
        if row == 0: ax.set_title(t, fontsize=10, fontweight="bold")
        rmse = np.sqrt(mean_squared_error(Y[:, col], preds[:, col]))
        ax.text(0.02, 0.93, f"RMSE={rmse:.4f}",
                transform=ax.transAxes, fontsize=8, color=COLORS[mname])

plt.tight_layout()
plt.savefig("figures/07_residuals.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/07_residuals.png")


# ─────────────────────────────────────────────────────────
# PLOT 8 — RF Feature Importance (Permutation proxy for single feature)
# ─────────────────────────────────────────────────────────
# Since we have one input, show importance of bamboo % on each property
fig, ax = plt.subplots(figsize=(9, 5))
rf_model.fit(X_scaled, Y_scaled)
# Use impurity decrease summed over all outputs for each target separately
importances = []
for i, t in enumerate(TARGETS):
    rf_single = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    rf_single.fit(X_scaled, Y_scaled[:, i])
    importances.append(rf_single.feature_importances_[0])

bars = ax.barh(TARGETS, importances, color=[COLORS["Random Forest"]]*4,
               edgecolor="white", height=0.5)
ax.set_xlabel("Feature Importance (Mean Impurity Decrease)", fontsize=11)
ax.set_title("Random Forest — Feature Importance\n(Bamboo Content → Each Property)", fontsize=12, fontweight="bold")
ax.set_xlim(0, 1.1)
for bar, v in zip(bars, importances):
    ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/08_feature_importance.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/08_feature_importance.png")


# ─────────────────────────────────────────────────────────
# PLOT 9 — Radar / Spider Chart: model comparison per property
# ─────────────────────────────────────────────────────────
from matplotlib.patches import FancyArrowPatch

categories = TARGETS
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")

for mname, mets in [("Random Forest", rf_metrics),
                     ("XGBoost",       xgb_metrics),
                     ("ANN",           ann_metrics)]:
    values = [mets[t]["R2"] for t in TARGETS]
    values += values[:1]
    ax.plot(angles, values, color=COLORS[mname], lw=2.5, label=mname)
    ax.fill(angles, values, color=COLORS[mname], alpha=0.12)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(TARGETS, fontsize=12, fontweight="bold")
ax.set_ylim(0.90, 1.005)
ax.set_yticks([0.92, 0.94, 0.96, 0.98, 1.00])
ax.set_yticklabels(["0.92", "0.94", "0.96", "0.98", "1.00"], fontsize=8)
ax.set_title("R² Radar Chart — Model vs Property", fontsize=13,
             fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)
plt.tight_layout()
plt.savefig("figures/09_radar_r2.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/09_radar_r2.png")


# ─────────────────────────────────────────────────────────
# PLOT 10 — Pairplot (properties coloured by bamboo loading)
# ─────────────────────────────────────────────────────────
df_plot = df.copy()
df_plot["Loading"] = df_plot["Bamboo_pct"].astype(str) + " wt%"
g = sns.pairplot(df_plot[TARGETS + ["Loading"]], hue="Loading",
                 plot_kws={"s": 70, "edgecolor": "white"},
                 palette=["#1b4332", "#52b788", "#95d5b2", "#d8f3dc"],
                 diag_kind="kde", corner=False)
g.figure.suptitle("Pairplot of Mechanical Properties by Bamboo Loading",
                  y=1.02, fontsize=13, fontweight="bold")
plt.savefig("figures/10_pairplot.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/10_pairplot.png")


# ─────────────────────────────────────────────────────────
# PLOT 11 — Metrics Heatmap (all models × properties)
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, metric_key, cmap in [
        (axes[0], "R²",   "Greens"),
        (axes[1], "RMSE", "Reds_r"),
        (axes[2], "MAE",  "Blues_r")]:
    heat_data = metrics_df.pivot(index="Model", columns="Property", values=metric_key)
    heat_data = heat_data.loc[["Random Forest", "XGBoost", "ANN"]]
    sns.heatmap(heat_data, annot=True, fmt=".4f", cmap=cmap, ax=ax,
                linewidths=0.5, annot_kws={"size": 9, "weight": "bold"},
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"{metric_key}", fontsize=12, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")

fig.suptitle("Performance Metrics Heatmap — All Models × All Properties",
             fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("figures/11_metrics_heatmap.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/11_metrics_heatmap.png")


# ─────────────────────────────────────────────────────────
# PLOT 12 — Combined property trend (line plot with markers)
# ─────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(11, 6))
ax2 = ax1.twinx()

ax1.plot(LOADINGS, mean_df["Tensile"],  "o-",  color="#2d6a4f", lw=2.2, ms=9, label="Tensile (MPa)")
ax1.plot(LOADINGS, mean_df["Flexural"], "s--", color="#52b788", lw=2.2, ms=9, label="Flexural (MPa)")
ax2.plot(LOADINGS, mean_df["Hardness"], "^:",  color="#e76f51", lw=2.2, ms=9, label="Hardness (Rockwell)")
ax2.plot(LOADINGS, mean_df["Impact"],   "D-.", color="#457b9d", lw=2.2, ms=9, label="Impact (J/mm²)")

ax1.set_xlabel("Bamboo Powder Content (%)", fontsize=12)
ax1.set_ylabel("Tensile / Flexural Strength (MPa)", fontsize=11, color="#2d6a4f")
ax2.set_ylabel("Hardness (Rockwell) / Impact (J/mm²)", fontsize=11, color="#e76f51")
ax1.set_xticks(LOADINGS)
ax1.set_title("Effect of Bamboo Powder Content on All Mechanical Properties",
              fontsize=13, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)
plt.tight_layout()
plt.savefig("figures/12_combined_trends.png", bbox_inches="tight")
plt.close()
print("  ✔ Saved: figures/12_combined_trends.png")


# ─────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  BEST MODEL PER PROPERTY (by R²)")
print("=" * 65)
for t in TARGETS:
    scores = {
        "Random Forest": rf_metrics[t]["R2"],
        "XGBoost":       xgb_metrics[t]["R2"],
        "ANN":           ann_metrics[t]["R2"],
    }
    best = max(scores, key=scores.get)
    print(f"  {t:<12}  →  {best:<15}  R²={scores[best]:.4f}")

print("\n" + "=" * 65)
print("  ALL FIGURES SAVED IN: ./figures/")
print("=" * 65)
print("""
  FIGURE GUIDE
  ─────────────────────────────────────────────
  01  Experimental bar charts with error bars
  02  ML prediction curves (RF / XGBoost / ANN)
  03  Actual vs Predicted scatter (LOOCV)
  04  R² grouped bar comparison
  05  RMSE & MAE grouped bar comparison
  06  Pearson correlation heatmap
  07  Residuals stem plots
  08  RF feature importance
  09  Radar chart (R² per model × property)
  10  Pairplot coloured by bamboo loading
  11  Performance metrics heatmap (all models)
  12  Combined property trends (dual-axis)
""")
