"""
================================================================================
 Nanofluid Thermophysical Property Prediction — ML Benchmark
 (with explicit NANOFLUID-TYPE EFFECT analysis)
================================================================================
 Trains SIX algorithms independently on each target property dataset:
   1. ANN      — Artificial Neural Network (MLP)
   2. ANFIS    — Adaptive Neuro-Fuzzy Inference System (first-order Takagi–Sugeno)
   3. GPR      — Gaussian Process Regression
   4. RF       — Random Forest
   5. XGBoost  — Extreme Gradient Boosting
   6. Ensemble — Stacking ensemble of all five base models (Ridge meta-learner)

 NANOFLUID-TYPE EFFECT (new in this version)
 -------------------------------------------
 The dataset contains TWO ternary hybrid nanofluids:
      Fe3O4-Al2O3-TiO2  (Nanofluid_Code = 1)
      Fe2O3-Al2O3-TiO2  (Nanofluid_Code = 0)
 The script now quantifies and visualizes how nanofluid type affects both
 the property values and the model performance:
   (a) PER-TYPE METRICS  — every model is scored on the whole test set AND
       separately on each nanofluid type's test samples
       -> extra '<prop> byType' sheets in metrics_summary.xlsx
   (b) TYPE-COLORED PARITY PLOTS — test points colored by nanofluid type
       so type-dependent bias is visible at a glance
   (c) TYPE-EFFECT CURVES — the best model predicts the property over the
       full 10–50 °C range at several concentrations for BOTH fluids on the
       same axes, plus the relative difference Fe3O4 vs Fe2O3 (%)
       -> results/type_effect_<prop>.png and type_effect_<prop>.csv
   (d) OPTIONAL PER-TYPE MODELS — set RUN_PER_TYPE=True to also run the full
       six-model benchmark separately for each nanofluid type (2-feature
       models: Temperature + Concentration), letting you compare a single
       combined model against dedicated per-fluid models.

 Outputs (in ./results/):
   metrics_summary.xlsx            overall + per-type metrics + rankings
   predictions_<prop>.csv          test predictions with type metadata
   parity_<prop>.png               parity plots, points colored by type
   metric_bars_<prop>.png          RMSE & R² comparison
   type_effect_<prop>.png/.csv     Fe3O4 vs Fe2O3 property curves & % diff
   models/<prop>_<model>.joblib    trained models

 DATA LAYOUT
 -----------
 ONE workbook 'nanofluid_data.xlsx' with FOUR sheets:
   Thermal_Conductivity | Viscosity | Electrical_Conductivity | pH
 Each sheet: the 7 metadata columns from your table + 1 target column
 (target auto-detected). No 'Split' column needed — an 80/20 split
 stratified jointly by Nanofluid_Type × Temperature is made (random_state=42).

 Run:  python nanofluid_ml_benchmark.py
================================================================================
"""
import os, re, sys, warnings, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RNG = 42
OUT = "results"
os.makedirs(f"{OUT}/models", exist_ok=True)

# ------------------------------------------------------------------ CONFIG ---
WORKBOOK = "nanofluid_data.xlsx"          # one file, four sheets (unchanged)

DATA_FILES = {
    "Thermal Conductivity":    dict(path=WORKBOOK, sheet="Thermal_Conductivity",
                                    target="Thermal Conductivity"),
    "Viscosity":               dict(path=WORKBOOK, sheet="Viscosity",
                                    target="Viscosity"),
    "Electrical Conductivity": dict(path=WORKBOOK, sheet="Electrical_Conductivity",
                                    target="Electrical Conductivity"),
    "pH":                      dict(path=WORKBOOK, sheet="pH",
                                    target="pH"),
}

# Columns (exactly as in your table)
ID_COL     = "Sample_ID"
TYPE_COL   = "Nanofluid_Type"
CODE_COL   = "Nanofluid_Code"
TEMP_COL   = "Temperature (°C)"
TEMPN_COL  = "Temperature_Norm"
CONC_COL   = "Concentration (vol.%)"
CONCN_COL  = "Concentration_Norm"
META_COLS  = {ID_COL, TYPE_COL, CODE_COL, TEMP_COL, TEMPN_COL,
              CONC_COL, CONCN_COL, "Split"}

# The two nanofluid types and their codes (edit here if you add more fluids;
# everything downstream — per-type metrics, colors, effect curves — follows)
NANOFLUID_TYPES = {
    1: "Fe3O4-Al2O3-TiO2",
    0: "Fe2O3-Al2O3-TiO2",
}
TYPE_COLORS = {"Fe3O4-Al2O3-TiO2": "#d62728",   # red  (magnetite hybrid)
               "Fe2O3-Al2O3-TiO2": "#1f77b4"}   # blue (hematite hybrid)

USE_NORMALIZED = True    # feed Temperature_Norm / Concentration_Norm as inputs
RUN_PER_TYPE   = False   # True -> ALSO run the full benchmark per nanofluid type
EFFECT_CONCS   = [0.00625, 0.05, 0.1, 0.3]   # vol.% curves in type-effect plot

# Raw ranges used to (de)normalize for the type-effect prediction grid
T_MIN, T_MAX = 10.0, 50.0
C_MIN, C_MAX = 0.00625, 0.3

# ------------------------------------------------------------- ANFIS MODEL ---
class ANFIS(RegressorMixin, BaseEstimator):
    """First-order Takagi–Sugeno ANFIS (grid partition, Gaussian MFs,
    regularized least-squares consequents)."""
    def __init__(self, n_mf=3, lam=1e-3):
        self.n_mf, self.lam = n_mf, lam
    def _design(self, X):
        mu = [np.exp(-0.5*((X[:, j:j+1]-self.cen_[j][None, :])/self.sig_[j])**2)
              for j in range(X.shape[1])]
        W = np.ones((X.shape[0], len(self.rules_)))
        for r, rule in enumerate(self.rules_):
            for j, m in enumerate(rule):
                W[:, r] *= mu[j][:, m]
        W /= (W.sum(1, keepdims=True) + 1e-12)
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.einsum("nr,nd->nrd", W, Xa).reshape(X.shape[0], -1)
    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        self.sx_ = StandardScaler().fit(X); Xs = self.sx_.transform(X)
        lo, hi = Xs.min(0), Xs.max(0)
        self.cen_ = [np.linspace(lo[j], hi[j], self.n_mf) for j in range(Xs.shape[1])]
        self.sig_ = [(hi[j]-lo[j])/(self.n_mf-1+1e-9)*0.75 + 1e-9 for j in range(Xs.shape[1])]
        self.rules_ = list(product(range(self.n_mf), repeat=Xs.shape[1]))
        Phi = self._design(Xs)
        A = Phi.T @ Phi + self.lam*np.eye(Phi.shape[1])
        self.theta_ = np.linalg.solve(A, Phi.T @ y)
        return self
    def predict(self, X):
        Xs = self.sx_.transform(np.asarray(X, float))
        return self._design(Xs) @ self.theta_

# ---------------------------------------------------------------- METRICS ---
def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    rmse = float(np.sqrt(np.mean(e**2)))
    mae  = float(np.mean(np.abs(e)))
    mape = float(np.nanmean(np.abs(e/np.where(np.abs(y) < 1e-12, np.nan, y)))*100)
    sst  = np.sum((y - y.mean())**2)
    r2   = float(1 - np.sum(e**2)/sst) if sst > 0 else np.nan
    nse  = r2
    theil = float(np.sqrt(np.mean(e**2)) /
                  (np.sqrt(np.mean(y**2)) + np.sqrt(np.mean(p**2)) + 1e-12))
    return dict(RMSE=rmse, MAE=mae, MAPE=mape, R2=r2, NSE=nse, TheilU=theil)

# ------------------------------------------------------------- MODEL ZOO ----
def build_models(n_features):
    ann = Pipeline([("sc", StandardScaler()),
                    ("m", MLPRegressor(activation="tanh", solver="lbfgs",
                                       max_iter=20000, random_state=RNG))])
    gpr = Pipeline([("sc", StandardScaler()),
                    ("m", GaussianProcessRegressor(
                        kernel=ConstantKernel(1.0)*RBF(np.ones(n_features))
                               + WhiteKernel(1e-3),
                        normalize_y=True, random_state=RNG))])
    rf  = RandomForestRegressor(random_state=RNG)
    xgb = XGBRegressor(random_state=RNG, objective="reg:squarederror",
                       verbosity=0, n_jobs=-1)
    return {
        "ANN":     (ann,  {"m__hidden_layer_sizes": [(8,), (12,), (10, 6)],
                           "m__alpha": [1e-4, 1e-3, 1e-2]}),
        "ANFIS":   (ANFIS(), {"n_mf": [2, 3], "lam": [1e-4, 1e-3, 1e-2]}),
        "GPR":     (gpr,  {}),
        "RF":      (rf,   {"n_estimators": [300, 600],
                           "max_depth": [None, 6, 10],
                           "min_samples_leaf": [1, 2]}),
        "XGBoost": (xgb,  {"n_estimators": [300, 600], "max_depth": [3, 5],
                           "learning_rate": [0.05, 0.1],
                           "subsample": [0.8, 1.0]}),
    }

# ------------------------------------------------------------ DATA LOADER ---
def _find_target(df, wanted):
    if wanted in df.columns:
        return wanted
    key = re.sub(r"\W+", "", wanted.split("(")[0]).lower()
    fuzzy = [c for c in df.columns
             if key in re.sub(r"\W+", "", str(c)).lower()]
    if fuzzy:
        return fuzzy[0]
    extra = [c for c in df.columns if c not in META_COLS
             and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
    if extra:
        return extra[0]
    raise KeyError(
        f"No target column found. Sheet columns = {list(df.columns)}. "
        "Each sheet must contain the 7 metadata columns PLUS one target column.")

def load_dataset(cfg, only_type=None):
    """Load one property sheet. If only_type is a nanofluid-type string,
    restrict to that fluid (used by the optional per-type benchmark)."""
    path, sheet, wanted = cfg["path"], cfg.get("sheet"), cfg["target"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Place your workbook beside this script "
            "or edit WORKBOOK at the top.")
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    required = [TYPE_COL, CODE_COL, TEMP_COL, CONC_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Sheet '{sheet}' is missing required columns {missing}. "
                       f"Found: {list(df.columns)}")

    if only_type is not None:
        df = df[df[TYPE_COL].astype(str).str.strip() == only_type]
        if df.empty:
            raise KeyError(f"No rows for nanofluid type '{only_type}' in '{sheet}'.")

    target = _find_target(df, wanted)

    if USE_NORMALIZED and TEMPN_COL in df.columns and CONCN_COL in df.columns:
        t_feat, c_feat = TEMPN_COL, CONCN_COL
    else:
        t_feat, c_feat = TEMP_COL, CONC_COL

    code = pd.to_numeric(df[CODE_COL], errors="coerce")
    if code.isna().any():
        code = pd.Series(pd.factorize(df[TYPE_COL])[0], index=df.index)

    feat = {
        "Temperature":   pd.to_numeric(df[t_feat], errors="coerce"),
        "Concentration": pd.to_numeric(df[c_feat], errors="coerce"),
    }
    if only_type is None:                 # combined model: type code is a feature
        feat["Nanofluid_Code"] = code.astype(float)
    X = pd.DataFrame(feat)
    y = pd.to_numeric(df[target], errors="coerce")
    ok = X.notna().all(1) & y.notna()
    X, y, df = X[ok], y[ok], df[ok]

    if "Split" in df.columns:
        s = df["Split"].astype(str).str.lower().str.strip()
        tr, te = s.str.startswith("train"), s.str.startswith("test")
        if tr.any() and te.any():
            return (X[tr].values, X[te].values, y[tr].values, y[te].values,
                    target, df[te])

    strat = df[TYPE_COL].astype(str) + "_" + df[TEMP_COL].astype(str)
    if strat.value_counts().min() < 2:
        strat = df[TYPE_COL] if only_type is None else df[TEMP_COL].astype(str)
    if strat.value_counts().min() < 2:
        strat = None
    Xtr, Xte, ytr, yte, _, dfe = train_test_split(
        X.values, y.values, df, test_size=0.2, random_state=RNG, stratify=strat)
    return Xtr, Xte, ytr, yte, target, dfe

# ------------------------------------------------ NANOFLUID-TYPE EFFECT -----
def norm_T(t):  return (np.asarray(t, float) - T_MIN) / (T_MAX - T_MIN)
def norm_C(c):  return (np.asarray(c, float) - C_MIN) / (C_MAX - C_MIN)

def _effect_grid(conc, code, T):
    """Prediction grid for one fluid at one concentration over temperatures T."""
    if USE_NORMALIZED:
        return np.column_stack([norm_T(T), np.full_like(T, norm_C(conc)),
                                np.full_like(T, float(code))])
    return np.column_stack([T, np.full_like(T, conc),
                            np.full_like(T, float(code))])

def type_effect_analysis(prop, tag, fitted, model_order, best_name, target):
    """Nanofluid-type effect for ALL SIX models (ANN, ANFIS, GPR, RF,
    XGBoost, Ensemble): each model predicts the property over the full
    temperature range at several concentrations for BOTH nanofluid types.
    Saves:
      type_effect_<prop>.png       2x3 grid, one panel per model, both fluids
      type_effect_diff_<prop>.png  relative Fe3O4-vs-Fe2O3 difference (%)
                                   from the best model
      type_effect_<prop>.csv       every model x fluid x T x conc prediction
    """
    T = np.linspace(T_MIN, T_MAX, 81)
    styles = ["-", "--", "-.", ":"]
    rows = []

    # ---- 2x3 grid: one panel per model, curves colored by nanofluid type ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5)); axes = axes.ravel()
    for ax, name in zip(axes, model_order):
        model = fitted[name]
        for si, conc in enumerate(EFFECT_CONCS):
            for code, tname in NANOFLUID_TYPES.items():
                p = model.predict(_effect_grid(conc, code, T))
                ax.plot(T, p, styles[si % len(styles)],
                        color=TYPE_COLORS.get(tname, None), lw=1.4)
                for t, v in zip(T, p):
                    rows.append(dict(Property=prop, Model=name,
                                     Nanofluid_Type=tname,
                                     Temperature_C=t, Concentration_vol=conc,
                                     Predicted=v))
        ax.set_title(name + ("  (best)" if name == best_name else ""),
                     fontsize=10)
        ax.set_xlabel("Temperature (°C)"); ax.set_ylabel(target)
    # one shared legend: colors = fluids, line styles = concentrations
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=TYPE_COLORS.get(tn), lw=2, label=tn)
               for tn in NANOFLUID_TYPES.values()]
    handles += [Line2D([0], [0], color="k", lw=1.4,
                       linestyle=styles[si % len(styles)],
                       label=f"{conc} vol.%")
                for si, conc in enumerate(EFFECT_CONCS)]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(f"Nanofluid-type effect on {prop} — all models", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(f"{OUT}/type_effect_{tag}.png", dpi=200); plt.close(fig)

    # ---- relative difference between fluids (best model) -------------------
    t1, t0 = list(NANOFLUID_TYPES.values())[0], list(NANOFLUID_TYPES.values())[1]
    best_model = fitted[best_name]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for si, conc in enumerate(EFFECT_CONCS):
        p1 = best_model.predict(_effect_grid(conc, 
                 [k for k, v in NANOFLUID_TYPES.items() if v == t1][0], T))
        p0 = best_model.predict(_effect_grid(conc,
                 [k for k, v in NANOFLUID_TYPES.items() if v == t0][0], T))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = 100.0 * (p1 - p0) / np.where(np.abs(p0) < 1e-12, np.nan, p0)
        ax.plot(T, rel, styles[si % len(styles)], lw=1.6, label=f"{conc} vol.%")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(f"({t1} − {t0}) / {t0}  (%)")
    ax.set_title(f"Relative type difference — {prop}\n(best model: {best_name})",
                 fontsize=10)
    ax.legend(fontsize=7.5, title="Concentration")
    fig.tight_layout()
    fig.savefig(f"{OUT}/type_effect_diff_{tag}.png", dpi=200); plt.close(fig)

    pd.DataFrame(rows).to_csv(f"{OUT}/type_effect_{tag}.csv", index=False)

# ------------------------------------------------------------- BENCHMARK ----
def run_property(prop, cfg, only_type=None):
    label = prop if only_type is None else f"{prop} [{only_type}]"
    print(f"\n{'='*70}\n {label}\n{'='*70}")
    Xtr, Xte, ytr, yte, target, df_te = load_dataset(cfg, only_type=only_type)
    print(f"  train n={len(ytr)}  test n={len(yte)}  target='{target}'  "
          f"features={'normalized' if USE_NORMALIZED else 'raw'}"
          + ("" if only_type is None else f"  (single-fluid model)"))
    zoo = build_models(Xtr.shape[1])
    cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
    fitted, rows, preds = {}, [], {}

    for name, (est, grid) in zoo.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=cv,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(Xtr, ytr); model = gs.best_estimator_
            print(f"  {name:<8s} best params: {gs.best_params_}")
        else:
            model = clone(est).fit(Xtr, ytr)
        fitted[name] = model
        p = model.predict(Xte); preds[name] = p
        rows.append(dict(Model=name, **metrics(yte, p)))

    stack = StackingRegressor(
        estimators=[(k, clone(v)) for k, v in fitted.items()],
        final_estimator=RidgeCV(alphas=np.logspace(-6, 2, 25)),
        cv=cv, n_jobs=-1).fit(Xtr, ytr)
    fitted["Ensemble"] = stack
    p = stack.predict(Xte); preds["Ensemble"] = p
    rows.append(dict(Model="Ensemble", **metrics(yte, p)))

    res = pd.DataFrame(rows).set_index("Model")
    ranks = pd.DataFrame({
        m: res[m].rank(ascending=m not in ("R2", "NSE")) for m in res.columns})
    res["Avg_Rank"] = ranks.mean(1)
    res["Overall_Rank"] = res["Avg_Rank"].rank(method="first").astype(int)
    res = res.sort_values("Overall_Rank")
    print(res.round(4).to_string())
    print(f"  >> Best model for {label}: {res.index[0]}")

    # ---------------- (a) per-nanofluid-type metrics on the test set --------
    res_type = None
    types_te = df_te[TYPE_COL].astype(str).str.strip().values
    if only_type is None and len(set(types_te)) > 1:
        trows = []
        for tname in NANOFLUID_TYPES.values():
            mask = types_te == tname
            if mask.sum() < 2:
                continue
            for name in res.index:
                trows.append(dict(Nanofluid_Type=tname, Model=name,
                                  n_test=int(mask.sum()),
                                  **metrics(yte[mask], preds[name][mask])))
        res_type = pd.DataFrame(trows)
        print("\n  Per-nanofluid-type test metrics (RMSE | R²):")
        for tname in NANOFLUID_TYPES.values():
            sub = res_type[res_type.Nanofluid_Type == tname]
            if sub.empty:
                continue
            brief = "  ".join(f"{r.Model}:{r.RMSE:.4g}|{r.R2:.3f}"
                              for r in sub.itertuples())
            print(f"    {tname}: {brief}")

    # ---------------- save predictions with type metadata -------------------
    tag = re.sub(r"\W+", "_", label.lower())
    out = pd.DataFrame({"y_true": yte, **preds})
    for c in (ID_COL, TYPE_COL, TEMP_COL, CONC_COL):
        if c in df_te.columns:
            out.insert(0, c, df_te[c].values)
    out.to_csv(f"{OUT}/predictions_{tag}.csv", index=False)
    for k, v in fitted.items():
        joblib.dump(v, f"{OUT}/models/{tag}_{k}.joblib")

    # ---------------- (b) parity plots colored by nanofluid type ------------
    fig, axes = plt.subplots(2, 3, figsize=(11, 7)); axes = axes.ravel()
    lo = min(yte.min(), *(np.min(p) for p in preds.values()))
    hi = max(yte.max(), *(np.max(p) for p in preds.values()))
    pad = 0.05*(hi-lo) if hi > lo else 0.05
    for ax, name in zip(axes, list(res.index)):
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], "k-", lw=0.8)
        if only_type is None:
            for tname in NANOFLUID_TYPES.values():
                mask = types_te == tname
                if mask.any():
                    ax.scatter(yte[mask], preds[name][mask], s=20, alpha=0.8,
                               color=TYPE_COLORS.get(tname),
                               edgecolor="k", lw=0.3, label=tname)
        else:
            ax.scatter(yte, preds[name], s=20, alpha=0.8,
                       color=TYPE_COLORS.get(only_type, "#333"),
                       edgecolor="k", lw=0.3)
        ax.set_title(f"{name}  (R²={res.loc[name,'R2']:.3f})", fontsize=9)
        ax.set_xlabel("Measured"); ax.set_ylabel("Predicted")
    if only_type is None:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8,
                   frameon=False)
        fig.suptitle(f"Parity plots — {prop} (colored by nanofluid type)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        fig.suptitle(f"Parity plots — {label}", fontsize=11)
        fig.tight_layout()
    fig.savefig(f"{OUT}/parity_{tag}.png", dpi=200); plt.close(fig)

    # ---------------- metric bars -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    res["RMSE"].plot.bar(ax=axes[0], edgecolor="k"); axes[0].set_ylabel("RMSE")
    res["R2"].plot.bar(ax=axes[1], color="#2ca02c", edgecolor="k"); axes[1].set_ylabel("R²")
    for ax in axes: ax.tick_params(axis="x", rotation=45)
    fig.suptitle(label); fig.tight_layout()
    fig.savefig(f"{OUT}/metric_bars_{tag}.png", dpi=200); plt.close(fig)

    # ---------------- grouped metric bars per nanofluid type ----------------
    if res_type is not None and not res_type.empty:
        piv_rmse = res_type.pivot(index="Model", columns="Nanofluid_Type",
                                  values="RMSE").reindex(res.index)
        piv_r2   = res_type.pivot(index="Model", columns="Nanofluid_Type",
                                  values="R2").reindex(res.index)
        colors = [TYPE_COLORS.get(c, None) for c in piv_rmse.columns]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        piv_rmse.plot.bar(ax=axes[0], color=colors, edgecolor="k", width=0.75)
        axes[0].set_ylabel("RMSE"); axes[0].set_title("RMSE by nanofluid type",
                                                      fontsize=10)
        piv_r2.plot.bar(ax=axes[1], color=colors, edgecolor="k", width=0.75)
        axes[1].set_ylabel("R²"); axes[1].set_title("R² by nanofluid type",
                                                    fontsize=10)
        ymin = max(0.0, np.nanmin(piv_r2.values) - 0.02)
        axes[1].set_ylim(ymin, 1.0)
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=7, title=None)
        fig.suptitle(f"{prop} — model performance per nanofluid type",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{OUT}/metric_bars_by_type_{tag}.png", dpi=200)
        plt.close(fig)

    # ------- (c) nanofluid-type effect curves for ALL SIX models ------------
    if only_type is None:
        type_effect_analysis(prop, tag, fitted, list(res.index),
                             res.index[0], target)

    return res, res_type

# ------------------------------------------------------------------- MAIN ---
if __name__ == "__main__":
    all_res, all_type_res, best = {}, {}, []
    for prop, cfg in DATA_FILES.items():
        try:
            r, rt = run_property(prop, cfg)
            all_res[prop] = r
            if rt is not None:
                all_type_res[prop] = rt
            best.append(dict(Property=prop, Best_Model=r.index[0],
                             RMSE=r.iloc[0]["RMSE"], R2=r.iloc[0]["R2"],
                             MAPE=r.iloc[0]["MAPE"]))
            # (d) optional dedicated single-fluid benchmarks
            if RUN_PER_TYPE:
                for tname in NANOFLUID_TYPES.values():
                    rp, _ = run_property(prop, cfg, only_type=tname)
                    all_res[f"{prop} [{tname}]"] = rp
        except (FileNotFoundError, KeyError) as e:
            print(f"[SKIP] {prop}: {e}")

    if all_res:
        with pd.ExcelWriter(f"{OUT}/metrics_summary.xlsx") as xw:
            for prop, r in all_res.items():
                r.round(6).to_excel(xw, sheet_name=prop[:31])
            for prop, rt in all_type_res.items():
                rt.round(6).to_excel(xw, sheet_name=(prop[:24] + " byType")[:31],
                                     index=False)
            pd.DataFrame(best).to_excel(xw, sheet_name="Best_per_Property",
                                        index=False)
        print(f"\nAll results saved in ./{OUT}/  "
              "(metrics_summary.xlsx incl. per-type sheets, predictions_*.csv, "
              "parity/metric plots, type_effect_*.png/.csv, models/)")
    else:
        print("\nNo dataset processed — check that 'nanofluid_data.xlsx' has the "
              "four sheets and that each sheet has the 7 metadata columns + 1 target.")
