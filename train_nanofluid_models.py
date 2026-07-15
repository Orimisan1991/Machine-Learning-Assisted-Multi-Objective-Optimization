"""
=============================================================================
 Nanofluid Thermophysical Property Prediction - ML Training & Evaluation
=============================================================================
 Trains 6 algorithms independently for each target property:
   1. Artificial Neural Network (ANN)
   2. Adaptive Neuro-Fuzzy Inference System (ANFIS)  [custom Takagi-Sugeno]
   3. Gaussian Process Regression (GPR)
   4. Random Forest (RF)
   5. Extreme Gradient Boosting (XGBoost)
   6. Ensemble Learning (stacked average of all base models)

 Evaluates each model with:
   RMSE, MAE, MAPE, R2, NSE, Theil's U
 and ranks the models per target property.

 USAGE:
   python train_nanofluid_models.py <data_file.csv|.xlsx>

 Data file requirements:
   - Input feature columns  : Temperature_Norm, Concentration_Norm
                              (+ Nanofluid_Code if >1 fluid present)
   - Target column(s)       : any column containing 'Conductivity',
                              'Viscosity', 'Density', 'Specific Heat', ...
                              (auto-detected; edit TARGET_KEYWORDS below)
   - Optional 'Split' column with values 'Train' / 'Test'.
     If absent (or no Test rows), an 80/20 random split is used.
=============================================================================
"""

import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

# Keywords used to auto-detect target property columns
TARGET_KEYWORDS = ["conductivity", "viscosity", "density", "specific heat",
                   "heat capacity"]

# Preferred (normalised) feature columns; falls back to raw columns
PREFERRED_FEATURES = ["Temperature_Norm", "Concentration_Norm"]
FALLBACK_FEATURES = ["Temperature (°C)", "Concentration (vol.%)"]


# ---------------------------------------------------------------------------
# 1. ANFIS  (first-order Takagi–Sugeno, Gaussian MFs, hybrid learning)
# ---------------------------------------------------------------------------
class ANFIS:
    """Compact Takagi-Sugeno ANFIS.

    - Gaussian membership functions per input (grid partition)
    - Premise parameters tuned by gradient descent
    - Consequent (linear) parameters solved by least squares each epoch
    """

    def __init__(self, n_mfs=3, epochs=300, lr=0.01, seed=RNG):
        self.n_mfs = n_mfs
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    # -- membership degrees for every input/MF ---------------------------
    def _memberships(self, X):
        # (n_samples, n_features, n_mfs)
        diff = X[:, :, None] - self.c[None, :, :]
        return np.exp(-0.5 * (diff / (self.s[None, :, :] + 1e-12)) ** 2)

    def _firing(self, X):
        mu = self._memberships(X)                    # (N, d, m)
        N, d, m = mu.shape
        # rule firing strength = product over inputs of chosen MF
        w = np.ones((N, self.n_rules))
        for r, combo in enumerate(self.rule_index):
            for j in range(d):
                w[:, r] *= mu[:, j, combo[j]]
        wsum = w.sum(axis=1, keepdims=True) + 1e-12
        return w / wsum, w

    def _design_matrix(self, X, wn):
        N, d = X.shape
        Xa = np.hstack([X, np.ones((N, 1))])         # affine inputs
        # (N, n_rules*(d+1))
        return (wn[:, :, None] * Xa[:, None, :]).reshape(N, -1)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        N, d = X.shape
        m = self.n_mfs

        # init MF centers evenly across each feature's range
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
        self.c = np.linspace(0, 1, m)[None, :] * span[:, None] + lo[:, None]
        self.s = np.tile((span / (2 * max(m - 1, 1)) + 1e-3)[:, None], (1, m))

        # full rule grid
        from itertools import product
        self.rule_index = list(product(range(m), repeat=d))
        self.n_rules = len(self.rule_index)

        for _ in range(self.epochs):
            wn, _ = self._firing(X)
            Phi = self._design_matrix(X, wn)
            # LSE for consequents (ridge-stabilised)
            A = Phi.T @ Phi + 1e-6 * np.eye(Phi.shape[1])
            self.theta = np.linalg.solve(A, Phi.T @ y)
            y_hat = Phi @ self.theta
            err = y_hat - y

            # numeric gradient step on premise params (fast, robust for small d)
            for P in (self.c, self.s):
                g = np.zeros_like(P)
                eps = 1e-4
                for idx in np.ndindex(P.shape):
                    old = P[idx]
                    P[idx] = old + eps
                    wn2, _ = self._firing(X)
                    e2 = self._design_matrix(X, wn2) @ self.theta - y
                    g[idx] = (np.mean(e2 ** 2) - np.mean(err ** 2)) / eps
                    P[idx] = old
                P -= self.lr * g
            self.s = np.clip(self.s, 1e-3, None)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        wn, _ = self._firing(X)
        return self._design_matrix(X, wn) @ self.theta


# ---------------------------------------------------------------------------
# 2. Performance metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / np.where(y_true == 0, 1e-12, y_true))) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    r2 = 1 - ss_res / ss_tot
    nse = 1 - ss_res / ss_tot          # NSE == R2 vs mean benchmark
    theil_u = rmse / (np.sqrt(np.mean(y_true ** 2)) +
                      np.sqrt(np.mean(y_pred ** 2)) + 1e-12)
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape,
            "R2": r2, "NSE": nse, "Theil's U": theil_u}


# ---------------------------------------------------------------------------
# 3. Model factory
# ---------------------------------------------------------------------------
def build_models(n_train):
    small = n_train < 30   # relax complexity for tiny datasets
    ann = MLPRegressor(hidden_layer_sizes=(16, 8) if small else (64, 32),
                       activation="relu", solver="lbfgs",
                       max_iter=5000, random_state=RNG)
    anfis = ANFIS(n_mfs=2 if small else 3, epochs=200 if small else 400)
    gpr = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)),
        normalize_y=True, n_restarts_optimizer=10, random_state=RNG)
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None if not small else 4,
        min_samples_leaf=1, random_state=RNG)
    xgb = XGBRegressor(
        n_estimators=400 if not small else 150,
        max_depth=4 if not small else 2,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=RNG, verbosity=0)
    return {"ANN": ann, "ANFIS": anfis, "GPR": gpr,
            "RF": rf, "XGBoost": xgb}


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------
def main(path):
    # ---- load -----------------------------------------------------------
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        # sniff separator (handles tab-separated pastes too)
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}\n")

    # ---- features -------------------------------------------------------
    feats = [c for c in PREFERRED_FEATURES if c in df.columns]
    if len(feats) < 2:
        feats = [c for c in FALLBACK_FEATURES if c in df.columns]
    if "Nanofluid_Code" in df.columns and df["Nanofluid_Code"].nunique() > 1:
        feats = ["Nanofluid_Code"] + feats
    print(f"Input features: {feats}")

    # ---- targets --------------------------------------------------------
    targets = [c for c in df.columns
               if any(k in c.lower() for k in TARGET_KEYWORDS)]
    print(f"Target properties detected: {targets}\n")

    # ---- split ----------------------------------------------------------
    if "Split" in df.columns and (df["Split"].str.lower() == "test").any():
        tr = df[df["Split"].str.lower() == "train"]
        te = df[df["Split"].str.lower() == "test"]
        print(f"Using provided Split column: {len(tr)} train / {len(te)} test")
    else:
        tr, te = train_test_split(df, test_size=0.2, random_state=RNG)
        print("No usable 'Test' rows found -> random 80/20 split "
              f"({len(tr)} train / {len(te)} test)")

    all_results = {}
    for target in targets:
        print("\n" + "=" * 70)
        print(f"TARGET PROPERTY: {target}")
        print("=" * 70)

        Xtr_raw, ytr = tr[feats].values.astype(float), tr[target].values
        Xte_raw, yte = te[feats].values.astype(float), te[target].values

        scaler = StandardScaler().fit(Xtr_raw)
        Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

        models = build_models(len(Xtr))
        preds_train, preds_test, rows = {}, {}, []

        for name, model in models.items():
            model.fit(Xtr, ytr)
            preds_train[name] = np.asarray(model.predict(Xtr)).ravel()
            preds_test[name] = np.asarray(model.predict(Xte)).ravel()

        # ---- Ensemble: ridge-stacked combination of base predictions ----
        P_tr = np.column_stack(list(preds_train.values()))
        P_te = np.column_stack(list(preds_test.values()))
        meta = RidgeCV(alphas=np.logspace(-4, 2, 20)).fit(P_tr, ytr)
        preds_train["Ensemble"] = meta.predict(P_tr)
        preds_test["Ensemble"] = meta.predict(P_te)

        # ---- evaluate ----------------------------------------------------
        for name in list(models) + ["Ensemble"]:
            m_tr = compute_metrics(ytr, preds_train[name])
            m_te = compute_metrics(yte, preds_test[name])
            rows.append({"Model": name,
                         **{f"Train {k}": v for k, v in m_tr.items()},
                         **{f"Test {k}": v for k, v in m_te.items()}})

        res = pd.DataFrame(rows).set_index("Model")

        # ---- rank on TEST metrics (lower better except R2 / NSE) ---------
        rank = pd.DataFrame(index=res.index)
        for col, asc in [("Test RMSE", True), ("Test MAE", True),
                         ("Test MAPE (%)", True), ("Test R2", False),
                         ("Test NSE", False), ("Test Theil's U", True)]:
            rank[col + " rank"] = res[col].rank(ascending=asc)
        res["Avg Rank"] = rank.mean(axis=1)
        res["Overall Rank"] = res["Avg Rank"].rank().astype(int)
        res = res.sort_values("Overall Rank")

        pd.set_option("display.float_format", lambda v: f"{v:,.5f}")
        print("\n--- Test-set performance (ranked) ---")
        show_cols = ["Test RMSE", "Test MAE", "Test MAPE (%)",
                     "Test R2", "Test NSE", "Test Theil's U",
                     "Avg Rank", "Overall Rank"]
        print(res[show_cols].to_string())
        best = res.index[0]
        print(f"\n>>> Best-performing algorithm for '{target}': {best}")

        safe = "".join(ch if ch.isalnum() else "_" for ch in target)[:40]
        out = f"model_evaluation_{safe}.csv"
        res.to_csv(out)
        print(f"Full metrics table saved to: {out}")
        all_results[target] = res

    return all_results


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "nanofluid_data.csv"
    main(data_path)
