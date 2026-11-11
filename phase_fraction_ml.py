"""
Phase-fraction regression benchmark for the 128 ThermoCalc alloy compositions.

What this script does
---------------------
1. Reads the ThermoCalc export (one sheet per temperature, 800-1000 °C).
2. Keeps every phase that is non-zero for at least one alloy at at least one
   temperature; wherever that phase is absent it is filled with 0.
3. Splits the *alloy compositions* 80/20 into train/test (random, seeded).
   Only AFTER the split are the temperatures attached, so every alloy carries
   all 9 temperatures and no alloy appears in both train and test.
4. Trains every switched-on regressor on the same training set, scores them on
   the same grouped cross-validation folds (grouped by alloy) and on the same
   test set.
5. Writes comparison tables, per-phase metrics, predictions and plots.

Inputs  (X): alloy composition in mass % (+ evaluation temperature)
Outputs (y): equilibrium phase amounts [mol] (≈ mole fractions, sum ≈ 1)

Models (each has an on/off switch, see MODEL_SWITCHES or --on / --off)
-----------------------------------------------------------------------
    mean, ridge, extratrees, xgboost, gpr, mlp

"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ID_COL = "Composition_id"
FAMILY_COL = "Family"
TEMP_COL = "Temperature_C"
PHASE_PREFIX = "Amount of "
COMP_PREFIX = "Mass percent "


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def load_thermocalc(path):
    """Read every temperature sheet and stack them into one long table."""
    sheets = pd.read_excel(path, sheet_name=None)
    frames = []
    for name, df in sheets.items():
        df = df.copy()
        # Use the temperature stored in the sheet; fall back to the sheet name.
        if "Evaluation temperature" in df.columns:
            df[TEMP_COL] = df["Evaluation temperature"].astype(float)
        else:
            df[TEMP_COL] = float(name)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    phase_cols = [c for c in data.columns if c.startswith(PHASE_PREFIX)]
    comp_cols = [c for c in data.columns if c.startswith(COMP_PREFIX)]
    data[phase_cols] = data[phase_cols].fillna(0.0)
    return data, phase_cols, comp_cols


FCC_SETS = ["Amount of FCC_A1 [mol]", "Amount of FCC_A1#2 [mol]", "Amount of FCC_L12 [mol]"]
AUSTENITE = "Amount of FCC_austenite [mol]"
FCC_MINOR = "Amount of FCC_MC_carbide [mol]"


def harmonize_fcc(data, phase_cols):
    """
    ThermoCalc numbers FCC composition sets (FCC_A1, FCC_A1#2, occasionally
    FCC_L12) in the order it finds them, so the austenite matrix is stored
    under FCC_A1 for some alloys/temperatures and under FCC_A1#2 for others
    (83 of the 128 alloys switch label between 800 and 1000 °C). The second
    FCC set is the MC carbide/nitride (V,Nb,Ti)(C,N), which is also FCC.

    To give the models a consistent target, per row the LARGEST FCC set is
    taken as austenite and the remaining FCC sets are summed as MC carbide.
    """
    present = [c for c in FCC_SETS if c in data.columns]
    if len(present) < 2:
        return data, phase_cols
    F = data[present].to_numpy(float)
    largest = F.max(axis=1)
    data = data.copy()
    data[AUSTENITE] = largest
    data[FCC_MINOR] = F.sum(axis=1) - largest
    # Rows where a lone, small FCC set could be either phase -> just report.
    n_single_small = int((((F > 0).sum(axis=1) == 1) & (largest < 0.03)).sum())
    print(f"FCC sets merged -> FCC_austenite (largest set) + FCC_MC_carbide (rest). "
          f"Rows with one small FCC set (<0.03, treated as austenite): {n_single_small}")
    data = data.drop(columns=present)
    phase_cols = [c for c in phase_cols if c not in present] + [AUSTENITE, FCC_MINOR]
    return data, phase_cols


def select_phases(data, phase_cols):
    """Keep a phase if it is non-zero anywhere (any alloy, any temperature)."""
    keep = [c for c in phase_cols if (data[c] != 0).any()]
    return keep


def clean_name(col):
    """'Amount of FCC_A1 [mol]' -> 'FCC_A1', 'Mass percent Cr' -> 'Cr'."""
    return col.replace(PHASE_PREFIX, "").replace(" [mol]", "").replace(COMP_PREFIX, "")


# ---------------------------------------------------------------------------
# 2. Alloy-level split, THEN attach temperatures
# ---------------------------------------------------------------------------
def split_by_alloy(data, test_size, seed):
    alloys = data[[ID_COL, FAMILY_COL]].drop_duplicates(ID_COL)
    train_ids, test_ids = train_test_split(
        alloys[ID_COL].values, test_size=test_size, random_state=seed, shuffle=True
    )
    # Attach all temperatures to each alloy *after* the split.
    train = data[data[ID_COL].isin(train_ids)].sort_values([ID_COL, TEMP_COL])
    test = data[data[ID_COL].isin(test_ids)].sort_values([ID_COL, TEMP_COL])

    # Safety checks: no leakage, every alloy has the full temperature set.
    assert set(train[ID_COL]).isdisjoint(set(test[ID_COL])), "alloy leakage!"
    n_temps = data[TEMP_COL].nunique()
    assert (train.groupby(ID_COL)[TEMP_COL].nunique() == n_temps).all()
    assert (test.groupby(ID_COL)[TEMP_COL].nunique() == n_temps).all()
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------
# On/off switch per model: True = train & evaluate, False = skip.
# Command-line --on / --off override these defaults.
MODEL_SWITCHES = {
    "mean":       True,   # 1_Mean (Dummy) baseline
    "ridge":      True,   # 2_Ridge
    "extratrees": True,   # 3_ExtraTrees
    "xgboost":    True,   # 4_XGBoost
    "gpr":        True,   # 5_GPR
    "mlp":        True,   # 6_NeuralNet (MLP)
}


def resolve_switches(on=None, off=None):
    """Start from MODEL_SWITCHES, then apply --on and --off (--off wins on conflict)."""
    switches = dict(MODEL_SWITCHES)
    for key in on or []:
        switches[key] = True
    for key in off or []:
        switches[key] = False
    return switches


def build_models(n_features, seed, switches, gpr_restarts=0):
    gp_kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-1, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    )
    # key -> (display name, constructor). Only switched-on models are built.
    factories = {
        "mean": ("1_Mean (Dummy)", lambda: DummyRegressor(strategy="mean")),
        "ridge": ("2_Ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "extratrees": ("3_ExtraTrees", lambda: ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=1, n_jobs=-1, random_state=seed
        )),
        "xgboost": ("4_XGBoost", lambda: XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )),
        "gpr": ("5_GPR", lambda: make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=gp_kernel,
                normalize_y=True,
                n_restarts_optimizer=gpr_restarts,
                random_state=seed,
            ),
        )),
        "mlp": ("6_NeuralNet (MLP)", lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64),
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=3000,
                early_stopping=True,
                random_state=seed,
            ),
        )),
    }
    models = {name: make() for key, (name, make) in factories.items() if switches.get(key, False)}
    if not models:
        raise SystemExit("All models are switched off - enable at least one (MODEL_SWITCHES or --on).")
    return models


def predict(model, X, clip):
    y = np.asarray(model.predict(X))
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if clip:  # phase amounts are physically bounded to [0, 1]
        y = np.clip(y, 0.0, 1.0)
    return y


def gpr_std(model, X):
    """Predictive standard deviation from the GPR pipeline (per output)."""
    scaler, gpr = model.steps[0][1], model.steps[-1][1]
    _, std = gpr.predict(scaler.transform(X), return_std=True)
    std = np.asarray(std)
    return std.reshape(len(X), -1)


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------
def overall_metrics(y_true, y_pred):
    return {
        "R2_variance_weighted": r2_score(y_true, y_pred, multioutput="variance_weighted"),
        "R2_flattened": r2_score(y_true.ravel(), y_pred.ravel()),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MaxAbsError": np.max(np.abs(y_true - y_pred)),
    }


def per_phase_metrics(y_true, y_pred, phase_names):
    rows = []
    for j, ph in enumerate(phase_names):
        t, p = y_true[:, j], y_pred[:, j]
        rows.append(
            {
                "phase": ph,
                "R2": r2_score(t, p) if np.var(t) > 0 else np.nan,
                "MAE": mean_absolute_error(t, p),
                "RMSE": np.sqrt(mean_squared_error(t, p)),
                "true_max": t.max(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------
def plot_summary(summary, out):
    names = summary["model"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title in zip(
        axes,
        ["test_R2_variance_weighted", "test_MAE", "test_RMSE"],
        ["Test R² (variance-weighted)", "Test MAE [mol]", "Test RMSE [mol]"],
    ):
        ax.bar(names, summary[col], color="#4C72B0")
        cv_col = col.replace("test_", "cv_")
        if cv_col + "_mean" in summary:
            ax.errorbar(
                np.arange(len(names)) + 0.25,
                summary[cv_col + "_mean"],
                yerr=summary[cv_col + "_std"],
                fmt="o",
                color="#DD8452",
                label="grouped CV (mean ± sd)",
            )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(min(0, summary["test_R2_variance_weighted"].min() - 0.05), 1.02)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_parity(y_test, preds, phase_names, out, top_n=4):
    order = np.argsort(-y_test.var(axis=0))[:top_n]  # most variable phases
    model_names = list(preds)
    fig, axes = plt.subplots(
        len(order), len(model_names), figsize=(3.2 * len(model_names), 3.0 * len(order)),
        squeeze=False,
    )
    for r, j in enumerate(order):
        lo, hi = 0, max(y_test[:, j].max(), 1e-6) * 1.05
        for c, m in enumerate(model_names):
            ax = axes[r, c]
            ax.scatter(y_test[:, j], preds[m][:, j], s=8, alpha=0.6)
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if r == 0:
                ax.set_title(m, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{phase_names[j]}\npredicted", fontsize=9)
            if r == len(order) - 1:
                ax.set_xlabel("ThermoCalc", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "parity_plots.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("data", nargs="?",
                    default=str(script_dir / "Equilibrium phase constitution_128_All Phases.xls"),
                    help="ThermoCalc .xls/.xlsx file (one sheet per temperature)")
    ap.add_argument("--out", default=str(script_dir / "ml_results"), help="output folder")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-folds", type=int, default=5, help="grouped CV folds on the training alloys")
    ap.add_argument("--keep-fe", action="store_true",
                    help="keep Fe as a feature (dropped by default: it is the balance element)")
    ap.add_argument("--no-clip", action="store_true", help="do not clip predictions to [0, 1]")
    ap.add_argument("--on", nargs="+", choices=list(MODEL_SWITCHES), metavar="MODEL",
                    help=f"switch models ON (choices: {', '.join(MODEL_SWITCHES)})")
    ap.add_argument("--off", nargs="+", choices=list(MODEL_SWITCHES), metavar="MODEL",
                    help="switch models OFF (same choices; overrides --on)")
    ap.add_argument("--raw-fcc-labels", action="store_true",
                    help="keep ThermoCalc's FCC_A1 / FCC_A1#2 / FCC_L12 labels as separate targets\n"
                         "(default: merge into FCC_austenite + FCC_MC_carbide, see harmonize_fcc)")
    ap.add_argument("--gpr-restarts", type=int, default=0,
                    help="extra random restarts of the GPR kernel optimiser (slower, default 0)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clip = not args.no_clip

    # ---- data -------------------------------------------------------------
    data, phase_cols, comp_cols = load_thermocalc(args.data)
    if not args.raw_fcc_labels:
        data, phase_cols = harmonize_fcc(data, phase_cols)
    target_cols = select_phases(data, phase_cols)
    feature_cols = [c for c in comp_cols if args.keep_fe or c != COMP_PREFIX + "Fe"]
    feature_cols = feature_cols + [TEMP_COL]

    phase_names = [clean_name(c) for c in target_cols]
    feat_names = [clean_name(c) for c in feature_cols]

    n_alloys = data[ID_COL].nunique()
    temps = sorted(data[TEMP_COL].unique())
    print(f"Alloys: {n_alloys} | temperatures: {[int(t) for t in temps]} | rows: {len(data)}")
    print(f"Phases kept ({len(target_cols)} of {len(phase_cols)}): {phase_names}")
    print(f"Features ({len(feature_cols)}): {feat_names}")

    # ---- split ------------------------------------------------------------
    train, test = split_by_alloy(data, args.test_size, args.seed)
    print(f"Train: {train[ID_COL].nunique()} alloys -> {len(train)} rows | "
          f"Test: {test[ID_COL].nunique()} alloys -> {len(test)} rows")

    keep_cols = [ID_COL, FAMILY_COL] + feature_cols + target_cols
    rename = {c: clean_name(c) for c in feature_cols + target_cols}
    train[keep_cols].rename(columns=rename).to_csv(out / "dataset_train.csv", index=False)
    test[keep_cols].rename(columns=rename).to_csv(out / "dataset_test.csv", index=False)

    X_train, y_train = train[feature_cols].to_numpy(float), train[target_cols].to_numpy(float)
    X_test, y_test = test[feature_cols].to_numpy(float), test[target_cols].to_numpy(float)
    groups = train[ID_COL].to_numpy()

    # Same grouped CV folds for every model (an alloy never spans folds).
    cv = GroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    folds = list(cv.split(X_train, y_train, groups))

    # ---- models (on/off switches) -----------------------------------------
    switches = resolve_switches(args.on, args.off)
    print("Model switches: " + ", ".join(f"{k}={'ON' if v else 'off'}" for k, v in switches.items()))
    models = build_models(len(feature_cols), args.seed, switches, args.gpr_restarts)

    summary_rows, phase_tables, preds = [], [], {}
    pred_long = test[[ID_COL, FAMILY_COL, TEMP_COL]].copy()

    for name, proto in models.items():
        print(f"\n=== {name} ===", flush=True)
        # -- grouped CV on the training alloys
        cv_scores = []
        for k, (tr, va) in enumerate(folds):
            m = clone(proto).fit(X_train[tr], y_train[tr])
            cv_scores.append(overall_metrics(y_train[va], predict(m, X_train[va], clip)))
        cv_df = pd.DataFrame(cv_scores)

        # -- final fit on all training alloys, evaluate on held-out alloys
        t0 = time.time()
        model = clone(proto).fit(X_train, y_train)
        fit_time = time.time() - t0
        y_pred = predict(model, X_test, clip)
        preds[name] = y_pred
        test_m = overall_metrics(y_test, y_pred)

        row = {"model": name, "fit_time_s": fit_time}
        row.update({f"test_{k}": v for k, v in test_m.items()})
        for k in cv_df.columns:
            row[f"cv_{k}_mean"] = cv_df[k].mean()
            row[f"cv_{k}_std"] = cv_df[k].std()

        # -- GPR uncertainty: mean σ and how often the truth lies within ±2σ
        if name.endswith("GPR"):
            std = gpr_std(model, X_test)
            if std.shape[1] == 1:
                std = np.repeat(std, y_test.shape[1], axis=1)
            row["gpr_mean_std"] = std.mean()
            row["gpr_coverage_2sigma"] = np.mean(np.abs(y_test - y_pred) <= 2 * std + 1e-12)
            for j, ph in enumerate(phase_names):
                pred_long[f"{ph}__{name}__std"] = std[:, j]

        summary_rows.append(row)
        pt = per_phase_metrics(y_test, y_pred, phase_names)
        pt.insert(0, "model", name)
        phase_tables.append(pt)
        for j, ph in enumerate(phase_names):
            pred_long[f"{ph}__{name}"] = y_pred[:, j]

        print(f"  CV  R²(vw) = {cv_df['R2_variance_weighted'].mean():.4f} ± "
              f"{cv_df['R2_variance_weighted'].std():.4f} | MAE = {cv_df['MAE'].mean():.5f}")
        print(f"  Test R²(vw) = {test_m['R2_variance_weighted']:.4f} | MAE = {test_m['MAE']:.5f} | "
              f"RMSE = {test_m['RMSE']:.5f} | fit {fit_time:.1f}s")

    # ---- outputs ----------------------------------------------------------
    for j, ph in enumerate(phase_names):
        pred_long[f"{ph}__ThermoCalc"] = y_test[:, j]
    pred_long = pred_long[[ID_COL, FAMILY_COL, TEMP_COL] + sorted(pred_long.columns[3:])]
    pred_long.to_csv(out / "test_predictions.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "model_comparison.csv", index=False)
    per_phase = pd.concat(phase_tables, ignore_index=True)
    per_phase.to_csv(out / "per_phase_metrics.csv", index=False)

    # MAE per temperature (does accuracy depend on T?)
    temp_rows = []
    for name, yp in preds.items():
        for T in temps:
            mask = test[TEMP_COL].to_numpy() == T
            temp_rows.append({"model": name, "temperature_C": T,
                              "MAE": mean_absolute_error(y_test[mask], yp[mask])})
    pd.DataFrame(temp_rows).pivot(index="temperature_C", columns="model", values="MAE") \
        .to_csv(out / "mae_per_temperature.csv")

    pd.DataFrame({"split": ["train"] * train[ID_COL].nunique() + ["test"] * test[ID_COL].nunique(),
                  ID_COL: list(train[ID_COL].unique()) + list(test[ID_COL].unique())}) \
        .to_csv(out / "alloy_split.csv", index=False)

    plot_summary(summary, out)
    plot_parity(y_test, preds, phase_names, out)

    cols = ["model", "test_R2_variance_weighted", "test_MAE", "test_RMSE",
            "cv_R2_variance_weighted_mean", "cv_MAE_mean", "fit_time_s"]
    print("\n" + "=" * 80)
    print("SUMMARY (test set = held-out alloys, all temperatures)")
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nResults written to: {out.resolve()}")


if __name__ == "__main__":
    main()