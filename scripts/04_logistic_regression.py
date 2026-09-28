"""
04_logistic_regression.py
=========================
Binary logistic regression: one reference group vs one or more treatment groups.
Top-N features are selected per timepoint from RF model importances.
Leave-one-out cross-validation (LOO-CV) estimates accuracy.
Bootstrap confidence intervals and p-values are computed for all coefficients.

Outputs (saved to --output_dir):
  logreg_performance.csv   — LOO-CV accuracy per timepoint
  logreg_coefficients.csv  — coefficients, 95% CI, odds ratios, p-values
  logreg_probabilities.csv — per-animal classification probabilities

Usage
-----
    python scripts/04_logistic_regression.py \\
        --feature_file data/results/fastest_trial_features.csv \\
        --reference_group WT \\
        --treatment_groups SOD1 [options]

Examples
--------
    # Single treatment group
    python scripts/04_logistic_regression.py \\
        --feature_file data/results/fastest_trial_features.csv \\
        --reference_group WT \\
        --treatment_groups SOD1

    # Multiple treatment groups collapsed into one class
    python scripts/04_logistic_regression.py \\
        --feature_file data/results/fastest_trial_features.csv \\
        --reference_group WT \\
        --treatment_groups SOD1_early SOD1_late
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.features import FEATURE_LABELS, find_closest_model_tp, parse_age

REFERENCE_LABEL = "Reference"
TREATMENT_LABEL = "Treatment"


def main():
    parser = argparse.ArgumentParser(
        description="Logistic regression with LOO-CV and bootstrap CIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feature_file",     required=True,
                        help="Feature CSV from 02_extract_features.py")
    parser.add_argument("--reference_group",  required=True,
                        help="Reference group name (e.g. WT)")
    parser.add_argument("--treatment_groups", nargs="+", required=True,
                        help="Treatment group(s); multiple groups are collapsed into one class")
    parser.add_argument("--model_dir",        default="models",
                        help="Folder containing RF .joblib files (for feature selection)")
    parser.add_argument("--output_dir",       default="data/results",
                        help="Output folder")
    parser.add_argument("--n_top_features",   type=int, default=5,
                        help="Number of top RF features to use per timepoint")
    parser.add_argument("--n_bootstrap",      type=int, default=1000,
                        help="Bootstrap iterations for coefficient CIs")
    parser.add_argument("--C",                type=float, default=1.0,
                        help="Logistic regression regularization strength (inverse)")
    parser.add_argument("--random_seed",      type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)

    # ── Load RF models (for feature importance ranking) ────────────────────────
    model_dir   = Path(args.model_dir)
    model_files = sorted(model_dir.glob("step10b_rf_model_*.joblib"))
    if not model_files:
        print(f"No RF model files found in {model_dir}/")
        sys.exit(1)

    model_cache, available_tps = {}, []
    for mf in model_files:
        tp = mf.stem.replace("step10b_rf_model_", "")
        try:
            model_cache[tp] = joblib.load(mf)
            available_tps.append(tp)
        except Exception as e:
            print(f"  ! Could not load {mf.name}: {e}")
    available_tps = sorted(available_tps, key=parse_age)
    print(f"RF models loaded: {', '.join(available_tps)}")

    # ── Load feature data ──────────────────────────────────────────────────────
    df = pd.read_csv(args.feature_file)
    df.columns = df.columns.str.strip()

    ref_groups   = [args.reference_group]
    treat_groups = args.treatment_groups
    all_selected = ref_groups + treat_groups

    df_lr = df[df["group"].isin(all_selected)].copy()
    df_lr["group_binary"] = df_lr["group"].apply(
        lambda g: REFERENCE_LABEL if g in ref_groups else TREATMENT_LABEL
    )

    ref_display   = " + ".join(ref_groups)
    treat_display = " + ".join(treat_groups)
    n_ref   = int((df_lr["group_binary"] == REFERENCE_LABEL).sum())
    n_treat = int((df_lr["group_binary"] == TREATMENT_LABEL).sum())
    print(f"\nReference ({ref_display}): n={n_ref}")
    print(f"Treatment ({treat_display}): n={n_treat}")

    mir_tps    = sorted(df_lr["timepoint_label"].unique(), key=parse_age)
    tp_mapping = {tp: find_closest_model_tp(tp, available_tps) for tp in mir_tps}

    BINARY_GROUPS = [REFERENCE_LABEL, TREATMENT_LABEL]
    LR_PARAMS = dict(
        C=args.C, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=args.random_seed,
    )

    perf_rows, coef_rows, prob_rows = [], [], []
    skip_log = []

    print(f"\nRunning LOO-CV across {len(mir_tps)} timepoints "
          f"(top {args.n_top_features} RF features each)...\n")

    for tp in mir_tps:
        model_tp      = tp_mapping[tp]
        bundle        = model_cache[model_tp]
        all_feat_cols = bundle["feature_cols"]
        imps          = bundle["model"].feature_importances_
        top_idx       = np.argsort(imps)[::-1][:args.n_top_features]
        feature_cols  = [all_feat_cols[i] for i in top_idx]

        tp_df = df_lr[df_lr["timepoint_label"] == tp].copy()
        feature_cols = [f for f in feature_cols if f in tp_df.columns]
        if not feature_cols:
            skip_log.append(f"{tp}: no features available — skipped")
            continue

        tp_df = tp_df.dropna(subset=feature_cols + ["group_binary"])
        n = len(tp_df)
        if n < 3 or tp_df["group_binary"].nunique() < 2:
            skip_log.append(f"{tp}: insufficient data (n={n}) — skipped")
            continue

        le         = LabelEncoder().fit(BINARY_GROUPS)
        X          = tp_df[feature_cols].values
        y          = le.transform(tp_df["group_binary"].values)
        mouse_ids  = tp_df["animal_id"].values
        orig_groups = tp_df["group"].values
        ref_idx    = list(le.classes_).index(REFERENCE_LABEL)
        trt_idx    = list(le.classes_).index(TREATMENT_LABEL)

        # LOO-CV
        y_true, y_pred = [], []
        for train_idx, test_idx in LeaveOneOut().split(X):
            if len(np.unique(y[train_idx])) < 2:
                continue
            clf  = LogisticRegression(**LR_PARAMS)
            clf.fit(X[train_idx], y[train_idx])
            pred  = clf.predict(X[test_idx])
            probs = clf.predict_proba(X[test_idx])[0]
            y_true.append(y[test_idx[0]])
            y_pred.append(pred[0])
            prob_rows.append({
                "timepoint_label":        tp,
                "model_timepoint":        model_tp,
                "animal_id":              mouse_ids[test_idx[0]],
                "original_group":         orig_groups[test_idx[0]],
                "group_binary":           tp_df["group_binary"].values[test_idx[0]],
                "predicted_label":        le.inverse_transform([pred[0]])[0],
                "correct":                int(y[test_idx[0]] == pred[0]),
                f"p_{REFERENCE_LABEL}":   float(probs[ref_idx]),
                f"p_{TREATMENT_LABEL}":   float(probs[trt_idx]),
            })

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        overall_acc = accuracy_score(y_true, y_pred) if len(y_true) else np.nan
        per_group_acc = {}
        for g in BINARY_GROUPS:
            g_enc = le.transform([g])[0]
            mask  = y_true == g_enc
            if mask.sum() > 0:
                per_group_acc[g] = accuracy_score(y_true[mask], y_pred[mask])

        perf_rows.append({
            "timepoint_label": tp,
            "model_timepoint": model_tp,
            "n_total":         n,
            "n_features":      len(feature_cols),
            "overall_acc":     round(overall_acc, 3),
            **{f"acc_{g}": round(per_group_acc.get(g, np.nan), 3) for g in BINARY_GROUPS},
        })
        print(f"  {tp}  (n={n}, model={model_tp})  LOO-CV accuracy={overall_acc:.3f}")

        # Final model on all data + bootstrap CIs
        clf_final = LogisticRegression(**LR_PARAMS)
        clf_final.fit(X, y)
        coef_point = clf_final.coef_[0]

        boot_coefs = {f: [] for f in feature_cols}
        for _ in range(args.n_bootstrap):
            boot_idx = resample(np.arange(n), replace=True, n_samples=n,
                                random_state=int(rng.integers(0, 1_000_000)))
            X_b, y_b = X[boot_idx], y[boot_idx]
            if len(np.unique(y_b)) < 2:
                continue
            try:
                clf_b = LogisticRegression(**LR_PARAMS)
                clf_b.fit(X_b, y_b)
                for fi, feat in enumerate(feature_cols):
                    boot_coefs[feat].append(clf_b.coef_[0, fi])
            except Exception:
                continue

        for fi, feat in enumerate(feature_cols):
            point_coef = float(coef_point[fi])
            boot_vals  = np.array(boot_coefs[feat])
            if len(boot_vals) >= 50:
                ci_low  = float(np.percentile(boot_vals, 2.5))
                ci_high = float(np.percentile(boot_vals, 97.5))
                p_val   = float(min(
                    2 * np.mean(boot_vals <= 0) if point_coef >= 0
                    else 2 * np.mean(boot_vals >= 0), 1.0
                ))
            else:
                ci_low = ci_high = p_val = np.nan

            coef_rows.append({
                "timepoint_label":  tp,
                "model_timepoint":  model_tp,
                "comparison":       f"{TREATMENT_LABEL}_vs_{REFERENCE_LABEL}",
                "feature":          feat,
                "feature_label":    FEATURE_LABELS.get(feat, feat),
                "coefficient":      round(point_coef, 4),
                "ci_low":           round(ci_low,  4) if not np.isnan(ci_low)  else np.nan,
                "ci_high":          round(ci_high, 4) if not np.isnan(ci_high) else np.nan,
                "p_value":          round(p_val,   4) if not np.isnan(p_val)   else np.nan,
                "odds_ratio":       round(np.exp(point_coef), 4),
                "odds_ratio_low":   round(np.exp(ci_low),  4) if not np.isnan(ci_low)  else np.nan,
                "odds_ratio_high":  round(np.exp(ci_high), 4) if not np.isnan(ci_high) else np.nan,
                "n_valid_boot":     len(boot_vals),
                "significant":      "*" if (not np.isnan(p_val) and p_val < 0.05) else "",
            })

    # ── Skip log ───────────────────────────────────────────────────────────────
    if skip_log:
        print(f"\n{len(skip_log)} timepoint(s) skipped:")
        for msg in skip_log:
            print(f"  !  {msg}")

    if not perf_rows:
        print("\nNo results computed. Verify group names and feature columns.")
        sys.exit(1)

    # ── Save outputs ───────────────────────────────────────────────────────────
    perf_df = pd.DataFrame(perf_rows)
    coef_df = pd.DataFrame(coef_rows)
    prob_df = pd.DataFrame(prob_rows)

    perf_df.to_csv(output_dir / "logreg_performance.csv",   index=False)
    coef_df.to_csv(output_dir / "logreg_coefficients.csv",  index=False)
    prob_df.to_csv(output_dir / "logreg_probabilities.csv", index=False)

    print(f"\nSaved to {output_dir}/")
    print("  • logreg_performance.csv")
    print("  • logreg_coefficients.csv")
    print("  • logreg_probabilities.csv")

    sig_df = coef_df[coef_df["significant"] == "*"]
    if not sig_df.empty:
        print(f"\nSignificant features (bootstrap p < 0.05):")
        cols = ["timepoint_label", "feature_label", "coefficient", "ci_low", "ci_high", "p_value"]
        print(sig_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
