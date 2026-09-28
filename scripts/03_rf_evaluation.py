"""
03_rf_evaluation.py
===================
Score animals against pre-trained WT/SOD1 Random Forest models.
Applies closest-timepoint matching when the animal's timepoint does not
exactly match a trained model (e.g. p50 → nearest model p45 or p58).
Outputs P(SOD1) per animal per timepoint.

Usage
-----
    python scripts/03_rf_evaluation.py --feature_file <path> [options]

Examples
--------
    python scripts/03_rf_evaluation.py \\
        --feature_file data/results/fastest_trial_features.csv \\
        --model_dir models \\
        --output_dir data/results
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.features import find_closest_model_tp, parse_age


def main():
    parser = argparse.ArgumentParser(
        description="Score animals with pre-trained WT/SOD1 Random Forest models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feature_file", required=True,
                        help="Feature CSV from 02_extract_features.py "
                             "(fastest_trial_features.csv or averaged_trial_features.csv)")
    parser.add_argument("--model_dir",   default="models",
                        help="Folder containing step10b_rf_model_p*.joblib files")
    parser.add_argument("--output_dir",  default="data/results",
                        help="Output folder")
    args = parser.parse_args()

    model_dir  = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load RF models ─────────────────────────────────────────────────────────
    model_files = sorted(model_dir.glob("step10b_rf_model_*.joblib"))
    if not model_files:
        print(f"No model files found in {model_dir}/")
        print("Expected files named: step10b_rf_model_p<age>.joblib")
        sys.exit(1)

    model_cache, available_tps = {}, []
    for mf in model_files:
        tp = mf.stem.replace("step10b_rf_model_", "")
        try:
            bundle = joblib.load(mf)
            model_cache[tp] = bundle
            available_tps.append(tp)
            n_wt   = bundle.get("n_wt",   "?")
            n_sod1 = bundle.get("n_sod1", "?")
            n_feat = len(bundle.get("feature_cols", []))
            print(f"  Loaded {mf.name}  (WT={n_wt}, SOD1={n_sod1}, features={n_feat})")
        except Exception as e:
            print(f"  ! Could not load {mf.name}: {e}")

    available_tps = sorted(available_tps, key=parse_age)
    print(f"\nModels available: {', '.join(available_tps)}")

    # ── Load feature data ──────────────────────────────────────────────────────
    df = pd.read_csv(args.feature_file)
    df.columns = df.columns.str.strip()

    if "timepoint_label" not in df.columns:
        print("Feature file must contain a 'timepoint_label' column.")
        sys.exit(1)

    new_tps    = sorted(df["timepoint_label"].unique(), key=parse_age)
    tp_mapping = {tp: find_closest_model_tp(tp, available_tps) for tp in new_tps}

    print(f"\nTimepoint mapping (your data → closest RF model):")
    for new_tp, model_tp in tp_mapping.items():
        gap = abs(parse_age(new_tp) - parse_age(model_tp))
        suffix = f"  [gap: {gap} days]" if gap > 0 else ""
        print(f"  {new_tp:>6}  →  {model_tp}{suffix}")

    # ── Score ──────────────────────────────────────────────────────────────────
    print("\nScoring animals...")
    prob_rows = []

    for new_tp in new_tps:
        model_tp  = tp_mapping[new_tp]
        bundle    = model_cache[model_tp]
        clf       = bundle["model"]
        feat_cols = bundle["feature_cols"]

        tp_df   = df[df["timepoint_label"] == new_tp].copy()
        missing = [f for f in feat_cols if f not in tp_df.columns]
        if missing:
            print(f"  !  {new_tp}: missing features {missing} — skipped")
            continue

        tp_df = tp_df.dropna(subset=feat_cols)
        if len(tp_df) == 0:
            print(f"  !  {new_tp}: no complete rows after dropping NaN — skipped")
            continue

        X     = tp_df[feat_cols].values
        probs = clf.predict_proba(X)[:, 1]   # P(SOD1)

        for idx, (_, row) in enumerate(tp_df.iterrows()):
            prob_rows.append({
                "timepoint_label": new_tp,
                "model_timepoint": model_tp,
                "age_gap_days":    abs(parse_age(new_tp) - parse_age(model_tp)),
                "animal_id":       row.get("animal_id", "?"),
                "group":           row.get("group", "unknown"),
                "p_sod1":          round(float(probs[idx]), 4),
                "p_wt":            round(float(1 - probs[idx]), 4),
            })
        print(f"  ✓  {new_tp}: scored {len(tp_df)} animals")

    if not prob_rows:
        print("\nNo probabilities computed. Check that feature column names match model expectations.")
        sys.exit(1)

    prob_df  = pd.DataFrame(prob_rows)
    out_path = output_dir / "rf_probabilities_p_sod1.csv"
    prob_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    summary = (
        prob_df.groupby(["timepoint_label", "group"])["p_sod1"]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
    )
    summary.columns = ["Timepoint", "Group", "Mean P(SOD1)", "SD", "N"]
    print("\nMean P(SOD1) per group × timepoint:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
