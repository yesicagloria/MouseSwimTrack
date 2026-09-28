"""
02_extract_features.py
======================
Extracts 24 bilateral swim features from cleaned tracking CSVs,
merges with experimental metadata, and saves four output files:
  - all_trials_features.csv
  - fastest_trial_features.csv   (fastest trial per animal × timepoint)
  - averaged_trial_features.csv  (mean across trials)
  - removed_short_trials.csv     (trials below min_frames threshold)

Usage
-----
    python scripts/02_extract_features.py [options]

Examples
--------
    # Defaults
    python scripts/02_extract_features.py

    # Custom acquisition parameters
    python scripts/02_extract_features.py \\
        --fps 120 \\
        --cm_per_px 0.040207 \\
        --lane_length 50 \\
        --min_frames 120
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.features import (
    build_bilateral_features,
    compute_averaged,
    compute_fastest,
    extract_trial_features,
)

# Default signal-processing parameters (match app defaults)
_DEFAULTS = dict(
    fps=120.0,
    cm_per_px=0.040207,
    swap_lr=True,
    smooth_w=5,
    lane_length_cm=50.0,
    min_peak_sep_s=0.13,
    prominence_trough=0.15,
    seg_smooth_s=0.08,
    min_seg_len_frames=8,
    prominence_speed=0.7,
    distance_speed=10,
    left_threshold=0.75,
    right_threshold=-0.75,
    nperseg=96,
    noverlap=72,
    freq_min=2.0,
    freq_max=15.0,
)

_META_SKIP = {"Video_name", "Animal_ID", "Trial_number", "Group", "Timepoint"}


def main():
    parser = argparse.ArgumentParser(
        description="Extract swim features from cleaned tracking CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clean_dir",  default="data/cleaned",
                        help="Folder containing cleaned CSVs (output of 01_clean_data.py)")
    parser.add_argument("--metadata",   default="data/metadata/metadata.csv",
                        help="Metadata CSV linking Video_name to Animal_ID, Group, Timepoint, etc.")
    parser.add_argument("--output_dir", default="data/results",
                        help="Output folder for feature CSVs")
    parser.add_argument("--min_frames", type=int, default=120,
                        help="Minimum frame count; shorter trials are excluded")
    # Acquisition parameters
    parser.add_argument("--fps",        type=float, default=120.0,
                        help="Camera frame rate (frames per second)")
    parser.add_argument("--cm_per_px",  type=float, default=0.040207,
                        help="Spatial calibration (cm per pixel)")
    parser.add_argument("--lane_length",type=float, default=50.0,
                        help="Swim lane length in cm")
    parser.add_argument("--no_swap_lr", action="store_true",
                        help="Disable L/R swap (use if camera is NOT mirrored)")
    args = parser.parse_args()

    clean_dir  = Path(args.clean_dir)
    output_dir = Path(args.output_dir)
    meta_path  = Path(args.metadata)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        print(f"Metadata not found: {meta_path}")
        sys.exit(1)

    meta = pd.read_csv(meta_path)
    meta.columns = meta.columns.str.strip()

    clean_files = sorted(clean_dir.glob("*.csv"))
    if not clean_files:
        print(f"No cleaned CSV files found in {clean_dir}")
        sys.exit(1)

    params = {
        **_DEFAULTS,
        "fps":           args.fps,
        "cm_per_px":     args.cm_per_px,
        "lane_length_cm":args.lane_length,
        "swap_lr":       not args.no_swap_lr,
    }

    print(f"Found {len(clean_files)} cleaned files")
    print(f"Metadata: {len(meta)} rows, {meta['Animal_ID'].nunique()} animals\n")

    all_rows = []
    for fpath in clean_files:
        stem     = fpath.stem.replace("_clean_segment", "")
        row_meta = meta[meta["Video_name"] == stem]

        if row_meta.empty:
            print(f"  !  {fpath.name}: no metadata match for '{stem}' — skipped")
            continue

        rm = row_meta.iloc[0]
        try:
            feats = extract_trial_features(
                csv_path=str(fpath),
                animal_id=rm.get("Animal_ID", "unknown"),
                trial=rm.get("Trial_number", 0),
                group=rm.get("Group", rm.get("Genotype", "unknown")),
                timepoint_label=rm.get("Timepoint", "unknown"),
                **params,
            )
            feats["file"] = fpath.name
            # Carry through any extra metadata columns (Sex, Camera, etc.)
            for col in rm.index:
                key = col.lower()
                if col not in _META_SKIP and key not in feats:
                    feats[key] = rm[col]
            all_rows.append(feats)
            print(f"  ✓  {fpath.name}: {feats['n_frames']} frames | "
                  f"{feats['swim_speed']:.1f} cm/s")
        except Exception as e:
            print(f"  ✗  {fpath.name}: {e}")

    if not all_rows:
        print("\nNo features extracted. Check cleaned files and metadata.")
        sys.exit(1)

    all_df    = build_bilateral_features(pd.DataFrame(all_rows))
    short_mask = all_df["n_frames"] < args.min_frames
    short_df  = all_df[short_mask].copy()
    all_df    = all_df[~short_mask].reset_index(drop=True)

    fastest_df  = compute_fastest(all_df)
    averaged_df = compute_averaged(all_df)

    outputs = {
        "all_trials_features.csv":     all_df,
        "fastest_trial_features.csv":  fastest_df,
        "averaged_trial_features.csv": averaged_df,
        "removed_short_trials.csv":    short_df,
    }
    for fname, df in outputs.items():
        df.to_csv(output_dir / fname, index=False)

    print(f"\nExtracted {len(all_df)} trials from {all_df['animal_id'].nunique()} animals")
    print(f"Short trials removed (<{args.min_frames} frames): {len(short_df)}")
    print(f"\nOutput saved to {output_dir}/")
    for fname in outputs:
        print(f"  • {fname}")


if __name__ == "__main__":
    main()
