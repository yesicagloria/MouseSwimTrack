"""
01_clean_data.py
================
Cleans LightningPose / DeepLabCut tracking CSVs:
  - Interpolates low-confidence keypoints (below likelihood cutoff)
  - Smooths trajectories with a Savitzky-Golay filter
  - Detects sternum crossings of start/finish lane markers
  - Segments the active swim portion
  - Saves one cleaned CSV per input file to the output folder

Usage
-----
    python scripts/01_clean_data.py [options]

Examples
--------
    # Defaults (120 fps, cutoff=0.7, window=5)
    python scripts/01_clean_data.py

    # Custom paths and parameters
    python scripts/01_clean_data.py \\
        --raw_dir data/raw \\
        --clean_dir data/cleaned \\
        --cutoff 0.7 \\
        --wl 5 \\
        --sternum sternum \\
        --start start \\
        --finish finish
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ── Cleaning functions ─────────────────────────────────────────────────────────

def dlc_savgol_filter(traj, WL_jitter=5, polyorder=3):
    """Apply Savitzky-Golay filter; NaNs become 0."""
    return np.nan_to_num(savgol_filter(traj, window_length=WL_jitter, polyorder=polyorder))


def dlc_interpolate(dlc_dat, bp_number, nan_array, coord="y_pos", cutoff=0.7):
    """Replace low-likelihood frames with NaN, then interpolate linearly."""
    traj = np.where(
        dlc_dat[bp_number]["likelihood"] < cutoff,
        [nan_array],
        [dlc_dat[bp_number][coord]],
    )[0]
    return np.array(pd.Series(traj).interpolate(limit_direction="both"))


def process_trajectory(dlc_dat_list, coord, nan_array, cutoff, WL_jitter):
    """Interpolate + smooth all body parts for one coordinate axis."""
    bp_traj, limbs = {}, {}
    for i, bp_info in enumerate(dlc_dat_list):
        limbs[bp_info["joint_name"]] = i
        bp_traj[i] = dlc_savgol_filter(
            dlc_interpolate(dlc_dat_list, i, nan_array, coord, cutoff),
            WL_jitter=WL_jitter, polyorder=3,
        )
    return bp_traj, limbs


def find_crossings(sig, line):
    """Return frame indices where sig crosses line (zero-crossing logic)."""
    sig = np.asarray(sig, dtype=float)
    rel = sig - line
    before, after = rel[:-1], rel[1:]
    return np.where((before < 0) & (after >= 0) | (before > 0) & (after <= 0))[0] + 1


def clean_file(
    csv_path: Path,
    save_dir: Path,
    cutoff: float = 0.7,
    WL_jitter: int = 5,
    BP_STERNUM: str = "sternum",
    BP_START: str = "start",
    BP_FINISH: str = "finish",
) -> dict:
    """
    Clean a single tracking CSV and save the segmented result.

    Returns
    -------
    dict with keys:
        status  : 'ok' | 'warn' | 'err'
        msg     : human-readable summary
        n_frames: number of frames in cleaned segment (only on 'ok')
    """
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        scorer    = df.columns.get_level_values(0)[0]
        bodyParts = np.unique(df.columns.get_level_values(1))

        dlc_dat_list = [
            {
                "joint_name": bp,
                "x_pos":      df[(scorer, bp, "x")].values,
                "y_pos":      df[(scorer, bp, "y")].values,
                "likelihood": df[(scorer, bp, "likelihood")].values,
            }
            for bp in bodyParts
            if (scorer, bp, "x") in df.columns
        ]
        if not dlc_dat_list:
            return {"status": "warn", "msg": "No body parts found"}

        nan_array  = np.full(len(dlc_dat_list[0]["x_pos"]), np.nan)
        bp_clean_x, limbs = process_trajectory(dlc_dat_list, "x_pos", nan_array, cutoff, WL_jitter)
        bp_clean_y, _     = process_trajectory(dlc_dat_list, "y_pos", nan_array, cutoff, WL_jitter)

        missing = [k for k in [BP_STERNUM, BP_START, BP_FINISH] if k not in limbs]
        if missing:
            return {"status": "warn", "msg": f"Missing keypoints: {missing}"}

        sternum_x   = bp_clean_x[limbs[BP_STERNUM]]
        start_line  = np.nanmean(bp_clean_x[limbs[BP_START]])
        finish_line = np.nanmean(bp_clean_x[limbs[BP_FINISH]])

        start_crossings  = find_crossings(sternum_x, start_line)
        finish_crossings = find_crossings(sternum_x, finish_line)

        if start_crossings.size == 0 or finish_crossings.size == 0:
            return {"status": "warn", "msg": "No sternum crossings detected"}

        last_finish = finish_crossings[-1]
        valid_start = start_crossings[start_crossings <= last_finish]
        if valid_start.size == 0:
            return {"status": "warn", "msg": "No start crossing before last finish"}

        seg = slice(int(valid_start[0]), int(last_finish) + 1)

        # Build output dataframe with cleaned, segmented trajectories
        rows = {}
        for i, bp_info in enumerate(dlc_dat_list):
            bp = bp_info["joint_name"]
            rows[(scorer, bp, "x")]          = bp_clean_x[i][seg]
            rows[(scorer, bp, "y")]          = bp_clean_y[i][seg]
            rows[(scorer, bp, "likelihood")] = bp_info["likelihood"][seg]

        out_df = pd.DataFrame(rows)
        out_df.columns = pd.MultiIndex.from_tuples(out_df.columns)

        out_path = save_dir / (csv_path.stem + "_clean_segment.csv")
        out_df.to_csv(out_path)
        return {"status": "ok", "msg": f"{len(out_df)} frames", "n_frames": len(out_df)}

    except Exception as e:
        return {"status": "err", "msg": str(e)}


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clean LightningPose tracking CSVs (interpolate → smooth → segment).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_dir",   default="data/raw",     help="Folder with raw tracking CSVs")
    parser.add_argument("--clean_dir", default="data/cleaned", help="Output folder for cleaned CSVs")
    parser.add_argument("--cutoff",    type=float, default=0.7, help="Likelihood cutoff for interpolation")
    parser.add_argument("--wl",        type=int,   default=5,   help="Savitzky-Golay window length (must be odd)")
    parser.add_argument("--sternum",   default="sternum",       help="Sternum keypoint name")
    parser.add_argument("--start",     default="start",         help="Start line keypoint name")
    parser.add_argument("--finish",    default="finish",        help="Finish line keypoint name")
    args = parser.parse_args()

    if args.wl % 2 == 0:
        args.wl += 1
        print(f"Window length adjusted to {args.wl} (must be odd).")

    raw_dir   = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} files in {raw_dir}")
    print(f"Output → {clean_dir}\n")

    ok = warn = err = 0
    for f in csv_files:
        result = clean_file(f, clean_dir, args.cutoff, args.wl,
                            args.sternum, args.start, args.finish)
        icon = {"ok": "✓", "warn": "!", "err": "✗"}[result["status"]]
        print(f"  {icon}  {f.name}: {result['msg']}")
        if   result["status"] == "ok":   ok   += 1
        elif result["status"] == "warn": warn += 1
        else:                            err  += 1

    print(f"\nDone — {ok} cleaned, {warn} warnings, {err} errors")


if __name__ == "__main__":
    main()
