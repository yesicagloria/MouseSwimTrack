"""
core/features.py
All feature-extraction logic adapted from the original ALS swim pipeline.
Filename-parsing and hardcoded group IDs removed — metadata drives those.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter, correlate, spectrogram
from sklearn.decomposition import PCA

# ── Feature metadata ───────────────────────────────────────────────────────────

FEATURE_COLS = [
    "hl_frequency", "hl_amp_avg", "hl_efficiency_avg",
    "hl_amp_regularity_cv", "hl_ext_speed_avg",
    "hl_ext_speed_regularity_cv", "hl_flex_speed_avg",
    "hl_flex_speed_regularity_cv", "hl_loop_area_avg",
    "hl_loop_area_regularity_cv", "hl_loop_shape_regularity_cv",
    "hl_lag", "hl_corr_strength", "hl_symmetry",
    "tail_frequency", "tail_oscillation_speed",
    "tail_lag_avg", "tail_coord_strength_avg",
    "forelimb_use_both",
    "body_axis_angle_mean", "body_axis_angle_std",
    "body_curvature_mean", "body_curvature_std",
    "swim_speed",
]

FEATURE_LABELS = {
    "hl_frequency":                "Hindlimb stroke frequency",
    "hl_amp_avg":                  "Hindlimb stroke amplitude",
    "hl_efficiency_avg":           "Hindlimb stroke efficiency",
    "hl_amp_regularity_cv":        "Hindlimb stroke amplitude regularity",
    "hl_ext_speed_avg":            "Hindlimb extension speed",
    "hl_ext_speed_regularity_cv":  "Hindlimb extension speed regularity",
    "hl_flex_speed_avg":           "Hindlimb flexion speed",
    "hl_flex_speed_regularity_cv": "Hindlimb flexion speed regularity",
    "hl_loop_area_avg":            "Hindlimb stroke loop area",
    "hl_loop_area_regularity_cv":  "Hindlimb stroke loop area regularity",
    "hl_loop_shape_regularity_cv": "Hindlimb stroke shape regularity",
    "hl_lag":                      "Hindlimb coordination lag",
    "hl_corr_strength":            "Hindlimb coordination strength",
    "hl_symmetry":                 "Hindlimb coordination symmetry",
    "tail_frequency":              "Tail oscillation frequency",
    "tail_oscillation_speed":      "Tail oscillation speed",
    "tail_lag_avg":                "Hindlimb-tail coordination lag",
    "tail_coord_strength_avg":     "Hindlimb-tail coordination strength",
    "forelimb_use_both":           "Forelimb use",
    "body_axis_angle_mean":        "Body heading",
    "body_axis_angle_std":         "Heading stability",
    "body_curvature_mean":         "Body curvature",
    "body_curvature_std":          "Curvature stability",
    "swim_speed":                  "Swim speed",
}

FEATURE_UNITS = {
    "hl_frequency":                "Hz",
    "hl_amp_avg":                  "cm",
    "hl_efficiency_avg":           "cm",
    "hl_amp_regularity_cv":        "CV (%)",
    "hl_ext_speed_avg":            "cm/s",
    "hl_ext_speed_regularity_cv":  "CV (%)",
    "hl_flex_speed_avg":           "cm/s",
    "hl_flex_speed_regularity_cv": "CV (%)",
    "hl_loop_area_avg":            "cm²",
    "hl_loop_area_regularity_cv":  "CV (%)",
    "hl_loop_shape_regularity_cv": "CV (%)",
    "hl_lag":                      "s",
    "hl_corr_strength":            "r",
    "hl_symmetry":                 "index",
    "tail_frequency":              "Hz",
    "tail_oscillation_speed":      "cm/s",
    "tail_lag_avg":                "s",
    "tail_coord_strength_avg":     "r",
    "forelimb_use_both":           "%",
    "body_axis_angle_mean":        "°",
    "body_axis_angle_std":         "°",
    "body_curvature_mean":         "°",
    "body_curvature_std":          "°",
    "swim_speed":                  "cm/s",
}


def parse_age(tp_label):
    """'p45' → 45"""
    digits = "".join(c for c in str(tp_label) if c.isdigit())
    return int(digits) if digits else 0


def find_closest_model_tp(tp_label, available_tps):
    """Return the trained timepoint label whose age is closest."""
    age = parse_age(tp_label)
    ages = np.array([parse_age(tp) for tp in available_tps])
    idx = int(np.argmin(np.abs(ages - age)))
    return available_tps[idx]


# ── Low-level signal utilities ─────────────────────────────────────────────────

def smooth_len_preserve(a, w):
    a = np.asarray(a, dtype=float)
    if w <= 1:
        return a
    return pd.Series(a).rolling(w, min_periods=1, center=True).mean().to_numpy()


def pick_scorer(df):
    scorers = df.columns.get_level_values(0).unique().tolist()
    preferred = ["heatmap_tracker", "heatmap_tra"]
    for p in preferred:
        if p in scorers:
            return p
    for s in scorers:
        s_str = str(s)
        if not s_str.startswith("Unnamed") and s_str not in ("scorer", "bodyparts", "coords"):
            return s
    raise ValueError(f"Could not identify scorer: {scorers}")


def get_series_cm(df, scorer, bodypart, coord, cm_per_px, smooth_w):
    col = (scorer, bodypart, coord)
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}")
    arr = df[col].to_numpy(float) * cm_per_px
    return smooth_len_preserve(arr, smooth_w)


def cv_percent(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    m = np.mean(x)
    if np.isclose(m, 0):
        return np.nan
    return np.std(x, ddof=1) / m * 100.0


def smooth_for_seg(y, seg_smooth_s, fps):
    y = np.asarray(y, dtype=float)
    win = int(round(seg_smooth_s * fps))
    if win < 5:      win = 5
    if win % 2 == 0: win += 1
    if win >= len(y):
        win = len(y) - 1 if len(y) % 2 == 0 else len(y)
    if win < 5:
        return y.copy()
    return savgol_filter(y, window_length=win, polyorder=3)


def detect_troughs(y, seg_smooth_s, fps, min_peak_sep_s, prominence_trough):
    y_s  = smooth_for_seg(y, seg_smooth_s, fps)
    dist = max(1, int(round(min_peak_sep_s * fps)))
    troughs, props = find_peaks(-y_s, distance=dist, prominence=prominence_trough)
    return y_s, troughs, props


def segment_strokes(y, seg_smooth_s, fps, min_peak_sep_s, prominence_trough, min_seg_len_frames):
    y_s  = smooth_for_seg(y, seg_smooth_s, fps)
    dist = max(1, int(round(min_peak_sep_s * fps)))
    troughs, _ = find_peaks(-y_s, distance=dist, prominence=prominence_trough)
    segments = []
    for i in range(len(troughs) - 1):
        s, e = int(troughs[i]), int(troughs[i + 1])
        if (e - s) >= min_seg_len_frames:
            segments.append((s, e))
    return segments


def detrend_x(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return np.full_like(x, np.nan)
    trend = np.linspace(x[0], x[-1], len(x))
    x_d   = x - trend
    return x_d - np.nanmean(x_d)


def center_y(y):
    return np.asarray(y, dtype=float) - np.nanmean(y)


def compute_area(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.nan
    xc = np.r_[x, x[0]]
    yc = np.r_[y, y[0]]
    return 0.5 * np.abs(np.sum(xc[:-1] * yc[1:] - xc[1:] * yc[:-1]))


def compute_circularity(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.nan
    coords = np.column_stack([x, y])
    if np.any(~np.isfinite(coords)):
        return np.nan
    try:
        pca_local = PCA(n_components=2)
        pca_local.fit(coords)
        var1, var2 = pca_local.explained_variance_
        if var1 <= 0 or var2 <= 0:
            return np.nan
        return np.sqrt(var2 / var1)
    except Exception:
        return np.nan


def process_limb_segments(x_rel, y_rel, seg_smooth_s, fps, min_peak_sep_s,
                           prominence_trough, min_seg_len_frames):
    segments = segment_strokes(y_rel, seg_smooth_s, fps, min_peak_sep_s,
                                prominence_trough, min_seg_len_frames)
    areas, circs = [], []
    for s, e in segments:
        xs = np.asarray(x_rel[s:e+1], dtype=float)
        ys = np.asarray(y_rel[s:e+1], dtype=float)
        if len(xs) < min_seg_len_frames:
            continue
        if np.any(~np.isfinite(xs)) or np.any(~np.isfinite(ys)):
            continue
        area = compute_area(detrend_x(xs), center_y(ys))
        circ = compute_circularity(detrend_x(xs), center_y(ys))
        if np.isfinite(area): areas.append(area)
        if np.isfinite(circ): circs.append(circ)
    return areas, circs, len(segments)


def analyze_hindlimb_from_troughs(hx, tail_x, sternum_x,
                                   seg_smooth_s, fps, min_peak_sep_s,
                                   prominence_trough, min_seg_len_frames):
    signal     = np.asarray(hx - tail_x, dtype=float)
    sternum_x  = np.asarray(sternum_x, dtype=float)
    empty = {"n_cycles": 0, "freq_hz": np.nan,
             "avg_amplitude_cm": np.nan, "cv_amplitude": np.nan,
             "avg_efficiency_cm": np.nan, "cv_efficiency": np.nan}
    if len(signal) < 3:
        return empty
    signal_s, troughs, _ = detect_troughs(signal, seg_smooth_s, fps,
                                           min_peak_sep_s, prominence_trough)
    if len(troughs) < 2:
        return empty
    cycle_times, amplitudes, efficiencies = [], [], []
    for i in range(len(troughs) - 1):
        s, e = int(troughs[i]), int(troughs[i + 1])
        if (e - s) < min_seg_len_frames:
            continue
        seg = signal_s[s:e+1]
        if len(seg) < 3 or np.any(~np.isfinite(seg)):
            continue
        cycle_t = (e - s) / fps
        amp     = np.nanmax(seg) - np.nanmin(seg)
        eff     = sternum_x[e] - sternum_x[s]
        if np.isfinite(cycle_t) and cycle_t > 0:
            cycle_times.append(cycle_t)
        if np.isfinite(amp):
            amplitudes.append(amp)
        if np.isfinite(eff):
            efficiencies.append(eff)
    cycle_times  = np.asarray(cycle_times, dtype=float)
    amplitudes   = np.asarray(amplitudes, dtype=float)
    efficiencies = np.asarray(efficiencies, dtype=float)
    freq_hz = (1.0 / np.nanmean(cycle_times)) if len(cycle_times) > 0 else np.nan
    return {
        "n_cycles":          int(len(cycle_times)),
        "freq_hz":           float(freq_hz) if np.isfinite(freq_hz) else np.nan,
        "avg_amplitude_cm":  float(np.nanmean(amplitudes))   if len(amplitudes)   else np.nan,
        "cv_amplitude":      cv_percent(amplitudes),
        "avg_efficiency_cm": float(np.nanmean(efficiencies)) if len(efficiencies) else np.nan,
        "cv_efficiency":     cv_percent(efficiencies),
    }


def abs_slope_event_to_next(event_idx, next_idx, y_trace, fps):
    event_idx = np.asarray(event_idx, dtype=int)
    next_idx  = np.asarray(next_idx, dtype=int)
    if len(event_idx) == 0 or len(next_idx) == 0:
        return np.array([], dtype=float)
    event_idx.sort(); next_idx.sort()
    j     = np.searchsorted(next_idx, event_idx, side="right")
    valid = j < len(next_idx)
    e, n  = event_idx[valid], next_idx[j[valid]]
    if len(e) == 0:
        return np.array([], dtype=float)
    dt   = (n - e) / fps
    dy   = y_trace[n] - y_trace[e]
    good = dt > 0
    return np.abs(dy[good] / dt[good])


def mean_cv(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    m  = np.mean(arr)
    sd = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
    cv = (sd / m) * 100 if m != 0 else np.nan
    return m, cv


def xcorr_metrics(sig1, sig2, fps):
    sig1 = np.asarray(sig1, dtype=float)
    sig2 = np.asarray(sig2, dtype=float)
    valid = np.isfinite(sig1) & np.isfinite(sig2)
    sig1, sig2 = sig1[valid], sig2[valid]
    if len(sig1) < 3:
        return np.nan, np.nan, np.nan
    sig1 -= np.mean(sig1); sig2 -= np.mean(sig2)
    std1, std2 = np.std(sig1), np.std(sig2)
    if std1 == 0 or std2 == 0:
        return np.nan, np.nan, np.nan
    xcorr    = correlate(sig1, sig2, mode="full") / (std1 * std2 * len(sig1))
    lags_sec = np.arange(-len(sig1) + 1, len(sig1)) / fps
    peak_idx = np.argmax(np.abs(xcorr))
    abs_peak_corr = float(np.abs(xcorr[peak_idx]))
    abs_lag_s     = float(np.abs(lags_sec[peak_idx]))
    mid = len(xcorr) // 2
    pos = xcorr[mid + 1:]
    neg = xcorr[:mid][::-1]
    n   = min(len(pos), len(neg))
    denom = np.sum(np.abs(xcorr))
    if n == 0 or denom == 0:
        symmetry_index = np.nan
    else:
        symmetry_index = float(np.sum(np.abs(pos[:n] - neg[:n])) / denom)
    return abs_lag_s, abs_peak_corr, symmetry_index


def compute_mean_freq_and_rms_speed(tail_traj_y, fps, nperseg=96, noverlap=72,
                                     freq_min=2.0, freq_max=15.0):
    y = np.asarray(tail_traj_y, float)
    if len(y) < 4 or np.all(np.isnan(y)):
        return np.nan, np.nan, 0
    y = y[np.isfinite(y)]
    if len(y) < 4:
        return np.nan, np.nan, 0
    this_nperseg = min(nperseg, len(y))
    if this_nperseg < 8:
        mean_freq, n_windows = np.nan, 0
    else:
        this_noverlap = min(noverlap, this_nperseg - 1)
        f, _, Sxx = spectrogram(y, fs=fps, nperseg=this_nperseg, noverlap=this_noverlap)
        mask = (f >= freq_min) & (f <= freq_max)
        if np.any(mask) and Sxx.shape[1] > 0:
            freq_over_time = f[mask][np.argmax(Sxx[mask], axis=0)]
            mean_freq  = float(np.mean(freq_over_time)) if len(freq_over_time) else np.nan
            n_windows  = int(len(freq_over_time))
        else:
            mean_freq, n_windows = np.nan, 0
    rms_tail_speed = float(np.sqrt(np.mean(np.gradient(y, 1 / fps) ** 2)))
    return mean_freq, rms_tail_speed, n_windows


# ── Main extraction function ───────────────────────────────────────────────────

def extract_trial_features(csv_path, animal_id, trial, group, timepoint_label,
                            fps=120.0, cm_per_px=0.040207, swap_lr=True,
                            smooth_w=5, lane_length_cm=50.0,
                            min_peak_sep_s=0.13, prominence_trough=0.15,
                            seg_smooth_s=0.08, min_seg_len_frames=8,
                            prominence_speed=0.7, distance_speed=10,
                            left_threshold=0.75, right_threshold=-0.75,
                            nperseg=96, noverlap=72,
                            freq_min=2.0, freq_max=15.0):
    """
    Extract all swim features from a single cleaned CSV.
    animal_id, trial, group, timepoint_label come from metadata.
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    top0 = df.columns.get_level_values(0).astype(str)
    df   = df.loc[:, (~top0.str.startswith("Unnamed")) & (top0 != "scorer")]
    scorer = pick_scorer(df)

    if swap_lr:
        left_hl, right_hl = "R_toe_HL", "L_toe_HL"
        left_fl, right_fl = "R_palm_FL", "L_palm_FL"
    else:
        left_hl, right_hl = "L_toe_HL", "R_toe_HL"
        left_fl, right_fl = "L_palm_FL", "R_palm_FL"

    kw = dict(cm_per_px=cm_per_px, smooth_w=smooth_w)
    L_toe_x   = get_series_cm(df, scorer, left_hl,     "x", **kw)
    L_toe_y   = get_series_cm(df, scorer, left_hl,     "y", **kw)
    R_toe_x   = get_series_cm(df, scorer, right_hl,    "x", **kw)
    R_toe_y   = get_series_cm(df, scorer, right_hl,    "y", **kw)
    tail_x    = get_series_cm(df, scorer, "tail_base", "x", **kw)
    tail_y    = get_series_cm(df, scorer, "tail_base", "y", **kw)
    sternum_x = get_series_cm(df, scorer, "sternum",   "x", **kw)
    sternum_y = get_series_cm(df, scorer, "sternum",   "y", **kw)
    nose_x    = get_series_cm(df, scorer, "nose",      "x", **kw)
    nose_y    = get_series_cm(df, scorer, "nose",      "y", **kw)
    L_palm_y  = get_series_cm(df, scorer, left_fl,     "y", **kw)
    R_palm_y  = get_series_cm(df, scorer, right_fl,    "y", **kw)

    n_frames   = len(tail_x)
    duration_s = n_frames / fps if n_frames else np.nan

    seg_kw = dict(seg_smooth_s=seg_smooth_s, fps=fps,
                  min_peak_sep_s=min_peak_sep_s,
                  prominence_trough=prominence_trough,
                  min_seg_len_frames=min_seg_len_frames)

    left_basic  = analyze_hindlimb_from_troughs(L_toe_x, tail_x, sternum_x, **seg_kw)
    right_basic = analyze_hindlimb_from_troughs(R_toe_x, tail_x, sternum_x, **seg_kw)

    HLstroke = L_toe_x - tail_x
    HRstroke = R_toe_x - tail_x
    HL_peaks,   _ = find_peaks( HLstroke, prominence=prominence_speed, distance=distance_speed)
    HL_troughs, _ = find_peaks(-HLstroke, prominence=prominence_speed, distance=distance_speed)
    HR_peaks,   _ = find_peaks( HRstroke, prominence=prominence_speed, distance=distance_speed)
    HR_troughs, _ = find_peaks(-HRstroke, prominence=prominence_speed, distance=distance_speed)

    HL_pt_mean, HL_pt_cv = mean_cv(abs_slope_event_to_next(HL_peaks,   HL_troughs, HLstroke, fps))
    HL_tp_mean, HL_tp_cv = mean_cv(abs_slope_event_to_next(HL_troughs, HL_peaks,   HLstroke, fps))
    HR_pt_mean, HR_pt_cv = mean_cv(abs_slope_event_to_next(HR_peaks,   HR_troughs, HRstroke, fps))
    HR_tp_mean, HR_tp_cv = mean_cv(abs_slope_event_to_next(HR_troughs, HR_peaks,   HRstroke, fps))

    areas_L, circs_L, _ = process_limb_segments(L_toe_x - tail_x, L_toe_y - tail_y, **seg_kw)
    areas_R, circs_R, _ = process_limb_segments(R_toe_x - tail_x, R_toe_y - tail_y, **seg_kw)

    hl_lag, hl_corr, hl_sym = xcorr_metrics(HLstroke, HRstroke, fps)

    body_y      = smooth_len_preserve((nose_y + sternum_y) / 2.0, smooth_w)
    tail_traj_y = tail_y - body_y
    tail_freq, tail_osc_speed, _ = compute_mean_freq_and_rms_speed(
        tail_traj_y, fps=fps, nperseg=nperseg, noverlap=noverlap,
        freq_min=freq_min, freq_max=freq_max)

    L_tail_lag, L_tail_corr, _ = xcorr_metrics(HLstroke, tail_traj_y, fps)
    R_tail_lag, R_tail_corr, _ = xcorr_metrics(HRstroke, tail_traj_y, fps)

    L_rel = L_palm_y - sternum_y
    R_rel = R_palm_y - sternum_y
    left_mask  = L_rel > left_threshold
    right_mask = R_rel < right_threshold
    either_mask = left_mask | right_mask
    left_pct  = float(np.sum(left_mask)   / n_frames * 100) if n_frames else np.nan
    right_pct = float(np.sum(right_mask)  / n_frames * 100) if n_frames else np.nan
    both_pct  = float(np.sum(either_mask) / n_frames * 100) if n_frames else np.nan

    dx_axis    = tail_x - nose_x
    dy_axis    = tail_y - nose_y
    body_angle = np.degrees(np.arctan2(dy_axis, dx_axis))
    body_axis_mean = float(np.mean(body_angle))
    body_axis_std  = float(np.std(body_angle, ddof=1))

    v1 = np.column_stack([nose_x - sternum_x, nose_y - sternum_y])
    v2 = np.column_stack([tail_x - sternum_x, tail_y - sternum_y])
    dot       = np.sum(v1 * v2, axis=1)
    norm1     = np.linalg.norm(v1, axis=1)
    norm2     = np.linalg.norm(v2, axis=1)
    cos_angle = np.clip(dot / (norm1 * norm2 + 1e-8), -1, 1)
    curv_angles    = np.degrees(np.arccos(cos_angle))
    body_curv_mean = float(np.mean(curv_angles))
    body_curv_std  = float(np.std(curv_angles, ddof=1))

    swim_speed = lane_length_cm / duration_s if (np.isfinite(duration_s) and duration_s > 0) else np.nan

    return {
        "timepoint_label":             str(timepoint_label),
        "animal_id":                   str(animal_id),
        "trial":                       int(trial),
        "group":                       str(group),
        "n_frames":                    n_frames,
        "duration_s":                  duration_s,
        "swim_speed":                  swim_speed,
        "lh_frequency":                left_basic["freq_hz"],
        "lh_amp_avg":                  left_basic["avg_amplitude_cm"],
        "lh_efficiency_avg":           left_basic["avg_efficiency_cm"],
        "lh_amp_regularity_cv":        left_basic["cv_amplitude"],
        "lh_efficiency_regularity_cv": left_basic["cv_efficiency"],
        "lh_ext_speed_avg":            HL_pt_mean,
        "lh_ext_speed_regularity_cv":  HL_pt_cv,
        "lh_flex_speed_avg":           HL_tp_mean,
        "lh_flex_speed_regularity_cv": HL_tp_cv,
        "lh_loop_area_avg":            float(np.nanmean(areas_L)) if len(areas_L) else np.nan,
        "lh_loop_area_regularity_cv":  cv_percent(areas_L),
        "lh_loop_shape_regularity_cv": cv_percent(circs_L),
        "rh_frequency":                right_basic["freq_hz"],
        "rh_amp_avg":                  right_basic["avg_amplitude_cm"],
        "rh_efficiency_avg":           right_basic["avg_efficiency_cm"],
        "rh_amp_regularity_cv":        right_basic["cv_amplitude"],
        "rh_efficiency_regularity_cv": right_basic["cv_efficiency"],
        "rh_ext_speed_avg":            HR_pt_mean,
        "rh_ext_speed_regularity_cv":  HR_pt_cv,
        "rh_flex_speed_avg":           HR_tp_mean,
        "rh_flex_speed_regularity_cv": HR_tp_cv,
        "rh_loop_area_avg":            float(np.nanmean(areas_R)) if len(areas_R) else np.nan,
        "rh_loop_area_regularity_cv":  cv_percent(areas_R),
        "rh_loop_shape_regularity_cv": cv_percent(circs_R),
        "hl_lag":                      hl_lag,
        "hl_corr_strength":            hl_corr,
        "hl_symmetry":                 hl_sym,
        "tail_frequency":              tail_freq,
        "tail_oscillation_speed":      tail_osc_speed,
        "left_tail_lag":               L_tail_lag,
        "left_tail_coord_strength":    L_tail_corr,
        "right_tail_lag":              R_tail_lag,
        "right_tail_coord_strength":   R_tail_corr,
        "forelimb_use_left":           left_pct,
        "forelimb_use_right":          right_pct,
        "forelimb_use_both":           both_pct,
        "body_axis_angle_mean":        body_axis_mean,
        "body_axis_angle_std":         body_axis_std,
        "body_curvature_mean":         body_curv_mean,
        "body_curvature_std":          body_curv_std,
    }


def build_bilateral_features(df):
    """Average left/right hindlimb columns into bilateral hl_* columns."""
    df = df.copy()
    pairs = [
        ("frequency",                "hl_frequency"),
        ("amp_avg",                  "hl_amp_avg"),
        ("efficiency_avg",           "hl_efficiency_avg"),
        ("amp_regularity_cv",        "hl_amp_regularity_cv"),
        ("efficiency_regularity_cv", "hl_efficiency_regularity_cv"),
        ("ext_speed_avg",            "hl_ext_speed_avg"),
        ("ext_speed_regularity_cv",  "hl_ext_speed_regularity_cv"),
        ("flex_speed_avg",           "hl_flex_speed_avg"),
        ("flex_speed_regularity_cv", "hl_flex_speed_regularity_cv"),
        ("loop_area_avg",            "hl_loop_area_avg"),
        ("loop_area_regularity_cv",  "hl_loop_area_regularity_cv"),
        ("loop_shape_regularity_cv", "hl_loop_shape_regularity_cv"),
    ]
    for suffix, new_name in pairs:
        l, r = f"lh_{suffix}", f"rh_{suffix}"
        if l in df.columns and r in df.columns:
            df[new_name] = np.nanmean(df[[l, r]].values, axis=1)
    df["tail_lag_avg"] = np.nanmean(
        df[["left_tail_lag", "right_tail_lag"]].values, axis=1)
    df["tail_coord_strength_avg"] = np.nanmean(
        df[["left_tail_coord_strength", "right_tail_coord_strength"]].values, axis=1)
    return df


def compute_fastest(df):
    """Select the trial with highest swim_speed per animal × timepoint."""
    group_keys = ["animal_id", "timepoint_label"]
    trial_counts = (df.groupby(group_keys)["trial"]
                    .count().rename("n_trials").reset_index())
    fastest = (df.sort_values("swim_speed", ascending=False)
               .groupby(group_keys, sort=False)
               .first()
               .reset_index())
    fastest = fastest.merge(trial_counts, on=group_keys)
    return fastest.sort_values(["timepoint_label", "animal_id"]).reset_index(drop=True)


def compute_averaged(df):
    """Average all feature columns per animal × timepoint."""
    group_keys = ["animal_id", "timepoint_label"]
    feat_cols  = [c for c in FEATURE_COLS if c in df.columns]
    trial_counts = (df.groupby(group_keys)["trial"]
                    .count().rename("n_trials").reset_index())
    # Carry through all metadata columns (first value per group)
    meta_cols = [c for c in ["group", "genotype", "sex"]
                 if c in df.columns]
    meta = df.groupby(group_keys)[meta_cols].first().reset_index()
    agg  = df.groupby(group_keys)[feat_cols].mean().reset_index()
    avg  = meta.merge(agg, on=group_keys).merge(trial_counts, on=group_keys)
    return avg.sort_values(["timepoint_label", "animal_id"]).reset_index(drop=True)
