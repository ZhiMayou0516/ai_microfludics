from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

from .config import CurveConfig


def robust_std(x: np.ndarray) -> float:
    """Median absolute deviation based robust standard deviation."""

    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(1.4826 * mad)


def fill_nan(y: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(y, dtype=float))
    s = s.interpolate(method="linear", limit_direction="both")
    return s.to_numpy(dtype=float)


def smooth_curve(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Robust median + mean smoothing without scipy dependency."""

    arr = fill_nan(y)
    n = len(arr)
    if n < 3 or window <= 1:
        return arr
    w = int(min(window, n))
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return arr
    s = pd.Series(arr)
    med = s.rolling(w, center=True, min_periods=1).median()
    mean = med.rolling(w, center=True, min_periods=1).mean()
    return mean.to_numpy(dtype=float)


def _linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(y) < 1e-12:
        return 0.0
    coef = np.polyfit(x, y, deg=1)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _crossing_index(y: np.ndarray, threshold: float, increasing: bool = True) -> float:
    if increasing:
        idx = np.where(y >= threshold)[0]
    else:
        idx = np.where(y <= threshold)[0]
    if len(idx) == 0:
        return math.nan
    return float(idx[0])


def extract_curve_features(
    y_raw: np.ndarray,
    config: CurveConfig | None = None,
) -> Dict[str, float]:
    """Extract numeric descriptors from one amplification curve."""

    config = config or CurveConfig()
    raw = fill_nan(np.asarray(y_raw, dtype=float))
    y = smooth_curve(raw, config.smooth_window)
    n = len(y)
    if n < 5:
        raise ValueError("曲线点数过少，至少需要 5 个时间点。")

    x = np.arange(n, dtype=float)
    bw = max(3, int(round(n * config.baseline_frac)))
    ew = max(3, int(round(n * config.final_frac)))
    bw = min(bw, n)
    ew = min(ew, n)

    baseline_region = y[:bw]
    final_region = y[-ew:]
    baseline = float(np.median(baseline_region))
    final = float(np.median(final_region))
    baseline_std = robust_std(baseline_region)
    final_std = robust_std(final_region)
    amplitude = final - baseline
    abs_amplitude = abs(amplitude)
    rel_amplitude = amplitude / (abs(baseline) + 1e-9)

    dy = np.diff(y)
    raw_dy = np.diff(raw)
    max_slope = float(np.max(dy)) if len(dy) else 0.0
    min_slope = float(np.min(dy)) if len(dy) else 0.0
    max_abs_delta_raw = float(np.max(np.abs(raw_dy))) if len(raw_dy) else 0.0
    diff_noise = robust_std(raw_dy)
    slope_noise = robust_std(dy)
    peak_idx = int(np.argmax(dy)) if len(dy) else 0
    peak_pos = peak_idx / max(n - 2, 1)

    # Shape descriptors.
    tolerance = max(50.0, 0.25 * max(baseline_std, slope_noise, 1.0))
    monotonic_ratio = float(np.mean(dy >= -tolerance))
    pos_slope_ratio = float(np.mean(dy > tolerance))
    roughness = float(np.sum(np.abs(dy)) / (abs(y[-1] - y[0]) + 1.0))
    linear_r2 = _linear_r2(x, y)
    corr = float(np.corrcoef(x, y)[0, 1]) if np.nanstd(y) > 1e-12 else 0.0

    if amplitude > 0:
        y10 = baseline + 0.10 * amplitude
        y50 = baseline + 0.50 * amplitude
        y90 = baseline + 0.90 * amplitude
        t10 = _crossing_index(y, y10, increasing=True)
        t50 = _crossing_index(y, y50, increasing=True)
        t90 = _crossing_index(y, y90, increasing=True)
    else:
        y10 = baseline + 0.10 * amplitude
        y50 = baseline + 0.50 * amplitude
        y90 = baseline + 0.90 * amplitude
        t10 = _crossing_index(y, y10, increasing=False)
        t50 = _crossing_index(y, y50, increasing=False)
        t90 = _crossing_index(y, y90, increasing=False)

    rise_time_10_90 = float(t90 - t10) if np.isfinite(t10) and np.isfinite(t90) else math.nan
    t50_norm = float(t50 / max(n - 1, 1)) if np.isfinite(t50) else math.nan

    # S-curve proxy: large amplitude, centered max slope, increasing, and non-linear.
    peak_center_score = max(0.0, 1.0 - abs(peak_pos - 0.50) / 0.50)
    nonlinear_score = max(0.0, 1.0 - max(linear_r2, 0.0))
    s_shape_score = float(
        max(0.0, rel_amplitude)
        * monotonic_ratio
        * peak_center_score
        * (0.5 + 0.5 * nonlinear_score)
    )

    return {
        "n_points": float(n),
        "raw_start": float(raw[0]),
        "raw_end": float(raw[-1]),
        "smooth_start": float(y[0]),
        "smooth_end": float(y[-1]),
        "baseline": baseline,
        "baseline_std": baseline_std,
        "final": final,
        "final_std": final_std,
        "min_value": float(np.min(y)),
        "max_value": float(np.max(y)),
        "mean_value": float(np.mean(y)),
        "std_value": float(np.std(y)),
        "amplitude": float(amplitude),
        "abs_amplitude": float(abs_amplitude),
        "rel_amplitude": float(rel_amplitude),
        "snr_amplitude": float(amplitude / (baseline_std + 1.0)),
        "max_slope": max_slope,
        "min_slope": min_slope,
        "slope_noise": slope_noise,
        "diff_noise_raw": diff_noise,
        "max_abs_delta_raw": max_abs_delta_raw,
        "peak_slope_pos": float(peak_pos),
        "monotonic_ratio": monotonic_ratio,
        "pos_slope_ratio": pos_slope_ratio,
        "roughness": roughness,
        "linear_r2": float(linear_r2),
        "time_corr": corr,
        "t10": float(t10) if np.isfinite(t10) else math.nan,
        "t50": float(t50) if np.isfinite(t50) else math.nan,
        "t90": float(t90) if np.isfinite(t90) else math.nan,
        "t50_norm": t50_norm,
        "rise_time_10_90": rise_time_10_90,
        "s_shape_score": s_shape_score,
    }


def extract_features_from_dataframe(
    curves_df: pd.DataFrame,
    config: CurveConfig | None = None,
) -> pd.DataFrame:
    """Extract features from all wells in a dataframe."""

    rows = []
    for i, col in enumerate(curves_df.columns, start=1):
        feats = extract_curve_features(curves_df[col].to_numpy(dtype=float), config=config)
        row = {"well": col, "well_index": i}
        row.update(feats)
        rows.append(row)
    return pd.DataFrame(rows)
