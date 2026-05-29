from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CurveConfig, RuleConfig
from .features import extract_curve_features, extract_features_from_dataframe, smooth_curve


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def interpret_feature_row(row: pd.Series, rule_config: RuleConfig | None = None) -> dict:
    """Interpret one curve from extracted features."""

    cfg = rule_config or RuleConfig()
    amp = float(row["amplitude"])
    abs_amp = float(row["abs_amplitude"])
    rel_amp = float(row["rel_amplitude"])
    snr = float(row["snr_amplitude"])
    max_slope = float(row["max_slope"])
    peak_pos = float(row["peak_slope_pos"])
    mono = float(row["monotonic_ratio"])
    rough = float(row["roughness"])
    jump = float(row["max_abs_delta_raw"])
    noise = max(float(row["diff_noise_raw"]), 1.0)

    reasons = []

    positive_votes = [
        amp >= cfg.positive_min_abs_amp,
        rel_amp >= cfg.positive_min_rel_amp,
        snr >= cfg.positive_min_snr,
        max_slope >= cfg.positive_min_max_slope,
        mono >= cfg.positive_min_monotonic_ratio,
        cfg.positive_peak_min_pos <= peak_pos <= cfg.positive_peak_max_pos,
    ]
    positive_score = sum(positive_votes) / len(positive_votes)

    negative_votes = [
        abs_amp <= cfg.negative_max_abs_amp or abs(rel_amp) <= cfg.negative_max_rel_amp,
        max_slope <= cfg.negative_max_max_slope,
        rough <= cfg.abnormal_roughness,
    ]
    negative_score = sum(negative_votes) / len(negative_votes)

    strong_drop = (amp <= -cfg.abnormal_min_abs_drop) and (abs(rel_amp) >= cfg.abnormal_min_rel_drop)
    isolated_jump = (jump >= cfg.abnormal_jump_abs) and (jump / noise >= cfg.abnormal_jump_noise_ratio)
    very_rough = rough >= cfg.abnormal_roughness

    if strong_drop:
        reasons.append("整体明显下降，疑似气泡、蒸发、读数漂移或反应异常")
    if isolated_jump and positive_score < 0.85:
        reasons.append("存在非典型大幅跳变，不符合稳定 S 型扩增")
    if very_rough and positive_score < 0.85:
        reasons.append("曲线粗糙度过高，波动过大")

    abnormal_score = 0.0
    abnormal_score += 0.45 if strong_drop else 0.0
    abnormal_score += 0.35 if isolated_jump and positive_score < 0.85 else 0.0
    abnormal_score += 0.20 if very_rough and positive_score < 0.85 else 0.0

    if positive_score >= 0.84 and not strong_drop:
        label = "positive"
        confidence = _clip01(0.55 + 0.45 * positive_score)
        reasons.append("呈持续上升并具有明显起峰，符合典型阳性扩增曲线")
    elif abnormal_score >= 0.45:
        label = "abnormal"
        confidence = _clip01(0.55 + abnormal_score)
        if not reasons:
            reasons.append("曲线形态异常，建议复核原始数据和孔位状态")
    elif negative_score >= 0.67:
        label = "negative"
        confidence = _clip01(0.55 + 0.35 * negative_score)
        reasons.append("未见稳定起峰，整体以小幅波动或缓慢漂移为主")
    else:
        label = "uncertain"
        confidence = _clip01(0.45 + 0.20 * max(positive_score, negative_score, abnormal_score))
        reasons.append("规则证据不足，建议结合阴阳性对照或使用机器学习模型复核")

    return {
        "rule_label": label,
        "rule_confidence": round(confidence, 4),
        "positive_score": round(float(positive_score), 4),
        "negative_score": round(float(negative_score), 4),
        "abnormal_score": round(float(abnormal_score), 4),
        "reason": "；".join(reasons),
    }


def rule_interpret_dataframe(
    curves_df: pd.DataFrame,
    curve_config: CurveConfig | None = None,
    rule_config: RuleConfig | None = None,
) -> pd.DataFrame:
    """Run rule-based interpretation for all wells."""

    feature_df = extract_features_from_dataframe(curves_df, config=curve_config)
    result_rows = []
    for _, row in feature_df.iterrows():
        result_rows.append(interpret_feature_row(row, rule_config=rule_config))
    result_df = pd.concat([feature_df, pd.DataFrame(result_rows)], axis=1)
    front_cols = [
        "well",
        "well_index",
        "rule_label",
        "rule_confidence",
        "positive_score",
        "negative_score",
        "abnormal_score",
        "reason",
    ]
    other_cols = [c for c in result_df.columns if c not in front_cols]
    return result_df[front_cols + other_cols]


def smooth_dataframe(curves_df: pd.DataFrame, curve_config: CurveConfig | None = None) -> pd.DataFrame:
    """Return smoothed curves for plotting."""

    cfg = curve_config or CurveConfig()
    return pd.DataFrame(
        {col: smooth_curve(curves_df[col].to_numpy(dtype=float), cfg.smooth_window) for col in curves_df.columns}
    )
