from dataclasses import dataclass


@dataclass
class CurveConfig:
    """Common preprocessing configuration."""

    drop_first_rows: int = 5
    smooth_window: int = 5
    baseline_frac: float = 0.20
    final_frac: float = 0.20


@dataclass
class RuleConfig:
    """Thresholds for rule-based curve interpretation.

    These defaults are intentionally conservative and should be calibrated with
    your own negative controls, positive controls and clinical samples.
    """

    # Typical positive: sustained S-shaped rising curve.
    positive_min_abs_amp: float = 1500.0
    positive_min_rel_amp: float = 0.10
    positive_min_snr: float = 6.0
    positive_min_max_slope: float = 250.0
    positive_min_monotonic_ratio: float = 0.60
    positive_peak_min_pos: float = 0.15
    positive_peak_max_pos: float = 0.90

    # Typical negative: nearly flat curve with only small-scale fluctuation.
    negative_max_abs_amp: float = 1000.0
    negative_max_rel_amp: float = 0.08
    negative_max_max_slope: float = 650.0

    # Abnormal: abrupt non-sigmoidal jump, strong downward drift, excessive roughness.
    abnormal_min_abs_drop: float = 1800.0
    abnormal_min_rel_drop: float = 0.12
    abnormal_jump_abs: float = 2500.0
    abnormal_jump_noise_ratio: float = 10.0
    abnormal_roughness: float = 8.0
