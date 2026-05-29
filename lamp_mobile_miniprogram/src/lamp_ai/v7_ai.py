from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .config import CurveConfig
from .features import extract_features_from_dataframe

# Keep this deliberately close to V4. Do not include rule scores, rule labels,
# sample labels, file paths, or other answer/leakage columns as AI features.
V4_CURVE_FEATURES = [
    "n_points", "raw_start", "raw_end", "smooth_start", "smooth_end",
    "baseline", "baseline_std", "final", "final_std", "min_value",
    "max_value", "mean_value", "std_value", "amplitude", "abs_amplitude",
    "rel_amplitude", "snr_amplitude", "max_slope", "min_slope",
    "slope_noise", "diff_noise_raw", "max_abs_delta_raw", "peak_slope_pos",
    "monotonic_ratio", "pos_slope_ratio", "roughness", "linear_r2",
    "time_corr", "t10", "t50", "t90", "t50_norm",
    "rise_time_10_90", "s_shape_score",
]

DISPLAY_MAP = {
    "positive": "阳性",
    "negative": "阴性",
    "abnormal": "异常",
    "uncertain": "需复核",
    "review": "需复核",
}


def display_label(label: object) -> str:
    return DISPLAY_MAP.get(str(label), str(label))


def normalize_teacher_label(x: object) -> str:
    text = str(x).strip().lower()
    if text in {"positive", "pos", "阳性"}:
        return "positive"
    if text in {"negative", "neg", "阴性"}:
        return "negative"
    if text in {"abnormal", "异常"}:
        return "abnormal"
    if text in {"uncertain", "review", "需复核", "可疑"}:
        return "review"
    return text or "review"


def find_feature_columns(df: pd.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
    if requested is not None:
        cols = [c for c in requested if c in df.columns]
        if cols:
            return cols
    return [c for c in V4_CURVE_FEATURES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def make_model(model_type: str = "extra_trees", random_state: int = 42) -> Pipeline:
    if model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced_subsample",
            min_samples_leaf=1,
            n_jobs=-1,
        )
    else:
        clf = ExtraTreesClassifier(
            n_estimators=700,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=1,
            n_jobs=-1,
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", clf),
    ])


def split_train_test(df: pd.DataFrame, y: np.ndarray, test_size: float, random_state: int):
    n = len(df)
    if n < 20 or len(np.unique(y)) < 2:
        idx = np.arange(n)
        return idx, idx, "no_holdout_small_dataset"

    if "sample_id" in df.columns and df["sample_id"].notna().nunique() >= 10:
        groups = df["sample_id"].astype(str).fillna(df.index.astype(str))
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, y, groups=groups))
        return train_idx, test_idx, "GroupShuffleSplit(sample_id)"

    counts = pd.Series(y).value_counts()
    if counts.min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, y))
        return train_idx, test_idx, "StratifiedShuffleSplit"

    idx_train, idx_test = train_test_split(np.arange(n), test_size=test_size, random_state=random_state)
    return idx_train, idx_test, "RandomShuffleSplit"


def train_v7_ai(
    pseudo_df: pd.DataFrame,
    model_type: str = "extra_trees",
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[Dict, Dict, pd.DataFrame]:
    df = pseudo_df.copy()
    if "use_for_training" in df.columns:
        mask = df["use_for_training"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        train_df = df[mask].copy()
    else:
        train_df = df.copy()

    label_col = "v7_train_label" if "v7_train_label" in train_df.columns else "label"
    if label_col not in train_df.columns:
        raise ValueError("pseudo table must contain v7_train_label or label")

    train_df[label_col] = train_df[label_col].map(normalize_teacher_label)
    train_df = train_df[train_df[label_col].isin(["positive", "negative", "abnormal", "review"])]
    if train_df.empty:
        raise ValueError("No pseudo-labeled rows available for V7 AI training")

    cols = find_feature_columns(train_df)
    if not cols:
        raise ValueError("No V4 curve feature columns found for V7 AI")

    encoder = LabelEncoder()
    y = encoder.fit_transform(train_df[label_col].astype(str))
    X = train_df[cols]

    pipeline = make_model(model_type=model_type, random_state=random_state)
    train_idx, test_idx, split_strategy = split_train_test(train_df, y, test_size, random_state)
    pipeline.fit(X.iloc[train_idx], y[train_idx])

    metrics: Dict = {
        "model_type": f"v7_rule_teacher_student_{model_type}",
        "principle": "V4 rule is kept unchanged. AI student is trained from high-confidence V4 rule pseudo labels.",
        "n_training_wells": int(len(train_df)),
        "train_label_counts": {str(k): int(v) for k, v in train_df[label_col].value_counts().to_dict().items()},
        "classes": list(encoder.classes_),
        "n_features": int(len(cols)),
        "feature_columns": cols,
        "split_strategy": split_strategy,
        "important_warning": "Metrics are against V4 rule pseudo labels, not manual well-level truth. Rule判读与AI判读在前端中保持分开。",
    }

    if split_strategy != "no_holdout_small_dataset":
        y_test = y[test_idx]
        y_pred = pipeline.predict(X.iloc[test_idx])
        metrics["holdout_report_against_rule_pseudo_labels"] = classification_report(
            y_test, y_pred, labels=np.arange(len(encoder.classes_)), target_names=encoder.classes_, output_dict=True, zero_division=0
        )
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred, labels=np.arange(len(encoder.classes_))).tolist()
        if hasattr(pipeline, "predict_proba") and len(encoder.classes_) == 2:
            proba = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            try:
                metrics["roc_auc_against_rule_pseudo_labels"] = float(roc_auc_score(y_test, proba))
                metrics["average_precision_against_rule_pseudo_labels"] = float(average_precision_score(y_test, proba))
            except Exception:
                pass

    bundle = {
        "pipeline": pipeline,
        "label_encoder": encoder,
        "feature_columns": cols,
        "model_type": "v7_rule_teacher_student",
        "display_map": DISPLAY_MAP,
        "rule_version": "v4_rule_unchanged",
        "version": "v7",
    }

    scored = predict_feature_table(df, bundle)
    return bundle, metrics, scored


def save_bundle(bundle: Dict, model_path: str | Path) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def load_bundle(model_path: str | Path) -> Dict:
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        return obj
    # Fallback for a bare sklearn estimator.
    return {"pipeline": obj, "label_encoder": None, "feature_columns": V4_CURVE_FEATURES, "version": "bare_sklearn"}


def predict_feature_table(feature_df: pd.DataFrame, bundle: Dict) -> pd.DataFrame:
    df = feature_df.copy()
    cols = bundle.get("feature_columns") or find_feature_columns(df)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("No usable feature columns for V7 AI prediction")

    pipeline = bundle.get("pipeline") or bundle.get("model")
    if pipeline is None:
        raise ValueError("Model bundle has no pipeline/model")
    encoder = bundle.get("label_encoder")

    X = df.reindex(columns=cols)
    pred_idx = pipeline.predict(X)
    if encoder is not None and hasattr(encoder, "inverse_transform"):
        labels = encoder.inverse_transform(pred_idx)
        classes = list(encoder.classes_)
    else:
        labels = np.asarray(pred_idx).astype(str)
        classes = sorted(set(labels))

    out_cols = [c for c in ["well", "well_index", "sample_id", "file", "relative_path"] if c in df.columns]
    out = df[out_cols].copy() if out_cols else pd.DataFrame(index=df.index)
    out["ai_label"] = labels
    out["ai_result"] = [display_label(x) for x in labels]
    out["ai_confidence"] = np.nan

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        out["ai_confidence"] = np.max(proba, axis=1).round(4)
        for i, cls in enumerate(classes):
            if i < proba.shape[1]:
                out[f"prob_{cls}"] = proba[:, i].round(4)
        if "positive" in classes:
            pos_idx = classes.index("positive")
            if pos_idx < proba.shape[1]:
                out["prob_positive"] = proba[:, pos_idx].round(4)

    # Keep features behind the result so the user can download/debug if needed.
    keep_meta = set(out.columns)
    rest = [c for c in df.columns if c not in keep_meta]
    return pd.concat([out, df[rest]], axis=1)


def predict_curves_dataframe(curves_df: pd.DataFrame, bundle: Dict, curve_config: CurveConfig | None = None) -> pd.DataFrame:
    feats = extract_features_from_dataframe(curves_df, config=curve_config or CurveConfig())
    return predict_feature_table(feats, bundle)
