from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import CurveConfig
from .data_io import read_lamp_file
from .features import extract_features_from_dataframe
from .label_utils import read_label_csv

NON_FEATURE_COLUMNS = {
    "file", "file_path", "relative_path", "parent_dir", "suffix",
    "well", "well_index", "label", "raw_label", "sample", "sample_name", "sample_id",
    "source_png", "image_position", "targets", "pathogen_targets", "resistance_targets", "other_targets",
    "summary_row", "parse_status", "reason", "rule_label",
}


def feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [c for c in feature_df.columns if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(feature_df[c])]


def build_dataset_from_label_table(
    csv_dir: str | Path,
    labels_df: pd.DataFrame,
    curve_config: CurveConfig | None = None,
) -> pd.DataFrame:
    """Build ML dataset from a label table.

    labels_df must include:
    - well: 1-based well index or Well_01-like name;
    - label: normalized class label;
    - optional file: csv filename. If omitted, csv_dir must contain exactly one CSV.
    """

    cfg = curve_config or CurveConfig()
    csv_dir = Path(csv_dir)
    if "file" not in labels_df.columns:
        csv_files = sorted(csv_dir.glob("*.csv"))
        if len(csv_files) != 1:
            raise ValueError("标签文件没有 file 列时，csv_dir 中必须只有一个 CSV。")
        labels_df = labels_df.copy()
        labels_df["file"] = csv_files[0].name

    rows = []
    for file_name, sub in labels_df.groupby("file"):
        csv_path = csv_dir / str(file_name)
        if not csv_path.exists():
            raise FileNotFoundError(f"标签文件引用的数据文件不存在：{csv_path}")
        curves = read_lamp_file(csv_path, drop_first_rows=cfg.drop_first_rows)
        feats = extract_features_from_dataframe(curves, config=cfg)

        for _, label_row in sub.iterrows():
            well = str(label_row["well"]).strip()
            if well in feats["well"].values:
                one = feats[feats["well"] == well].iloc[0].to_dict()
            else:
                idx = int(float(well))
                if 1 <= idx <= len(feats):
                    one = feats.iloc[idx - 1].to_dict()
                elif 0 <= idx < len(feats):
                    one = feats.iloc[idx].to_dict()
                else:
                    raise IndexError(f"{file_name} 的孔位 {well} 超出范围。")
            one["file"] = str(file_name)
            one["label"] = label_row["label"]
            if "raw_label" in label_row:
                one["raw_label"] = label_row["raw_label"]
            rows.append(one)

    if not rows:
        raise ValueError("没有构建出任何训练样本，请检查标签文件。")
    return pd.DataFrame(rows)


def train_model(
    dataset_df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[Dict, Dict]:
    """Train a random-forest classifier from extracted feature table."""

    cols = feature_columns(dataset_df)
    if not cols:
        raise ValueError("没有可用于训练的数值特征。")

    X = dataset_df[cols]
    y_text = dataset_df["label"].astype(str)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    min_samples_leaf=1,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    metrics: Dict = {"classes": list(encoder.classes_), "n_samples": int(len(dataset_df))}
    class_counts = y_text.value_counts().to_dict()
    metrics["class_counts"] = {str(k): int(v) for k, v in class_counts.items()}

    # Stratified split only when every class has at least two samples.
    can_split = len(dataset_df) >= 10 and min(class_counts.values()) >= 2 and len(class_counts) >= 2
    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics["holdout_report"] = classification_report(
            y_test,
            y_pred,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    else:
        pipeline.fit(X, y)
        metrics["holdout_report"] = None
        metrics["confusion_matrix"] = None
        metrics["note"] = "样本量或类别数不足，未划分验证集；已使用全部数据训练。"

    bundle = {
        "pipeline": pipeline,
        "label_encoder": encoder,
        "feature_columns": cols,
        "version": "0.1.0",
    }
    return bundle, metrics


def save_model(bundle: Dict, model_path: str | Path) -> None:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def load_model(model_path: str | Path) -> Dict:
    return joblib.load(model_path)


def save_metrics(metrics: Dict, metrics_path: str | Path) -> None:
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def predict_dataframe(curves_df: pd.DataFrame, model_bundle: Dict, curve_config: CurveConfig | None = None) -> pd.DataFrame:
    """Predict all wells in one numeric CSV using a trained model."""

    cfg = curve_config or CurveConfig()
    feats = extract_features_from_dataframe(curves_df, config=cfg)
    cols = model_bundle["feature_columns"]
    X = feats.reindex(columns=cols)
    pipeline = model_bundle["pipeline"]
    encoder = model_bundle["label_encoder"]
    pred_idx = pipeline.predict(X)
    labels = encoder.inverse_transform(pred_idx)

    out = feats[["well", "well_index"]].copy()
    out["ml_label"] = labels
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        out["ml_confidence"] = np.max(proba, axis=1).round(4)
        for i, cls in enumerate(encoder.classes_):
            out[f"prob_{cls}"] = proba[:, i].round(4)
    else:
        out["ml_confidence"] = np.nan
    return pd.concat([out, feats.drop(columns=["well", "well_index"])], axis=1)


def train_from_files(
    csv_dir: str | Path,
    label_csv: str | Path,
    model_path: str | Path,
    metrics_path: str | Path | None = None,
    label_mode: str = "binary",
    curve_config: CurveConfig | None = None,
) -> tuple[pd.DataFrame, Dict]:
    labels = read_label_csv(label_csv, label_mode=label_mode)
    dataset = build_dataset_from_label_table(csv_dir, labels, curve_config=curve_config)
    bundle, metrics = train_model(dataset)
    save_model(bundle, model_path)
    if metrics_path is not None:
        save_metrics(metrics, metrics_path)
    return dataset, metrics
