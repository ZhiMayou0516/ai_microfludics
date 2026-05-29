from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamp_ai.config import CurveConfig, RuleConfig
from lamp_ai.data_io import read_lamp_file
from lamp_ai.rules import rule_interpret_dataframe, smooth_dataframe
from lamp_ai.v7_ai import display_label, load_bundle, predict_curves_dataframe

APP_TITLE = "LAMP 曲线智能判读后端"
DEFAULT_MODEL_PATH = ROOT / "models" / "lamp_v7_well_ai.joblib"
LABEL_CN = {
    "positive": "阳性",
    "negative": "阴性",
    "abnormal": "异常",
    "uncertain": "需复核",
    "review": "需复核",
}

app = FastAPI(title=APP_TITLE, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4)
    except Exception:
        return None


def _safe_value(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return _safe_float(x)
    if isinstance(x, float):
        return _safe_float(x)
    if pd.isna(x):
        return None
    return x


def _count_labels(labels: list[str]) -> dict[str, int]:
    keys = ["positive", "negative", "abnormal", "review", "uncertain"]
    counts = {k: 0 for k in keys}
    for x in labels:
        k = "review" if x == "uncertain" else str(x)
        counts[k] = counts.get(k, 0) + 1
    counts["review"] = counts.get("review", 0) + counts.pop("uncertain", 0)
    return counts


def _overall_from_counts(counts: dict[str, int]) -> str:
    # 医学检测页面上，阳性优先，其次异常/复核，最后阴性。
    if counts.get("positive", 0) > 0:
        return "positive"
    if counts.get("abnormal", 0) > 0:
        return "abnormal"
    if counts.get("review", 0) > 0:
        return "review"
    return "negative"


def _downsample_indices(n: int, max_points: int = 120) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).round().astype(int)


def _curves_payload(curves: pd.DataFrame, curve_cfg: CurveConfig, max_points: int = 120) -> dict[str, Any]:
    smooth = smooth_dataframe(curves, curve_cfg)
    idx = _downsample_indices(len(curves), max_points=max_points)
    time = [int(i) for i in idx.tolist()]
    wells = []
    for col in curves.columns:
        raw = curves[col].to_numpy(dtype=float)[idx]
        sm = smooth[col].to_numpy(dtype=float)[idx]
        wells.append({
            "well": str(col),
            "raw": [_safe_float(x) for x in raw],
            "smooth": [_safe_float(x) for x in sm],
        })
    return {"time": time, "wells": wells, "x_label": "time", "y_label": "荧光"}


def _rule_wells(result_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in result_df.iterrows():
        label = str(row.get("rule_label", "review"))
        rows.append({
            "well": str(row.get("well", "")),
            "well_index": int(row.get("well_index", len(rows) + 1)),
            "label": label,
            "label_text": LABEL_CN.get(label, label),
            "confidence": _safe_float(row.get("rule_confidence")),
            "positive_score": _safe_float(row.get("positive_score")),
            "negative_score": _safe_float(row.get("negative_score")),
            "abnormal_score": _safe_float(row.get("abnormal_score")),
            "reason": str(row.get("reason", "")),
        })
    return rows


def _ai_wells(pred_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
    for _, row in pred_df.iterrows():
        label = str(row.get("ai_label", "review"))
        item = {
            "well": str(row.get("well", "")),
            "well_index": int(row.get("well_index", len(rows) + 1)),
            "label": label,
            "label_text": display_label(label),
            "confidence": _safe_float(row.get("ai_confidence")),
        }
        for c in prob_cols:
            item[c] = _safe_float(row.get(c))
        rows.append(item)
    return rows


def _save_upload_to_tmp(file: UploadFile, original_filename: str = "") -> str:
    display_name = original_filename or file.filename or "curve.csv"
    suffix = Path(display_name).suffix.lower() or ".csv"
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="只支持 CSV/XLSX/XLS 文件。")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with tmp:
            shutil.copyfileobj(file.file, tmp)
        return tmp.name
    except Exception:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        raise


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "message": "LAMP backend is running",
        "model_exists": DEFAULT_MODEL_PATH.exists(),
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    original_filename: str = Form(""),
    mode: str = Form("rule"),
    sample_id: str = Form(""),
    drop_first_rows: int = Form(5),
    smooth_window: int = Form(5),
    drop_time_column: bool = Form(False),
) -> dict[str, Any]:
    tmp_path = _save_upload_to_tmp(file, original_filename=original_filename)
    try:
        curve_cfg = CurveConfig(drop_first_rows=0, smooth_window=int(smooth_window))
        curves = read_lamp_file(
            tmp_path,
            drop_first_rows=int(drop_first_rows),
            drop_time_column=bool(drop_time_column),
        )
        mode = (mode or "rule").lower().strip()
        if mode == "ai":
            if not DEFAULT_MODEL_PATH.exists():
                raise HTTPException(status_code=500, detail=f"模型文件不存在：{DEFAULT_MODEL_PATH}")
            try:
                bundle = load_bundle(DEFAULT_MODEL_PATH)
                pred = predict_curves_dataframe(curves, bundle, curve_config=curve_cfg)
            except Exception as model_exc:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "AI 模型加载或预测失败。这个 joblib 模型建议使用 scikit-learn==1.6.1；"
                        "可以先运行 pip install scikit-learn==1.6.1 后重启后端。"
                        f" 原始错误：{model_exc}"
                    ),
                )
            wells = _ai_wells(pred)
            result_table = pred
            method_text = "AI 判读"
        else:
            result = rule_interpret_dataframe(curves, curve_config=curve_cfg, rule_config=RuleConfig())
            wells = _rule_wells(result)
            result_table = result
            method_text = "规则判读"

        labels = [w["label"] for w in wells]
        counts = _count_labels(labels)
        overall = _overall_from_counts(counts)
        payload = {
            "ok": True,
            "sample_id": sample_id or Path(original_filename or file.filename or "").stem or "未填写",
            "filename": original_filename or file.filename,
            "mode": mode,
            "method_text": method_text,
            "overall_result": overall,
            "overall_text": LABEL_CN.get(overall, display_label(overall)),
            "counts": counts,
            "n_wells": len(wells),
            "n_points": int(curves.shape[0]),
            "wells": wells,
            "curves": _curves_payload(curves, curve_cfg),
            "table_columns": list(result_table.columns),
            "note": "结果由程序自动判读，正式诊断应结合对照孔、实验记录和人工复核。",
        }
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
