from __future__ import annotations

import re
from typing import IO, Union

import pandas as pd

PathLikeOrBuffer = Union[str, IO[bytes], IO[str]]


STANDARD_COLUMNS = {
    "file": ["file", "filename", "csv", "csv_file", "文件", "文件名", "数据文件"],
    "well": ["well", "well_id", "孔", "孔位", "列", "column", "col"],
    "label": ["label", "result", "target", "检测结果", "结果", "判读结果", "类别"],
    "sample": ["sample", "样品", "样本", "样本名", "sample_name"],
}


def normalize_text(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip().replace("'", "").replace("\"", "")


def normalize_label(label: object, mode: str = "binary") -> str:
    """Normalize Chinese/English labels.

    mode="binary": negative / positive / abnormal.
    mode="multiclass": keep pathogen or target names as positive classes; only
    negative and abnormal are normalized.
    """

    text = normalize_text(label)
    compact = re.sub(r"\s+", "", text).lower()
    if not compact:
        return "unknown"

    negative_keys = ["阴性", "negative", "neg", "未检出", "无扩增", "平稳", "无起峰"]
    abnormal_keys = ["异常", "无效", "invalid", "abnormal", "污染", "跳变", "大规模跃动"]
    positive_keys = ["阳性", "positive", "pos", "检出", "起峰"]

    if any(k in compact for k in abnormal_keys):
        return "abnormal"
    if any(k in compact for k in negative_keys):
        return "negative"
    if mode == "binary":
        if any(k in compact for k in positive_keys):
            return "positive"
        # In the user's screenshot, labels such as “鲍曼不动杆菌” or
        # “碳青霉烯类...” indicate a positive target rather than a negative call.
        if compact not in {"未知", "unknown", "na", "nan", "none"}:
            return "positive"
        return "unknown"
    return text


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for std, candidates in STANDARD_COLUMNS.items():
        for c in candidates:
            key = c.lower()
            if key in lower_map:
                rename[lower_map[key]] = std
                break
    return df.rename(columns=rename)


def read_label_csv(source: PathLikeOrBuffer, label_mode: str = "binary") -> pd.DataFrame:
    """Read label CSV for machine-learning training.

    Recommended format:
        file,well,label,sample
        314pc.csv,1,positive,unknown
        314pc.csv,2,negative,unknown

    For one CSV only, `file` can be omitted:
        well,label
        1,negative
        2,positive

    Chinese headers such as 文件名、孔位、检测结果 are also supported.
    """

    if hasattr(source, "seek"):
        source.seek(0)
    try:
        df = pd.read_csv(source, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, encoding="gbk")

    df = _rename_columns(df)
    if "label" not in df.columns:
        raise ValueError("标签文件必须包含 label/result/检测结果/结果 等标签列。")
    if "well" not in df.columns:
        raise ValueError("标签文件必须包含 well/孔位/列 等孔位列。")

    df["well"] = df["well"].apply(normalize_text)
    df["raw_label"] = df["label"].apply(normalize_text)
    df["label"] = df["label"].apply(lambda x: normalize_label(x, mode=label_mode))
    df = df[df["label"] != "unknown"].copy()
    return df
