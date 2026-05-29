from __future__ import annotations

from pathlib import Path
from typing import IO, Union

import numpy as np
import pandas as pd

PathLikeOrBuffer = Union[str, Path, IO[bytes], IO[str]]

SUPPORTED_CURVE_EXTS = {".csv", ".xlsx", ".xls"}


def _reset_buffer_if_possible(source: PathLikeOrBuffer) -> None:
    if hasattr(source, "seek"):
        try:
            source.seek(0)
        except Exception:
            pass


def _clean_numeric_frame(
    raw: pd.DataFrame,
    drop_first_rows: int = 5,
    drop_time_column: bool = False,
) -> pd.DataFrame:
    """Convert a raw table into numeric LAMP curve dataframe.

    The project convention is:
    - columns = wells / reaction holes;
    - rows = time points;
    - optional first rows can be removed;
    - optional first numeric column can be removed when it is only a time/index column.
    """

    numeric = raw.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if numeric.empty:
        raise ValueError("文件中没有读到有效数字，请检查分隔符、工作表或文件内容。")

    if drop_time_column and numeric.shape[1] > 1:
        numeric = numeric.iloc[:, 1:].copy()

    if drop_first_rows > 0:
        if len(numeric) <= drop_first_rows:
            raise ValueError(
                f"数据只有 {len(numeric)} 行，无法去除前 {drop_first_rows} 行。"
            )
        numeric = numeric.iloc[drop_first_rows:].copy()

    numeric = numeric.reset_index(drop=True)
    numeric.columns = [f"Well_{i + 1:02d}" for i in range(numeric.shape[1])]
    return numeric.astype(float)


def read_lamp_csv(
    source: PathLikeOrBuffer,
    drop_first_rows: int = 5,
    encoding: str = "utf-8-sig",
    drop_time_column: bool = False,
) -> pd.DataFrame:
    """Read pure numeric LAMP fluorescence CSV.

    Input format:
    - each column = one well / one reaction hole;
    - each row = one time point;
    - no label column is required;
    - trailing empty columns/rows are automatically removed;
    - the first `drop_first_rows` time points are removed by default because the
      early fluorescence values are often unstable.
    """

    _reset_buffer_if_possible(source)
    read_errors = []
    for enc in (encoding, "gbk", "utf-8", "latin1"):
        for sep in (None, ",", "\t", ";"):
            _reset_buffer_if_possible(source)
            try:
                raw = pd.read_csv(
                    source,
                    header=None,
                    sep=sep,
                    engine="python",
                    encoding=enc,
                    skip_blank_lines=True,
                )
                return _clean_numeric_frame(raw, drop_first_rows, drop_time_column)
            except Exception as exc:
                read_errors.append(f"encoding={enc}, sep={sep}: {exc}")

    raise ValueError("CSV 读取失败：" + " | ".join(read_errors[:6]))


def read_lamp_excel(
    source: PathLikeOrBuffer,
    drop_first_rows: int = 5,
    sheet_name: int | str = 0,
    drop_time_column: bool = False,
) -> pd.DataFrame:
    """Read LAMP curve data from Excel.

    This is for clinical exported files such as `检测结果 (1).xlsx`.
    The function deliberately treats the sheet as a numeric matrix and ignores
    non-numeric cells.
    """

    _reset_buffer_if_possible(source)
    raw = pd.read_excel(source, header=None, sheet_name=sheet_name)
    if isinstance(raw, dict):
        # When sheet_name=None is passed by mistake, use the first sheet.
        raw = next(iter(raw.values()))
    return _clean_numeric_frame(raw, drop_first_rows, drop_time_column)


def read_lamp_file(
    source: PathLikeOrBuffer,
    drop_first_rows: int = 5,
    sheet_name: int | str = 0,
    drop_time_column: bool = False,
) -> pd.DataFrame:
    """Read a LAMP curve file from CSV or Excel by file suffix.

    File-like objects uploaded through Streamlit have a `.name`; when suffix is
    unavailable the function falls back to CSV parsing.
    """

    suffix = ""
    if isinstance(source, (str, Path)):
        suffix = Path(source).suffix.lower()
    else:
        suffix = Path(getattr(source, "name", "")).suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return read_lamp_excel(source, drop_first_rows=drop_first_rows, sheet_name=sheet_name, drop_time_column=drop_time_column)
    return read_lamp_csv(source, drop_first_rows=drop_first_rows, drop_time_column=drop_time_column)


def save_result_csv(result_df: pd.DataFrame, output_path: PathLikeOrBuffer) -> None:
    """Save result table with utf-8-sig so Excel can open Chinese labels."""

    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def get_well_series(df: pd.DataFrame, well: str | int) -> np.ndarray:
    """Get one well curve from dataframe.

    `well` supports Well_01, 1-based integer, or zero-based integer-like string.
    """

    if isinstance(well, str) and well in df.columns:
        return df[well].to_numpy(dtype=float)
    if isinstance(well, str) and well.lower().startswith("well_"):
        return df[well].to_numpy(dtype=float)

    idx = int(well)
    if 1 <= idx <= df.shape[1]:
        return df.iloc[:, idx - 1].to_numpy(dtype=float)
    if 0 <= idx < df.shape[1]:
        return df.iloc[:, idx].to_numpy(dtype=float)
    raise IndexError(f"孔位 {well} 超出范围，当前共有 {df.shape[1]} 个孔位。")
