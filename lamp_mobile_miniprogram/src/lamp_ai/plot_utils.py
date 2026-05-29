from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import CurveConfig
from .rules import smooth_dataframe


def plot_all_curves(curves_df: pd.DataFrame, output_path: str | Path | None = None, smooth: bool = True) -> None:
    """Plot all well curves into one figure."""

    data = smooth_dataframe(curves_df, CurveConfig()) if smooth else curves_df
    plt.figure(figsize=(11, 6))
    for col in data.columns:
        plt.plot(data.index, data[col], linewidth=1, alpha=0.8)
    plt.xlabel("Time point after removing early unstable rows")
    plt.ylabel("Fluorescence")
    plt.title("LAMP amplification curves")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_one_curve(
    curves_df: pd.DataFrame,
    well: str,
    output_path: str | Path | None = None,
    smooth: bool = True,
) -> None:
    """Plot raw and smoothed curve for one well."""

    if well not in curves_df.columns:
        raise ValueError(f"找不到孔位：{well}")
    plt.figure(figsize=(8, 4.5))
    plt.plot(curves_df.index, curves_df[well], marker="o", linewidth=1, label="raw")
    if smooth:
        sm = smooth_dataframe(curves_df[[well]], CurveConfig())
        plt.plot(sm.index, sm[well], linewidth=2, label="smoothed")
    plt.xlabel("Time point after removing early unstable rows")
    plt.ylabel("Fluorescence")
    plt.title(well)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=200)
        plt.close()
    else:
        plt.show()
