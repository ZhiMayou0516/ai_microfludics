from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import CurveConfig, RuleConfig
from .data_io import SUPPORTED_CURVE_EXTS, read_lamp_file
from .label_utils import normalize_label, normalize_text
from .rules import rule_interpret_dataframe

PATHOGEN_TARGETS = [
    "肺炎克雷伯菌",
    "大肠埃希氏菌",
    "鲍曼不动杆菌",
    "铜绿假单胞菌",
    "金黄色葡萄球菌",
    "屎肠球菌",
    "粪肠球菌",
    "沙门菌",
    "阴沟肠杆菌",
]

RESISTANCE_TARGETS = [
    "碳青霉烯类（A类丝氨酸酶）",
    "碳青霉烯类（D类丝氨酸酶）",
    "喹诺酮类",
    "利奈唑胺类",
    "四环素类",
    "多粘菌素类",
    "头孢菌素类",
]

META_COLUMNS = {
    "sample_id",
    "sample_name",
    "source_png",
    "image_position",
    "raw_label",
    "label",
    "targets",
    "pathogen_targets",
    "resistance_targets",
    "file",
    "file_path",
    "relative_path",
    "parent_dir",
    "summary_row",
    "parse_status",
    "well",
    "well_index",
    "reason",
    "rule_label",
}


def split_targets(raw_label: object) -> list[str]:
    """Split a clinical result such as `大肠埃希氏菌+四环素类` into targets."""

    text = normalize_text(raw_label)
    if not text or normalize_label(text, mode="binary") == "negative":
        return []
    if text in {"阳性", "positive", "Positive", "POS"}:
        return ["阳性"]
    return [p.strip() for p in re.split(r"\s*\+\s*", text) if p.strip()]


def classify_targets(targets: Iterable[str]) -> tuple[list[str], list[str], list[str]]:
    targets = list(targets)
    pathogens = [t for t in targets if any(k in t for k in PATHOGEN_TARGETS)]
    resistances = [t for t in targets if any(k in t for k in RESISTANCE_TARGETS)]
    others = [t for t in targets if t not in pathogens and t not in resistances]
    return pathogens, resistances, others


def _parse_summary_line(text: str) -> list[dict]:
    """Parse one line in the user's `结果汇总表.xlsx`.

    Expected line examples:
    结果1-2.png: 上图样品='未知', 检测结果='阴性'; 下图样品='未知', 检测结果='阴性'
    """

    text = normalize_text(text)
    if not text or "检测结果" not in text:
        return []

    head = re.search(r"结果\s*(\d+)\s*[-—–]\s*(\d+)\s*\.\s*png", text, flags=re.I)
    if not head:
        return []
    ids = [int(head.group(1)), int(head.group(2))]
    source_png = f"结果{ids[0]}-{ids[1]}.png"

    item_re = re.compile(
        r"(上图|下图)\s*样品\s*=\s*[\'\"‘’“”]?([^\'\"‘’“”;；,，]*)[\'\"‘’“”]?\s*[,，]\s*检测结果\s*=\s*[\'\"‘’“”]?([^\'\"‘’“”;；]*)[\'\"‘’“”]?"
    )
    items = item_re.findall(text)
    if len(items) < 2:
        return []

    rows = []
    position_to_id = {"上图": ids[0], "下图": ids[1]}
    for pos, sample_name, raw_label in items[:2]:
        sid = position_to_id.get(pos, ids[0] if pos == "上图" else ids[1])
        targets = split_targets(raw_label)
        pathogens, resistances, others = classify_targets(targets)
        rows.append(
            {
                "sample_id": sid,
                "source_png": source_png,
                "image_position": pos,
                "sample_name": normalize_text(sample_name) or "未知",
                "raw_label": normalize_text(raw_label),
                "targets": "+".join(targets),
                "pathogen_targets": "+".join(pathogens),
                "resistance_targets": "+".join(resistances),
                "other_targets": "+".join(others),
                "parse_status": "ok",
            }
        )
    return rows


def parse_clinical_summary_excel(
    summary_path: str | Path,
    label_mode: str = "binary",
) -> pd.DataFrame:
    """Parse `结果汇总表.xlsx` into one row per clinical sample.

    Parameters
    ----------
    label_mode:
        - `binary`: 阴性 -> negative; any concrete pathogen/resistance result -> positive.
        - `multiclass`: 阴性 -> negative; concrete combinations are kept as their raw Chinese label.
    """

    summary_path = Path(summary_path)
    raw = pd.read_excel(summary_path, header=None, dtype=str)
    rows: list[dict] = []
    for i, val in enumerate(raw.iloc[:, 0].fillna(""), start=1):
        parsed = _parse_summary_line(str(val))
        for row in parsed:
            row["summary_row"] = i
            row["label"] = normalize_label(row["raw_label"], mode=label_mode)
            rows.append(row)
    if not rows:
        raise ValueError(f"没有从汇总表中解析到检测结果：{summary_path}")
    df = pd.DataFrame(rows)
    # Keep the last occurrence if the same sample appears twice; also make duplicates visible to the user.
    dup_counts = df["sample_id"].value_counts()
    df["sample_id_duplicate_count"] = df["sample_id"].map(dup_counts).astype(int)
    df = df.sort_values("sample_id").reset_index(drop=True)
    return df


def extract_sample_id_from_filename(path: str | Path) -> int | None:
    """Extract the local sample/file number from names such as `检测结果 (12).xlsx`."""

    stem = Path(path).stem
    patterns = [
        r"检测结果\s*[\(（]\s*(\d+)\s*[\)）]",
        r"检测结果\s*(\d+)",
        r"^结果\s*(\d+)$",
        r"^(\d+)$",
    ]
    for pat in patterns:
        m = re.search(pat, stem)
        if m:
            return int(m.group(1))
    return None


def _parse_result_range_folder(name: str) -> tuple[int, int] | None:
    """Parse folder names such as `结果1-1600` or `结果3857-5444`."""

    m = re.match(r"^结果\s*(\d+)\s*[-—–]\s*(\d+)\s*$", str(name).strip())
    if not m:
        return None
    start, end = int(m.group(1)), int(m.group(2))
    if end < start:
        start, end = end, start
    return start, end


def infer_global_sample_id(path: str | Path, root_dir: str | Path, local_sample_id: int, sample_id_mode: str = "batch_offset") -> dict:
    """Infer the global sample id used by `结果汇总表.xlsx`.

    In the user's clinical folder, files are often stored under batch folders such as
    `结果1601-3856` and local filenames restart from `检测结果 (1)`.  Therefore the
    global sample id should usually be `batch_start + local_id - 1`, not the local
    filename number itself.  Keeping the local id would silently attach wrong labels.
    """

    p = Path(path)
    root = Path(root_dir)
    try:
        parts = p.relative_to(root).parts
    except Exception:
        parts = p.parts

    candidates: list[tuple[int, int, str]] = []
    for part in parts:
        rng = _parse_result_range_folder(part)
        if rng is not None:
            start, end = rng
            candidates.append((start, end, part))

    # When nested folders also contain tiny ranges, use the widest batch range.
    batch_start = None
    batch_end = None
    batch_name = ""
    if candidates:
        batch_start, batch_end, batch_name = max(candidates, key=lambda x: (x[1] - x[0], x[0]))

    if sample_id_mode == "local" or batch_start is None:
        sample_id = local_sample_id
        mode_used = "local"
    elif sample_id_mode == "batch_offset":
        sample_id = batch_start + local_sample_id - 1
        mode_used = "batch_offset"
    else:
        raise ValueError(f"Unsupported sample_id_mode: {sample_id_mode}")

    warning = ""
    if batch_start is not None and sample_id_mode == "batch_offset" and sample_id > batch_end:
        warning = f"global sample_id {sample_id} exceeds batch range {batch_start}-{batch_end}"

    return {
        "sample_id": int(sample_id),
        "local_sample_id": int(local_sample_id),
        "batch_start": batch_start,
        "batch_end": batch_end,
        "batch_name": batch_name,
        "sample_id_mode": mode_used,
        "sample_id_warning": warning,
    }



def _parse_override_map(batch_start_override: dict | str | None) -> dict[str, int]:
    """Parse manual batch start overrides.

    Accepted formats:
    - dict: {"结果3857-5444": 1601}
    - string: "结果3857-5444=1601;结果1-1600=1"
    """
    if not batch_start_override:
        return {}
    if isinstance(batch_start_override, dict):
        return {str(k): int(v) for k, v in batch_start_override.items()}
    out: dict[str, int] = {}
    for part in str(batch_start_override).split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _is_numeric_range_dir(name: str) -> bool:
    """Keep normal grouping folders such as 1-9, 10-99, 1000-2030."""
    return re.match(r"^\s*\d+\s*[-—–]\s*\d+\s*$", str(name).strip()) is not None


def infer_global_sample_id(
    path: str | Path,
    root_dir: str | Path,
    local_sample_id: int,
    sample_id_mode: str = "batch_offset",
    batch_start_override: dict | str | None = None,
) -> dict:
    """Infer the global sample id used by `结果汇总表.xlsx`.

    v4 adds manual `batch_start_override`, because the current clinical folder has
    a misleading batch name: `结果3857-5444` contains local files 1-2030 and appears
    to actually continue the sample sequence after sample 1600.
    """

    p = Path(path)
    root = Path(root_dir)
    try:
        parts = p.relative_to(root).parts
    except Exception:
        parts = p.parts

    candidates: list[tuple[int, int, str]] = []
    for part in parts:
        rng = _parse_result_range_folder(part)
        if rng is not None:
            start, end = rng
            candidates.append((start, end, part))

    batch_start = None
    batch_end = None
    batch_name = ""
    if candidates:
        batch_start, batch_end, batch_name = max(candidates, key=lambda x: (x[1] - x[0], x[0]))

    overrides = _parse_override_map(batch_start_override)
    if batch_name and batch_name in overrides:
        batch_start = int(overrides[batch_name])

    if sample_id_mode == "local" or batch_start is None:
        sample_id = local_sample_id
        mode_used = "local"
    elif sample_id_mode == "batch_offset":
        sample_id = batch_start + local_sample_id - 1
        mode_used = "batch_offset"
    elif sample_id_mode == "two_per_file":
        # This returns the first sample ID of the two-sample file; the build step expands it.
        sample_id = batch_start + 2 * (local_sample_id - 1)
        mode_used = "two_per_file"
    else:
        raise ValueError(f"Unsupported sample_id_mode: {sample_id_mode}")

    warning = ""
    if batch_start is not None and batch_end is not None:
        if sample_id_mode == "batch_offset" and sample_id > batch_end:
            warning = f"global sample_id {sample_id} exceeds batch range {batch_start}-{batch_end}"
        if sample_id_mode == "two_per_file" and batch_name not in overrides:
            # Folder range is only advisory in two_per_file mode, because users may override it.
            second_sid = sample_id + 1
            if second_sid > batch_end:
                warning = f"two_per_file sample_ids {sample_id}-{second_sid} exceed folder range {batch_start}-{batch_end}"

    return {
        "sample_id": int(sample_id),
        "local_sample_id": int(local_sample_id),
        "batch_start": batch_start,
        "batch_end": batch_end,
        "batch_name": batch_name,
        "sample_id_mode": mode_used,
        "sample_id_warning": warning,
    }


def scan_clinical_curve_files(
    root_dir: str | Path,
    sample_id_mode: str = "batch_offset",
    batch_start_override: dict | str | None = None,
    numeric_parent_only: bool = False,
) -> pd.DataFrame:
    """Scan the nested clinical folder and find curve data files.

    `numeric_parent_only=True` skips ad-hoc folders such as `PQH-1102` and
    `25-8-21-王-3-C15`, while keeping normal folders like `1-9`, `10-99`,
    `100-999`, `1000-2030`.
    """

    root = Path(root_dir)
    rows = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith(".") or p.name.startswith("~$"):
            continue
        suffix = p.suffix.lower()
        if suffix not in SUPPORTED_CURVE_EXTS:
            continue
        name = p.name
        if "汇总" in name or "样品" in p.stem:
            continue
        if "检测结果" not in p.stem:
            continue
        if numeric_parent_only and not _is_numeric_range_dir(p.parent.name):
            continue
        local_sid = extract_sample_id_from_filename(p)
        if local_sid is None:
            continue
        sid_info = infer_global_sample_id(
            p,
            root,
            local_sid,
            sample_id_mode=sample_id_mode,
            batch_start_override=batch_start_override,
        )
        rows.append(
            {
                **sid_info,
                "file": p.name,
                "file_path": str(p),
                "relative_path": str(p.relative_to(root)),
                "parent_dir": p.parent.name,
                "suffix": suffix,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["sample_id", "local_sample_id", "file", "file_path", "relative_path", "parent_dir", "suffix"])
    return pd.DataFrame(rows).sort_values(["sample_id", "relative_path"]).reset_index(drop=True)


def expand_curve_files_to_samples(
    curves: pd.DataFrame,
    sample_layout: str = "one_per_file",
    wells_per_sample: int = 15,
) -> pd.DataFrame:
    """Expand a curve-file index to sample-level rows.

    In the current clinical summary, one line is `结果1-2.png` with 上图/下图 two
    samples. The corresponding raw file is usually `检测结果 (1).csv/.xlsx`, with
    30 wells. Therefore `sample_layout=two_per_file_15x2` maps:
      file 1 -> sample 1 wells 1-15; sample 2 wells 16-30
      file 2 -> sample 3 wells 1-15; sample 4 wells 16-30
    """
    if curves.empty:
        return curves.copy()
    rows: list[dict] = []
    if sample_layout == "one_per_file":
        for _, row in curves.iterrows():
            d = row.to_dict()
            d.update({"sample_slot": 1, "well_start": None, "well_end": None, "sample_layout": sample_layout})
            rows.append(d)
    elif sample_layout == "two_per_file_15x2":
        for _, row in curves.iterrows():
            base_id = int(row["sample_id"])
            for slot in (1, 2):
                d = row.to_dict()
                d["sample_id"] = base_id + slot - 1
                d["sample_slot"] = slot
                d["well_start"] = (slot - 1) * wells_per_sample + 1
                d["well_end"] = slot * wells_per_sample
                d["sample_layout"] = sample_layout
                rows.append(d)
    else:
        raise ValueError(f"Unsupported sample_layout: {sample_layout}")
    return pd.DataFrame(rows).sort_values(["sample_id", "relative_path", "sample_slot"]).reset_index(drop=True)


def aggregate_well_result_to_sample(
    well_result: pd.DataFrame,
    include_well_features: bool = True,
) -> dict:
    """Aggregate per-well rule/features into one sample-level feature row."""

    out: dict[str, float | int] = {}
    out["n_wells"] = int(len(well_result))

    if "rule_label" in well_result.columns:
        counts = well_result["rule_label"].value_counts().to_dict()
        for cls in ["positive", "negative", "abnormal", "uncertain"]:
            c = int(counts.get(cls, 0))
            out[f"rule_{cls}_count"] = c
            out[f"rule_{cls}_ratio"] = c / max(len(well_result), 1)

    numeric_cols = [
        c for c in well_result.columns
        if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(well_result[c])
    ]
    for c in numeric_cols:
        s = pd.to_numeric(well_result[c], errors="coerce")
        out[f"{c}_mean"] = float(s.mean())
        out[f"{c}_std"] = float(s.std(ddof=0))
        out[f"{c}_min"] = float(s.min())
        out[f"{c}_max"] = float(s.max())

    if include_well_features:
        keep = [
            "amplitude",
            "rel_amplitude",
            "snr_amplitude",
            "max_slope",
            "peak_slope_pos",
            "monotonic_ratio",
            "roughness",
            "t50_norm",
            "s_shape_score",
            "positive_score",
            "negative_score",
            "abnormal_score",
        ]
        # Use within-sample well order when a 30-well file is split into two 15-well samples.
        if "within_sample_well_index" in well_result.columns:
            sort_col = "within_sample_well_index"
        else:
            sort_col = "well_index"
        for _, row in well_result.sort_values(sort_col).iterrows():
            wi = int(row.get(sort_col, row.get("well_index", 0)))
            for c in keep:
                if c in row:
                    out[f"well_{wi:02d}_{c}"] = row[c]
    return out


def _subset_wells_for_sample(well_result: pd.DataFrame, well_start: object, well_end: object) -> pd.DataFrame:
    """Keep wells belonging to one sample slot and add within-sample well indices."""
    if pd.isna(well_start) or pd.isna(well_end) or well_start is None or well_end is None:
        wr = well_result.copy()
        wr["within_sample_well_index"] = wr["well_index"] if "well_index" in wr.columns else range(1, len(wr) + 1)
        return wr
    start = int(well_start)
    end = int(well_end)
    wr = well_result[(well_result["well_index"] >= start) & (well_result["well_index"] <= end)].copy()
    wr["within_sample_well_index"] = wr["well_index"].astype(int) - start + 1
    return wr


def build_clinical_dataset(
    root_dir: str | Path,
    summary_path: str | Path,
    label_mode: str = "binary",
    curve_config: CurveConfig | None = None,
    rule_config: RuleConfig | None = None,
    drop_time_column: bool = False,
    include_well_features: bool = True,
    max_files: int | None = None,
    sample_id_mode: str = "batch_offset",
    sample_layout: str = "one_per_file",
    wells_per_sample: int = 15,
    batch_start_override: dict | str | None = None,
    numeric_parent_only: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build sample-level and well-level datasets from clinical folders."""

    cfg = curve_config or CurveConfig()
    rcfg = rule_config or RuleConfig()
    labels = parse_clinical_summary_excel(summary_path, label_mode=label_mode)
    curves = scan_clinical_curve_files(
        root_dir,
        sample_id_mode=sample_id_mode,
        batch_start_override=batch_start_override,
        numeric_parent_only=numeric_parent_only,
    )
    if curves.empty:
        raise FileNotFoundError(
            "没有扫描到 `检测结果 (n).csv/.xlsx/.xls` 形式的曲线文件；请检查 root_dir 是否指向 medicaldata 根目录。"
        )

    expanded = expand_curve_files_to_samples(curves, sample_layout=sample_layout, wells_per_sample=wells_per_sample)
    merged = expanded.merge(labels, on="sample_id", how="left", suffixes=("", "_label"))
    matched = merged[merged["label"].notna()].copy()
    if max_files is not None:
        matched = matched.head(max_files).copy()

    sample_rows: list[dict] = []
    well_rows: list[pd.DataFrame] = []
    errors: list[dict] = []

    # Cache rule extraction per source file so two-per-file mode does not read each file twice.
    file_cache: dict[str, pd.DataFrame] = {}

    for k, row in matched.reset_index(drop=True).iterrows():
        if verbose and (k + 1) % 100 == 0:
            print(f"processed {k + 1}/{len(matched)} samples")
        file_path = Path(row["file_path"])
        try:
            file_key = str(file_path)
            if file_key not in file_cache:
                curves_df = read_lamp_file(
                    file_path,
                    drop_first_rows=cfg.drop_first_rows,
                    drop_time_column=drop_time_column,
                )
                file_cache[file_key] = rule_interpret_dataframe(curves_df, curve_config=cfg, rule_config=rcfg)
            well_result_all = file_cache[file_key]
            well_result = _subset_wells_for_sample(row.get("well_start"), row.get("well_end"), well_result_all) if False else _subset_wells_for_sample(well_result_all, row.get("well_start"), row.get("well_end"))
            meta = {
                "sample_id": int(row["sample_id"]),
                "local_sample_id": int(row.get("local_sample_id", row["sample_id"])),
                "sample_slot": int(row.get("sample_slot", 1)),
                "sample_layout": row.get("sample_layout", sample_layout),
                "well_start": row.get("well_start", ""),
                "well_end": row.get("well_end", ""),
                "batch_name": row.get("batch_name", ""),
                "batch_start": row.get("batch_start", ""),
                "batch_end": row.get("batch_end", ""),
                "sample_id_mode": row.get("sample_id_mode", sample_id_mode),
                "sample_id_warning": row.get("sample_id_warning", ""),
                "file": row["file"],
                "file_path": row["file_path"],
                "relative_path": row["relative_path"],
                "parent_dir": row["parent_dir"],
                "sample_name": row.get("sample_name", "未知"),
                "raw_label": row["raw_label"],
                "label": row["label"],
                "source_png": row.get("source_png", ""),
                "image_position": row.get("image_position", ""),
                "targets": row.get("targets", ""),
                "pathogen_targets": row.get("pathogen_targets", ""),
                "resistance_targets": row.get("resistance_targets", ""),
                "summary_row": int(row.get("summary_row", -1)),
            }
            wr = well_result.copy()
            for mk, mv in meta.items():
                wr[mk] = mv
            well_rows.append(wr)

            sample_feature = dict(meta)
            sample_feature.update(aggregate_well_result_to_sample(well_result, include_well_features=include_well_features))
            sample_rows.append(sample_feature)
        except Exception as exc:
            errors.append(
                {
                    "sample_id": row.get("sample_id"),
                    "file_path": str(file_path),
                    "error": repr(exc),
                }
            )

    sample_features = pd.DataFrame(sample_rows)
    well_results = pd.concat(well_rows, ignore_index=True) if well_rows else pd.DataFrame()

    report = {
        "n_labels": int(len(labels)),
        "n_curve_files": int(len(curves)),
        "n_expanded_sample_slots": int(len(expanded)),
        "n_matched_samples": int(len(matched)),
        "n_sample_feature_rows": int(len(sample_features)),
        "n_well_rows": int(len(well_results)),
        "n_errors": int(len(errors)),
        "sample_id_mode": sample_id_mode,
        "sample_layout": sample_layout,
        "wells_per_sample": int(wells_per_sample),
        "batch_start_override": _parse_override_map(batch_start_override),
        "numeric_parent_only": bool(numeric_parent_only),
        "curve_files_by_batch": curves["batch_name"].fillna("").value_counts().to_dict() if "batch_name" in curves.columns else {},
        "expanded_slots_by_batch": expanded["batch_name"].fillna("").value_counts().to_dict() if "batch_name" in expanded.columns else {},
        "curve_id_warning_count": int((curves.get("sample_id_warning", pd.Series(dtype=str)).fillna("") != "").sum()),
        "duplicated_expanded_sample_ids": int(expanded["sample_id"].duplicated().sum()) if "sample_id" in expanded.columns else 0,
        "label_counts": labels["label"].value_counts().to_dict(),
        "matched_label_counts": matched["label"].value_counts().to_dict() if "label" in matched.columns else {},
        "raw_label_top20": labels["raw_label"].value_counts().head(20).to_dict(),
        "unmatched_expanded_sample_slots": int(merged["label"].isna().sum()),
        "unmatched_labels": int(len(set(labels["sample_id"]) - set(expanded["sample_id"]))),
        "errors": errors[:50],
    }
    return sample_features, well_results, labels, expanded, report


def save_clinical_outputs(
    output_dir: str | Path,
    sample_features: pd.DataFrame,
    well_results: pd.DataFrame,
    labels: pd.DataFrame,
    curve_index: pd.DataFrame,
    report: dict,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample_features.to_csv(out / "clinical_sample_features.csv", index=False, encoding="utf-8-sig")
    well_results.to_csv(out / "clinical_well_rule_results.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(out / "clinical_label_index.csv", index=False, encoding="utf-8-sig")
    curve_index.to_csv(out / "clinical_curve_file_index.csv", index=False, encoding="utf-8-sig")
    with (out / "clinical_build_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
