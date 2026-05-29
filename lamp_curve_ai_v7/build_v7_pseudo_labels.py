#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lamp_ai.v7_ai import normalize_teacher_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V7 AI pseudo labels from V4 rule results without changing rule logic")
    parser.add_argument("--well_results", required=True, help="clinical_well_rule_results.csv or any rule result CSV")
    parser.add_argument("--out_dir", default="outputs/clinical_v7/v7_ai")
    parser.add_argument("--min_confidence", type=float, default=0.70, help="minimum rule_confidence to train AI student")
    parser.add_argument("--include_review", action="store_true", help="also train on uncertain/review pseudo labels; default excludes review")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.well_results, encoding="utf-8-sig", low_memory=False)
    if "rule_label" not in df.columns:
        raise ValueError("Input CSV must contain rule_label from V4 rule判读")

    df = df.copy()
    df["v7_teacher_label"] = df["rule_label"].map(normalize_teacher_label)
    df["v7_train_label"] = df["v7_teacher_label"]

    conf = pd.to_numeric(df.get("rule_confidence", 1.0), errors="coerce").fillna(0.0)
    allowed = {"positive", "negative", "abnormal"}
    if args.include_review:
        allowed.add("review")

    df["use_for_training"] = df["v7_train_label"].isin(allowed) & (conf >= float(args.min_confidence))
    df.loc[~df["use_for_training"], "v7_train_label"] = "exclude"
    df["v7_result_cn"] = df["v7_teacher_label"].map({"positive":"阳性","negative":"阴性","abnormal":"异常","review":"需复核"}).fillna("需复核")

    pseudo_path = out_dir / "v7_well_pseudo_labels.csv"
    summary_path = out_dir / "v7_pseudo_summary.json"
    df.to_csv(pseudo_path, index=False, encoding="utf-8-sig")

    summary = {
        "input": str(args.well_results),
        "n_well_rows": int(len(df)),
        "min_confidence": float(args.min_confidence),
        "teacher_label_counts": {str(k): int(v) for k, v in df["v7_teacher_label"].value_counts().to_dict().items()},
        "train_label_counts": {str(k): int(v) for k, v in df["v7_train_label"].value_counts().to_dict().items()},
        "use_for_training_counts": {str(k): int(v) for k, v in df["use_for_training"].value_counts().to_dict().items()},
        "outputs": {"pseudo_labels": str(pseudo_path), "summary": str(summary_path)},
        "important_warning": "V7 pseudo labels are produced only from the unchanged V4 rule判读. They are not manual well-level truth.",
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
