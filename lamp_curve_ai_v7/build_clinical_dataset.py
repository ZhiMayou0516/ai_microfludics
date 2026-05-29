#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lamp_ai.clinical import build_clinical_dataset, save_clinical_outputs, parse_clinical_summary_excel
from lamp_ai.config import CurveConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sample-level and well-level datasets from clinical LAMP folders")
    parser.add_argument("--root", required=True, help="clinical data root, e.g. Desktop/ai-test/medicaldata")
    parser.add_argument("--summary", required=True, help="结果汇总表.xlsx")
    parser.add_argument("--out_dir", default="outputs/clinical", help="output directory")
    parser.add_argument("--label_mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--drop_first_rows", type=int, default=5)
    parser.add_argument("--drop_time_column", action="store_true", help="drop the first numeric column if it is time/index")
    parser.add_argument("--sample_id_mode", choices=["batch_offset", "local", "two_per_file"], default="batch_offset", help="how to map 检测结果(n) to summary sample_id")
    parser.add_argument("--sample_layout", choices=["one_per_file", "two_per_file_15x2"], default="one_per_file", help="whether one curve file contains one sample or two 15-well samples")
    parser.add_argument("--wells_per_sample", type=int, default=15, help="used by two_per_file_15x2; default: first 15 wells and last 15 wells")
    parser.add_argument("--batch_start_override", default=None, help="manual batch start override, e.g. 结果3857-5444=1601;结果1-1600=1")
    parser.add_argument("--numeric_parent_only", action="store_true", help="skip ad-hoc folders whose parent dir is not a numeric range such as 1-9 or 1000-2030")
    parser.add_argument("--no_well_features", action="store_true", help="only use aggregate features, do not flatten per-well features")
    parser.add_argument("--max_files", type=int, default=None, help="debug only: limit number of files")
    parser.add_argument("--labels_only", action="store_true", help="only parse the summary Excel and save clinical_label_index.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.labels_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        labels = parse_clinical_summary_excel(args.summary, label_mode=args.label_mode)
        labels.to_csv(out_dir / "clinical_label_index.csv", index=False, encoding="utf-8-sig")
        print(f"Parsed labels: {len(labels)}")
        print(labels["label"].value_counts().to_string())
        print(f"Saved to: {out_dir / 'clinical_label_index.csv'}")
        return

    sample_features, well_results, labels, curve_index, report = build_clinical_dataset(
        root_dir=args.root,
        summary_path=args.summary,
        label_mode=args.label_mode,
        curve_config=CurveConfig(drop_first_rows=args.drop_first_rows),
        drop_time_column=args.drop_time_column,
        include_well_features=not args.no_well_features,
        max_files=args.max_files,
        sample_id_mode=args.sample_id_mode,
        sample_layout=args.sample_layout,
        wells_per_sample=args.wells_per_sample,
        batch_start_override=args.batch_start_override,
        numeric_parent_only=args.numeric_parent_only,
    )
    save_clinical_outputs(out_dir, sample_features, well_results, labels, curve_index, report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:5000])
    print(f"\nSaved clinical outputs to: {out_dir}")


if __name__ == "__main__":
    main()
