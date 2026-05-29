#!/usr/bin/env python
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lamp_ai.config import CurveConfig, RuleConfig
from lamp_ai.data_io import read_lamp_file, save_result_csv
from lamp_ai.rules import rule_interpret_dataframe
from lamp_ai.plot_utils import plot_all_curves


def main():
    parser = argparse.ArgumentParser(description="Rule-based LAMP curve interpretation")
    parser.add_argument("--file", "--csv", dest="file", required=True, help="pure numeric LAMP CSV/XLSX/XLS")
    parser.add_argument("--out", default="outputs/rule_results.csv", help="output result CSV")
    parser.add_argument("--drop_first_rows", type=int, default=5)
    parser.add_argument("--drop_time_column", action="store_true", help="drop first numeric column if it is time/index")
    parser.add_argument("--plot", default=None, help="optional output PNG for all curves")
    args = parser.parse_args()

    curves = read_lamp_file(args.file, drop_first_rows=args.drop_first_rows, drop_time_column=args.drop_time_column)
    result = rule_interpret_dataframe(curves, curve_config=CurveConfig(drop_first_rows=args.drop_first_rows), rule_config=RuleConfig())
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_result_csv(result, out_path)

    if args.plot:
        plot_path = ROOT / args.plot if not Path(args.plot).is_absolute() else Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_all_curves(curves, output_path=plot_path, smooth=True)

    print(f"Done. Results saved to: {out_path}")
    print(result[["well", "rule_label", "rule_confidence", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
