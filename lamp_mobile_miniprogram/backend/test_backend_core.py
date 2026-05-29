"""不用启动服务器，快速测试核心判读是否能跑通。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lamp_ai.config import CurveConfig, RuleConfig
from lamp_ai.data_io import read_lamp_file
from lamp_ai.rules import rule_interpret_dataframe

curves = read_lamp_file(ROOT / "examples" / "314pc.csv", drop_first_rows=5)
result = rule_interpret_dataframe(curves, curve_config=CurveConfig(smooth_window=5), rule_config=RuleConfig())
print(result[["well", "rule_label", "rule_confidence", "positive_score", "negative_score"]].head())
print("OK", curves.shape, result.shape)
