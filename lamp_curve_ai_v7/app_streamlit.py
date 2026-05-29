#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import io
import json
import sys
import tempfile

import joblib
import pandas as pd
import streamlit as st
import altair as alt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lamp_ai.clinical import build_clinical_dataset, parse_clinical_summary_excel, save_clinical_outputs
from lamp_ai.config import CurveConfig, RuleConfig
from lamp_ai.data_io import read_lamp_file
from lamp_ai.ml_model import train_model
from lamp_ai.rules import rule_interpret_dataframe, smooth_dataframe
from lamp_ai.v7_ai import load_bundle, predict_curves_dataframe, display_label


st.set_page_config(page_title="LAMP 曲线判读 V7", page_icon="🧬", layout="wide")

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 1280px;
}
h1, h2, h3 {
    letter-spacing: 0;
}
.main-title {
    display: block;
    overflow: visible;
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", Arial, sans-serif;
    font-size: clamp(1.85rem, 3.4vw, 2.35rem);
    line-height: 1.35;
    font-weight: 750;
    color: #172033;
    margin: 0 0 0.25rem 0;
    padding: 0.1rem 0 0.18rem 0;
    white-space: normal;
    text-rendering: geometricPrecision;
}
.sub-title {
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif;
    color: #667085;
    font-size: 0.98rem;
    line-height: 1.7;
    margin: 0.15rem 0 1.25rem 0;
}
.section-card {
    background: #ffffff;
    border: 1px solid #e8edf5;
    border-radius: 16px;
    padding: 1.05rem 1.15rem;
    margin: 0.65rem 0 1rem 0;
    box-shadow: 0 6px 20px rgba(30, 41, 59, 0.05);
}
.soft-panel {
    background: #f8fafc;
    border: 1px solid #edf2f7;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin: 0.7rem 0;
}
.metric-label {
    color: #667085;
    font-size: 0.82rem;
}
.stButton > button {
    border-radius: 10px;
    font-weight: 650;
}
div[data-testid="stSidebar"] {
    background: #f8fafc;
}
div[data-testid="stFileUploader"] {
    border-radius: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-title">LAMP 荧光扩增曲线判读系统</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">规则判读｜AI 判读｜批量建库｜模型训练</div>', unsafe_allow_html=True)

mode = st.sidebar.radio(
    "工作模式",
    ["规则判读", "AI判读", "临床批量建库/训练", "V7伪标签/AI训练", "说明"],
)
drop_first_rows = st.sidebar.number_input("读取时去除前 N 行", min_value=0, max_value=100, value=5, step=1)
smooth_window = st.sidebar.number_input("平滑窗口", min_value=1, max_value=21, value=5, step=2)
drop_time_column = st.sidebar.checkbox("第一列为时间/序号", value=False)
curve_cfg = CurveConfig(drop_first_rows=int(drop_first_rows), smooth_window=int(smooth_window))

RULE_CN = {
    "positive": "阳性",
    "negative": "阴性",
    "abnormal": "异常",
    "uncertain": "需复核",
    "review": "需复核",
}


def page_title(title: str, desc: str | None = None):
    st.markdown(f"## {title}")
    if desc:
        st.markdown(f'<div class="soft-panel">{desc}</div>', unsafe_allow_html=True)


def show_df(df: pd.DataFrame, height: int | None = None):
    kwargs = {"width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    try:
        st.dataframe(df, **kwargs)
    except TypeError:
        fallback_kwargs = {"use_container_width": True}
        if height is not None:
            fallback_kwargs["height"] = height
        st.dataframe(df, **fallback_kwargs)


def add_rule_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "rule_label" in out.columns:
        out.insert(3, "规则判读", out["rule_label"].map(RULE_CN).fillna(out["rule_label"].astype(str)))
    return out


def compact_rule_table(result: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c for c in [
            "well", "well_index", "规则判读", "rule_confidence",
            "positive_score", "negative_score", "abnormal_score", "reason"
        ] if c in result.columns
    ]
    return result[cols].copy()


def compact_ai_table(result: pd.DataFrame) -> pd.DataFrame:
    out = result.copy()
    if "ai_result" not in out.columns and "ai_label" in out.columns:
        out["ai_result"] = out["ai_label"].map(lambda x: display_label(x))
    cols = [
        c for c in [
            "well", "well_index", "ai_result", "ai_confidence",
            "prob_positive", "prob_negative", "prob_abnormal", "prob_review"
        ] if c in out.columns
    ]
    return out[cols].copy()


def show_curves(curves: pd.DataFrame):
    st.markdown("#### 曲线预览")
    show_smooth = st.checkbox("显示平滑曲线", value=True)
    plot_df = smooth_dataframe(curves, curve_cfg) if show_smooth else curves
    st.line_chart(plot_df)

    wells = list(curves.columns)
    if wells:
        selected = st.selectbox("单孔查看", wells)
        single_df = pd.DataFrame({
            "原始曲线": curves[selected],
            "平滑曲线": smooth_dataframe(curves[[selected]], curve_cfg)[selected],
        })
        st.line_chart(single_df)


def download_df_button(df: pd.DataFrame, name: str):
    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("下载 CSV", csv_bytes, file_name=name, mime="text/csv")


def load_model_from_upload_or_path(uploaded, local_path: str):
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".joblib"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        return load_bundle(tmp_path)
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"模型文件不存在：{p}")
    return load_bundle(p)


if mode == "规则判读":
    page_title("规则判读")
    file = st.file_uploader("上传 LAMP CSV/XLSX", type=["csv", "xlsx", "xls"], key="rule_file")
    if file is not None:
        curves = read_lamp_file(file, drop_first_rows=int(drop_first_rows), drop_time_column=drop_time_column)
        result = rule_interpret_dataframe(curves, curve_config=curve_cfg, rule_config=RuleConfig())
        result_show = add_rule_display(result)

        show_curves(curves)

        st.markdown("#### 判读结果")
        show_df(compact_rule_table(result_show))
        with st.expander("完整特征表"):
            show_df(result_show)
        download_df_button(result_show, "v7_rule_results.csv")

elif mode == "AI判读":
    page_title("AI判读")
    curve_file = st.file_uploader("上传待判读 LAMP CSV/XLSX", type=["csv", "xlsx", "xls"], key="ai_curve")
    model_file = st.file_uploader("上传模型 .joblib（可选）", type=["joblib", "pkl"], key="ai_model")
    local_model = st.text_input("本地模型路径", value="models/lamp_v7_well_ai.joblib")

    if curve_file is not None:
        curves = read_lamp_file(curve_file, drop_first_rows=int(drop_first_rows), drop_time_column=drop_time_column)
        show_curves(curves)

        if st.button("开始 AI 判读", type="primary"):
            try:
                bundle = load_model_from_upload_or_path(model_file, local_model)
                pred = predict_curves_dataframe(
                    curves,
                    bundle,
                    curve_config=CurveConfig(drop_first_rows=0, smooth_window=int(smooth_window)),
                )
                st.markdown("#### AI 判读结果")
                show_df(compact_ai_table(pred))
                with st.expander("完整特征表"):
                    show_df(pred)
                download_df_button(pred, "v7_ai_results.csv")
            except Exception as e:
                st.error(str(e))
                st.caption("请确认模型路径正确，并使用 V7 模型。")

elif mode == "临床批量建库/训练":
    page_title("临床批量建库 / 样本级训练")
    root_dir = st.text_input("临床数据根目录", value="..\\medicaldata")
    summary_path = st.text_input("结果汇总表路径", value="..\\medicaldata\\结果汇总表.xlsx")
    out_dir = st.text_input("输出目录", value="outputs/clinical_v7")

    c1, c2, c3 = st.columns(3)
    with c1:
        label_mode = st.selectbox("样本标签模式", ["binary", "multiclass"], index=0)
        sample_id_mode = st.selectbox("样本编号模式", ["two_per_file", "batch_offset", "local"], index=0)
    with c2:
        sample_layout = st.selectbox("样本布局", ["two_per_file_15x2", "one_per_file"], index=0)
        wells_per_sample = st.number_input("每个样本孔数", min_value=1, max_value=96, value=15, step=1)
    with c3:
        max_files = st.number_input("最多读取文件数（0 为全部）", min_value=0, value=0, step=10)
        include_well_features = st.checkbox("加入固定孔位特征", value=True)

    batch_start_override = st.text_input("批次起始编号", value="结果1-1600=1;结果3857-5444=1601")
    numeric_parent_only = st.checkbox("只读取数字范围父目录", value=True)
    run_train = st.checkbox("建库后训练样本级模型", value=False)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("解析汇总表标签"):
            labels = parse_clinical_summary_excel(summary_path, label_mode=label_mode)
            st.success(f"解析到 {len(labels)} 条样本标签")
            st.write(labels["label"].value_counts())
            show_df(labels.head(200))
    with col_b:
        if st.button("开始批量建库", type="primary"):
            sample_features, well_results, labels, curve_index, report = build_clinical_dataset(
                root_dir=root_dir,
                summary_path=summary_path,
                label_mode=label_mode,
                curve_config=curve_cfg,
                drop_time_column=drop_time_column,
                include_well_features=include_well_features,
                max_files=None if int(max_files) == 0 else int(max_files),
                sample_id_mode=sample_id_mode,
                sample_layout=sample_layout,
                wells_per_sample=int(wells_per_sample),
                batch_start_override=batch_start_override,
                numeric_parent_only=numeric_parent_only,
            )
            save_clinical_outputs(out_dir, sample_features, well_results, labels, curve_index, report)
            st.success(f"建库完成：{out_dir}")
            st.json(report)

            st.markdown("#### 样本级特征")
            show_df(sample_features.head(200))
            st.markdown("#### 孔位级规则判读")
            show_df(add_rule_display(well_results).head(200))

            if run_train:
                bundle, metrics = train_model(sample_features)
                model_buffer = io.BytesIO()
                joblib.dump(bundle, model_buffer)
                model_buffer.seek(0)
                st.success("样本级模型训练完成")
                st.json(metrics)
                st.download_button("下载样本级模型", model_buffer.getvalue(), file_name="lamp_sample_model.joblib")

elif mode == "V7伪标签/AI训练":
    page_title("V7 伪标签 / AI 学生模型训练")
    st.markdown("#### 推荐命令")
    st.code(
        "python build_v7_pseudo_labels.py --well_results outputs\\clinical_v7\\clinical_well_rule_results.csv --out_dir outputs\\clinical_v7\\v7_ai\n"
        "python train_v7_well_ai.py --pseudo outputs\\clinical_v7\\v7_ai\\v7_well_pseudo_labels.csv --model models\\lamp_v7_well_ai.joblib --metrics outputs\\clinical_v7\\v7_ai\\v7_well_ai_metrics.json --scores outputs\\clinical_v7\\v7_ai\\v7_well_ai_scores.csv",
        language="powershell",
    )
    metrics_path = st.text_input("训练结果路径", value="outputs/clinical_v7/v7_ai/v7_well_ai_metrics.json")
    if st.button("读取训练结果"):
        p = Path(metrics_path)
        if p.exists():
            st.json(json.loads(p.read_text(encoding="utf-8")))
        else:
            st.warning("未找到 metrics 文件。")

else:
    page_title("说明")
    st.markdown(
        """
<div class="section-card">
<b>当前版本：</b>V7<br>
<b>主要页面：</b>规则判读、AI 判读、临床批量建库、V7 训练。<br>
<b>推荐模型：</b><code>models/lamp_v7_well_ai.joblib</code>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("#### 常用命令")
    st.code(
        'python build_clinical_dataset.py --root "..\\medicaldata" --summary "..\\medicaldata\\结果汇总表.xlsx" --out_dir outputs\\clinical_v7 --label_mode binary --sample_id_mode two_per_file --sample_layout two_per_file_15x2 --batch_start_override "结果1-1600=1;结果3857-5444=1601" --numeric_parent_only\n'
        'python build_v7_pseudo_labels.py --well_results outputs\\clinical_v7\\clinical_well_rule_results.csv --out_dir outputs\\clinical_v7\\v7_ai\n'
        'python train_v7_well_ai.py --pseudo outputs\\clinical_v7\\v7_ai\\v7_well_pseudo_labels.csv --model models\\lamp_v7_well_ai.joblib --metrics outputs\\clinical_v7\\v7_ai\\v7_well_ai_metrics.json --scores outputs\\clinical_v7\\v7_ai\\v7_well_ai_scores.csv\n'
        'streamlit run app_streamlit.py',
        language="powershell",
    )
