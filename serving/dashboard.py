"""
Streamlit benchmarking dashboard.

Displays side-by-side comparison of all three model states
(base FP16, fine-tuned FP16, fine-tuned INT4) using results
from benchmarking/run_benchmark.py.

Run:
    streamlit run serving/dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS_DIR = Path("results")

st.set_page_config(page_title="ModelEdge Dashboard", layout="wide")


# ── helpers ──────────────────────────────────────────────────────────────────

def load_summary(results_dir: Path) -> dict:
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return _demo_data()


def _demo_data() -> dict:
    return {
        "base": {
            "p50_ms": 320, "p95_ms": 480, "vram_gb": 6.2,
            "accuracy": 0.42, "hallucination_rate": 0.38,
        },
        "fp16": {
            "p50_ms": 320, "p95_ms": 475, "vram_gb": 6.2,
            "accuracy": 0.71, "hallucination_rate": 0.12,
        },
        "int8": {
            "p50_ms": 210, "p95_ms": 310, "vram_gb": 3.8,
            "accuracy": 0.70, "hallucination_rate": 0.13,
        },
        "int4": {
            "p50_ms": 140, "p95_ms": 205, "vram_gb": 2.5,
            "accuracy": 0.68, "hallucination_rate": 0.14,
        },
    }


def results_to_df(data: dict) -> pd.DataFrame:
    rows = []
    for name, metrics in data.items():
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows)


# ── layout ───────────────────────────────────────────────────────────────────

st.title("ModelEdge — Optimization Benchmark Dashboard")
st.caption("Fine-tuning + quantization tradeoff analysis on Medical QA")

results = load_summary(RESULTS_DIR)
df = results_to_df(results)

tab_overview, tab_latency, tab_memory, tab_accuracy, tab_tradeoff = st.tabs(
    ["Overview", "Latency", "Memory", "Accuracy", "Tradeoff Curves"]
)

with tab_overview:
    st.subheader("Summary Table")
    display_cols = ["model", "p50_ms", "p95_ms", "vram_gb", "accuracy", "hallucination_rate"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available].set_index("model"), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    if "fp16" in results and "int4" in results:
        speedup = results["fp16"]["p50_ms"] / results["int4"]["p50_ms"]
        mem_reduction = 1 - results["int4"]["vram_gb"] / results["fp16"]["vram_gb"]
        acc_drop = results["fp16"]["accuracy"] - results["int4"]["accuracy"]
        hal_improvement = results.get("base", {}).get("hallucination_rate", 0) - results["fp16"]["hallucination_rate"]

        col1.metric("Latency speedup (FT INT4 vs FT FP16)", f"{speedup:.1f}×")
        col2.metric("VRAM reduction (INT4 vs FP16)", f"{mem_reduction*100:.0f}%")
        col3.metric("Accuracy drop (FP16 → INT4)", f"{acc_drop*100:.1f}pp")
        col4.metric("Hallucination ↓ (base → FT FP16)", f"{hal_improvement*100:.0f}pp")

with tab_latency:
    st.subheader("Latency — p50 and p95 (ms)")
    if "p50_ms" in df.columns and "p95_ms" in df.columns:
        lat_df = df[["model", "p50_ms", "p95_ms"]].melt(
            id_vars="model", var_name="percentile", value_name="latency_ms"
        )
        fig = px.bar(
            lat_df, x="model", y="latency_ms", color="percentile",
            barmode="group", labels={"latency_ms": "Latency (ms)"},
            color_discrete_map={"p50_ms": "#4C72B0", "p95_ms": "#DD8452"},
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_memory:
    st.subheader("VRAM Usage (GB)")
    if "vram_gb" in df.columns:
        fig = px.bar(
            df, x="model", y="vram_gb",
            labels={"vram_gb": "VRAM (GB)"},
            color="model", text_auto=".2f",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_accuracy:
    st.subheader("Accuracy vs Hallucination Rate")
    if "accuracy" in df.columns and "hallucination_rate" in df.columns:
        acc_df = df[["model", "accuracy", "hallucination_rate"]].melt(
            id_vars="model", var_name="metric", value_name="value"
        )
        fig = px.bar(
            acc_df, x="model", y="value", color="metric",
            barmode="group",
            color_discrete_map={"accuracy": "#2ca02c", "hallucination_rate": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_tradeoff:
    st.subheader("Tradeoff: Latency vs Accuracy")
    if "p50_ms" in df.columns and "accuracy" in df.columns:
        fig = px.scatter(
            df, x="p50_ms", y="accuracy", text="model",
            size="vram_gb", color="model",
            labels={"p50_ms": "p50 Latency (ms)", "accuracy": "Task Accuracy"},
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tradeoff: VRAM vs Accuracy")
    if "vram_gb" in df.columns and "accuracy" in df.columns:
        fig2 = px.scatter(
            df, x="vram_gb", y="accuracy", text="model",
            size="p50_ms", color="model",
            labels={"vram_gb": "VRAM (GB)", "accuracy": "Task Accuracy"},
        )
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("Run `python benchmarking/run_benchmark.py` to populate real results.")
