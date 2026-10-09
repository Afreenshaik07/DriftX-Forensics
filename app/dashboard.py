from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from app.metrics import final_metric, fmt_pct, fmt_num

st.set_page_config(page_title="DriftX - Command Center", page_icon="DX", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.metric-card {
    background: linear-gradient(145deg,#101a2d,#16233b);
    border: 1px solid #263957;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 12px;
}
.metric-label {color:#8fa6c6;font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:1px;}
.metric-value {color:#ffffff;font-size:30px;font-weight:750;margin-top:5px;}
.metric-sub {color:#6f87a8;font-size:12px;margin-top:4px;}
.section {
    color:#ffffff;
    font-size:20px;
    font-weight:700;
    margin-top:25px;
    margin-bottom:12px;
}
</style>
""", unsafe_allow_html=True)

forensic = final_metric("Forensic accuracy", 0)
baseline = final_metric("Temporal baseline accuracy", 0)
confidence = final_metric("Mean forensic confidence", 0)
engines = final_metric("Streaming test engines", 0)
champion_mae = final_metric("Champion MAE", 0)
adapted_mae = final_metric("Adapted MAE", 0)
gain = float(final_metric("MAE improvement %", 0))
decision = final_metric("Streaming safety decision", "UNKNOWN")

st.title("DriftX-Forensics")
st.caption("Cross-Stage Drift Diagnosis and Safety-Gated Autonomous Adaptation")

st.markdown("### System Status")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-label">Forensic Accuracy</div>
    <div class="metric-value">{fmt_pct(forensic)}</div>
    <div class="metric-sub">vs {fmt_pct(baseline)} temporal baseline</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-label">Forensic Confidence</div>
    <div class="metric-value">{fmt_pct(confidence)}</div>
    <div class="metric-sub">Mean diagnostic confidence</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-label">Adaptation Gain</div>
    <div class="metric-value">{gain:.2f}%</div>
    <div class="metric-sub">Aggregate streaming MAE improvement</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-label">Safety Decision</div>
    <div class="metric-value">{decision}</div>
    <div class="metric-sub">{int(float(engines))} streaming test engines</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section">Self-Healing Pipeline</div>', unsafe_allow_html=True)

steps = [
    ("01", "DATA", "Incoming data"),
    ("02", "DRIFT", "Multi-stage detection"),
    ("03", "FORENSICS", "Origin diagnosis"),
    ("04", "IMPACT", "Risk assessment"),
    ("05", "POLICY", "Monitor / Update / Retrain"),
    ("06", "CHALLENGER", "Candidate model"),
    ("07", "SAFETY GATE", "Promotion decision"),
    ("08", "CHAMPION", "Production model"),
]

cols = st.columns(8)
for col, (num, title, sub) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div class="metric-card" style="min-height:125px;text-align:center;">
        <div class="metric-label">{num}</div>
        <div style="color:#ffffff;font-weight:700;margin-top:8px;">{title}</div>
        <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section">Model Performance</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Champion MAE", fmt_num(champion_mae, 4))

with m2:
    st.metric("Adapted MAE", fmt_num(adapted_mae, 4))

with m3:
    st.metric("MAE Improvement", f"{gain:.2f}%")

st.info(
    "DriftX evaluates changing data, reconstructs the failure path across pipeline stages, "
    "selects an adaptation action, evaluates a challenger, and applies a safety-gated model decision."
)
