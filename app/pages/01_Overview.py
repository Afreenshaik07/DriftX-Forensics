from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from app.metrics import final_metric

st.set_page_config(page_title="DriftX - Overview", page_icon="DX", layout="wide")

def num(name, default=0.0):
    try:
        return float(final_metric(name, default))
    except:
        return default

forensic = num("Forensic accuracy")
baseline = num("Temporal baseline accuracy")
confidence = num("Mean forensic confidence")
engines = num("Streaming test engines")
champion_mae = num("Champion MAE")
adapted_mae = num("Adapted MAE")
gain = num("MAE improvement %")
decision = str(final_metric("Streaming safety decision", "UNKNOWN"))

adapt_path = ROOT / "results" / "autonomous_adaptation_experiment.csv"
adapt = pd.read_csv(adapt_path) if adapt_path.exists() else pd.DataFrame()

st.markdown("""
<style>
.block-container {padding-top:2rem;}
.card {
    background:linear-gradient(145deg,#101a2d,#16233b);
    border:1px solid #263957;
    border-radius:14px;
    padding:18px;
}
.label {
    color:#8fa6c6;
    font-size:12px;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:1px;
}
.value {
    color:#ffffff;
    font-size:28px;
    font-weight:750;
    margin-top:6px;
}
.sub {
    color:#7189aa;
    font-size:12px;
    margin-top:5px;
}
.section {
    color:#ffffff;
    font-size:20px;
    font-weight:700;
    margin-top:28px;
    margin-bottom:12px;
}
</style>
""", unsafe_allow_html=True)

st.title("DriftX-Forensics")
st.caption("Cross-stage drift diagnosis, autonomous adaptation and safety-gated model lifecycle.")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
    <div class="label">Forensic Accuracy</div>
    <div class="value">{forensic*100:.2f}%</div>
    <div class="sub">Cross-stage diagnosis</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
    <div class="label">Adaptation Improvement</div>
    <div class="value">{gain:.2f}%</div>
    <div class="sub">Champion -> adapted model</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
    <div class="label">Test Engines</div>
    <div class="value">{int(engines)}</div>
    <div class="sub">Streaming evaluation</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
    <div class="label">Safety Gate</div>
    <div class="value">{decision}</div>
    <div class="sub">Model promotion control</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section">DriftX Decision Pipeline</div>', unsafe_allow_html=True)

pipeline = [
    ("01","Data","Quality + feature drift"),
    ("02","Prediction","Output distribution"),
    ("03","Residual","Error behaviour"),
    ("04","Forensics","Origin diagnosis"),
    ("05","Adapt","Monitor / Update / Retrain"),
    ("06","Safety Gate","Promote / Reject"),
]

cols = st.columns(6)

for col,(n,title,sub) in zip(cols,pipeline):
    with col:
        st.markdown(f"""
        <div class="card" style="min-height:125px;text-align:center;">
        <div class="label">{n}</div>
        <div style="color:#fff;font-weight:700;margin-top:8px;">{title}</div>
        <div class="sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section">System State</div>', unsafe_allow_html=True)

a,b,c = st.columns(3)

with a:
    st.metric("Champion", "ACTIVE")

with b:
    st.metric("Registry", "READY")

with c:
    st.metric("Regimes", len(__import__("json").loads(
        (ROOT / "artifacts" / "regime_memory.json").read_text(encoding="utf-8")
    )) if (ROOT / "artifacts" / "regime_memory.json").exists() else 0)

st.markdown('<div class="section">Adaptation Performance</div>', unsafe_allow_html=True)

a,b,c,d = st.columns(4)

a.metric("Champion MAE", f"{champion_mae:.4f}")
b.metric("Adapted MAE", f"{adapted_mae:.4f}")
c.metric("Forensic Accuracy", f"{forensic*100:.2f}%")
d.metric("vs Baseline", f"{baseline*100:.2f}%")

st.markdown('<div class="section">Autonomous Adaptation Actions</div>', unsafe_allow_html=True)

if not adapt.empty and "action" in adapt.columns:
    counts = adapt["action"].astype(str).str.upper().value_counts()

    a,b,c = st.columns(3)

    with a:
        st.metric("MONITOR", int(counts.get("MONITOR",0)))

    with b:
        st.metric("UPDATE", int(counts.get("UPDATE",0)))

    with c:
        st.metric("RETRAIN", int(counts.get("RETRAIN",0)))

    display = adapt[
        ["scenario","predicted_origin","confidence","action","severity","risk_score"]
    ].copy()

    display.columns = [
        "Scenario","Origin","Confidence","Action","Severity","Risk Score"
    ]

    display["Confidence"] = display["Confidence"].map(lambda x: f"{float(x):.3f}")
    display["Risk Score"] = display["Risk Score"].map(lambda x: f"{float(x):.3f}")

    st.dataframe(display, width="stretch", hide_index=True)

st.success(
    f"DriftX is operational. Forensic accuracy is {forensic*100:.2f}%, "
    f"adaptation improvement is {gain:.2f}%, and the aggregate safety decision is {decision}."
)
