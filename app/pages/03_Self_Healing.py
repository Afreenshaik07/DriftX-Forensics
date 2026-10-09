from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from app.metrics import final_metric, fmt_pct, fmt_num

st.set_page_config(page_title="DriftX - Self-Healing", page_icon="DX", layout="wide")

decision = str(final_metric("Streaming safety decision", "UNKNOWN"))
gain = float(final_metric("MAE improvement %", 0))
champion_mae = float(final_metric("Champion MAE", 0))
adapted_mae = float(final_metric("Adapted MAE", 0))
champion_rmse = float(final_metric("Champion RMSE", 0))
adapted_rmse = float(final_metric("Adapted RMSE", 0))
champion_r2 = float(final_metric("Champion R2", 0))
adapted_r2 = float(final_metric("Adapted R2", 0))

adapt = pd.read_csv(ROOT / "results" / "autonomous_adaptation_experiment.csv")

st.markdown("""
<style>
.block-container {padding-top:2rem;}
.card {
background:linear-gradient(145deg,#101a2d,#16233b);
border:1px solid #263957;
border-radius:14px;
padding:20px;
margin-bottom:14px;
}
.label {color:#8fa6c6;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;}
.value {color:#fff;font-size:28px;font-weight:750;margin-top:5px;}
.sub {color:#7189aa;font-size:12px;margin-top:5px;}
.section {font-size:20px;font-weight:700;color:#fff;margin:25px 0 12px;}
</style>
""", unsafe_allow_html=True)

st.title("Self-Healing Control Center")
st.caption("Autonomous adaptation policy, challenger evaluation, and safety-gated promotion")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.markdown(f'<div class="card"><div class="label">System Decision</div><div class="value">{decision}</div><div class="sub">Aggregate streaming evaluation</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="card"><div class="label">MAE Improvement</div><div class="value">{gain:.2f}%</div><div class="sub">Champion to adapted model</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="card"><div class="label">Champion MAE</div><div class="value">{champion_mae:.4f}</div><div class="sub">Production baseline</div></div>', unsafe_allow_html=True)

with c4:
    st.markdown(f'<div class="card"><div class="label">Adapted MAE</div><div class="value">{adapted_mae:.4f}</div><div class="sub">Post-adaptation aggregate</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section">Self-Healing Decision Loop</div>', unsafe_allow_html=True)

steps = [
    ("1","Detect","Monitor incoming behaviour"),
    ("2","Diagnose","Identify drift origin"),
    ("3","Assess","Estimate risk and impact"),
    ("4","Adapt","Monitor / Update / Retrain"),
    ("5","Challenge","Evaluate candidate model"),
    ("6","Gate","Apply safety criteria"),
    ("7","Promote","Update champion only if safe"),
]

cols = st.columns(7)
for col,(n,title,sub) in zip(cols,steps):
    with col:
        st.markdown(f'<div class="card" style="min-height:125px;text-align:center;"><div class="label">{n}</div><div style="color:#fff;font-weight:700;margin-top:8px;">{title}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section">Aggregate Model Validation</div>', unsafe_allow_html=True)

m1,m2,m3 = st.columns(3)
with m1:
    st.metric("RMSE", f"{champion_rmse:.4f}", f"{adapted_rmse-champion_rmse:+.4f}")
with m2:
    st.metric("R²", f"{champion_r2:.4f}", f"{adapted_r2-champion_r2:+.4f}")
with m3:
    st.metric("MAE", f"{adapted_mae:.4f}", f"{-gain:.2f}%")

st.markdown('<div class="section">Adaptation Policy Decisions</div>', unsafe_allow_html=True)

display = adapt[
    ["scenario","predicted_origin","confidence","action","severity","risk_score","previous_regime"]
].copy()

display.columns = [
    "Scenario","Origin","Confidence","Action",
    "Severity","Risk Score","Previous Regime"
]

display["Confidence"] = display["Confidence"].map(lambda x: f"{float(x):.3f}")
display["Risk Score"] = display["Risk Score"].map(lambda x: f"{float(x):.3f}")

st.dataframe(display, width="stretch", hide_index=True)

st.markdown('<div class="section">Safety Gate</div>', unsafe_allow_html=True)

if decision.upper() == "PROMOTE":
    st.success(
        "PROMOTE - The aggregate adapted model improves MAE over the champion "
        "and the recorded streaming safety decision authorizes promotion."
    )
else:
    st.warning(f"{decision} - No aggregate promotion should be inferred from individual engine results.")

st.info(
    "Important: individual engines can improve or degrade independently. "
    "The system-level decision shown here is based on the aggregate streaming evaluation, "
    "not on a single engine."
)
