from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
from app.metrics import final_metric, fmt_pct, fmt_num

st.set_page_config(page_title="DriftX - Model Registry", page_icon="DX", layout="wide")

decision = str(final_metric("Streaming safety decision", "UNKNOWN"))
gain = float(final_metric("MAE improvement %", 0))
champion_mae = float(final_metric("Champion MAE", 0))
adapted_mae = float(final_metric("Adapted MAE", 0))
champion_rmse = float(final_metric("Champion RMSE", 0))
adapted_rmse = float(final_metric("Adapted RMSE", 0))
champion_r2 = float(final_metric("Champion R2", 0))
adapted_r2 = float(final_metric("Adapted R2", 0))

st.title("Model Registry")
st.caption("Champion model, challenger evaluation, and promotion governance")

st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)

with c1:
    st.markdown('<div class="card"><div class="label">Champion</div><div class="value">ACTIVE</div><div class="sub">Registry champion remains production reference</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown(f'<div class="card"><div class="label">Challenger Result</div><div class="value">{decision}</div><div class="sub">Aggregate streaming safety decision</div></div>', unsafe_allow_html=True)

with c3:
    st.markdown(f'<div class="card"><div class="label">MAE Improvement</div><div class="value">{gain:.2f}%</div><div class="sub">Aggregate adaptation improvement</div></div>', unsafe_allow_html=True)

st.subheader("Model Validation")

a,b,c,d = st.columns(4)
a.metric("Champion MAE", f"{champion_mae:.4f}")
b.metric("Adapted MAE", f"{adapted_mae:.4f}")
c.metric("Champion RMSE", f"{champion_rmse:.4f}")
d.metric("Adapted RMSE", f"{adapted_rmse:.4f}")

a,b = st.columns(2)
with a:
    st.metric("Champion R²", f"{champion_r2:.4f}")
with b:
    st.metric("Adapted R²", f"{adapted_r2:.4f}", f"{adapted_r2-champion_r2:+.4f}")

st.subheader("Promotion Governance")

if decision.upper() == "PROMOTE":
    st.success(
        f"PROMOTE: aggregate MAE improved by {gain:.2f}%. "
        "The recorded streaming safety decision is PROMOTE."
    )
else:
    st.warning(f"Safety decision: {decision}")

st.info(
    "The registry page uses the aggregate FINAL_RESULTS evaluation. "
    "Individual engine-level results are intentionally kept in Experiment Lab "
    "and are not treated as the global promotion decision."
)
