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

st.markdown("""
<style>
:root {
    --dx-bg: #0b1120;
    --dx-panel: #111c30;
    --dx-border: #263650;
    --dx-text: #e8eef9;
    --dx-muted: #93a6c5;
    --dx-blue: #6ba5ff;
}
.stApp {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(39,91,163,.15), transparent 38%),
        #0b1120;
    color: var(--dx-text);
}
.block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
h1 {
    font-size: 2.35rem !important;
    font-weight: 800 !important;
    letter-spacing: -1.2px;
    color: #f4f7ff !important;
}
h2, h3 {
    color: #eaf1ff !important;
    letter-spacing: -.35px;
}
p, label, .stCaption {
    color: var(--dx-muted);
}
[data-testid="stSidebar"] {
    background: #0e1728;
    border-right: 1px solid var(--dx-border);
}
[data-testid="stSidebar"] * {
    color: #dbe7fa;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #14223a, #101a2c);
    border: 1px solid var(--dx-border);
    border-radius: 14px;
    padding: 16px 18px;
}
[data-testid="stMetricLabel"] {
    color: #9bb0d1 !important;
    font-size: .83rem !important;
}
[data-testid="stMetricValue"] {
    color: #f4f7ff !important;
    font-weight: 750;
}
.metric-card {
    background: linear-gradient(145deg, #14223a, #101a2c);
    border: 1px solid #2a3d5c;
    border-radius: 15px;
    padding: 21px;
    margin-bottom: 12px;
    min-height: 125px;
    box-shadow: 0 8px 24px rgba(0,0,0,.12);
    transition: border-color .2s ease, transform .2s ease;
}
.metric-card:hover {
    border-color: #548fe5;
    transform: translateY(-2px);
}
.metric-label {
    color: #9bb0d1;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.1px;
}
.metric-value {
    color: #f5f8ff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 9px;
    overflow-wrap: anywhere;
}
.metric-sub {
    color: #8da2c2;
    font-size: 12px;
    margin-top: 7px;
    line-height: 1.5;
}
.section {
    color: #edf3ff;
    font-size: 21px;
    font-weight: 750;
    margin-top: 30px;
    margin-bottom: 15px;
    padding-bottom: 9px;
    border-bottom: 1px solid #263650;
}
div[data-testid="stForm"],
div[data-testid="stExpander"] {
    background: rgba(17,28,48,.65);
    border: 1px solid var(--dx-border);
    border-radius: 13px;
}
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 42px;
    transition: all .2s ease;
}
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(100deg, #3479e8, #5b91f2);
    border: 1px solid #6ba5ff;
    color: white;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #6ba5ff;
}
input, textarea {
    border-radius: 8px !important;
}
div[data-baseweb="select"] > div {
    border-radius: 8px;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
hr {
    border-color: #263650;
}
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }
    h1 {
        font-size: 1.8rem !important;
    }
    .metric-card {
        padding: 15px;
    }
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

st.markdown("""
<div style="background:linear-gradient(115deg,#14294a,#17233b,#202b49);
border:1px solid #5484bd;border-radius:18px;padding:28px 30px;
margin-bottom:22px;box-shadow:0 12px 32px rgba(0,0,0,.24)">
<div style="color:#8dbbff;font-size:11px;font-weight:800;
letter-spacing:2px;text-transform:uppercase">MLOps Intelligence Platform</div>
<div style="font-size:36px;font-weight:850;letter-spacing:-1px;
color:#f4f7ff;margin-top:10px">DriftX<span style="color:#70a9ff">-Forensics</span></div>
<div style="font-size:14px;color:#c0d0e8;margin-top:10px;line-height:1.8">
Cross-stage drift detection · Root-cause investigation ·
Confidence-aware remediation · Safety-gated adaptation</div>
<div style="margin-top:18px;color:#9edfc6;font-size:12px;font-weight:700">
● SAFETY-GATED WORKFLOW &nbsp; | &nbsp; CROSS-STAGE FORENSICS
&nbsp; | &nbsp; INCIDENT OBSERVABILITY</div>
</div>
""", unsafe_allow_html=True)

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


# INTEGRATED INCIDENT INVESTIGATION
from src.driftx.evaluation.cross_stage_forensics import diagnose_cross_stage_origin
from src.driftx.evaluation.incident_orchestrator import process_incident, save_incident_record

st.markdown('<div class="section">Integrated Incident Investigation</div>', unsafe_allow_html=True)
st.caption("Diagnose a cross-stage incident, obtain a remediation recommendation, and evaluate a challenger against the current champion.")

with st.expander("Investigate a drift incident", expanded=False):
    with st.form("driftx_incident_form"):
        scenario = st.text_input("Incident name", "Production drift incident")

        st.markdown("**Stage evidence**")
        c1, c2, c3, c4 = st.columns(4)
        feature_evidence = c1.slider("Feature drift", 0.0, 1.0, 0.90, 0.05)
        prediction_evidence = c2.slider("Prediction drift", 0.0, 1.0, 0.65, 0.05)
        residual_evidence = c3.slider("Residual drift", 0.0, 1.0, 0.30, 0.05)
        performance_evidence = c4.slider("Performance drift", 0.0, 1.0, 0.15, 0.05)

        st.markdown("**First observed stage-change times**")
        t1, t2, t3, t4 = st.columns(4)
        feature_time = t1.number_input("Feature time", value=1.0, min_value=0.0)
        prediction_time = t2.number_input("Prediction time", value=2.0, min_value=0.0)
        residual_time = t3.number_input("Residual time", value=3.0, min_value=0.0)
        performance_time = t4.number_input("Performance time", value=4.0, min_value=0.0)

        st.markdown("**Champion model metrics**")
        a, b, c = st.columns(3)
        champion_mae_input = a.number_input("Champion MAE", min_value=0.0, value=12.0)
        champion_rmse_input = b.number_input("Champion RMSE", min_value=0.0, value=18.0)
        champion_r2_input = c.number_input("Champion R?", min_value=-1.0, max_value=1.0, value=0.82)

        st.markdown("**Challenger model metrics**")
        d, e, f = st.columns(3)
        challenger_mae_input = d.number_input("Challenger MAE", min_value=0.0, value=10.5)
        challenger_rmse_input = e.number_input("Challenger RMSE", min_value=0.0, value=17.0)
        challenger_r2_input = f.number_input("Challenger R?", min_value=-1.0, max_value=1.0, value=0.83)

        run_incident = st.form_submit_button("Run incident investigation", type="primary")

    if run_incident:
        evidence = {
            "feature": feature_evidence,
            "prediction": prediction_evidence,
            "residual": residual_evidence,
            "performance": performance_evidence,
        }
        onset_lags = {
            "feature": feature_time,
            "prediction": prediction_time,
            "residual": residual_time,
            "performance": performance_time,
        }
        impact = dict(evidence)

        try:
            diagnosis = diagnose_cross_stage_origin(evidence, onset_lags, impact)
            incident = process_incident(
                scenario=scenario,
                diagnosis=diagnosis,
                evidence=evidence,
                impact=impact,
                champion_metrics={
                    "mae": champion_mae_input,
                    "rmse": champion_rmse_input,
                    "r2": champion_r2_input,
                },
                challenger_metrics={
                    "mae": challenger_mae_input,
                    "rmse": challenger_rmse_input,
                    "r2": challenger_r2_input,
                },
            )

            history_path = save_incident_record(incident)
            st.success(f"Incident saved to {history_path}")

            st.markdown("### Incident diagnosis")
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted origin", diagnosis["predicted_origin"].upper())
            r2.metric("Diagnostic confidence", f'{diagnosis["confidence"]:.1%}')
            r3.metric("Pipeline path", " ? ".join(diagnosis["propagation_path"]) or "Not established")

            st.markdown("### Recommended remediation")
            remediation = incident["remediation"]
            st.subheader(remediation["title"])
            st.write(remediation["reason"])
            st.write("**Recommended steps**")
            for step in remediation["steps"]:
                st.write(f"- {step}")
            st.info(f'Verification: {remediation["verification"]}')

            st.markdown("### Challenger safety gate")
            gate = incident["safety_gate"]
            if gate["decision"] == "PROMOTE":
                st.success("PASS ? challenger satisfies the configured safety checks.")
            else:
                st.error("REJECT ? challenger did not satisfy the configured safety checks.")
            st.write(gate["reason"])
            st.metric("MAE improvement", f'{gate["improvement"]:.4f}')
            st.metric("MAE improvement ratio", f'{gate["improvement_ratio"]:.1%}')
            st.warning("This is a decision-support result. Model deployment is not performed automatically.")

            import json
            st.download_button(
                "Download incident report (JSON)",
                data=json.dumps(incident, indent=2, default=str),
                file_name="driftx_incident_report.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(f"Incident investigation failed: {exc}")
            st.caption("Check the incident engine imports and supplied metric values.")


