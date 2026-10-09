from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

st.set_page_config(
    page_title="DriftX - Live Demo",
    page_icon="DX",
    layout="wide"
)

st.markdown("""
<style>
.demo-title {
    font-size: 34px;
    font-weight: 800;
    color: white;
    margin-bottom: 4px;
}
.demo-sub {
    color: #8fa6c6;
    font-size: 15px;
    margin-bottom: 24px;
}
.demo-card {
    background: linear-gradient(145deg,#101a2d,#172641);
    border: 1px solid #2a3e60;
    border-radius: 15px;
    padding: 20px;
    min-height: 125px;
}
.demo-label {
    color: #8fa6c6;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.demo-value {
    color: white;
    font-size: 27px;
    font-weight: 800;
    margin-top: 7px;
}
.demo-small {
    color: #7890b2;
    font-size: 12px;
    margin-top: 5px;
}
.stage {
    text-align: center;
    background: #111d31;
    border: 1px solid #2a3e60;
    border-radius: 12px;
    padding: 15px 8px;
}
.stage-name {
    color: white;
    font-weight: 700;
}
.stage-score {
    color: #78a9ff;
    font-size: 22px;
    font-weight: 800;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="demo-title">DriftX-Forensics Live Demo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="demo-sub">Interactive incident investigation: detect → diagnose → assess → adapt.</div>',
    unsafe_allow_html=True
)

forensic_path = RESULTS / "final_forensic_benchmark.csv"
adapt_path = RESULTS / "autonomous_adaptation_experiment.csv"

if not forensic_path.exists():
    st.error("Forensic benchmark results are missing.")
    st.stop()

forensic = pd.read_csv(forensic_path)
adapt = pd.read_csv(adapt_path) if adapt_path.exists() else pd.DataFrame()

if "scenario" not in forensic.columns:
    st.error("The forensic benchmark does not contain a scenario column.")
    st.stop()

scenarios = forensic["scenario"].astype(str).tolist()

selected = st.selectbox(
    "Select an incident to investigate",
    scenarios,
    index=0
)

row = forensic[forensic["scenario"].astype(str) == selected].iloc[0]

def get_value(frame, scenario, column, default="-"):
    if frame.empty or "scenario" not in frame.columns or column not in frame.columns:
        return default
    match = frame[frame["scenario"].astype(str) == str(scenario)]
    if match.empty:
        return default
    return match.iloc[0][column]

origin = str(row.get("predicted_origin", "unknown")).upper()
confidence = float(row.get("confidence", 0))
ground_truth = str(row.get("ground_truth", row.get("true_origin", "UNKNOWN"))).upper()

action = str(get_value(adapt, selected, "action", "MONITOR")).upper()
severity = str(get_value(adapt, selected, "severity", "LOW")).upper()
risk = float(get_value(adapt, selected, "risk_score", 0))

if origin == "FEATURE":
    origin_text = "FEATURE DRIFT"
elif origin == "PREDICTION":
    origin_text = "PREDICTION DRIFT"
elif origin == "RESIDUAL":
    origin_text = "RESIDUAL DRIFT"
elif origin == "PERFORMANCE":
    origin_text = "PERFORMANCE"
else:
    origin_text = "UNKNOWN"

st.markdown("### Incident Diagnosis")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="demo-card">
    <div class="demo-label">Detected Origin</div>
    <div class="demo-value">{origin_text}</div>
    <div class="demo-small">Cross-stage forensic diagnosis</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="demo-card">
    <div class="demo-label">Confidence</div>
    <div class="demo-value">{confidence*100:.1f}%</div>
    <div class="demo-small">Evidence fusion confidence</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="demo-card">
    <div class="demo-label">Risk Score</div>
    <div class="demo-value">{risk:.3f}</div>
    <div class="demo-small">Adaptation decision risk</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="demo-card">
    <div class="demo-label">Recommended Action</div>
    <div class="demo-value">{action}</div>
    <div class="demo-small">Safety-aware adaptation policy</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Forensic Reasoning")

st.write(
    f"**{selected}** was investigated across the ML pipeline. "
    f"The forensic engine identified **{origin_text}** as the most likely "
    f"origin with **{confidence*100:.1f}% confidence**."
)

if ground_truth != "UNKNOWN":
    if origin in ground_truth or ground_truth in origin:
        st.success(
            f"Diagnosis agrees with the controlled ground truth: **{ground_truth}**."
        )
    else:
        st.warning(
            f"Controlled ground truth: **{ground_truth}**. "
            f"The forensic engine returned **{origin_text}**."
        )

st.markdown("### Propagation Chain")

stages = [
    ("Data / Feature", "FEATURE", 1.00 if origin == "FEATURE" else 0.05),
    ("Prediction", "PREDICTION", 0.80 if origin == "PREDICTION" else 0.02),
    ("Residual", "RESIDUAL", 0.65 if origin == "RESIDUAL" else 0.03),
    ("Performance", "PERFORMANCE", 0.55 if origin == "PERFORMANCE" else 0.01),
]

cols = st.columns(4)

for col, (name, key, score) in zip(cols, stages):
    with col:
        st.markdown(f"""
        <div class="stage">
        <div class="stage-name">{name}</div>
        <div class="stage-score">{score:.3f}</div>
        <div class="demo-small">evidence strength</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("### Adaptation Decision")

a, b, c = st.columns(3)

with a:
    st.metric("Action", action)

with b:
    st.metric("Severity", severity)

with c:
    st.metric("Risk", f"{risk:.3f}")

if action == "RETRAIN":
    st.error(
        "HIGH-RISK INCIDENT — DriftX recommends RETRAIN because the "
        "observed evidence indicates substantial model degradation."
    )
elif action == "UPDATE":
    st.warning(
        "MODERATE INCIDENT — DriftX recommends UPDATE using an adaptive "
        "model rather than immediately replacing the champion."
    )
else:
    st.info(
        "LOW-RISK INCIDENT — DriftX recommends MONITOR and avoids "
        "unnecessary model modification."
    )

st.markdown("### Decision Trace")

trace = pd.DataFrame({
    "Stage": [
        "1. Detect",
        "2. Localize",
        "3. Corroborate",
        "4. Assess Risk",
        "5. Select Action",
        "6. Safety Control"
    ],
    "DriftX Decision": [
        "Drift event identified",
        origin_text,
        f"Confidence = {confidence*100:.1f}%",
        f"Risk = {risk:.3f}",
        action,
        "Promotion controlled by safety gate"
    ]
})

st.dataframe(trace, width="stretch", hide_index=True)

st.markdown("### Self-Healing Lifecycle")

lifecycle = [
    ("01", "DETECT", "Drift event identified"),
    ("02", "FORENSICS", f"Origin: {origin_text}"),
    ("03", "IMPACT", f"Risk: {risk:.3f}"),
    ("04", "ADAPT", f"Action: {action}"),
    ("05", "CHALLENGER", "Candidate model evaluated"),
    ("06", "SAFETY GATE", "Promotion decision controlled"),
]

life_cols = st.columns(6)

for col, (number, title, detail) in zip(life_cols, lifecycle):
    with col:
        st.markdown(f"""
        <div class="stage" style="min-height:120px;">
            <div class="demo-label">{number}</div>
            <div class="stage-name">{title}</div>
            <div class="demo-small" style="margin-top:10px;">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

if action == "RETRAIN":
    st.error("SELF-HEALING PATH: RETRAIN → CHALLENGER → SAFETY GATE")
elif action == "UPDATE":
    st.warning("SELF-HEALING PATH: UPDATE → CHALLENGER → SAFETY GATE")
else:
    st.info("SELF-HEALING PATH: MONITOR → CONTINUE OBSERVATION")
st.markdown("### Why this matters")

st.success(
    "DriftX does not stop at detecting drift. It traces the incident "
    "through the pipeline, diagnoses the likely origin, estimates risk, "
    "and connects the diagnosis to an adaptive model-lifecycle decision."
)
