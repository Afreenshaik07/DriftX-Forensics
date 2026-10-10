from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.driftx.evaluation.remediation_recommender import recommend_remediation

st.set_page_config(page_title="Incident Workbench | DriftX", layout="wide")
st.title("Incident Workbench")
st.caption("Evidence ? Diagnosis ? Remediation ? Safety review")

RESULTS = ROOT / "results"

def load_csv(name):
    path = RESULTS / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()

forensic = load_csv("forensic_full_benchmark.csv")
adaptation = load_csv("adaptation_benchmark.csv")
safety = load_csv("safety_gate_stress_test_summary.csv")

if forensic.empty:
    forensic = load_csv("final_forensic_benchmark.csv")

if forensic.empty:
    st.error("No forensic benchmark evidence is available. Check the results directory.")
    st.stop()

st.subheader("1 ? Select a diagnostic incident")
scenario_col = "scenario" if "scenario" in forensic.columns else forensic.columns[0]
scenarios = forensic[scenario_col].astype(str).tolist()
selected = st.selectbox("Scenario", scenarios)
row = forensic.loc[forensic[scenario_col].astype(str) == selected].iloc[0]

def value(column, default=None):
    return row[column] if column in row.index else default

origin = str(value("predicted_origin", "unknown")).strip().lower()
if origin in ("nan", "", "none"):
    origin = "unknown"

try:
    confidence = float(value("forensic_confidence", 0.0))
except (TypeError, ValueError):
    confidence = 0.0

confidence = max(0.0, min(1.0, confidence))

a, b, c = st.columns(3)
a.metric("Scenario", selected)
b.metric("Diagnosed origin", origin.upper())
c.metric("Forensic confidence", f"{confidence:.1%}")

st.subheader("2 ? Evidence available")
evidence_columns = {
    "feature": "feature_ks",
    "prediction": "prediction_ks",
    "residual": "residual_ks",
    "performance": "performance_score",
}
evidence = {}
cols = st.columns(4)
for (stage, column), panel in zip(evidence_columns.items(), cols):
    raw = value(column, 0.0)
    try:
        number = float(raw)
    except (TypeError, ValueError):
        number = 0.0
    evidence[stage] = max(0.0, min(1.0, number))
    panel.metric(f"{stage.title()} signal", f"{number:.3f}")

st.subheader("3 ? Recommended remediation")
recommendation = recommend_remediation(
    origin=origin,
    confidence=confidence,
    evidence=evidence,
)
st.markdown(f"### {recommendation['title']}")
st.write(recommendation["reason"])
st.info(f"Recommended action: {recommendation['action']}")
for step in recommendation["steps"]:
    st.markdown(f"- {step}")
st.caption("Verification: " + recommendation["verification"])

st.subheader("4 ? Adaptation policy evidence")
if not adaptation.empty:
    if "scenario" in adaptation.columns:
        matching = adaptation[
            adaptation["scenario"].astype(str) == selected
        ]
    else:
        matching = pd.DataFrame()

    if not matching.empty:
        st.dataframe(matching, width='stretch', hide_index=True)
    else:
        st.dataframe(adaptation, width='stretch', hide_index=True)
else:
    st.warning("Adaptation benchmark results are not available.")

st.subheader("5 ? Safety gate evidence")
if not safety.empty:
    st.dataframe(safety, width='stretch', hide_index=True)
else:
    safety_file = RESULTS / "safety_gate_stress_test.csv"
    if safety_file.exists():
        st.dataframe(pd.read_csv(safety_file), width='stretch', hide_index=True)
    else:
        st.warning("No safety-gate result file was found.")

st.subheader("6 ? Incident decision")
st.write(
    "This workbench provides evidence-backed recommendations. "
    "It does not automatically deploy, promote, or modify a production model."
)
st.download_button(
    "Export incident evidence",
    data=pd.DataFrame([{
        "scenario": selected,
        "origin": origin,
        "confidence": confidence,
        "recommended_action": recommendation["action"],
        "recommendation": recommendation["title"],
        **{f"{k}_evidence": v for k, v in evidence.items()},
    }]).to_csv(index=False),
    file_name="driftx_incident_evidence.csv",
    mime="text/csv",
)
