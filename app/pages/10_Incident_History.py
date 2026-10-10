from pathlib import Path
import json
import pandas as pd
import streamlit as st
from src.driftx.evaluation.incident_report import generate_incident_pdf

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "results" / "incident_history.jsonl"

st.set_page_config(page_title="DriftX | Incident History", page_icon="DX", layout="wide")
st.title("Incident History")
st.caption("Forensic investigations, root-cause evidence, remediation recommendations, and model safety decisions.")

if not HISTORY.exists() or not HISTORY.read_text(encoding="utf-8").strip():
    st.info("No saved incidents yet. Run an investigation from the Incident Workbench.")
    st.stop()

records = []
with HISTORY.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        except json.JSONDecodeError:
            st.warning(f"Malformed history entry skipped: line {line_number}")

if not records:
    st.warning("No valid incident records were found.")
    st.stop()

def get_severity(record):
    severity = record.get("severity", "UNCLASSIFIED")
    if isinstance(severity, dict):
        severity = severity.get("severity", "UNCLASSIFIED")
    return str(severity).upper()

rows = []
for record in records:
    diagnosis = record.get("diagnosis") or {}
    remediation = record.get("remediation") or {}
    gate = record.get("safety_gate") or {}
    rows.append({
        "Recorded UTC": record.get("recorded_at_utc", "Unknown"),
        "Incident": record.get("scenario", "Unnamed incident"),
        "Severity": get_severity(record),
        "Root cause": diagnosis.get("predicted_origin", "unknown"),
        "Confidence (%)": round(float(diagnosis.get("confidence", 0) or 0) * 100, 1),
        "Remediation": remediation.get("action", "Not available"),
        "Safety decision": gate.get("decision", "NOT_EVALUATED"),
    })

df = pd.DataFrame(rows)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total incidents", len(records))
m2.metric("Promoted candidates", int((df["Safety decision"] == "PROMOTE").sum()))
m3.metric("Rejected candidates", int((df["Safety decision"] == "REJECT").sum()))
m4.metric("Not evaluated", int((df["Safety decision"] == "NOT_EVALUATED").sum()))

st.subheader("Investigation records")
filter_choice = st.selectbox(
    "Filter by safety decision",
    ["All"] + sorted(df["Safety decision"].astype(str).unique().tolist()),
)
filtered = df if filter_choice == "All" else df[df["Safety decision"] == filter_choice]
st.dataframe(filtered, width="stretch", hide_index=True, use_container_width=False)

st.download_button(
    "Download history as CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="driftx_incident_history.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Incident investigation details")

selected_index = st.selectbox(
    "Select an incident",
    range(len(records)),
    format_func=lambda i: (
        f'{records[i].get("scenario", "Incident")} — '
        f'{records[i].get("recorded_at_utc", "Timestamp unavailable")}'
    ),
)
record = records[selected_index]
diagnosis = record.get("diagnosis") or {}
remediation = record.get("remediation") or {}
gate = record.get("safety_gate") or {}

left, right = st.columns([2, 1])
with left:
    st.markdown("#### Incident")
    st.write(record.get("scenario", "Unnamed incident"))
    st.caption(f'ID: {record.get("incident_id", "Not recorded")}')
    st.caption(f'Recorded UTC: {record.get("recorded_at_utc", "Unknown")}')
with right:
    st.markdown("#### Status")
    st.metric("Severity", get_severity(record))
    st.metric("Safety decision", gate.get("decision", "NOT_EVALUATED"))
    st.caption("Deployment authorized: " + ("Yes" if record.get("deployment_authorized") else "No"))

st.divider()
st.markdown("#### Root-cause diagnosis")
d1, d2, d3 = st.columns(3)
d1.metric("Predicted origin", str(diagnosis.get("predicted_origin", "unknown")).title())
d2.metric("Confidence", f'{float(diagnosis.get("confidence", 0) or 0) * 100:.1f}%')
d3.metric("Method", str(diagnosis.get("method", "Not recorded")).replace("_", " ").title())

ranking = diagnosis.get("ranking", [])
if ranking:
    st.markdown("**Stage evidence ranking**")
    ranking_rows = []
    for item in ranking:
        if isinstance(item, dict):
            ranking_rows.append({
                "Stage": item.get("stage", "unknown"),
                "Forensic score": item.get("forensic_score", 0),
                "Local evidence": item.get("local_evidence", 0),
                "Temporal precedence": item.get("temporal_precedence", 0),
                "Upstream isolation": item.get("upstream_isolation", 0),
                "Impact evidence": item.get("impact_evidence", 0),
            })
    if ranking_rows:
        st.dataframe(pd.DataFrame(ranking_rows), hide_index=True, width="stretch")

path = diagnosis.get("propagation_path", [])
if path:
    st.markdown("**Observed propagation path**")
    st.write(" → ".join(str(stage).title() for stage in path))

st.divider()
st.markdown("#### Remediation recommendation")
st.markdown(f"**{remediation.get('title', 'No recommendation recorded')}**")
st.write(remediation.get("reason", "No remediation rationale recorded."))

steps = remediation.get("steps", [])
if steps:
    st.markdown("**Recommended actions**")
    for step in steps:
        st.markdown(f"- {step}")

if remediation.get("verification"):
    st.markdown("**Verification criteria**")
    st.write(remediation["verification"])

st.caption(
    "Automatic execution: "
    + ("Enabled" if remediation.get("automatic_execution") else "Disabled — recommendation only")
)

st.divider()
st.markdown("#### Challenger safety evaluation")
if not gate.get("evaluated", False):
    st.info(gate.get("reason", "The safety gate was not evaluated for this incident."))
else:
    st.write(gate.get("reason", "No decision rationale recorded."))
    metric_rows = []
    for label, key in [
        ("MAE", "mae"),
        ("RMSE", "rmse"),
        ("R²", "r2"),
    ]:
        metric_rows.append({
            "Metric": label,
            "Champion": gate.get(f"champion_{key}", "N/A"),
            "Challenger": gate.get(f"challenger_{key}", "N/A"),
        })
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")
    if "improvement_ratio" in gate:
        st.metric("MAE improvement", f'{float(gate["improvement_ratio"]) * 100:.2f}%')

st.divider()
st.markdown("#### Full record")
with st.expander("View raw JSON (debugging only)"):
    st.json(record)

st.subheader("Export incident report")
try:
    pdf = generate_incident_pdf(record)
    st.download_button(
        "Download incident report (PDF)",
        data=pdf,
        file_name=f"driftx_incident_{record.get('incident_id', 'report')}.pdf",
        mime="application/pdf",
    )
except Exception as exc:
    st.error(f"PDF export failed: {exc}")

st.caption("Incident history is stored locally. A PROMOTE decision does not deploy a model.")
