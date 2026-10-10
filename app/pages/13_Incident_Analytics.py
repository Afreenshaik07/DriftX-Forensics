from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "results" / "incident_history.jsonl"

st.set_page_config(page_title="DriftX | Incident Analytics", page_icon="DX", layout="wide")
st.title("Incident Analytics")
st.caption("Historical analysis of recorded investigations—not live production telemetry.")

if not HISTORY.exists():
    st.info("No incident history found. Run an investigation from Incident Workbench.")
    st.stop()

records = []
with HISTORY.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
        except json.JSONDecodeError:
            st.warning(f"Skipped malformed history line {line_number}.")

if not records:
    st.info("No valid incident records found.")
    st.stop()

def severity_of(record):
    value = record.get("severity", "UNCLASSIFIED")
    if isinstance(value, dict):
        value = value.get("severity", "UNCLASSIFIED")
    return str(value).upper()

rows = []
for record in records:
    diagnosis = record.get("diagnosis") or {}
    gate = record.get("safety_gate") or {}
    confidence = float(diagnosis.get("confidence", 0) or 0)
    rows.append({
        "Incident ID": record.get("incident_id", "legacy"),
        "Scenario": record.get("scenario", "Unspecified"),
        "Severity": severity_of(record),
        "Origin": str(diagnosis.get("predicted_origin", "unknown")).title(),
        "Confidence (%)": round(confidence * 100, 1),
        "Safety decision": str(gate.get("decision", "NOT_EVALUATED")),
        "Gate evaluated": bool(gate.get("evaluated", False)),
        "Recorded UTC": record.get("recorded_at_utc", ""),
    })

df = pd.DataFrame(rows)
df["Recorded UTC"] = pd.to_datetime(df["Recorded UTC"], errors="coerce", utc=True)

a, b, c, d = st.columns(4)
a.metric("Total incidents", len(df))
b.metric("Critical incidents", int((df["Severity"] == "CRITICAL").sum()))
c.metric("Rejected challengers", int((df["Safety decision"] == "REJECT").sum()))
d.metric("Mean diagnosis confidence", f'{df["Confidence (%)"].mean():.1f}%')

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Incidents by severity")
    st.bar_chart(df["Severity"].value_counts())
with right:
    st.subheader("Diagnosed root-cause origins")
    st.bar_chart(df["Origin"].value_counts())

st.subheader("Safety-gate decisions")
st.bar_chart(df["Safety decision"].value_counts())

st.subheader("Confidence distribution")
st.bar_chart(df["Confidence (%)"].value_counts().sort_index())

st.subheader("Investigation records")
f1, f2 = st.columns(2)
with f1:
    severity_filter = st.selectbox(
        "Severity filter",
        ["All"] + sorted(df["Severity"].unique().tolist()),
    )
with f2:
    decision_filter = st.selectbox(
        "Decision filter",
        ["All"] + sorted(df["Safety decision"].unique().tolist()),
    )

view = df.copy()
if severity_filter != "All":
    view = view[view["Severity"] == severity_filter]
if decision_filter != "All":
    view = view[view["Safety decision"] == decision_filter]

display = view.copy()
display["Recorded UTC"] = display["Recorded UTC"].astype(str)
st.dataframe(display, width="stretch", hide_index=True)

st.download_button(
    "Download filtered analytics (CSV)",
    data=display.to_csv(index=False).encode("utf-8"),
    file_name="driftx_incident_analytics.csv",
    mime="text/csv",
)

st.caption("All metrics summarize saved incident records; they do not establish live monitoring or causal certainty.")
