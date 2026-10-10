from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "results" / "incident_history.jsonl"

st.set_page_config(page_title="DriftX | Incident Monitor", page_icon="DX", layout="wide")
st.title("Incident Monitor")
st.caption("Monitor recorded investigations, forensic findings, and challenger safety decisions.")

if st.button("Refresh incident data"):
    st.rerun()

if not HISTORY.exists():
    st.info("No incident history exists yet. Run an investigation from Incident Workbench.")
    st.stop()

records = []
with HISTORY.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
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
for i, record in enumerate(records):
    diagnosis = record.get("diagnosis") or {}
    gate = record.get("safety_gate") or {}
    rows.append({
        "Record index": i,
        "Recorded UTC": record.get("recorded_at_utc", "Unknown"),
        "Incident": record.get("scenario", "Unnamed incident"),
        "Severity": severity_of(record),
        "Root cause": str(diagnosis.get("predicted_origin", "unknown")).title(),
        "Confidence (%)": round(float(diagnosis.get("confidence", 0) or 0) * 100, 1),
        "Safety decision": str(gate.get("decision", "NOT_EVALUATED")),
        "Gate evaluated": bool(gate.get("evaluated", False)),
    })

df = pd.DataFrame(rows)
df["_timestamp"] = pd.to_datetime(df["Recorded UTC"], errors="coerce", utc=True)
df = df.sort_values("_timestamp", ascending=False, na_position="last")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Recorded incidents", len(records))
c2.metric("Promote decisions", int((df["Safety decision"] == "PROMOTE").sum()))
c3.metric("Rejected challengers", int((df["Safety decision"] == "REJECT").sum()))
c4.metric("Evaluated gates", int(df["Gate evaluated"].sum()))

st.divider()
st.subheader("Incident filters")
f1, f2 = st.columns(2)
with f1:
    severity_options = ["All"] + sorted(df["Severity"].unique().tolist())
    selected_severity = st.selectbox("Severity", severity_options)
with f2:
    decision_options = ["All"] + sorted(df["Safety decision"].unique().tolist())
    selected_decision = st.selectbox("Safety decision", decision_options)

view = df.copy()
if selected_severity != "All":
    view = view[view["Severity"] == selected_severity]
if selected_decision != "All":
    view = view[view["Safety decision"] == selected_decision]

st.subheader("Incident timeline")
st.dataframe(
    view.drop(columns=["_timestamp", "Record index"]),
    width="stretch",
    hide_index=True,
)

st.subheader("Safety decision distribution")
distribution = view["Safety decision"].value_counts()
if not distribution.empty:
    st.bar_chart(distribution)
else:
    st.info("No incidents match these filters.")

st.divider()
st.subheader("Incident investigation details")

available_indices = view["Record index"].astype(int).tolist()
if not available_indices:
    st.info("No incident matches the selected filters.")
    st.stop()

selected_index = st.selectbox(
    "Choose an incident",
    available_indices,
    format_func=lambda i: (
        f'{records[i].get("scenario", "Incident")} — '
        f'{records[i].get("recorded_at_utc", "Unknown time")}'
    ),
)
record = records[selected_index]
diagnosis = record.get("diagnosis") or {}
remediation = record.get("remediation") or {}
gate = record.get("safety_gate") or {}

a, b, c = st.columns(3)
a.metric("Root-cause origin", str(diagnosis.get("predicted_origin", "unknown")).title())
b.metric("Diagnosis confidence", f'{float(diagnosis.get("confidence", 0) or 0) * 100:.1f}%')
c.metric("Severity", severity_of(record))

st.markdown("#### Safety gate")
g1, g2, g3 = st.columns(3)
g1.metric("Decision", str(gate.get("decision", "NOT_EVALUATED")))
g2.metric("Evaluated", "Yes" if gate.get("evaluated", False) else "No")
g3.metric("Deployment authorized", "Yes" if record.get("deployment_authorized", False) else "No")
st.write(gate.get("reason", "No safety-gate rationale recorded."))

if gate.get("evaluated", False):
    metrics = []
    for label, key in [("MAE", "mae"), ("RMSE", "rmse"), ("R²", "r2")]:
        metrics.append({
            "Metric": label,
            "Champion": gate.get(f"champion_{key}", "N/A"),
            "Challenger": gate.get(f"challenger_{key}", "N/A"),
        })
    st.dataframe(pd.DataFrame(metrics), hide_index=True, width="stretch")

st.markdown("#### Forensic evidence")
ranking = diagnosis.get("ranking") or []
if ranking:
    st.dataframe(pd.DataFrame(ranking), hide_index=True, width="stretch")
else:
    st.info("No stage ranking was saved for this incident.")

path = diagnosis.get("propagation_path") or []
if path:
    st.markdown("**Propagation path:** " + " → ".join(map(str, path)))

st.markdown("#### Remediation recommendation")
st.markdown(f"**{remediation.get('title', 'No recommendation recorded')}**")
st.write(remediation.get("reason", "No remediation rationale recorded."))
for step in remediation.get("steps", []):
    st.markdown(f"- {step}")
if remediation.get("verification"):
    st.markdown("**Verification:** " + str(remediation["verification"]))
st.caption(
    "Automatic execution: "
    + ("Enabled" if remediation.get("automatic_execution") else "Disabled; recommendation only")
)

with st.expander("View raw incident JSON"):
    st.json(record)

st.download_button(
    "Download filtered incidents (CSV)",
    data=view.drop(columns=["_timestamp", "Record index"]).to_csv(index=False).encode("utf-8"),
    file_name="driftx_incident_monitor.csv",
    mime="text/csv",
)

st.download_button(
    "Download complete incident history (JSONL)",
    data=HISTORY.read_bytes(),
    file_name="driftx_incident_history.jsonl",
    mime="application/x-ndjson",
)

st.caption("This page reads the local incident log. It does not monitor external systems or deploy models.")
