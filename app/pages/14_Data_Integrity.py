from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "results" / "incident_history.jsonl"

st.set_page_config(page_title="DriftX Data Integrity", page_icon="DX", layout="wide")
st.title("Incident Data Integrity")
st.caption("Validate saved incident records before relying on historical analytics.")

if not HISTORY.exists():
    st.info("Incident history has not been created yet.")
    st.stop()

records = []
issues = []
seen_ids = set()
malformed = 0

with HISTORY.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            issues.append({
                "Line": line_number,
                "Severity": "CRITICAL",
                "Issue": "Malformed JSON",
                "Detail": "Record could not be parsed.",
            })
            continue

        records.append(record)
        incident_id = record.get("incident_id")

        if not incident_id:
            issues.append({
                "Line": line_number,
                "Severity": "MEDIUM",
                "Issue": "Missing incident ID",
                "Detail": "Record may be a legacy entry.",
            })
        elif incident_id in seen_ids:
            issues.append({
                "Line": line_number,
                "Severity": "HIGH",
                "Issue": "Duplicate incident ID",
                "Detail": str(incident_id),
            })
        else:
            seen_ids.add(incident_id)

        diagnosis = record.get("diagnosis") or {}
        confidence = diagnosis.get("confidence")
        if confidence is None:
            issues.append({
                "Line": line_number,
                "Severity": "MEDIUM",
                "Issue": "Missing confidence",
                "Detail": "Diagnostic confidence was not recorded.",
            })
        else:
            try:
                if not 0 <= float(confidence) <= 1:
                    raise ValueError
            except (TypeError, ValueError):
                issues.append({
                    "Line": line_number,
                    "Severity": "HIGH",
                    "Issue": "Invalid confidence",
                    "Detail": repr(confidence),
                })

        gate = record.get("safety_gate") or {}
        if gate.get("decision") in ("PROMOTE", "REJECT") and not gate.get("evaluated", False):
            issues.append({
                "Line": line_number,
                "Severity": "HIGH",
                "Issue": "Inconsistent safety-gate status",
                "Detail": "Decision exists but evaluated flag is false.",
            })

c1, c2, c3 = st.columns(3)
c1.metric("Records parsed", len(records))
c2.metric("Malformed lines", malformed)
c3.metric("Integrity issues", len(issues))

if not issues:
    st.success("No integrity issues detected by these checks.")
else:
    st.warning("Review the findings before using the affected records for analysis.")
    issue_df = pd.DataFrame(issues)
    st.dataframe(issue_df, width="stretch", hide_index=True)
    st.download_button(
        "Download integrity findings",
        data=issue_df.to_csv(index=False).encode("utf-8"),
        file_name="driftx_integrity_findings.csv",
        mime="text/csv",
    )

st.caption("These checks validate incident-log structure and selected fields; they do not prove that a diagnosis or model metric is scientifically correct.")
