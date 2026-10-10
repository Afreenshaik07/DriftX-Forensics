from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

st.set_page_config(page_title="DriftX System Health", page_icon="DX", layout="wide")
st.title("System Health")
st.caption("Operational visibility for local DriftX-Forensics artifacts.")

checks = [
    ("Forensic benchmark", ["forensic_full_benchmark.csv", "final_forensic_benchmark.csv"]),
    ("Adaptation benchmark", ["adaptation_benchmark.csv"]),
    ("Safety-gate results", ["safety_gate_stress_test_summary.csv", "safety_gate_stress_test.csv"]),
    ("Incident history", ["incident_history.jsonl"]),
]

rows = []
for label, candidates in checks:
    found = next((RESULTS / name for name in candidates if (RESULTS / name).is_file()), None)
    rows.append({
        "Component": label,
        "Status": "AVAILABLE" if found else "NOT FOUND",
        "Artifact": str(found.relative_to(ROOT)) if found else ", ".join(candidates),
        "Last modified": datetime.fromtimestamp(found.stat().st_mtime).isoformat(timespec="seconds") if found else "",
        "Size (KB)": round(found.stat().st_size / 1024, 2) if found else None,
    })

df = pd.DataFrame(rows)
available = int((df["Status"] == "AVAILABLE").sum())
c1, c2, c3 = st.columns(3)
c1.metric("Artifacts checked", len(checks))
c2.metric("Available", available)
c3.metric("Missing", len(checks) - available)

if available == len(checks):
    st.success("All monitored artifact categories are available.")
else:
    st.warning("Some artifacts are missing. Run the corresponding project workflow if those outputs are expected.")

st.dataframe(df, width="stretch", hide_index=True)

st.subheader("Recent result files")
if RESULTS.exists():
    files = sorted(
        (f for f in RESULTS.glob("*") if f.is_file()),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if files:
        recent = pd.DataFrame([{
            "File": f.name,
            "Modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
            "Size (KB)": round(f.stat().st_size / 1024, 2),
        } for f in files[:25]])
        st.dataframe(recent, width="stretch", hide_index=True)
    else:
        st.info("The results directory exists but contains no files.")
else:
    st.error("The results directory does not exist.")

st.caption("This page reports file availability and timestamps; it does not claim that missing artifacts are runtime failures.")
