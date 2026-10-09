import streamlit as st
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

st.set_page_config(
    page_title="DriftX â€¢ Audit Log",
    page_icon="AL",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top:2rem;}
.title {font-size:36px;font-weight:750;}
.subtitle {color:#8995b1;margin-bottom:25px;}
.card {
    padding:20px;
    border-radius:16px;
    background:#121a2d;
    border:1px solid #293653;
    margin-bottom:15px;
}
.label {
    font-size:12px;
    color:#8995b1;
    text-transform:uppercase;
    letter-spacing:1px;
}
.value {
    font-size:28px;
    font-weight:700;
    margin-top:6px;
}
.event {
    padding:16px;
    border-radius:14px;
    background:#121a2d;
    border:1px solid #293653;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">Audit Log</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Traceable record of drift detection, forensic diagnosis, adaptation decisions and model lifecycle events.</div>',
    unsafe_allow_html=True
)

adaptation_path = Path("results/autonomous_adaptation_experiment.csv")
forensic_path = Path("results/final_forensic_benchmark.csv")
registry_path = Path("artifacts/registry/promotion_history.json")

events = []

if forensic_path.exists():

    forensic = pd.read_csv(forensic_path)

    for _, row in forensic.iterrows():

        events.append({
            "Event": "FORENSIC_DIAGNOSIS",
            "Scenario": str(row.get("scenario", "UNKNOWN")),
            "Origin": str(
                row.get("predicted_origin", "UNKNOWN")
            ).upper(),
            "Action": "DIAGNOSE",
            "Severity": "ANALYSIS",
            "Confidence": float(
                row.get("forensic_confidence", 0)
            ),
            "Risk": float(
                row.get("performance_score", 0)
            ),
            "Source": "Cross-Stage Forensics"
        })

if adaptation_path.exists():

    adaptation = pd.read_csv(adaptation_path)

    for _, row in adaptation.iterrows():

        events.append({
            "Event": "ADAPTATION_DECISION",
            "Scenario": str(row.get("scenario", "UNKNOWN")),
            "Origin": str(
                row.get("predicted_origin", "UNKNOWN")
            ).upper(),
            "Action": str(
                row.get("action", "UNKNOWN")
            ).upper(),
            "Severity": str(
                row.get("severity", "UNKNOWN")
            ).upper(),
            "Confidence": float(
                row.get("confidence", 0)
            ),
            "Risk": float(
                row.get("risk_score", 0)
            ),
            "Source": "Adaptation Policy"
        })

if registry_path.exists():

    try:

        with open(registry_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        if isinstance(history, dict):

            if "history" in history:
                history = history["history"]
            else:
                history = [history]

        if isinstance(history, list):

            for item in history:

                if isinstance(item, dict):

                    events.append({
                        "Event": "MODEL_REGISTRY",
                        "Scenario": "MODEL_LIFECYCLE",
                        "Origin": "SYSTEM",
                        "Action": str(
                            item.get(
                                "action",
                                item.get(
                                    "decision",
                                    "UNKNOWN"
                                )
                            )
                        ).upper(),
                        "Severity": "GOVERNANCE",
                        "Confidence": 0.0,
                        "Risk": 0.0,
                        "Source": "Model Registry"
                    })

    except Exception:
        pass

audit_df = pd.DataFrame(events)

st.markdown("### Audit Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="card"><div class="label">Total Events</div>'
        f'<div class="value">{len(audit_df)}</div></div>',
        unsafe_allow_html=True
    )

with c2:
    diagnoses = (
        len(audit_df[
            audit_df["Event"] == "FORENSIC_DIAGNOSIS"
        ])
        if not audit_df.empty else 0
    )

    st.markdown(
        f'<div class="card"><div class="label">Forensic Events</div>'
        f'<div class="value">{diagnoses}</div></div>',
        unsafe_allow_html=True
    )

with c3:
    adaptations = (
        len(audit_df[
            audit_df["Event"] == "ADAPTATION_DECISION"
        ])
        if not audit_df.empty else 0
    )

    st.markdown(
        f'<div class="card"><div class="label">Adaptation Events</div>'
        f'<div class="value">{adaptations}</div></div>',
        unsafe_allow_html=True
    )

with c4:
    retrains = (
        len(audit_df[
            audit_df["Action"] == "RETRAIN"
        ])
        if not audit_df.empty else 0
    )

    st.markdown(
        f'<div class="card"><div class="label">Retrain Decisions</div>'
        f'<div class="value">{retrains}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("### Event Filters")

if not audit_df.empty:

    f1, f2, f3 = st.columns(3)

    with f1:
        event_filter = st.selectbox(
            "Event Type",
            ["ALL"] + sorted(
                audit_df["Event"].dropna().unique().tolist()
            )
        )

    with f2:
        action_filter = st.selectbox(
            "Action",
            ["ALL"] + sorted(
                audit_df["Action"].dropna().unique().tolist()
            )
        )

    with f3:
        origin_filter = st.selectbox(
            "Origin",
            ["ALL"] + sorted(
                audit_df["Origin"].dropna().unique().tolist()
            )
        )

    filtered = audit_df.copy()

    if event_filter != "ALL":
        filtered = filtered[
            filtered["Event"] == event_filter
        ]

    if action_filter != "ALL":
        filtered = filtered[
            filtered["Action"] == action_filter
        ]

    if origin_filter != "ALL":
        filtered = filtered[
            filtered["Origin"] == origin_filter
        ]

else:
    filtered = audit_df

st.markdown("### Event Timeline")

if filtered.empty:

    st.info("No audit events match the selected filters.")

else:

    for _, row in filtered.iterrows():

        confidence = float(row["Confidence"])
        risk = float(row["Risk"])

        st.markdown(
            f"""
            <div class="event">
                <div style="font-size:12px;color:#8995b1;">
                    {row["Event"]}
                </div>
                <div style="font-size:19px;font-weight:700;margin-top:5px;">
                    {row["Scenario"]}
                </div>
                <div style="margin-top:8px;">
                    Origin: <b>{row["Origin"]}</b>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    Action: <b>{row["Action"]}</b>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    Severity: <b>{row["Severity"]}</b>
                </div>
                <div style="margin-top:6px;color:#8995b1;">
                    Confidence: {confidence:.1%}
                    &nbsp;&nbsp; Risk: {risk:.3f}
                    &nbsp;&nbsp; Source: {row["Source"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("### Structured Audit Records")

if not filtered.empty:

    display = filtered.copy()

    for column in ["Confidence", "Risk"]:
        if column in display.columns:
            display[column] = display[column].round(4)

    st.dataframe(
        display,
        width="stretch",
        hide_index=True
    )

st.markdown("### Governance Trace")

st.info(
    "Every major DriftX decision can be traced through the pipeline: "
    "**drift evidence ? forensic diagnosis ? impact assessment ? "
    "adaptation policy ? safety-gated model lifecycle**."
)

st.success(
    "Audit visibility is active. DriftX decisions remain explainable and traceable."
)
