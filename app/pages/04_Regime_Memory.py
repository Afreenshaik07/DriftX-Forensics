from pathlib import Path
import sys, json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="DriftX - Regime Memory", page_icon="DX", layout="wide")

MEMORY = ROOT / "artifacts" / "regime_memory.json"
ADAPT = ROOT / "results" / "autonomous_adaptation_experiment.csv"

st.title("Regime Memory")
st.caption("Historical drift regimes used to recognize recurring operating conditions")

if MEMORY.exists():
    try:
        raw = json.loads(MEMORY.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            regimes = raw.get("regimes", raw.get("memory", []))
        else:
            regimes = raw
    except Exception:
        regimes = []
else:
    regimes = []

if not isinstance(regimes, list):
    regimes = []

adapt = pd.read_csv(ADAPT) if ADAPT.exists() else pd.DataFrame()

st.subheader("Memory Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Known Regimes", len(regimes))

with c2:
    if adapt.empty or "previous_regime" not in adapt.columns:
        matches = 0
    else:
        matches = int((adapt["previous_regime"].astype(str) != "NONE").sum())
    st.metric("Memory Matches", matches)

with c3:
    if adapt.empty or "action" not in adapt.columns:
        updates = 0
    else:
        updates = int((adapt["action"] == "UPDATE").sum())
    st.metric("UPDATE Decisions", updates)

with c4:
    if adapt.empty or "action" not in adapt.columns:
        retrains = 0
    else:
        retrains = int((adapt["action"] == "RETRAIN").sum())
    st.metric("RETRAIN Decisions", retrains)

st.subheader("Memory Signature")

st.write(
    "Each regime is represented using four behavioural dimensions: "
    "feature drift, prediction drift, residual drift, and performance degradation."
)

signature = pd.DataFrame({
    "Signal": [
        "Feature Drift",
        "Prediction Drift",
        "Residual Drift",
        "Performance Degradation"
    ],
    "Purpose": [
        "Detect changes in input distributions",
        "Detect changes in model output distribution",
        "Detect changes in prediction error behaviour",
        "Measure degradation in model performance"
    ]
})

st.dataframe(signature, width="stretch", hide_index=True)

st.subheader("Stored Regimes")

if regimes:
    rows = []

    for i, regime in enumerate(regimes, 1):
        if not isinstance(regime, dict):
            continue

        signature_data = regime.get("signature", {})
        if not isinstance(signature_data, dict):
            signature_data = {}

        rows.append({
            "Regime": regime.get("regime_id", f"REGIME_{i:03d}"),
            "Origin": regime.get("origin", "UNKNOWN"),
            "Drift Type": regime.get("drift_type", "UNKNOWN"),
            "Action": regime.get("action", "UNKNOWN"),
            "Confidence": round(float(regime.get("confidence", 0)), 3),
            "Risk Score": round(float(regime.get("risk_score", 0)), 3),
            "Model Version": regime.get("model_version", "unknown"),
            "Adapter": regime.get("adapter", "none"),
            "Feature Drift": round(float(signature_data.get("feature_drift", 0)), 4),
            "Prediction Drift": round(float(signature_data.get("prediction_drift", 0)), 4),
            "Residual Drift": round(float(signature_data.get("residual_drift", 0)), 4),
            "Performance Degradation": round(float(signature_data.get("performance_degradation", 0)), 4),
        })

    regime_df = pd.DataFrame(rows)

    if not regime_df.empty:
        st.dataframe(regime_df, width="stretch", hide_index=True)

        st.subheader("Regime Details")

        selected = st.selectbox(
            "Select a regime",
            regime_df["Regime"].tolist()
        )

        selected_row = regime_df[regime_df["Regime"] == selected].iloc[0]

        a, b, c, d = st.columns(4)

        a.metric("Origin", str(selected_row["Origin"]))
        b.metric("Action", str(selected_row["Action"]))
        c.metric("Confidence", f'{selected_row["Confidence"]:.3f}')
        d.metric("Risk Score", f'{selected_row["Risk Score"]:.3f}')

        st.write("Behavioural signature")

        sig_df = pd.DataFrame({
            "Signal": [
                "Feature Drift",
                "Prediction Drift",
                "Residual Drift",
                "Performance Degradation"
            ],
            "Value": [
                selected_row["Feature Drift"],
                selected_row["Prediction Drift"],
                selected_row["Residual Drift"],
                selected_row["Performance Degradation"]
            ]
        })

        st.dataframe(sig_df, width="stretch", hide_index=True)
else:
    st.info("No regime memory records are currently stored.")

if not adapt.empty:
    st.subheader("Memory-Assisted Adaptation")

    cols = [
        "scenario",
        "previous_regime",
        "regime_distance",
        "action",
        "severity",
        "risk_score"
    ]

    available = [c for c in cols if c in adapt.columns]

    if available:
        memory_df = adapt[available].copy()
        memory_df.columns = [c.replace("_", " ").title() for c in available]

        if "Regime Distance" in memory_df.columns:
            memory_df["Regime Distance"] = pd.to_numeric(
                memory_df["Regime Distance"], errors="coerce"
            ).round(4)

        st.dataframe(memory_df, width="stretch", hide_index=True)

st.success(
    "Regime Memory is now rendered as structured data. "
    "Stored historical regimes can be matched against new drift signatures "
    "to support consistent adaptation decisions."
)
