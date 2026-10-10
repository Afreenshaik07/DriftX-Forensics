import streamlit as st
from src.driftx.evaluation.remediation_recommender import recommend_remediation
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="DriftX ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢ Forensics", page_icon="??", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:2rem;}
.title {font-size:36px;font-weight:750;}
.subtitle {color:#8995b1;margin-bottom:25px;}
.card {padding:20px;border-radius:16px;background:#121a2d;border:1px solid #293653;margin-bottom:15px;}
.label {font-size:12px;color:#8995b1;text-transform:uppercase;letter-spacing:1px;}
.value {font-size:27px;font-weight:700;margin-top:6px;}
.stage {padding:18px;border-radius:14px;background:#121a2d;border:1px solid #293653;text-align:center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">?? Cross-Stage Forensics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Trace drift from the first affected stage to downstream model failure.</div>',
    unsafe_allow_html=True
)

path = Path("results/final_forensic_benchmark.csv")

if not path.exists():
    st.error("Forensic benchmark not found.")
    st.stop()

df = pd.read_csv(path)

scenario = st.selectbox(
    "Select incident",
    df["scenario"].astype(str).tolist()
)

row = df[df["scenario"].astype(str) == scenario].iloc[0]

origin = str(row.get("predicted_origin", "unknown")).upper()
confidence = float(row.get("forensic_confidence", 0))
true_cause = str(row.get("true_cause", "UNKNOWN"))

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="card"><div class="label">Detected Origin</div>'
        f'<div class="value">{origin}</div></div>',
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f'<div class="card"><div class="label">Confidence</div>'
        f'<div class="value">{confidence:.1%}</div></div>',
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f'<div class="card"><div class="label">Ground Truth</div>'
        f'<div class="value">{true_cause}</div></div>',
        unsafe_allow_html=True
    )

with c4:
    score = float(row.get("performance_score", 0))
    st.markdown(
        f'<div class="card"><div class="label">Performance Impact</div>'
        f'<div class="value">{score:.3f}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("### Propagation Chain")

stages = [
    ("?", "Feature", "feature_ks"),
    ("?", "Prediction", "prediction_ks"),
    ("?", "Residual", "residual_ks"),
    ("?", "Performance", "performance_score"),
]

cols = st.columns(4)

for col, (number, name, metric) in zip(cols, stages):
    value = float(row.get(metric, 0))

    with col:
        st.markdown(
            f"""
            <div class="stage">
                <div style="font-size:24px;">{number}</div>
                <div style="font-size:18px;font-weight:650;">{name}</div>
                <div style="font-size:25px;font-weight:700;margin-top:8px;">
                    {value:.3f}
                </div>
                <div style="color:#8995b1;font-size:12px;">drift evidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("### Evidence Profile")

evidence = {
    "Feature Drift": float(row.get("feature_ks", 0)),
    "Prediction Drift": float(row.get("prediction_ks", 0)),
    "Residual Drift": float(row.get("residual_ks", 0)),
    "Performance Impact": float(row.get("performance_score", 0)),
}

evidence_df = pd.DataFrame(
    {"Evidence": list(evidence.values())},
    index=list(evidence.keys())
)

st.bar_chart(evidence_df)

st.markdown("### Incident Details")

details = {
    "Scenario": scenario,
    "True Cause": true_cause,
    "Predicted Origin": origin,
    "Confidence": confidence,
    "Change Point": row.get("change_point", "N/A"),
    "Reference MAE": row.get("reference_mae", "N/A"),
    "Current MAE": row.get("current_mae", "N/A"),
    "MAE Change %": row.get("mae_change_pct", "N/A"),
}

st.dataframe(
    pd.DataFrame(
        {"Value": [str(v) for v in details.values()]},
        index=list(details.keys())
    ),
    width="stretch"
)

st.success(
    f"DriftX forensic conclusion: **{origin}** is the most likely "
    f"origin of the observed failure with **{confidence:.1%} confidence**."
)


st.divider()
st.subheader("Recommended Remediation")
origin = st.selectbox(
    "Diagnosed drift origin",
    ["feature", "prediction", "residual", "performance", "unknown", "none"]
)
confidence = st.slider("Diagnosis confidence", 0.0, 1.0, 0.8, 0.05)
recommendation = recommend_remediation(origin, confidence)
st.markdown("### " + recommendation["title"])
st.write(recommendation["reason"])
st.metric("Recommended action", recommendation["action"])
st.write("**Recommended steps**")
for step in recommendation["steps"]:
    st.write("- " + step)
st.info("Verification: " + recommendation["verification"])
st.caption("Recommendation only ? no model changes execute automatically.")
