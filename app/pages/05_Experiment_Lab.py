from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="DriftX — Experiment Lab",
    page_icon="LAB",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

st.markdown("""
<style>
.stApp { background:#f6f8fc; }
.block-container { max-width:1450px; padding:2rem 3rem 4rem; }

.hero {
    background:linear-gradient(135deg,#0b1630,#172b55,#274b7d);
    padding:34px 38px;
    border-radius:24px;
    color:white;
    margin-bottom:25px;
}
.hero h1 { font-size:40px; margin:0; font-weight:850; }
.hero p { color:#b9c8e5; margin:8px 0 0; }

.section {
    font-size:27px;
    font-weight:800;
    margin:30px 0 15px;
    color:#171c2b;
}

.card {
    background:#101a33;
    border-radius:18px;
    padding:23px;
    color:white;
    min-height:125px;
}
.label {
    color:#91a3c8;
    font-size:11px;
    letter-spacing:1.5px;
}
.value {
    font-size:30px;
    font-weight:850;
    margin-top:12px;
}
.sub {
    color:#aebbd5;
    font-size:12px;
    margin-top:6px;
}

.info {
    background:#eaf3ff;
    color:#174e91;
    border-left:5px solid #337de8;
    padding:18px 20px;
    border-radius:10px;
}
.good {
    background:#e8f8ef;
    color:#17683f;
    border-left:5px solid #20a568;
    padding:18px 20px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)


def load(name):
    path = RESULTS / name

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


forensic = load("final_forensic_benchmark.csv")
adaptation = load("autonomous_adaptation_experiment.csv")
final = load("FINAL_RESULTS.csv")
streaming = load("final_streaming_adaptation.csv")


def value(df, names, default=None):
    if "metric" in df.columns and "value" in df.columns:
        lookup = dict(zip(df["metric"], df["value"]))
        for n in names:
            if n in lookup:
                return lookup[n]

    if df.empty:
        return default

    for name in names:

        if name in df.columns:

            try:
                v = df[name].iloc[0]

                if pd.notna(v):
                    return v

            except Exception:
                pass

    return default


def pct(v):

    if v is None:
        return "—"

    try:

        x = float(v)

        if abs(x) <= 1:
            x *= 100

        return f"{x:.1f}%"

    except Exception:
        return str(v)


def num(v):

    if v is None:
        return "—"

    try:
        return f"{float(v):.3f}"

    except Exception:
        return str(v)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
    <h1>Experiment Lab</h1>
    <p>Controlled drift scenarios · forensic benchmarking · adaptation evaluation</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# BENCHMARK SUMMARY
# =========================================================

st.markdown(
    '<div class="section">Benchmark Summary</div>',
    unsafe_allow_html=True
)

forensic_accuracy = value(
    final,
    ["forensic_accuracy", "Forensic Accuracy"]
)

temporal_accuracy = value(
    final,
    ["temporal_baseline_accuracy", "Temporal baseline accuracy"]
)

confidence = value(
    final,
    ["mean_forensic_confidence", "Mean forensic confidence"]
)

improvement = value(
    final,
    ["mae_improvement_pct", "MAE improvement %", "mae_improvement"]
)

engines = value(
    final,
    ["streaming_test_engines", "test_engines"]
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="label">FORENSIC ACCURACY</div>
        <div class="value">{pct(forensic_accuracy)}</div>
        <div class="sub">Controlled benchmark</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="label">TEMPORAL BASELINE</div>
        <div class="value">{pct(temporal_accuracy)}</div>
        <div class="sub">Comparison baseline</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="label">MEAN CONFIDENCE</div>
        <div class="value">{pct(confidence)}</div>
        <div class="sub">Forensic confidence</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
        <div class="label">ADAPTATION GAIN</div>
        <div class="value">{pct(improvement)}</div>
        <div class="sub">Measured MAE improvement</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="card">
        <div class="label">TEST ENGINES</div>
        <div class="value">{num(engines)}</div>
        <div class="sub">Streaming evaluation</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# EXPERIMENT SELECTOR
# =========================================================

st.markdown(
    '<div class="section">Experiment Selector</div>',
    unsafe_allow_html=True
)

experiment = st.selectbox(
    "Choose experiment view",
    [
        "Cross-Stage Forensic Benchmark",
        "Autonomous Adaptation",
        "Streaming Model Adaptation"
    ]
)


# =========================================================
# FORENSIC BENCHMARK
# =========================================================

if experiment == "Cross-Stage Forensic Benchmark":

    st.markdown(
        '<div class="section">Cross-Stage Forensic Benchmark</div>',
        unsafe_allow_html=True
    )

    if forensic.empty:

        st.warning("Forensic benchmark results are not available.")

    else:

        scenario_column = (
            "scenario"
            if "scenario" in forensic.columns
            else forensic.columns[0]
        )

        selected = st.selectbox(
            "Inspect scenario",
            forensic[scenario_column].astype(str).tolist()
        )

        row = forensic[
            forensic[scenario_column].astype(str) == selected
        ].iloc[0]

        origin = str(
            row.get("predicted_origin", "UNKNOWN")
        ).upper()

        truth = str(
            row.get("true_cause", "UNKNOWN")
        ).upper()

        confidence_value = row.get(
            "forensic_confidence",
            row.get("confidence", None)
        )

        f1, f2, f3, f4 = st.columns(4)

        with f1:
            st.metric("Scenario", selected)

        with f2:
            st.metric("Ground Truth", truth)

        with f3:
            st.metric("Diagnosis", origin)

        with f4:
            st.metric(
                "Confidence",
                pct(confidence_value)
            )

        st.markdown("### Evidence Profile")

        evidence = {}

        for column, label in [
            ("feature_ks", "Feature Drift"),
            ("prediction_ks", "Prediction Drift"),
            ("residual_ks", "Residual Drift"),
            ("performance_score", "Performance Impact")
        ]:

            if column in row.index:

                try:
                    evidence[label] = float(row[column])
                except Exception:
                    pass

        if evidence:

            chart = pd.DataFrame(
                {"Evidence": evidence}
            )

            st.bar_chart(chart)

        st.markdown("### Scenario Record")

        record = pd.DataFrame({
            "Metric": [
                str(k)
                for k in row.index
            ],
            "Value": [
                str(v)
                for v in row.values
            ]
        })

        st.dataframe(
            record,
            width="stretch",
            hide_index=True
        )

        st.markdown("""
        <div class="info">
        This experiment evaluates whether DriftX can identify the most likely
        failure origin using evidence across multiple pipeline stages rather
        than relying on a single drift signal.
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# AUTONOMOUS ADAPTATION
# =========================================================

elif experiment == "Autonomous Adaptation":

    st.markdown(
        '<div class="section">Autonomous Adaptation Experiment</div>',
        unsafe_allow_html=True
    )

    if adaptation.empty:

        st.warning("Autonomous adaptation results are not available.")

    else:

        if "action" in adaptation.columns:

            actions = (
                adaptation["action"]
                .astype(str)
                .str.upper()
                .value_counts()
            )

            a1, a2, a3 = st.columns(3)

            with a1:
                st.metric(
                    "MONITOR",
                    int(actions.get("MONITOR", 0))
                )

            with a2:
                st.metric(
                    "UPDATE",
                    int(actions.get("UPDATE", 0))
                )

            with a3:
                st.metric(
                    "RETRAIN",
                    int(actions.get("RETRAIN", 0))
                )

            st.markdown("### Policy Distribution")

            st.bar_chart(actions)

        st.markdown("### Scenario Decisions")

        columns = [
            c for c in [
                "scenario",
                "predicted_origin",
                "action",
                "severity",
                "risk_score",
                "confidence",
                "previous_regime",
                "regime_distance"
            ]
            if c in adaptation.columns
        ]

        if columns:

            table = adaptation[columns].copy()

            rename = {
                "scenario": "Scenario",
                "predicted_origin": "Origin",
                "action": "Action",
                "severity": "Severity",
                "risk_score": "Risk",
                "confidence": "Confidence",
                "previous_regime": "Previous Regime",
                "regime_distance": "Regime Distance"
            }

            table = table.rename(columns=rename)

            st.dataframe(
                table,
                width="stretch",
                hide_index=True
            )

        st.markdown("""
        <div class="good">
        <b>Policy evaluation:</b> DriftX converts forensic evidence into
        explicit adaptation actions instead of treating every drift event
        as an automatic retraining trigger.
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# STREAMING ADAPTATION
# =========================================================

else:

    st.markdown(
        '<div class="section">Streaming Adaptation Experiment</div>',
        unsafe_allow_html=True
    )

    if streaming.empty:

        st.warning("Streaming adaptation results are not available.")

    else:

        row = streaming.iloc[0]

        champion_mae = row.get(
            "Champion MAE",
            row.get("champion_mae", None)
        )

        adapted_mae = row.get(
            "Adapted MAE",
            row.get("adapted_mae", None)
        )

        champion_rmse = row.get(
            "Champion RMSE",
            row.get("champion_rmse", None)
        )

        adapted_rmse = row.get(
            "Adapted RMSE",
            row.get("adapted_rmse", None)
        )

        decision = str(
            row.get("Decision", "UNKNOWN")
        ).upper()

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric(
                "Champion MAE",
                num(champion_mae)
            )

        with s2:
            st.metric(
                "Adapted MAE",
                num(adapted_mae)
            )

        with s3:
            try:
                gain = (
                    (float(champion_mae) - float(adapted_mae))
                    / float(champion_mae)
                ) * 100
                st.metric(
                    "MAE Improvement",
                    f"{gain:.2f}%"
                )
            except Exception:
                st.metric("MAE Improvement", "—")

        with s4:
            st.metric("Safety Decision", decision)

        st.markdown("### Error Comparison")

        comparison = pd.DataFrame({
            "Model": ["Champion", "Adapted"],
            "MAE": [
                float(champion_mae),
                float(adapted_mae)
            ],
            "RMSE": [
                float(champion_rmse),
                float(adapted_rmse)
            ]
        }).set_index("Model")

        st.bar_chart(comparison)

        if decision == "PROMOTE":

            st.markdown("""
            <div class="good">
            <b>Promotion condition satisfied.</b>
            The adapted model achieved lower recorded error than the champion
            in the streaming evaluation.
            </div>
            """, unsafe_allow_html=True)

        elif decision == "REJECT":

            st.warning(
                "The adapted model did not satisfy the recorded promotion condition."
            )


# =========================================================
# EXPERIMENT INTERPRETATION
# =========================================================

st.markdown(
    '<div class="section">Experiment Interpretation</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info">
<b>Why this laboratory exists:</b><br><br>

DriftX is evaluated using controlled scenarios so that the system knows the
actual injected failure condition. This creates a measurable environment for
testing forensic diagnosis, adaptation policy and safety-gated model changes.
<br><br>

The dashboard therefore exposes both the <b>decision</b> and the underlying
<b>experimental evidence</b>.
</div>
""", unsafe_allow_html=True)

st.caption(
    "DriftX-Forensics · Controlled benchmark and adaptation evaluation laboratory"
)
