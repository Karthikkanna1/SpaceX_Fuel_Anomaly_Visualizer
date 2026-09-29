import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils import (
    prepare_dataframe,
    compute_expected_curve,
    detect_anomalies,
    assign_severity,
    generate_anomaly_explanation,
    calculate_risk_score,
    generate_multivariate_telemetry,
    summarize_anomalies,
    detect_multivariate_anomalies,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SpaceX Fuel Burn Anomaly Visualizer",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 SpaceX Fuel Burn Anomaly Visualizer")

st.markdown(
    """
This app visualizes rocket **fuel burn** and flags **anomalies**
by comparing actual burn against an expected (nominal) curve.

- Works with the included sample telemetry or your own CSV.
- Minimum columns: `time_s` and one of:
  `burn_rate_kg_s` or `fuel_mass_kg`.
- Optional: `stage` and `event`.
"""
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header("Settings")

    smoothing_window = st.slider(
        "Smoothing window (seconds)",
        1, 30, 8, 1
    )

    z_thresh = st.slider(
        "Z-score threshold",
        1.0, 5.0, 3.0, 0.1
    )

    iso_frac = st.slider(
        "IsolationForest contamination (fraction)",
        0.0, 0.2, 0.02, 0.01
    )

    use_isolation_forest = st.checkbox(
        "Use IsolationForest (in addition to z-score)",
        True
    )

    show_expected = st.checkbox(
        "Show expected (nominal) curve",
        True
    )

    show_residuals = st.checkbox(
        "Show residuals panel",
        False
    )

    st.divider()

    st.caption(
        "Tip: Tweak thresholds to see how anomaly flags change."
    )


# ============================================================
# LOAD TELEMETRY
# ============================================================

st.subheader("1) Load Telemetry")

uploaded = st.file_uploader(
    "Upload CSV (or use the bundled sample)",
    type=["csv"]
)

if uploaded is None:

    st.info(
        "Using bundled sample data: "
        "`data/sample_spacex_telemetry.csv`"
    )

    df = pd.read_csv(
        "data/sample_spacex_telemetry.csv"
    )

else:

    try:

        df = pd.read_csv(uploaded)

    except Exception as e:

        st.error(
            f"Failed to read CSV: {e}"
        )

        st.stop()


# ============================================================
# DATA PROCESSING
# ============================================================

try:

    df = prepare_dataframe(df)

    df_expected = compute_expected_curve(
        df,
        smoothing_window=int(
            smoothing_window
        )
    )

    df_flagged = detect_anomalies(
        df_expected,
        z_thresh=z_thresh,
        use_iforest=use_isolation_forest,
        contamination=max(
            iso_frac,
            1e-6
        )
    )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    df_flagged = assign_severity(
        df_flagged
    )

    # --------------------------------------------------------
    # Automatic explanation
    # --------------------------------------------------------

    df_flagged["explanation"] = df_flagged.apply(
        generate_anomaly_explanation,
        axis=1
    )

    # --------------------------------------------------------
    # Multivariate telemetry
    # --------------------------------------------------------

    df_flagged = generate_multivariate_telemetry(
        df_flagged
    )

    # --------------------------------------------------------
    # Multivariate anomaly detection
    # --------------------------------------------------------

    df_flagged = detect_multivariate_anomalies(
        df_flagged
    )

    # --------------------------------------------------------
    # Mission risk
    # --------------------------------------------------------

    risk = calculate_risk_score(
        df_flagged
    )

    risk_score = risk["risk_score"]
    risk_level = risk["risk_level"]


except Exception as e:

    st.error(
        f"Data processing failed: {e}"
    )

    st.stop()


# ============================================================
# ANOMALY SUMMARY
# ============================================================

summary = summarize_anomalies(
    df_flagged
)

total_points = len(
    df_flagged
)

total_anomalies = int(
    df_flagged["is_anomaly"].sum()
)


# ============================================================
# SEVERITY COUNTS
# ============================================================

minor_count = int(
    (
        df_flagged["severity"] == "Minor"
    ).sum()
)

moderate_count = int(
    (
        df_flagged["severity"] == "Moderate"
    ).sum()
)

high_count = int(
    (
        df_flagged["severity"] == "High"
    ).sum()
)

critical_count = int(
    (
        df_flagged["severity"] == "Critical"
    ).sum()
)


# ============================================================
# ANOMALY PERCENTAGE
# ============================================================

if total_points > 0:

    anomaly_percentage = (
        total_anomalies
        / total_points
    ) * 100

else:

    anomaly_percentage = 0


# ============================================================
# MISSION HEALTH
# ============================================================

mission_health = 100.0

mission_health -= (
    anomaly_percentage * 2
)

mission_health = max(
    0,
    min(
        100,
        mission_health
    )
)

mission_health = round(
    mission_health
)


# ============================================================
# MISSION STATUS
# ============================================================

if mission_health >= 90:

    mission_status = "🟢 HEALTHY"

elif mission_health >= 70:

    mission_status = "🟡 WARNING"

elif mission_health >= 40:

    mission_status = "🟠 HIGH RISK"

else:

    mission_status = "🔴 CRITICAL"


# ============================================================
# ANOMALY SEVERITY OVERVIEW
# ============================================================

st.subheader(
    "🚨 Anomaly Severity Overview"
)

severity_col1, severity_col2, severity_col3, severity_col4 = st.columns(4)

with severity_col1:

    st.metric(
        "🔵 Minor",
        f"{minor_count:,}"
    )

with severity_col2:

    st.metric(
        "🟡 Moderate",
        f"{moderate_count:,}"
    )

with severity_col3:

    st.metric(
        "🟠 High",
        f"{high_count:,}"
    )

with severity_col4:

    st.metric(
        "🔴 Critical",
        f"{critical_count:,}"
    )


# ============================================================
# CURRENT STAGE
# ============================================================

if (
    "stage" in df_flagged.columns
    and not df_flagged.empty
):

    current_stage = (
        df_flagged.iloc[-1]["stage"]
    )

    try:

        current_stage = int(
            current_stage
        )

        current_stage_text = (
            f"Stage {current_stage}"
        )

    except Exception:

        current_stage_text = str(
            current_stage
        )

else:

    current_stage_text = "Unknown"


# ============================================================
# MISSION HEALTH DASHBOARD
# ============================================================

st.subheader(
    "📊 Mission Health Dashboard"
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Mission Risk",
        f"{risk_score}/100",
        delta=risk_level
    )

with col2:

    st.metric(
        "Mission Health",
        f"{mission_health}/100",
        delta=mission_status
    )

with col3:

    st.metric(
        "Telemetry Points",
        f"{total_points:,}"
    )

with col4:

    st.metric(
        "Detected Anomalies",
        f"{total_anomalies:,}"
    )

with col5:

    st.metric(
        "Current Stage",
        current_stage_text
    )


# ============================================================
# MISSION STATUS MESSAGE
# ============================================================

if mission_health >= 90:

    st.success(
        f"Mission Status: {mission_status} — "
        "Telemetry is mostly within the expected "
        "operating profile."
    )

elif mission_health >= 70:

    st.warning(
        f"Mission Status: {mission_status} — "
        f"{total_anomalies} anomalous telemetry "
        "points detected."
    )

elif mission_health >= 40:

    st.warning(
        f"Mission Status: {mission_status} — "
        "Several abnormal telemetry patterns "
        "require investigation."
    )

else:

    st.error(
        f"Mission Status: {mission_status} — "
        "A significant number of abnormal telemetry "
        "points were detected."
    )


# ============================================================
# RISK BREAKDOWN
# ============================================================

st.subheader(
    "📈 Risk Breakdown"
)

risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

with risk_col1:

    st.metric(
        "Minor Risk Events",
        risk["minor"]
    )

with risk_col2:

    st.metric(
        "Moderate Risk Events",
        risk["moderate"]
    )

with risk_col3:

    st.metric(
        "High Risk Events",
        risk["high"]
    )

with risk_col4:

    st.metric(
        "Critical Risk Events",
        risk["critical"]
    )


# ============================================================
# MULTIVARIATE TELEMETRY
# ============================================================

st.subheader(
    "📡 Multivariate Telemetry"
)

st.caption(
    "Additional telemetry values are simulated/derived "
    "for demonstration purposes and are not real "
    "SpaceX measurements."
)

telemetry_col1, telemetry_col2, telemetry_col3 = st.columns(3)

with telemetry_col1:

    st.metric(
        "🔥 Thrust",
        f"{df_flagged['thrust_kN'].iloc[-1]:.1f} kN"
    )

with telemetry_col2:

    st.metric(
        "🌡️ Temperature",
        f"{df_flagged['temperature_K'].iloc[-1]:.1f} K"
    )

with telemetry_col3:

    st.metric(
        "📊 Pressure",
        f"{df_flagged['pressure_bar'].iloc[-1]:.1f} bar"
    )


telemetry_col4, telemetry_col5, telemetry_col6 = st.columns(3)

with telemetry_col4:

    st.metric(
        "🛰️ Altitude",
        f"{df_flagged['altitude_m'].iloc[-1]:.1f} m"
    )

with telemetry_col5:

    st.metric(
        "🚀 Velocity",
        f"{df_flagged['velocity_m_s'].iloc[-1]:.1f} m/s"
    )

with telemetry_col6:

    st.metric(
        "📈 Acceleration",
        f"{df_flagged['acceleration_m_s2'].iloc[-1]:.2f} m/s²"
    )


# ============================================================
# MULTIVARIATE ANOMALY ANALYSIS
# ============================================================

st.subheader(
    "🔬 Multivariate Anomaly Analysis"
)

multivariate_anomalies = int(
    df_flagged[
        "multivariate_anomaly"
    ].sum()
)

normal_multivariate = (
    total_points
    - multivariate_anomalies
)

multi_col1, multi_col2, multi_col3 = st.columns(3)

with multi_col1:

    st.metric(
        "Multivariate Anomalies",
        f"{multivariate_anomalies:,}"
    )

with multi_col2:

    st.metric(
        "Normal Telemetry",
        f"{normal_multivariate:,}"
    )

with multi_col3:

    if total_points > 0:

        anomaly_rate = (
            multivariate_anomalies
            / total_points
        ) * 100

    else:

        anomaly_rate = 0

    st.metric(
        "Multivariate Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )

st.caption(
    "A multivariate anomaly is detected when at least "
    "two telemetry parameters show abnormal behavior "
    "at the same time."
)


# ============================================================
# MULTIVARIATE ANOMALY DETAILS
# ============================================================

if multivariate_anomalies > 0:

    st.markdown(
        "### 🚨 Detected Multivariate Events"
    )

    multivariate_details = df_flagged[
        df_flagged[
            "multivariate_anomaly"
        ]
    ][
        [
            "time_s",
            "stage",
            "multivariate_anomaly_count",
            "abnormal_parameters"
        ]
    ].copy()

    multivariate_details.columns = [
        "Time (s)",
        "Stage",
        "Abnormal Parameter Count",
        "Abnormal Parameters"
    ]

    st.dataframe(
        multivariate_details,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No simultaneous abnormal behavior was "
        "detected across multiple telemetry parameters."
    )


# ============================================================
# ANOMALY TIMELINE
# ============================================================

st.subheader(
    "⏱️ Mission Anomaly Timeline"
)

timeline_df = df_flagged[
    df_flagged["is_anomaly"]
].copy()

if not timeline_df.empty:

    timeline_df["timeline_label"] = (
        timeline_df["severity"]
        + " Anomaly"
    )

    fig_timeline = go.Figure()

    severity_levels = [
        "Minor",
        "Moderate",
        "High",
        "Critical"
    ]

    for severity in severity_levels:

        severity_data = timeline_df[
            timeline_df["severity"] == severity
        ]

        if not severity_data.empty:

            custom_data = severity_data[
                [
                    "stage",
                    "deviation_%",
                    "zscore_stage",
                    "explanation"
                ]
            ].to_numpy()

            fig_timeline.add_trace(
                go.Scatter(
                    x=severity_data["time_s"],

                    y=[
                        severity
                    ] * len(
                        severity_data
                    ),

                    mode="markers",

                    name=severity,

                    marker=dict(
                        size=12,
                        symbol="circle"
                    ),

                    customdata=custom_data,

                    hovertemplate=(
                        "<b>%{y} Anomaly</b><br>"
                        "Time: %{x:.1f} s<br>"
                        "Stage: %{customdata[0]}<br>"
                        "Fuel Deviation: "
                        "%{customdata[1]:.1f}%<br>"
                        "Z-score: "
                        "%{customdata[2]:.2f}<br>"
                        "Explanation: "
                        "%{customdata[3]}"
                        "<extra></extra>"
                    )
                )
            )

    fig_timeline.update_layout(
        title=(
            "Anomaly Events Across "
            "Mission Timeline"
        ),

        xaxis_title=(
            "Mission Time (seconds)"
        ),

        yaxis_title="Severity",

        height=450,

        hovermode="closest",

        showlegend=True
    )

    st.plotly_chart(
        fig_timeline,
        use_container_width=True
    )

else:

    st.success(
        "No fuel-burn anomalies detected "
        "during the mission."
    )


# ============================================================
# REAL-TIME TELEMETRY SIMULATION
# ============================================================

st.subheader(
    "📡 Real-Time Telemetry Simulation"
)

st.caption(
    "Simulated telemetry stream for demonstration "
    "purposes. Values are not real-time SpaceX data."
)


# ============================================================
# SIMULATION CONTROLS
# ============================================================

sim_col1, sim_col2 = st.columns(2)

with sim_col1:

    simulation_points = st.slider(
        "Number of telemetry points",
        min_value=10,
        max_value=100,
        value=30,
        step=10
    )

with sim_col2:

    start_simulation = st.button(
        "▶️ Start Telemetry Simulation"
    )


# ============================================================
# IMPORTANT:
# INITIALIZE COUNTERS BEFORE THE IF BLOCK
# ============================================================

normal_count = 0
warning_count = 0
critical_count = 0

live_telemetry = []


# ============================================================
# START SIMULATION
# ============================================================

if start_simulation:

    st.info(
        f"Simulation started with "
        f"{simulation_points} telemetry points."
    )

    # --------------------------------------------------------
    # Live metric columns
    # --------------------------------------------------------

    live_col1, live_col2, live_col3 = st.columns(3)

    live_col4, live_col5, live_col6 = st.columns(3)

    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    progress_bar = st.progress(0)

    # ========================================================
    # GENERATE TELEMETRY
    # ========================================================

    for i in range(
        simulation_points
    ):

        # ----------------------------------------------------
        # Mission time
        # ----------------------------------------------------

        current_time = i

        # ----------------------------------------------------
        # Normal simulated fuel burn
        # ----------------------------------------------------

        fuel_burn = (
            1200
            + np.random.normal(
                0,
                25
            )
        )

        # ----------------------------------------------------
        # CONTROLLED ABNORMAL EVENTS
        # ----------------------------------------------------
        # These are artificial events created only for
        # demonstration/testing of the anomaly detector.
        # They are NOT real SpaceX measurements.
        # ----------------------------------------------------

        if simulation_points >= 10:

            # Warning event
            if i == int(
                simulation_points * 0.30
            ):

                fuel_burn = 1280

            # Critical event
            elif i == int(
                simulation_points * 0.60
            ):

                fuel_burn = 1350

            # Warning event
            elif i == int(
                simulation_points * 0.80
            ):

                fuel_burn = 1285

        # ----------------------------------------------------
        # Thrust
        # ----------------------------------------------------

        thrust = (
            1050
            + np.random.normal(
                0,
                20
            )
        )

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        temperature = (
            330
            + np.random.normal(
                0,
                3
            )
        )

        # ----------------------------------------------------
        # Pressure
        # ----------------------------------------------------

        pressure = (
            52
            + np.random.normal(
                0,
                1
            )
        )

        # ----------------------------------------------------
        # Altitude
        # ----------------------------------------------------

        altitude = max(
            0,
            0.5 * current_time ** 2
        )

        # ----------------------------------------------------
        # Velocity
        # ----------------------------------------------------

        velocity = max(
            0,
            5 * current_time
        )

        # ----------------------------------------------------
        # Acceleration
        # ----------------------------------------------------

        acceleration = (
            5
            + np.random.normal(
                0,
                0.1
            )
        )

        # ====================================================
        # LIVE ANOMALY DETECTION
        # ====================================================

        fuel_deviation = (
            abs(
                fuel_burn - 1200
            )
            / 1200
        ) * 100

        if fuel_deviation > 10:

            live_status = "🔴 CRITICAL"

            critical_count += 1

        elif fuel_deviation > 5:

            live_status = "🟠 WARNING"

            warning_count += 1

        else:

            live_status = "🟢 NORMAL"

            normal_count += 1

        # ====================================================
        # STORE TELEMETRY
        # ====================================================

        live_telemetry.append({

            "time_s":
                current_time,

            "burn_rate_kg_s":
                fuel_burn,

            "thrust_kN":
                thrust,

            "temperature_K":
                temperature,

            "pressure_bar":
                pressure,

            "altitude_m":
                altitude,

            "velocity_m_s":
                velocity,

            "acceleration_m_s2":
                acceleration
        })

        # ====================================================
        # LIVE MISSION STATUS
        # ====================================================

        st.markdown(
            f"### Live Mission Status: "
            f"{live_status}"
        )

        # ====================================================
        # LIVE TELEMETRY DISPLAY
        # ====================================================

        with live_col1:

            st.metric(
                "🔥 Live Fuel Burn",
                f"{fuel_burn:.1f} kg/s"
            )

        with live_col2:

            st.metric(
                "🚀 Live Thrust",
                f"{thrust:.1f} kN"
            )

        with live_col3:

            st.metric(
                "🌡️ Live Temperature",
                f"{temperature:.1f} K"
            )

        with live_col4:

            st.metric(
                "📊 Live Pressure",
                f"{pressure:.1f} bar"
            )

        with live_col5:

            st.metric(
                "🛰️ Live Altitude",
                f"{altitude:.1f} m"
            )

        with live_col6:

            st.metric(
                "📈 Live Velocity",
                f"{velocity:.1f} m/s"
            )

        # ====================================================
        # PROGRESS
        # ====================================================

        progress_bar.progress(
            (i + 1)
            / simulation_points
        )


    # ========================================================
    # SIMULATION COMPLETE
    # ========================================================

    st.success(
        "✅ Real-time telemetry simulation completed."
    )


# ============================================================
# LIVE EVENT SUMMARY
# ============================================================
# This is outside the simulation block, but counters were
# initialized above, so there is NO NameError.
# ============================================================

if start_simulation:

    st.subheader(
        "📊 Live Simulation Event Summary"
    )

    alert_count = (
        warning_count
        + critical_count
    )

    alert_rate = (
        alert_count
        / simulation_points
    ) * 100

    alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)

    with alert_col1:

        st.metric(
            "🟢 Normal Events",
            normal_count
        )

    with alert_col2:

        st.metric(
            "🟠 Warning Events",
            warning_count
        )

    with alert_col3:

        st.metric(
            "🔴 Critical Events",
            critical_count
        )

    with alert_col4:

        st.metric(
            "🚨 Alert Rate",
            f"{alert_rate:.1f}%"
        )


# ============================================================
# FUEL BURN GRAPH
# ============================================================

st.subheader(
    "2) Fuel Burn Over Time"
)

fig = go.Figure()


# ============================================================
# ACTUAL BURN
# ============================================================

fig.add_trace(
    go.Scatter(
        x=df_flagged["time_s"],

        y=df_flagged["burn_rate_kg_s"],

        mode="lines",

        name="Actual burn rate"
    )
)


# ============================================================
# EXPECTED BURN
# ============================================================

if show_expected:

    fig.add_trace(
        go.Scatter(
            x=df_flagged[
                "time_s"
            ],

            y=df_flagged[
                "expected_burn_rate"
            ],

            mode="lines",

            name="Expected (nominal)"
        )
    )


# ============================================================
# ANOMALY POINTS
# ============================================================

anom = df_flagged[
    df_flagged[
        "is_anomaly"
    ] == True
]

fig.add_trace(
    go.Scatter(
        x=anom["time_s"],

        y=anom["burn_rate_kg_s"],

        mode="markers",

        name="Anomalies",

        marker=dict(
            size=8,
            symbol="x"
        )
    )
)


# ============================================================
# STAGE SHADING
# ============================================================

if (
    "stage" in df_flagged.columns
    and
    df_flagged[
        "stage"
    ].nunique() > 1
):

    for stage_val in sorted(
        df_flagged[
            "stage"
        ].dropna().unique()
    ):

        stage_df = df_flagged[
            df_flagged[
                "stage"
            ] == stage_val
        ]

        if stage_df.empty:

            continue

        x0 = stage_df[
            "time_s"
        ].min()

        x1 = stage_df[
            "time_s"
        ].max()

        fig.add_vrect(
            x0=x0,

            x1=x1,

            fillcolor=None,

            line_width=1,

            annotation_text=(
                f"Stage {int(stage_val)}"
            ),

            annotation_position=(
                "top left"
            ),

            layer="below"
        )


# ============================================================
# EVENT MARKERS
# ============================================================

if "event" in df_flagged.columns:

    max_y = (
        float(
            np.nanmax(
                df_flagged[
                    "burn_rate_kg_s"
                ]
            )
        )
        if len(df_flagged)
        else 0.0
    )

    for _, r in df_flagged.dropna(
        subset=["event"]
    ).iterrows():

        fig.add_vline(
            x=float(
                r["time_s"]
            ),

            line_dash="dot"
        )

        fig.add_annotation(
            x=float(
                r["time_s"]
            ),

            y=max_y,

            text=str(
                r["event"]
            ),

            showarrow=False,

            yshift=10
        )


# ============================================================
# GRAPH LAYOUT
# ============================================================

fig.update_layout(

    xaxis_title="Time (s)",

    yaxis_title="Burn rate (kg/s)",

    legend_title_text="Series",

    height=520
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# RESIDUALS
# ============================================================

if show_residuals:

    st.subheader(
        "Residuals (Actual - Expected)"
    )

    res_fig = go.Figure()

    res_fig.add_trace(
        go.Scatter(

            x=df_flagged[
                "time_s"
            ],

            y=df_flagged[
                "residual"
            ],

            mode="lines",

            name="Residual"
        )
    )

    res_fig.add_hline(
        y=0,

        line_dash="dash"
    )

    st.plotly_chart(
        res_fig,
        use_container_width=True
    )


# ============================================================
# ANOMALY TABLE
# ============================================================

st.subheader(
    "3) Anomalies Table"
)

st.dataframe(
    summary,
    use_container_width=True
)


# ============================================================
# DOWNLOAD ANOMALY REPORT
# ============================================================

st.download_button(

    label="⬇️ Download anomalies as CSV",

    data=summary.to_csv(
        index=False
    ).encode("utf-8"),

    file_name="anomalies_summary.csv",

    mime="text/csv"
)


# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader(
    "4) Data Preview"
)

st.dataframe(
    df_flagged.head(100),
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Built with ❤️ using "
    "Streamlit + Plotly + scikit-learn"
)