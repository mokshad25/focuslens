"""
FocusLens — dashboard.py
=========================
Main Streamlit application entry point.

Pages:
  1. Overview      — today's split, weekly bar, streak
  2. Focus Patterns — heatmap, trend, peak window label
  3. Distraction   — top domains, spirals, context switches
  4. AI Report     — LLM-generated coaching report

Run:
    streamlit run dashboard.py
    streamlit run dashboard.py -- --demo   (demo mode flag passed after --)
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.database import (
    init_database,
    query_daily_summary,
    query_hourly_features,
    query_cluster_assignments,
    query_raw_visits,
    query_distraction_spirals,
    get_cached_report,
    save_ai_report,
    save_domain_override,
    get_domain_overrides,
)
from src.visualizer import (
    chart_today_pie,
    chart_weekly_stacked_bar,
    chart_top_distraction_domains,
    chart_context_switches,
    chart_distraction_spirals,
    chart_pca_scatter,
    chart_elbow_curve,
    chart_cluster_distribution,
    chart_anomaly_calendar,
    chart_week_comparison,
)
from src.insights_generator import build_weekly_summary, generate_report
from pipeline import run_pipeline

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise in dashboard

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FocusLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — modern SaaS dashboard theme ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  :root {
    --bg: #0f1014;
    --surface: #16181d;
    --surface-2: #1d2027;
    --surface-3: #242833;
    --border: #2b303b;
    --border-soft: rgba(255,255,255,0.07);
    --text: #f4f6f8;
    --muted: #9aa3b2;
    --muted-2: #6f7887;
    --accent: #14b8a6;
    --accent-soft: rgba(20,184,166,0.12);
    --danger: #f87171;
    --danger-soft: rgba(248,113,113,0.11);
    --warning: #fbbf24;
    --shadow: 0 18px 55px rgba(0,0,0,0.28);
    --radius: 10px;
  }

  /* App shell */
  html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: radial-gradient(circle at top left, rgba(20,184,166,0.08), transparent 30%),
                linear-gradient(180deg, #111216 0%, var(--bg) 42%, #0b0c0f 100%);
    color: var(--text);
  }
  .stAppHeader,
  header[data-testid="stHeader"] {
    background: transparent;
  }
  #MainMenu,
  footer,
  [data-testid="stToolbar"],
  [data-testid="stDecoration"] {
    visibility: hidden;
    height: 0;
  }
  .main .block-container {
    max-width: 1420px;
    padding: 2rem 2.4rem 4rem;
  }
  [data-testid="stVerticalBlock"] {
    gap: 1rem;
  }
  [data-testid="stHorizontalBlock"] {
    gap: 1rem;
    align-items: stretch;
  }
  hr {
    border: none;
    height: 1px;
    background: var(--border-soft);
    margin: 1.4rem 0;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #15171c 0%, #101116 100%);
    border-right: 1px solid var(--border-soft);
    box-shadow: 18px 0 55px rgba(0,0,0,0.22);
  }
  [data-testid="stSidebar"] > div:first-child {
    padding: 1.4rem 1rem 2rem;
  }
  [data-testid="stSidebar"] h2 {
    border: 0;
    margin: 0 0 0.15rem;
    padding: 0;
    color: var(--text);
    font-size: 1.2rem;
    letter-spacing: 0;
  }
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stCaption {
    color: var(--muted);
    font-size: 0.82rem;
  }
  [data-testid="stSidebar"] [role="radiogroup"] {
    gap: 0.4rem;
  }
  [data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.55rem 0.65rem;
    transition: background 140ms ease, border-color 140ms ease, color 140ms ease;
  }
  [data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.045);
    border-color: var(--border-soft);
  }
  [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: var(--accent-soft);
    border-color: rgba(20,184,166,0.35);
    color: var(--text);
  }
  [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
    display: none;
  }
  [data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
    color: inherit;
    font-weight: 650;
    line-height: 1.2;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
    color: var(--text);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Typography */
  h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    letter-spacing: 0;
  }
  h1 {
    color: var(--text);
    font-size: clamp(1.85rem, 2vw, 2.35rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 0.45rem;
  }
  h2 {
    color: var(--text);
    border-bottom: 1px solid var(--border-soft);
    font-size: 1.02rem;
    font-weight: 700;
    margin: 1.65rem 0 0.75rem;
    padding-bottom: 0.55rem;
  }
  h3 {
    color: var(--text);
    font-size: 0.96rem;
    font-weight: 700;
    margin-top: 1.15rem;
  }
  p, li, label, .stMarkdown, .stCaption {
    color: var(--muted);
  }
  .stCaption {
    font-size: 0.82rem;
  }

  /* Cards and chart frames */
  [data-testid="stMetric"],
  [data-testid="stPlotlyChart"],
  [data-testid="stDataFrame"],
  [data-testid="stJson"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.018));
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    box-shadow: 0 12px 32px rgba(0,0,0,0.16);
  }
  [data-testid="stPlotlyChart"] {
    padding: 0.75rem 0.8rem 0.55rem;
  }
  [data-testid="stMetric"] {
    min-height: 118px;
    padding: 1rem 1rem 0.9rem;
  }
  [data-testid="stMetric"] label {
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    line-height: 1.35;
    text-transform: uppercase;
  }
  [data-testid="stMetricValue"] {
    color: var(--text);
    font-size: clamp(1.45rem, 2.2vw, 2rem);
    font-weight: 800;
    line-height: 1.1;
    padding-top: 0.35rem;
  }
  [data-testid="stMetricDelta"] {
    color: var(--muted);
    font-size: 0.78rem;
  }

  /* Buttons */
  .stButton > button {
    min-height: 2.55rem;
    background: linear-gradient(180deg, #242833 0%, #1d2027 100%);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.16);
    font-family: 'Inter', sans-serif;
    font-size: 0.86rem;
    font-weight: 700;
    padding: 0.62rem 1rem;
    transition: transform 140ms ease, box-shadow 140ms ease, background 140ms ease, border-color 140ms ease;
  }
  .stButton > button:hover {
    background: linear-gradient(180deg, #14b8a6 0%, #0f9f92 100%);
    border-color: rgba(20,184,166,0.65);
    color: #061110;
    box-shadow: 0 14px 28px rgba(20,184,166,0.18);
    transform: translateY(-1px);
  }
  .stButton > button:active {
    transform: translateY(0);
  }
  .stButton > button:focus:not(:active) {
    border-color: rgba(20,184,166,0.8);
    box-shadow: 0 0 0 3px rgba(20,184,166,0.16);
  }

  /* Inputs */
  [data-baseweb="input"],
  [data-baseweb="select"] > div,
  [data-baseweb="base-input"],
  [data-baseweb="datepicker"] input {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    min-height: 2.55rem;
  }
  input,
  textarea,
  [data-baseweb="select"] span {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
  }
  input::placeholder {
    color: var(--muted-2) !important;
  }
  [data-baseweb="slider"] div {
    color: var(--muted);
  }
  [data-baseweb="slider"] [role="slider"] {
    background-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(20,184,166,0.16);
  }

  /* Alerts and expanders */
  .stAlert {
    border-radius: var(--radius);
    border: 1px solid var(--border-soft);
    box-shadow: 0 12px 32px rgba(0,0,0,0.15);
  }
  [data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 0 !important;
    color: var(--text) !important;
    font-weight: 700;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    padding: 0.25rem;
    gap: 0.25rem;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: var(--muted);
    font-family: 'Inter', sans-serif;
    font-size: 0.86rem;
    font-weight: 700;
  }
  .stTabs [aria-selected="true"] {
    background: var(--surface-3) !important;
    color: var(--text) !important;
  }

  /* Focus score badge */
  .focus-badge {
    display: inline-block;
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .badge-excellent { background: rgba(20,184,166,0.14); color: #5eead4; border: 1px solid rgba(20,184,166,0.42); }
  .badge-good      { background: rgba(251,191,36,0.12); color: #fcd34d; border: 1px solid rgba(251,191,36,0.38); }
  .badge-average   { background: rgba(248,113,113,0.12); color: #fca5a5; border: 1px solid rgba(248,113,113,0.38); }
  .badge-poor      { background: rgba(154,163,178,0.12); color: var(--muted); border: 1px solid rgba(154,163,178,0.34); }

  /* AI report text */
  .ai-report {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid var(--border-soft);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 1.25rem 1.4rem;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #d7dce5;
    white-space: pre-wrap;
  }

  /* Distraction warning */
  .dist-warning {
    background: var(--danger-soft);
    border: 1px solid rgba(248,113,113,0.28);
    border-radius: var(--radius);
    padding: 0.9rem 1rem;
    color: #fca5a5;
    font-size: 0.88rem;
    margin: 0.6rem 0 1rem;
  }

  /* Spiral cards */
  .spiral-card {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-left: 3px solid var(--danger);
    border-radius: var(--radius);
    box-shadow: 0 10px 24px rgba(0,0,0,0.13);
    padding: 0.85rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.83rem;
    color: #d7dce5;
  }

  /* Dataframes */
  [data-testid="stDataFrame"] {
    overflow: hidden;
  }
  [data-testid="stDataFrame"] div {
    font-family: 'Inter', sans-serif;
  }

  @media (max-width: 900px) {
    .main .block-container {
      padding: 1.2rem 1rem 3rem;
    }
    [data-testid="stMetric"] {
      min-height: 104px;
    }
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_all_data():
    """Load all tables from SQLite (cached for 5 minutes)."""
    try:
        init_database()
        daily    = query_daily_summary()
        hourly   = query_hourly_features()
        clusters = query_cluster_assignments()
        raw      = query_raw_visits()
        spirals  = query_distraction_spirals()
        return daily, hourly, clusters, raw, spirals
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def focus_badge(score: float) -> str:
    if score >= config.FOCUS_SCORE_THRESHOLDS["excellent"]:
        return f'<span class="focus-badge badge-excellent">Excellent</span>'
    elif score >= config.FOCUS_SCORE_THRESHOLDS["good"]:
        return f'<span class="focus-badge badge-good">Good</span>'
    elif score >= config.FOCUS_SCORE_THRESHOLDS["average"]:
        return f'<span class="focus-badge badge-average">Average</span>'
    return f'<span class="focus-badge badge-poor">Needs Work</span>'


def compute_streak(daily_df: pd.DataFrame, threshold: float = 50) -> int:
    """Count consecutive recent days with focus_score >= threshold."""
    if daily_df.empty or "focus_score" not in daily_df.columns:
        return 0
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=False)
    streak = 0
    for _, row in df.iterrows():
        if row["focus_score"] >= threshold:
            streak += 1
        else:
            break
    return streak


def filter_date_range(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return only the last `days` rows by date column."""
    if df.empty or "date" not in df.columns:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].reset_index(drop=True)


def _safe_val(df, col, default=0):
    """Safely get last value of a column from a DataFrame."""
    if df.empty or col not in df.columns:
        return default
    return df[col].iloc[-1]


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔬 FocusLens")
        st.markdown('<p style="color:#9aa3b2;font-size:0.78rem;margin-top:-10px;">Personal Productivity Analytics</p>', unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["📊 Overview", "🎯 Focus Patterns", "😵 Distraction Analysis",
             "✨ AI Report"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**⚙️ Settings**")

        lookback = st.selectbox("Lookback window", [7, 14, 30], index=2, format_func=lambda x: f"Last {x} days")

        dist_limit = st.slider(
            "Daily distraction limit (min)",
            min_value=15, max_value=180,
            value=config.DAILY_DISTRACTION_LIMIT_MINUTES, step=15,
        )

        st.markdown("---")
        st.markdown("**🔄 Pipeline**")

        col1, col2 = st.columns(2)
        with col1:
            run_real = st.button("Run (Real)", use_container_width=True)
        with col2:
            run_demo = st.button("Run (Demo)", use_container_width=True)

        if run_real:
            with st.spinner("Running pipeline…"):
                run_pipeline(use_demo=False)
            st.cache_data.clear()
            st.success("Done ✓")
            st.rerun()

        if run_demo:
            with st.spinner("Generating demo data…"):
                run_pipeline(use_demo=True)
            st.cache_data.clear()
            st.success("Done ✓")
            st.rerun()

        st.markdown("---")
        st.markdown("**➕ Custom Domain**")
        custom_domain = st.text_input("Domain (e.g. notion.so)", placeholder="example.com")
        custom_cat = st.selectbox("Category", ["productive", "distraction", "neutral"])
        if st.button("Add Domain", use_container_width=True):
            if custom_domain.strip():
                save_domain_override(custom_domain.strip(), custom_cat)
                st.success(f"Saved: {custom_domain} → {custom_cat}")
            else:
                st.warning("Enter a domain name first.")

        st.markdown("---")
        st.caption("🔒 100% local · No data leaves your machine")

    return page, lookback, dist_limit


# ── Page 1: Overview ──────────────────────────────────────────────────────────

def page_overview(daily_df, raw_df, lookback):
    st.markdown("# 📊 Overview")

    if daily_df.empty:
        st.info("No data yet. Click **Run (Demo)** in the sidebar to load sample data.")
        return

    recent = filter_date_range(daily_df, lookback)
    today  = recent.iloc[-1] if not recent.empty else None

    # ── KPI row ───────────────────────────────────────────────────────────
    streak = compute_streak(recent)
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        score = float(today["focus_score"]) if today is not None else 0
        st.metric("Today's Focus Score", f"{score:.0f}/100")
    with c2:
        prod = float(today["productive_minutes"]) if today is not None else 0
        st.metric("Productive Today", f"{prod:.0f} min")
    with c3:
        dist = float(today["distraction_minutes"]) if today is not None else 0
        delta_color = "inverse"
        st.metric("Distraction Today", f"{dist:.0f} min")
    with c4:
        st.metric("Focus Streak", f"{streak} days 🔥")
    with c5:
        avg_score = recent["focus_score"].mean() if not recent.empty else 0
        st.metric(f"{lookback}-day Avg Score", f"{avg_score:.0f}/100")

    # Badge
    if today is not None:
        st.markdown(focus_badge(float(today["focus_score"])), unsafe_allow_html=True)

    # Distraction limit warning
    if today is not None and float(today["distraction_minutes"]) > config.DAILY_DISTRACTION_LIMIT_MINUTES:
        excess = float(today["distraction_minutes"]) - config.DAILY_DISTRACTION_LIMIT_MINUTES
        st.markdown(
            f'<div class="dist-warning">⚠️  Distraction limit exceeded by <strong>{excess:.0f} min</strong> today. '
            f'Limit is set to {config.DAILY_DISTRACTION_LIMIT_MINUTES} min/day.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Charts row ────────────────────────────────────────────────────────
    col_pie, col_bar = st.columns([1, 2])
    with col_pie:
        st.markdown("## Today's Split")
        st.plotly_chart(chart_today_pie(daily_df), use_container_width=True, config={"displayModeBar": False})
    with col_bar:
        st.markdown("## Weekly Breakdown")
        st.plotly_chart(chart_weekly_stacked_bar(recent.tail(14)), use_container_width=True, config={"displayModeBar": False})

    # ── Week comparison ───────────────────────────────────────────────────
    if len(daily_df) >= 14:
        st.markdown("## This Week vs Last Week")
        st.plotly_chart(chart_week_comparison(daily_df), use_container_width=True, config={"displayModeBar": False})


# ── Page 2: Focus Patterns ────────────────────────────────────────────────────

def page_focus_patterns(daily_df, hourly_df, raw_df, lookback):
    st.markdown("# 🎯 Focus Patterns")

    if hourly_df.empty:
        st.info("No hourly data. Run the pipeline first.")
        return

    recent_hourly = filter_date_range(hourly_df, lookback)
    recent_daily  = filter_date_range(daily_df, lookback)

    # ── Peak focus window ──────────────────────────────────────────────────
    if not recent_hourly.empty:
        best_hour_row = recent_hourly.groupby("hour")["productive_ratio"].mean().idxmax()
        best_dow_row  = recent_hourly.groupby("day_of_week")["productive_ratio"].mean().idxmax()
        day_map = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hour_label = f"{best_hour_row:02d}:00 – {(best_hour_row+1)%24:02d}:00"
        day_label  = day_map[int(best_dow_row)]

        st.markdown(
            f'<div style="background:linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02));'
            f'border:1px solid rgba(255,255,255,0.07);border-left:3px solid #14b8a6;'
            f'border-radius:10px;padding:16px 20px;margin-bottom:18px;box-shadow:0 12px 32px rgba(0,0,0,0.16);">'
            f'<span style="color:#9aa3b2;font-size:0.76rem;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;">Peak Focus Window</span><br>'
            f'<span style="font-family:Inter,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;font-size:1.12rem;color:#5eead4;font-weight:800;">'
            f'{day_label} · {hour_label}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Productive vs distraction by hour (bar) ───────────────────────────
    st.markdown("## Average Activity by Hour")
    if not recent_hourly.empty:
        hour_agg = recent_hourly.groupby("hour")[["productive_ratio", "distraction_ratio"]].mean().reset_index()
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_bar(x=hour_agg["hour"], y=hour_agg["productive_ratio"],
                    name="Productive", marker_color="#14b8a6",
                    hovertemplate="Hour %{x}:00<br>%{y:.1%}<extra></extra>")
        fig.add_bar(x=hour_agg["hour"], y=hour_agg["distraction_ratio"],
                    name="Distraction", marker_color="#FF6B6B",
                    hovertemplate="Hour %{x}:00<br>%{y:.1%}<extra></extra>")
        fig.update_layout(
            barmode="group",
            xaxis=dict(title="Hour of Day", tickmode="linear", dtick=2),
            yaxis=dict(title="Ratio", tickformat=".0%"),
            paper_bgcolor="#0f1014", plot_bgcolor="#0f1014",
            font=dict(family="Inter, sans-serif", color="#f4f6f8"),
            legend=dict(bgcolor="#16181d", bordercolor="#2b303b", borderwidth=1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page 3: Distraction Analysis ──────────────────────────────────────────────

def page_distraction(raw_df, hourly_df, spirals_df, daily_df, lookback, dist_limit):
    st.markdown("# 😵 Distraction Analysis")

    if raw_df.empty:
        st.info("No visit data. Run the pipeline first.")
        return

    recent_raw     = raw_df.copy()
    recent_hourly  = filter_date_range(hourly_df, lookback)
    recent_daily   = filter_date_range(daily_df,  lookback)

    if "visit_time" in recent_raw.columns:
        recent_raw["visit_time"] = pd.to_datetime(recent_raw["visit_time"])
        cutoff = recent_raw["visit_time"].max() - pd.Timedelta(days=lookback)
        recent_raw = recent_raw[recent_raw["visit_time"] >= cutoff]

    # ── Days over limit ────────────────────────────────────────────────────
    if not recent_daily.empty and "distraction_minutes" in recent_daily.columns:
        over_days = (recent_daily["distraction_minutes"] > dist_limit).sum()
        if over_days > 0:
            st.markdown(
                f'<div class="dist-warning">⚠️  You exceeded the {dist_limit}-min distraction limit on '
                f'<strong>{over_days} of {len(recent_daily)} days</strong> in this period.</div>',
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("## Top Distraction Sites")
        st.plotly_chart(chart_top_distraction_domains(recent_raw), use_container_width=True)

    with col2:
        st.markdown("## Context Switches by Hour")
        st.plotly_chart(chart_context_switches(recent_hourly), use_container_width=True)

    # ── Distraction spirals ────────────────────────────────────────────────
    st.markdown("## Distraction Spirals")
    st.caption("A spiral = 3+ consecutive distraction visits. The longer the bar, the worse the spiral.")

    recent_spirals = spirals_df.copy()
    if not recent_spirals.empty and "start_time" in recent_spirals.columns:
        recent_spirals["start_time"] = pd.to_datetime(recent_spirals["start_time"])
        cutoff = recent_spirals["start_time"].max() - pd.Timedelta(days=lookback)
        recent_spirals = recent_spirals[recent_spirals["start_time"] >= cutoff]

    st.plotly_chart(chart_distraction_spirals(recent_spirals), use_container_width=True)

    # Spiral detail cards
    if not recent_spirals.empty:
        st.markdown("### Spiral Details")
        for _, row in recent_spirals.sort_values("duration_minutes", ascending=False).head(10).iterrows():
            st.markdown(
                f'<div class="spiral-card">'
                f'<strong>{row.get("date", "")}</strong> · '
                f'{str(row.get("start_time", ""))[:16]} → {str(row.get("end_time", ""))[:16]}<br>'
                f'⏱ {row.get("duration_minutes", 0):.0f} min · {row.get("visit_count", 0)} visits<br>'
                f'🌐 {row.get("domains", "")}</div>',
                unsafe_allow_html=True,
            )


# ── Page 4: AI Report ─────────────────────────────────────────────────────────

def page_ai_report(daily_df, raw_df, hourly_df):
    st.markdown("# ✨ AI Productivity Report")
    st.caption("Generated by Claude. Only aggregated metrics are sent — no URLs or personal data.")

    if daily_df.empty:
        st.info("No data yet. Run the pipeline first.")
        return

    # ── Week selector ──────────────────────────────────────────────────────
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    min_date = daily_df["date"].min().date()
    max_date = daily_df["date"].max().date()

    week_start = st.date_input(
        "Analyse week starting:",
        value=max_date - timedelta(days=6),
        min_value=min_date,
        max_value=max_date,
    )

    week_start_str = str(week_start)
    week_end_str   = str(week_start + timedelta(days=6))

    # ── Build summary ──────────────────────────────────────────────────────
    summary = build_weekly_summary(daily_df, raw_df, hourly_df, week_start=week_start_str)

    # Show summary stats
    with st.expander("📊 Summary data sent to AI (privacy-safe)"):
        st.json(summary)

    # ── Cached or generate report ──────────────────────────────────────────
    cached = get_cached_report(week_start_str)

    col_gen, col_dry = st.columns([1, 1])
    with col_gen:
        generate_btn = st.button("🪄 Generate / Regenerate Report", use_container_width=True)
    with col_dry:
        dryrun_btn = st.button("⚡ Quick Report (No API)", use_container_width=True)

    report_text = None

    if generate_btn:
        with st.spinner("Asking Claude for your productivity coaching…"):
            report_text = generate_report(summary, dry_run=False)
            if report_text and not report_text.startswith("No data"):
                save_ai_report(week_start_str, week_end_str, report_text)
        st.success("Report generated ✓")

    elif dryrun_btn:
        report_text = generate_report(summary, dry_run=True)

    elif cached:
        report_text = cached
        st.caption("Showing cached report. Click **Generate** to refresh.")

    # ── Display report ──────────────────────────────────────────────────────
    if report_text:
        st.markdown("---")
        st.markdown("### Your Weekly Productivity Report")
        st.markdown(f'<div class="ai-report">{report_text}</div>', unsafe_allow_html=True)
    elif not cached:
        st.info(
            "No report yet for this week. Click **Generate Report** to get a personalised "
            "coaching analysis, or **Quick Report** for an instant offline version."
        )


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    page, lookback, dist_limit = render_sidebar()

    # Load data
    daily_df, hourly_df, cluster_df, raw_df, spirals_df = load_all_data()

    # Check if DB is empty → prompt to run pipeline
    db_is_empty = daily_df.empty and raw_df.empty

    if db_is_empty and page != "✨ AI Report":
        st.markdown("# 🔬 FocusLens")
        st.markdown(
            '<p style="color:#9aa3b2;font-size:1rem;">Your personal productivity analytics dashboard.</p>',
            unsafe_allow_html=True,
        )
        st.info(
            "**No data found.** To get started:\n\n"
            "- Click **Run (Demo)** in the sidebar to load 30 days of demo data.\n"
            "- Click **Run (Real)** to analyse your actual Chrome history (close Chrome first)."
        )
        return

    # Route to page
    if page == "📊 Overview":
        page_overview(daily_df, raw_df, lookback)
    elif page == "🎯 Focus Patterns":
        page_focus_patterns(daily_df, hourly_df, raw_df, lookback)
    elif page == "😵 Distraction Analysis":
        page_distraction(raw_df, hourly_df, spirals_df, daily_df, lookback, dist_limit)
    elif page == "✨ AI Report":
        page_ai_report(daily_df, raw_df, hourly_df)


if __name__ == "__main__":
    main()
