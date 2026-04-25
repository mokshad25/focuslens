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

# ── Custom CSS — dark, minimal, professional ──────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0D1117;
    color: #E6EDF3;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #21262D;
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stRadio label,
  [data-testid="stSidebar"] p { color: #8B949E; font-size: 0.85rem; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 16px 20px;
  }
  [data-testid="stMetric"] label { color: #8B949E; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
  [data-testid="stMetricValue"]  { color: #E6EDF3; font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; }
  [data-testid="stMetricDelta"]  { font-size: 0.8rem; }

  /* Section headers */
  h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #E6EDF3; letter-spacing: -0.02em; }
  h2 { font-family: 'IBM Plex Mono', monospace; font-size: 1.15rem; font-weight: 500; color: #8B949E; border-bottom: 1px solid #21262D; padding-bottom: 6px; margin-top: 28px; }
  h3 { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.95rem; font-weight: 500; color: #C9D1D9; }

  /* Buttons */
  .stButton > button {
    background: #21262D;
    color: #E6EDF3;
    border: 1px solid #30363D;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    padding: 6px 16px;
    transition: all 0.15s;
  }
  .stButton > button:hover {
    background: #00C9A7;
    color: #0D1117;
    border-color: #00C9A7;
  }

  /* Warning / info banners */
  .stAlert { border-radius: 6px; }

  /* Expanders */
  .streamlit-expanderHeader { background: #161B22; border: 1px solid #21262D; border-radius: 6px; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #161B22; border-radius: 8px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 6px; color: #8B949E; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; }
  .stTabs [aria-selected="true"] { background: #21262D !important; color: #E6EDF3 !important; }

  /* Focus score badge */
  .focus-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
  }
  .badge-excellent { background: rgba(0,201,167,0.15); color: #00C9A7; border: 1px solid #00C9A7; }
  .badge-good      { background: rgba(255,217,61,0.15); color: #FFD93D; border: 1px solid #FFD93D; }
  .badge-average   { background: rgba(255,107,107,0.12); color: #FF6B6B; border: 1px solid #FF6B6B; }
  .badge-poor      { background: rgba(139,148,158,0.15); color: #8B949E; border: 1px solid #8B949E; }

  /* AI report text */
  .ai-report {
    background: #161B22;
    border: 1px solid #21262D;
    border-left: 3px solid #00C9A7;
    border-radius: 8px;
    padding: 20px 24px;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #C9D1D9;
    white-space: pre-wrap;
  }

  /* Distraction warning */
  .dist-warning {
    background: rgba(255,107,107,0.08);
    border: 1px solid rgba(255,107,107,0.3);
    border-radius: 8px;
    padding: 12px 18px;
    color: #FF6B6B;
    font-size: 0.88rem;
    margin: 8px 0;
  }

  /* Spiral cards */
  .spiral-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-left: 3px solid #FF6B6B;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.83rem;
  }

  /* Dataframes */
  [data-testid="stDataFrame"] { border: 1px solid #21262D; border-radius: 6px; }
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
        st.markdown('<p style="color:#8B949E;font-size:0.78rem;margin-top:-10px;">Personal Productivity Analytics</p>', unsafe_allow_html=True)
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
            f'<div style="background:#161B22;border:1px solid #21262D;border-left:3px solid #00C9A7;'
            f'border-radius:8px;padding:14px 20px;margin-bottom:16px;">'
            f'<span style="color:#8B949E;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;">Peak Focus Window</span><br>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;color:#00C9A7;font-weight:600;">'
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
                    name="Productive", marker_color="#00C9A7",
                    hovertemplate="Hour %{x}:00<br>%{y:.1%}<extra></extra>")
        fig.add_bar(x=hour_agg["hour"], y=hour_agg["distraction_ratio"],
                    name="Distraction", marker_color="#FF6B6B",
                    hovertemplate="Hour %{x}:00<br>%{y:.1%}<extra></extra>")
        fig.update_layout(
            barmode="group",
            xaxis=dict(title="Hour of Day", tickmode="linear", dtick=2),
            yaxis=dict(title="Ratio", tickformat=".0%"),
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            font=dict(color="#E6EDF3"),
            legend=dict(bgcolor="#161B22", bordercolor="#30363D", borderwidth=1),
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
            '<p style="color:#8B949E;font-size:1rem;">Your personal productivity analytics dashboard.</p>',
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
