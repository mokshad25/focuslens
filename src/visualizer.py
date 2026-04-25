"""
FocusLens — visualizer.py
==========================
All Plotly chart functions for the dashboard.

Design philosophy:
  - Dark theme with a consistent color palette (defined in config.py)
  - Each chart function is self-contained: takes DataFrames, returns fig
  - No Streamlit calls here — pure visualization layer
  - Charts are interactive via Plotly (hover, zoom, filter)
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ─── Shared Theme ─────────────────────────────────────────────────────────────

DARK_THEME = dict(
    paper_bgcolor="#0D1117",
    plot_bgcolor="#0D1117",
    font=dict(family="'IBM Plex Mono', monospace", color="#E6EDF3", size=12),
    title_font=dict(family="'IBM Plex Mono', monospace", color="#E6EDF3", size=14),
    xaxis=dict(gridcolor="#21262D", linecolor="#30363D", zerolinecolor="#30363D"),
    yaxis=dict(gridcolor="#21262D", linecolor="#30363D", zerolinecolor="#30363D"),
    legend=dict(bgcolor="#161B22", bordercolor="#30363D", borderwidth=1),
)

CAT_COLORS = config.CATEGORY_COLORS
CLUSTER_COLORS = config.CLUSTER_COLORS


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the FocusLens dark theme to any figure."""
    fig.update_layout(
        **DARK_THEME,
        title=dict(text=title, x=0.02, xanchor="left"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ─── Overview Charts ──────────────────────────────────────────────────────────

def chart_today_pie(daily_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart: today's time split between productive / neutral / distraction.
    Shows the user's focus ratio at a glance.
    """
    if daily_df.empty:
        return _empty_chart("No data for today")
    
    today = daily_df.iloc[-1]
    
    labels = ["Productive", "Neutral", "Distraction"]
    values = [
        max(today.get("productive_minutes", 0), 0),
        max(today.get("neutral_minutes", 0), 0),
        max(today.get("distraction_minutes", 0), 0),
    ]
    colors = [CAT_COLORS["productive"], CAT_COLORS["neutral"], CAT_COLORS["distraction"]]
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0D1117", width=3)),
        textinfo="percent",
        textfont=dict(size=13, color="#E6EDF3"),
        hovertemplate="<b>%{label}</b><br>%{value:.0f} min<extra></extra>",
    ))
    
    # Center annotation: focus score
    focus = today.get("focus_score", 0)
    fig.add_annotation(
        text=f"<b>{focus:.0f}</b><br><span style='font-size:10px'>Focus Score</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="#E6EDF3"),
    )
    
    return _apply_theme(fig, "Today's Activity Split")


def chart_weekly_stacked_bar(daily_df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart: productive / neutral / distraction minutes per day.
    Reveals the week's rhythm and which days were most focused.
    """
    if daily_df.empty:
        return _empty_chart("No weekly data available")
    
    df = daily_df.copy()
    df["date_label"] = pd.to_datetime(df["date"]).dt.strftime("%a %b %d")
    
    fig = go.Figure()
    
    for cat, color, col in [
        ("Productive", CAT_COLORS["productive"], "productive_minutes"),
        ("Neutral", CAT_COLORS["neutral"], "neutral_minutes"),
        ("Distraction", CAT_COLORS["distraction"], "distraction_minutes"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Bar(
                name=cat,
                x=df["date_label"],
                y=df[col],
                marker_color=color,
                hovertemplate=f"<b>{cat}</b><br>%{{y:.0f}} min<extra></extra>",
            ))
    
    fig.update_layout(
        barmode="stack",
        xaxis_title="",
        yaxis_title="Minutes",
        legend=dict(orientation="h", y=-0.15),
    )
    
    return _apply_theme(fig, "Weekly Activity Breakdown")


# ─── Distraction Analysis Charts ──────────────────────────────────────────────

def chart_top_distraction_domains(raw_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Horizontal bar chart: top distraction domains by total time spent.
    Shows where the attention is actually going.
    """
    if raw_df.empty or "category" not in raw_df.columns:
        return _empty_chart("No distraction data")
    
    dist_df = raw_df[raw_df["category"] == "distraction"].copy()
    if dist_df.empty:
        return _empty_chart("No distraction sites found — great job! 🎯")
    
    domain_col = "clean_domain" if "clean_domain" in dist_df.columns else "domain"
    domain_time = (
        dist_df.groupby(domain_col)["duration_seconds"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    domain_time_min = domain_time / 60
    
    # Color gradient: most time = most red
    colors = px.colors.sequential.Reds[2:][::-1]
    n = len(domain_time_min)
    bar_colors = [colors[int(i * (len(colors) - 1) / max(n - 1, 1))] for i in range(n)]
    
    fig = go.Figure(go.Bar(
        x=domain_time_min.values[::-1],
        y=domain_time_min.index.tolist()[::-1],
        orientation="h",
        marker=dict(color=bar_colors[::-1]),
        hovertemplate="<b>%{y}</b><br>%{x:.1f} minutes<extra></extra>",
        text=[f"{v:.0f} min" for v in domain_time_min.values[::-1]],
        textposition="outside",
        textfont=dict(color="#E6EDF3", size=11),
    ))
    
    fig.update_layout(
        xaxis_title="Total Time (minutes)",
        yaxis_title="",
        height=max(300, n * 40),
    )
    
    return _apply_theme(fig, f"Top {top_n} Distraction Sites by Time")


def chart_context_switches(hourly_df: pd.DataFrame) -> go.Figure:
    """
    Line chart: average context switch rate by hour of day.
    Peaks reveal times when attention is most fragmented.
    """
    if hourly_df.empty:
        return _empty_chart("No context switch data")
    
    hourly_mean = (
        hourly_df.groupby("hour")["context_switch_rate"]
        .mean()
        .reset_index()
    )
    
    fig = go.Figure(go.Scatter(
        x=hourly_mean["hour"],
        y=hourly_mean["context_switch_rate"],
        mode="lines+markers",
        line=dict(color="#FFD93D", width=2.5),
        marker=dict(size=6, color="#FFD93D"),
        fill="tozeroy",
        fillcolor="rgba(255, 217, 61, 0.10)",
        hovertemplate="<b>Hour %{x}:00</b><br>Switch rate: %{y:.2f}<extra></extra>",
    ))
    
    fig.update_layout(
        xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Context Switch Rate"),
    )
    
    return _apply_theme(fig, "Context Switch Rate by Hour")


def chart_distraction_spirals(spirals_df: pd.DataFrame) -> go.Figure:
    """
    Timeline chart: distraction spirals shown as horizontal spans.
    Duration = width, so longer spirals are visually obvious.
    """
    if spirals_df.empty:
        return _empty_chart("No distraction spirals detected this period 🎉")
    
    df = spirals_df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    df["date_label"] = df["start_time"].dt.strftime("%b %d")
    
    fig = go.Figure()
    
    for i, row in df.iterrows():
        intensity = min(row["duration_minutes"] / 60, 1.0)
        r = int(255 * 0.4 + 255 * 0.6 * intensity)
        g = int(107 * (1 - intensity))
        b = int(107 * (1 - intensity))
        
        fig.add_trace(go.Bar(
            x=[row["duration_minutes"]],
            y=[row["date_label"]],
            orientation="h",
            marker_color=f"rgb({r},{g},{b})",
            name="",
            showlegend=False,
            hovertemplate=(
                f"<b>{row['date_label']} {row['start_time'].strftime('%H:%M')}</b><br>"
                f"Duration: {row['duration_minutes']:.0f} min<br>"
                f"Sites: {row['domains']}<br>"
                f"Visits: {row['visit_count']}<extra></extra>"
            ),
        ))
    
    fig.update_layout(
        xaxis_title="Duration (minutes)",
        yaxis_title="Date",
        barmode="overlay",
        showlegend=False,
    )
    
    return _apply_theme(fig, "Distraction Spirals (3+ Consecutive Distraction Visits)")


# ─── ML Cluster Charts ────────────────────────────────────────────────────────

def chart_pca_scatter(cluster_df: pd.DataFrame) -> go.Figure:
    """
    2D PCA scatter plot colored by KMeans cluster.
    Reveals the natural groupings in browsing behavior space.
    
    cluster_df must have: pca_x, pca_y, kmeans_label, kmeans_name, date, hour
    """
    if cluster_df.empty:
        return _empty_chart("No cluster data available")
    
    fig = go.Figure()
    
    for label in sorted(cluster_df["kmeans_label"].unique()):
        subset = cluster_df[cluster_df["kmeans_label"] == label]
        name = subset["kmeans_name"].iloc[0] if "kmeans_name" in subset.columns else f"Cluster {label}"
        color = CLUSTER_COLORS.get(label, "#888888")
        
        fig.add_trace(go.Scatter(
            x=subset["pca_x"],
            y=subset["pca_y"],
            mode="markers",
            name=name,
            marker=dict(
                color=color,
                size=6,
                opacity=0.75,
                line=dict(color="#0D1117", width=0.5),
            ),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "Date: %{customdata[0]}<br>"
                "Hour: %{customdata[1]}:00<extra></extra>"
            ),
            customdata=list(zip(subset["date"], subset["hour"])),
        ))
    
    # Highlight DBSCAN anomalies
    if "dbscan_label" in cluster_df.columns:
        anomalies = cluster_df[cluster_df["dbscan_label"] == -1]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["pca_x"],
                y=anomalies["pca_y"],
                mode="markers",
                name="⚠️ Anomaly (DBSCAN)",
                marker=dict(
                    color=CLUSTER_COLORS.get(-1, "#9B59B6"),
                    size=10,
                    symbol="diamond",
                    line=dict(color="#E6EDF3", width=1.5),
                ),
            ))
    
    fig.update_layout(
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        legend=dict(orientation="v"),
    )
    
    return _apply_theme(fig, "Behavior Clusters — PCA Projection")


def chart_elbow_curve(elbow_df: pd.DataFrame) -> go.Figure:
    """
    Elbow method + silhouette score chart.
    Justifies the choice of k=4 clusters.
    """
    if elbow_df.empty:
        return _empty_chart("No elbow data")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=elbow_df["k"], y=elbow_df["inertia"],
        mode="lines+markers",
        name="Inertia",
        line=dict(color=CAT_COLORS["productive"], width=2),
        marker=dict(size=7),
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=elbow_df["k"], y=elbow_df["silhouette_score"],
        mode="lines+markers",
        name="Silhouette Score",
        line=dict(color="#FFD93D", width=2, dash="dash"),
        marker=dict(size=7),
    ), secondary_y=True)
    
    # Highlight chosen k
    fig.add_vline(
        x=config.KMEANS_CLUSTERS,
        line_dash="dot",
        line_color="#FF6B6B",
        annotation_text=f"  k={config.KMEANS_CLUSTERS} (selected)",
        annotation_font_color="#FF6B6B",
    )
    
    fig.update_yaxes(title_text="Inertia", secondary_y=False,
                     gridcolor="#21262D", color="#E6EDF3")
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True,
                     color="#FFD93D")
    fig.update_xaxes(title_text="Number of Clusters (k)", tickmode="linear")
    
    return _apply_theme(fig, "KMeans Elbow Method & Silhouette Analysis")


def chart_cluster_distribution(cluster_df: pd.DataFrame) -> go.Figure:
    """
    Pie chart showing what fraction of hourly buckets fall into each cluster.
    Gives a quick read on the user's dominant behavior mode.
    """
    if cluster_df.empty:
        return _empty_chart("No cluster distribution data")
    
    counts = cluster_df.groupby(["kmeans_label", "kmeans_name"]).size().reset_index(name="count")
    
    colors = [CLUSTER_COLORS.get(int(row["kmeans_label"]), "#888") for _, row in counts.iterrows()]
    
    fig = go.Figure(go.Pie(
        labels=counts["kmeans_name"],
        values=counts["count"],
        marker=dict(colors=colors, line=dict(color="#0D1117", width=2)),
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>%{value} hours (%{percent})<extra></extra>",
    ))
    
    return _apply_theme(fig, "Behavior Mode Distribution")


def chart_anomaly_calendar(daily_df: pd.DataFrame, anomaly_dates: pd.DataFrame) -> go.Figure:
    """
    Calendar heatmap of focus scores with anomaly dates highlighted in red.
    """
    if daily_df.empty:
        return _empty_chart("No calendar data")
    
    df = daily_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt")
    
    anomaly_date_set = set()
    if not anomaly_dates.empty and "date" in anomaly_dates.columns:
        anomaly_date_set = set(pd.to_datetime(anomaly_dates["date"]).dt.date)
    
    colors = []
    hover = []
    for _, row in df.iterrows():
        d = pd.to_datetime(row["date"]).date()
        is_anomaly = d in anomaly_date_set
        score = row.get("focus_score", 0)
        
        if is_anomaly:
            colors.append("#9B59B6")  # DBSCAN anomaly = purple
        elif score >= 70:
            colors.append(CAT_COLORS["productive"])
        elif score >= 45:
            colors.append("#FFD93D")
        else:
            colors.append(CAT_COLORS["distraction"])
        
        hover.append(
            f"Date: {d}<br>Focus: {score:.0f}/100"
            + (" ⚠️ Anomaly" if is_anomaly else "")
        )
    
    fig = go.Figure(go.Bar(
        x=df["date_dt"],
        y=df["focus_score"],
        marker_color=colors,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Focus Score",
        yaxis=dict(range=[0, 105]),
        showlegend=False,
    )
    
    return _apply_theme(fig, "Daily Focus Score (Purple = DBSCAN Anomaly)")


# ─── Week Comparison ──────────────────────────────────────────────────────────

def chart_week_comparison(daily_df: pd.DataFrame) -> go.Figure:
    """
    Side-by-side bar chart: this week vs last week metrics.
    Delta annotations show improvement or regression.
    """
    if daily_df.empty or len(daily_df) < 7:
        return _empty_chart("Need at least 14 days of data for comparison")
    
    df = daily_df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt")
    
    today = df["date_dt"].max()
    this_week = df[df["date_dt"] > today - pd.Timedelta(days=7)]
    last_week = df[
        (df["date_dt"] <= today - pd.Timedelta(days=7)) &
        (df["date_dt"] > today - pd.Timedelta(days=14))
    ]
    
    if this_week.empty or last_week.empty:
        return _empty_chart("Insufficient data for week comparison")
    
    metrics = {
        "Avg Focus Score": ("focus_score", "mean"),
        "Productive (min/day)": ("productive_minutes", "mean"),
        "Distraction (min/day)": ("distraction_minutes", "mean"),
        "Context Switches/day": ("context_switches", "mean"),
    }
    
    this_vals = []
    last_vals = []
    labels = []
    
    for label, (col, agg) in metrics.items():
        if col in this_week.columns and col in last_week.columns:
            this_v = getattr(this_week[col], agg)()
            last_v = getattr(last_week[col], agg)()
            this_vals.append(round(this_v, 1))
            last_vals.append(round(last_v, 1))
            labels.append(label)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Last Week",
        x=labels,
        y=last_vals,
        marker_color="#30363D",
        text=[f"{v:.1f}" for v in last_vals],
        textposition="outside",
        textfont=dict(color="#8B949E"),
    ))
    
    fig.add_trace(go.Bar(
        name="This Week",
        x=labels,
        y=this_vals,
        marker_color=CAT_COLORS["productive"],
        text=[f"{v:.1f}" for v in this_vals],
        textposition="outside",
        textfont=dict(color="#E6EDF3"),
    ))
    
    fig.update_layout(
        barmode="group",
        yaxis_title="Value",
        legend=dict(orientation="h", y=-0.15),
    )
    
    return _apply_theme(fig, "This Week vs Last Week Comparison")


# ─── Utilities ────────────────────────────────────────────────────────────────

def _empty_chart(message: str) -> go.Figure:
    """Return a clean empty figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color="#8B949E"),
    )
    fig.update_layout(
        **DARK_THEME,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,
    )
    return fig


if __name__ == "__main__":
    from src.history_reader import generate_demo_history
    from src.categorizer import categorize_dataframe
    from src.feature_engineer import (
        assign_sessions, build_hourly_features, build_daily_summary, build_feature_matrix
    )
    from src.clusterer import run_full_clustering, get_cluster_summary
    
    df = generate_demo_history(days=30)
    df = categorize_dataframe(df)
    df = assign_sessions(df)
    hourly = build_hourly_features(df)
    daily = build_daily_summary(df)
    X, feat_df = build_feature_matrix(hourly)
    results = run_full_clustering(X, feat_df)
    
    print("Charts generated (no errors):")
    figs = [
        ("Today pie", chart_today_pie(daily)),
        ("Weekly bar", chart_weekly_stacked_bar(daily.tail(7))),
        ("Top distraction", chart_top_distraction_domains(df)),
        ("Elbow", chart_elbow_curve(results["elbow_df"])),
    ]
    for name, fig in figs:
        print(f"  ✓ {name}")
