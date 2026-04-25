"""
FocusLens — feature_engineer.py
================================
Transforms raw categorized visit data into ML-ready feature vectors.

Feature extraction happens at two levels:
  1. Hourly buckets: aggregate stats per (date, hour) for clustering
  2. Daily summaries: aggregate stats per date for dashboard metrics

All features are designed to capture behavioral patterns, not just counts.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# ─── Session Detection ────────────────────────────────────────────────────────

def assign_sessions(df: pd.DataFrame, gap_minutes: int = None) -> pd.DataFrame:
    """
    Assign a session_id to each visit based on time gaps.
    
    A new session begins when there is a gap > gap_minutes between visits.
    This simulates user stepping away from the computer.
    
    Args:
        df: Categorized DataFrame sorted by visit_time.
        gap_minutes: Inactivity gap that ends a session.
    
    Returns:
        DataFrame with added 'session_id' column.
    """
    if gap_minutes is None:
        gap_minutes = config.SESSION_GAP_MINUTES
    
    df = df.sort_values("visit_time").copy()
    
    # Calculate time delta between consecutive visits
    df["time_delta_minutes"] = (
        df["visit_time"].diff().dt.total_seconds() / 60
    ).fillna(0)
    
    # New session starts when gap exceeds threshold
    df["new_session"] = (df["time_delta_minutes"] > gap_minutes).astype(int)
    df["session_id"] = df["new_session"].cumsum()
    
    df.drop(columns=["time_delta_minutes", "new_session"], inplace=True)
    logger.info(f"Sessions detected: {df['session_id'].nunique()} sessions")
    return df


# ─── Hourly Feature Engineering ───────────────────────────────────────────────

def build_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate visit data into hourly buckets and compute behavioral features.
    
    Each row represents one (date, hour) bucket with features:
      - productive_ratio: fraction of visits that are productive
      - distraction_ratio: fraction of visits that are distraction
      - context_switches: # category transitions (productive↔distraction↔neutral)
      - session_length_minutes: avg session length in this hour
      - unique_domains: # distinct domains visited
      - distraction_streak: longest consecutive distraction visit run
      - total_visits: raw visit count
      - total_minutes: total browsing time in minutes
      - hour_of_day: 0–23 (for temporal patterns)
      - day_of_week: 0=Monday, 6=Sunday
      - is_weekend: binary flag
    
    Returns:
        DataFrame indexed by (date, hour) with one row per bucket.
    """
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df["date"] = df["visit_time"].dt.date
    df["hour"] = df["visit_time"].dt.hour
    df["day_of_week"] = df["visit_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Group by date + hour
    groups = df.groupby(["date", "hour"])
    
    rows = []
    for (date, hour), group in groups:
        group = group.sort_values("visit_time")
        cats = group["category"].tolist()
        
        n = len(group)
        prod_n = (group["category"] == "productive").sum()
        dist_n = (group["category"] == "distraction").sum()
        neut_n = (group["category"] == "neutral").sum()
        
        # Context switches: count category changes in sequence
        switches = sum(1 for i in range(1, len(cats)) if cats[i] != cats[i - 1])
        
        # Distraction streak: longest run of consecutive distraction visits
        dist_streak = _max_streak(cats, "distraction")
        
        # Session length: use pre-computed session_id if available
        if "session_id" in group.columns:
            session_lengths = group.groupby("session_id")["duration_seconds"].sum() / 60
            avg_session_min = session_lengths.mean() if not session_lengths.empty else 0
        else:
            avg_session_min = group["duration_seconds"].sum() / 60
        
        total_minutes = group["duration_seconds"].sum() / 60
        
        rows.append({
            "date": date,
            "hour": hour,
            "day_of_week": group["day_of_week"].iloc[0],
            "is_weekend": group["is_weekend"].iloc[0],
            "total_visits": n,
            "total_minutes": round(total_minutes, 2),
            "productive_visits": int(prod_n),
            "distraction_visits": int(dist_n),
            "neutral_visits": int(neut_n),
            "productive_ratio": round(prod_n / n, 4) if n > 0 else 0.0,
            "distraction_ratio": round(dist_n / n, 4) if n > 0 else 0.0,
            "neutral_ratio": round(neut_n / n, 4) if n > 0 else 0.0,
            "context_switches": int(switches),
            "context_switch_rate": round(switches / n, 4) if n > 0 else 0.0,
            "distraction_streak": int(dist_streak),
            "unique_domains": int(group["clean_domain"].nunique()),
            "avg_session_minutes": round(avg_session_min, 2),
        })
    
    hourly_df = pd.DataFrame(rows)
    
    # Fill any gaps in the date-hour grid with zeros (hours with no activity)
    hourly_df = _fill_time_grid(hourly_df, df)
    
    logger.info(f"Hourly features: {len(hourly_df)} buckets built")
    return hourly_df


def _max_streak(sequence: list, target: str) -> int:
    """Find the maximum consecutive run of `target` in a sequence."""
    max_run = 0
    current_run = 0
    for item in sequence:
        if item == target:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _fill_time_grid(hourly_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing (date, hour) combinations with zero-activity rows.
    
    Ensures the ML model sees a complete time grid — not just hours with visits.
    Missing hours = inactive periods, which are their own behavioral signal.
    """
    if hourly_df.empty:
        return hourly_df
    
    all_dates = pd.date_range(
        start=original_df["visit_time"].min().date(),
        end=original_df["visit_time"].max().date(),
        freq="D"
    ).date
    all_hours = range(24)
    
    full_index = pd.MultiIndex.from_product(
        [all_dates, all_hours], names=["date", "hour"]
    )
    
    hourly_df = hourly_df.set_index(["date", "hour"])
    hourly_df = hourly_df.reindex(full_index, fill_value=0)
    
    # Fix day_of_week and is_weekend for filled rows
    hourly_df = hourly_df.reset_index()
    mask = hourly_df["day_of_week"] == 0  # might be legitimately 0 (Monday), use another check
    # Recompute for all rows to be safe
    dates_as_dt = pd.to_datetime(hourly_df["date"])
    hourly_df["day_of_week"] = dates_as_dt.dt.dayofweek
    hourly_df["is_weekend"] = (hourly_df["day_of_week"] >= 5).astype(int)
    
    return hourly_df


# ─── Daily Summary ────────────────────────────────────────────────────────────

def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-day summary metrics for the dashboard overview.
    
    Returns one row per day with:
      - productive/distraction/neutral minutes
      - focus_score: weighted 0–100 score
      - distraction_limit_exceeded: boolean
      - top_productive_domain, top_distraction_domain
    """
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df["date"] = df["visit_time"].dt.date
    
    rows = []
    for date, group in df.groupby("date"):
        prod_min  = group[group["category"] == "productive"]["duration_seconds"].sum() / 60
        dist_min  = group[group["category"] == "distraction"]["duration_seconds"].sum() / 60
        neut_min  = group[group["category"] == "neutral"]["duration_seconds"].sum() / 60
        total_min = group["duration_seconds"].sum() / 60
        
        n = len(group)
        prod_ratio = prod_min / total_min if total_min > 0 else 0
        dist_ratio = dist_min / total_min if total_min > 0 else 0
        
        # Context switches across the full day
        cats = group.sort_values("visit_time")["category"].tolist()
        switches = sum(1 for i in range(1, len(cats)) if cats[i] != cats[i - 1])
        
        top_prod = (
            group[group["category"] == "productive"]["clean_domain"]
            .value_counts().index[0]
            if (group["category"] == "productive").any() else "—"
        )
        top_dist = (
            group[group["category"] == "distraction"]["clean_domain"]
            .value_counts().index[0]
            if (group["category"] == "distraction").any() else "—"
        )
        
        focus_score = compute_focus_score(prod_ratio, dist_ratio, switches, n)
        
        rows.append({
            "date": date,
            "productive_minutes": round(prod_min, 1),
            "distraction_minutes": round(dist_min, 1),
            "neutral_minutes": round(neut_min, 1),
            "total_minutes": round(total_min, 1),
            "productive_ratio": round(prod_ratio, 4),
            "distraction_ratio": round(dist_ratio, 4),
            "context_switches": switches,
            "unique_domains": group["clean_domain"].nunique(),
            "total_visits": n,
            "focus_score": focus_score,
            "distraction_limit_exceeded": dist_min > config.DAILY_DISTRACTION_LIMIT_MINUTES,
            "top_productive_domain": top_prod,
            "top_distraction_domain": top_dist,
        })
    
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compute_focus_score(
    productive_ratio: float,
    distraction_ratio: float,
    context_switches: int,
    total_visits: int,
) -> float:
    """
    Compute a 0–100 focus score for a time period.
    
    Formula (weighted):
      + productive_ratio  × 45   (main driver)
      − distraction_ratio × 35   (penalizes distraction)
      + context_switch_score × 10  (fewer switches = more focused)
      + baseline 10 points for any activity
    
    Score is clipped to [0, 100].
    """
    if total_visits == 0:
        return 0.0
    
    # Normalize context switches: 0 switches = 1.0, many switches → 0.0
    # Use sigmoid-like decay: score = 1 / (1 + switches/10)
    switch_score = 1.0 / (1.0 + context_switches / 10.0)
    
    score = (
        productive_ratio * 45
        + (1 - distraction_ratio) * 35
        + switch_score * 10
        + 10  # baseline for having any activity
    )
    
    return round(min(max(score, 0.0), 100.0), 1)


# ─── ML Feature Matrix ────────────────────────────────────────────────────────

def build_feature_matrix(hourly_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract the numerical feature matrix for ML clustering.
    
    Selects only the behavioral features (no date/time index columns).
    Returns both the numpy array (for sklearn) and the feature DataFrame (for interpretation).
    
    Features selected for clustering:
      - productive_ratio
      - distraction_ratio
      - context_switch_rate
      - avg_session_minutes
      - unique_domains (normalized by total_visits)
      - distraction_streak
      - is_weekend
      - hour_of_day (sinusoidal encoded for cyclical nature)
    
    Sin/cos encoding for hour: hour 0 and 23 should be "close" — not far apart.
    """
    df = hourly_df.copy()
    
    # Only keep rows with at least some activity for meaningful clustering
    active_df = df[df["total_visits"] > 0].copy()
    
    if active_df.empty:
        raise ValueError("No active hours found. Cannot build feature matrix.")
    
    # Cyclical encoding of hour_of_day (0–23 wraps around)
    active_df["hour_sin"] = np.sin(2 * np.pi * active_df["hour"] / 24)
    active_df["hour_cos"] = np.cos(2 * np.pi * active_df["hour"] / 24)
    
    # Normalize unique_domains by total_visits (relative domain diversity)
    active_df["domain_diversity"] = (
        active_df["unique_domains"] / active_df["total_visits"].clip(lower=1)
    )
    
    # Normalize distraction_streak by total_visits
    active_df["streak_ratio"] = (
        active_df["distraction_streak"] / active_df["total_visits"].clip(lower=1)
    )
    
    feature_cols = [
        "productive_ratio",
        "distraction_ratio",
        "context_switch_rate",
        "avg_session_minutes",
        "domain_diversity",
        "streak_ratio",
        "is_weekend",
        "hour_sin",
        "hour_cos",
    ]
    
    feature_df = active_df[["date", "hour"] + feature_cols].copy()
    X = active_df[feature_cols].values
    
    logger.info(f"Feature matrix shape: {X.shape} | Features: {feature_cols}")
    return X, feature_df


# ─── Distraction Spiral Detection ─────────────────────────────────────────────

def detect_distraction_spirals(df: pd.DataFrame, min_streak: int = 3) -> pd.DataFrame:
    """
    Detect distraction spirals: sequences of 3+ consecutive distraction visits.
    
    Returns a DataFrame with one row per spiral, containing:
      - start_time: when the spiral began
      - end_time: when it ended
      - duration_minutes: how long it lasted
      - domains: which distraction sites were visited
      - visit_count: number of visits in the spiral
    """
    if df.empty or "category" not in df.columns:
        return pd.DataFrame()
    
    df = df.sort_values("visit_time").copy()
    spirals = []
    
    i = 0
    while i < len(df):
        if df.iloc[i]["category"] == "distraction":
            # Start of a potential spiral
            j = i
            while j < len(df) and df.iloc[j]["category"] == "distraction":
                j += 1
            
            streak_length = j - i
            if streak_length >= min_streak:
                spiral_group = df.iloc[i:j]
                spirals.append({
                    "start_time": spiral_group["visit_time"].iloc[0],
                    "end_time": spiral_group["visit_time"].iloc[-1],
                    "duration_minutes": round(
                        spiral_group["duration_seconds"].sum() / 60, 1
                    ),
                    "visit_count": streak_length,
                    "domains": ", ".join(
                        spiral_group["clean_domain"].value_counts().index[:5].tolist()
                    ),
                    "date": spiral_group["visit_time"].iloc[0].date(),
                })
            i = j
        else:
            i += 1
    
    if not spirals:
        return pd.DataFrame()
    
    return pd.DataFrame(spirals).sort_values("start_time").reset_index(drop=True)


if __name__ == "__main__":
    from src.history_reader import generate_demo_history
    from src.categorizer import categorize_dataframe
    
    df = generate_demo_history(days=14)
    df = categorize_dataframe(df)
    df = assign_sessions(df)
    
    hourly = build_hourly_features(df)
    daily = build_daily_summary(df)
    X, feat_df = build_feature_matrix(hourly)
    spirals = detect_distraction_spirals(df)
    
    print("\n=== Feature Engineer Test ===")
    print(f"Hourly features shape: {hourly.shape}")
    print(f"Daily summary shape: {daily.shape}")
    print(f"ML feature matrix shape: {X.shape}")
    print(f"Distraction spirals detected: {len(spirals)}")
    print(f"\nFocus scores:\n{daily[['date', 'focus_score']].tail(7)}")
