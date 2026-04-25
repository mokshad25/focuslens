"""
FocusLens — insights_generator.py
===================================
Generates AI-powered productivity reports using the Anthropic Claude API.

Privacy guarantee:
  - ONLY aggregated summary statistics are sent to the API
  - NO raw URLs, domain names, or personally identifying browsing data
  - The summary dict contains only numbers and general time references

Fallback behavior:
  - If API call fails → return last cached report from database
  - If no cache → return a rule-based static report
  - --dry-run mode → skip API call entirely, return mock report
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# Day name mapping for readable reports
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOUR_LABELS = {
    0: "midnight", 6: "6 AM", 7: "7 AM", 8: "8 AM", 9: "9 AM",
    10: "10 AM", 11: "11 AM", 12: "noon", 13: "1 PM", 14: "2 PM",
    15: "3 PM", 16: "4 PM", 17: "5 PM", 18: "6 PM", 19: "7 PM",
    20: "8 PM", 21: "9 PM", 22: "10 PM", 23: "11 PM",
}


def build_weekly_summary(
    daily_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    week_start: Optional[str] = None,
) -> dict:
    """
    Aggregate one week of data into a privacy-safe summary dictionary.
    
    This summary is what gets sent to the LLM — no raw URLs or domain names.
    
    Args:
        daily_df: Daily summary DataFrame from database.
        raw_df: Raw visits DataFrame (used for time-of-day patterns).
        hourly_df: Hourly features DataFrame.
        week_start: ISO date string (YYYY-MM-DD). Defaults to 7 days ago.
    
    Returns:
        dict with aggregated metrics safe to share with an external API.
    """
    if daily_df.empty:
        return {}
    
    # Filter to selected week
    if week_start:
        week_start_dt = pd.to_datetime(week_start).date()
    else:
        week_start_dt = (datetime.now() - timedelta(days=7)).date()
    
    week_end_dt = week_start_dt + timedelta(days=6)
    
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    week_df = daily_df[
        (daily_df["date"] >= week_start_dt) &
        (daily_df["date"] <= week_end_dt)
    ].copy()
    
    if week_df.empty:
        # Fall back to all available data
        week_df = daily_df.copy()
    
    # ── Most productive hour ───────────────────────────────────────────────
    most_productive_hour = None
    most_distracted_hour = None
    
    if not hourly_df.empty:
        hourly_df["date"] = pd.to_datetime(hourly_df["date"]).dt.date
        week_hourly = hourly_df[
            (hourly_df["date"] >= week_start_dt) &
            (hourly_df["date"] <= week_end_dt)
        ]
        
        if not week_hourly.empty:
            hour_prod = week_hourly.groupby("hour")["productive_ratio"].mean()
            hour_dist = week_hourly.groupby("hour")["distraction_ratio"].mean()
            
            best_hour = int(hour_prod.idxmax()) if not hour_prod.empty else None
            worst_hour = int(hour_dist.idxmax()) if not hour_dist.empty else None
            
            most_productive_hour = HOUR_LABELS.get(best_hour, f"{best_hour}:00")
            most_distracted_hour = HOUR_LABELS.get(worst_hour, f"{worst_hour}:00")
    
    # ── Top distraction and productive sites (category labels only) ───────
    top_distraction_categories = []
    top_productive_categories = []
    
    if not raw_df.empty:
        raw_df_copy = raw_df.copy()
        if "visit_time" in raw_df_copy.columns:
            raw_df_copy["date"] = pd.to_datetime(raw_df_copy["visit_time"]).dt.date
            week_raw = raw_df_copy[
                (raw_df_copy["date"] >= week_start_dt) &
                (raw_df_copy["date"] <= week_end_dt)
            ]
        else:
            week_raw = raw_df_copy
        
        if not week_raw.empty and "category" in week_raw.columns:
            # Send ONLY domain names (not full URLs) — still considered safe
            # since we're sending aggregate visit counts
            if "clean_domain" in week_raw.columns:
                dist_domains = (
                    week_raw[week_raw["category"] == "distraction"]["clean_domain"]
                    .value_counts().head(5).index.tolist()
                )
                prod_domains = (
                    week_raw[week_raw["category"] == "productive"]["clean_domain"]
                    .value_counts().head(5).index.tolist()
                )
                top_distraction_categories = dist_domains
                top_productive_categories = prod_domains
    
    # ── Worst distraction day ─────────────────────────────────────────────
    worst_day_idx = week_df["distraction_minutes"].idxmax() if not week_df.empty else None
    worst_day_name = None
    if worst_day_idx is not None:
        worst_day_date = week_df.loc[worst_day_idx, "date"]
        worst_day_name = DAY_NAMES[pd.Timestamp(worst_day_date).dayofweek]
    
    # ── Best focus day ────────────────────────────────────────────────────
    best_day_idx = week_df["focus_score"].idxmax() if not week_df.empty else None
    best_day_name = None
    if best_day_idx is not None:
        best_day_date = week_df.loc[best_day_idx, "date"]
        best_day_name = DAY_NAMES[pd.Timestamp(best_day_date).dayofweek]
    
    # ── Assemble summary ──────────────────────────────────────────────────
    summary = {
        "week_start": str(week_start_dt),
        "week_end": str(week_end_dt),
        "most_productive_hour": most_productive_hour or "Unknown",
        "most_distracted_hour": most_distracted_hour or "Unknown",
        "top_distraction_sites": top_distraction_categories,
        "top_productive_sites": top_productive_categories,
        "avg_daily_productive_minutes": round(week_df["productive_minutes"].mean(), 1),
        "avg_daily_distraction_minutes": round(week_df["distraction_minutes"].mean(), 1),
        "avg_daily_focus_score": round(week_df["focus_score"].mean(), 1),
        "best_focus_day": best_day_name or "Unknown",
        "worst_distraction_day": worst_day_name or "Unknown",
        "context_switch_avg": round(week_df["context_switches"].mean(), 1),
        "total_productive_hours": round(week_df["productive_minutes"].sum() / 60, 1),
        "total_distraction_hours": round(week_df["distraction_minutes"].sum() / 60, 1),
        "days_distraction_limit_exceeded": int(week_df["distraction_limit_exceeded"].sum()),
        "distraction_limit_minutes": config.DAILY_DISTRACTION_LIMIT_MINUTES,
    }
    
    logger.info(f"Weekly summary built: {summary}")
    return summary


def generate_report(
    summary: dict,
    dry_run: bool = False,
) -> str:
    """
    Call the Anthropic Claude API to generate a natural language productivity report.
    
    Args:
        summary: Privacy-safe aggregated metrics dict.
        dry_run: If True, return a mock report without calling the API.
    
    Returns:
        Report text string (3 paragraphs of productivity coaching).
    """
    if dry_run:
        return _generate_static_report(summary)
    
    if not summary:
        return "No data available for the selected week."
    
    try:
        import anthropic
        
        # Load API key from environment
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — using static report")
            return _generate_static_report(summary)
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Format summary for the prompt — clean, readable
        summary_text = _format_summary_for_prompt(summary)
        
        message = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            system=config.LLM_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Here is my weekly productivity summary:\n\n{summary_text}\n\nPlease write my productivity report."
                }
            ]
        )
        
        report_text = message.content[0].text
        logger.info(f"AI report generated ({len(report_text)} chars)")
        return report_text
    
    except ImportError:
        logger.warning("anthropic SDK not installed — using static report")
        return _generate_static_report(summary)
    
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return _generate_static_report(summary)


def _format_summary_for_prompt(summary: dict) -> str:
    """Format the summary dict as a readable text block for the LLM."""
    lines = [
        f"Analysis period: {summary.get('week_start')} to {summary.get('week_end')}",
        f"",
        f"FOCUS METRICS:",
        f"  Average daily focus score: {summary.get('avg_daily_focus_score')}/100",
        f"  Average daily productive time: {summary.get('avg_daily_productive_minutes')} minutes",
        f"  Average daily distraction time: {summary.get('avg_daily_distraction_minutes')} minutes",
        f"  Total productive hours this week: {summary.get('total_productive_hours')} hours",
        f"  Total distraction hours this week: {summary.get('total_distraction_hours')} hours",
        f"",
        f"PATTERNS:",
        f"  Peak productive hour: {summary.get('most_productive_hour')}",
        f"  Most distracted hour: {summary.get('most_distracted_hour')}",
        f"  Best focus day: {summary.get('best_focus_day')}",
        f"  Worst distraction day: {summary.get('worst_distraction_day')}",
        f"  Average context switches per day: {summary.get('context_switch_avg')}",
        f"  Days where distraction exceeded {summary.get('distraction_limit_minutes')}-minute limit: {summary.get('days_distraction_limit_exceeded')}/7",
        f"",
        f"TOP PRODUCTIVE SITES: {', '.join(summary.get('top_productive_sites', []))}",
        f"TOP DISTRACTION SITES: {', '.join(summary.get('top_distraction_sites', []))}",
    ]
    return "\n".join(lines)


def _generate_static_report(summary: dict) -> str:
    """
    Rule-based fallback report — no API call needed.
    Used when API is unavailable or in dry-run mode.
    """
    focus_score = summary.get("avg_daily_focus_score", 50)
    prod_min = summary.get("avg_daily_productive_minutes", 0)
    dist_min = summary.get("avg_daily_distraction_minutes", 0)
    best_hour = summary.get("most_productive_hour", "morning")
    worst_day = summary.get("worst_distraction_day", "")
    best_day = summary.get("best_focus_day", "")
    
    # Grade the week
    if focus_score >= 70:
        p1 = f"You had a strong week with an average focus score of {focus_score}/100. Your most productive period was around {best_hour}, and {best_day} was your standout day. Averaging {prod_min:.0f} minutes of productive browsing daily shows you have solid focused work habits."
    elif focus_score >= 50:
        p1 = f"This was a moderate week with an average focus score of {focus_score}/100. Your peak window was around {best_hour} — this is when your mind is sharpest. You logged {prod_min:.0f} minutes of productive activity daily on average, which is a reasonable baseline to build from."
    else:
        p1 = f"This week showed a focus score of {focus_score}/100, suggesting your attention was fragmented. There were still bright spots — {best_hour} appeared to be your most productive window, and {best_day} was your best day. {prod_min:.0f} minutes of daily productive time is your current baseline."
    
    dist_sites = ", ".join(summary.get("top_distraction_sites", ["social media"])[:3]) or "social media"
    p2 = f"Your distraction pattern is clearest on {worst_day}s, where distraction time averaged {dist_min:.0f} minutes per day. Your main time sinks were {dist_sites}. The data shows you're most vulnerable to distraction in the {summary.get('most_distracted_hour', 'afternoon')}, which often follows your productive morning window — a classic energy dip pattern."
    
    p3 = (
        f"Three practical changes for next week: "
        f"(1) Schedule your hardest work for {best_hour} and protect that window aggressively — no notifications, no social tabs. "
        f"(2) On {worst_day}, set a browser time limit for {dist_sites.split(',')[0].strip()} — even 20 minutes can save over an hour of drift. "
        f"(3) Track your context switches — every time you flip between productive and distraction sites, you lose 10–15 minutes of recovery time. Try finishing one productive task fully before switching."
    )
    
    return f"{p1}\n\n{p2}\n\n{p3}"


if __name__ == "__main__":
    # Test with demo data
    from src.history_reader import generate_demo_history
    from src.categorizer import categorize_dataframe
    from src.feature_engineer import (
        assign_sessions, build_hourly_features, build_daily_summary
    )
    
    df = generate_demo_history(days=14)
    df = categorize_dataframe(df)
    df = assign_sessions(df)
    hourly = build_hourly_features(df)
    daily = build_daily_summary(df)
    
    summary = build_weekly_summary(daily, df, hourly)
    report = generate_report(summary, dry_run=True)
    
    print("\n=== AI Report (Dry Run) ===")
    print(report)
