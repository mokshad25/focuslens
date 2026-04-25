"""
FocusLens — history_reader.py
==============================
Reads browser history from Chrome/Firefox SQLite files.
Always operates on a COPY of the history file — never touches the original.

Chrome stores timestamps as microseconds since January 1, 1601 (Windows FILETIME epoch).
We convert these to standard Python datetimes.
"""

import sys
import shutil
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Chrome epoch starts Jan 1, 1601 — 11644473600 seconds before Unix epoch (Jan 1, 1970)
CHROME_EPOCH_OFFSET_SECONDS = 11_644_473_600


def _chrome_timestamp_to_datetime(chrome_ts: int) -> datetime:
    """
    Convert Chrome's microsecond timestamp (since 1601-01-01) to Python datetime.
    
    Chrome stores: microseconds since January 1, 1601
    Unix epoch:    seconds since January 1, 1970
    Difference:    11644473600 seconds
    """
    if not chrome_ts or chrome_ts == 0:
        return None
    unix_seconds = (chrome_ts / 1_000_000) - CHROME_EPOCH_OFFSET_SECONDS
    try:
        return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).replace(tzinfo=None)
    except (OSError, OverflowError, ValueError):
        return None


def _get_chrome_history_path() -> Path:
    """Detect Chrome History file path based on current operating system."""
    platform = sys.platform
    
    # Normalize Linux variants
    if platform.startswith("linux"):
        platform = "linux"
    
    path = config.CHROME_HISTORY_PATHS.get(platform)
    if path is None:
        raise OSError(f"Unsupported platform: {platform}. Cannot locate Chrome history.")
    return path


def copy_history_file(source_path: Path = None) -> Path:
    """
    Copy the browser history file to data/ directory.
    
    Chrome locks its History file while running — we must copy it first.
    If source_path is None, auto-detect based on OS.
    
    Returns:
        Path to the copied history file.
    Raises:
        FileNotFoundError: If Chrome history file doesn't exist.
        PermissionError: If Chrome is open and the file is locked (Windows).
    """
    if source_path is None:
        source_path = _get_chrome_history_path()
    
    source_path = Path(source_path)
    
    if not source_path.exists():
        raise FileNotFoundError(
            f"Chrome History file not found at:\n  {source_path}\n"
            "Make sure Google Chrome is installed and has been opened at least once."
        )
    
    dest_path = config.HISTORY_COPY_PATH
    
    try:
        shutil.copy2(source_path, dest_path)
        logger.info(f"History copied: {source_path} → {dest_path}")
        return dest_path
    except PermissionError:
        raise PermissionError(
            "Cannot read Chrome History file — Chrome is currently open.\n"
            "Please close Chrome completely and try again."
        )


def _read_chrome_history(db_path: Path, lookback_days: int) -> pd.DataFrame:
    """
    Query Chrome's SQLite history database and return raw visit records.
    
    Chrome schema (relevant tables):
      urls(id, url, title, visit_count, last_visit_time)
      visits(id, url, visit_time, from_visit, transition, duration)
    
    duration in visits is in microseconds.
    """
    cutoff_dt = datetime.now() - timedelta(days=lookback_days)
    # Convert cutoff back to Chrome microsecond timestamp
    cutoff_chrome_ts = int(
        (cutoff_dt.timestamp() + CHROME_EPOCH_OFFSET_SECONDS) * 1_000_000
    )
    
    query = """
        SELECT
            v.id          AS visit_id,
            u.url         AS url,
            u.title       AS title,
            v.visit_time  AS chrome_visit_time,
            v.visit_duration AS chrome_duration
        FROM visits v
        JOIN urls u ON v.url = u.id
        WHERE v.visit_time >= ?
        ORDER BY v.visit_time ASC
    """
    
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=(cutoff_chrome_ts,))
    
    logger.info(f"Raw visits loaded: {len(df):,} records")
    return df


def _clean_and_transform(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw Chrome history data into a clean, analysis-ready DataFrame.
    
    Transformations applied:
    1. Convert Chrome timestamps → Python datetime
    2. Convert duration from microseconds → seconds
    3. Filter out extremely short visits (< MIN_VISIT_DURATION_SECONDS)
    4. Extract domain from URL (basic extraction; tldextract used in categorizer)
    5. Drop rows with null visit times
    6. Reset index cleanly
    """
    if raw_df.empty:
        logger.warning("No history data found for the lookback period.")
        return pd.DataFrame(columns=["url", "domain", "title", "visit_time", "duration_seconds"])
    
    df = raw_df.copy()
    
    # Step 1: Convert Chrome timestamp → datetime
    df["visit_time"] = df["chrome_visit_time"].apply(_chrome_timestamp_to_datetime)
    df = df.dropna(subset=["visit_time"])
    
    # Step 2: Convert duration (microseconds → seconds)
    # Chrome sets duration=0 for the last visit in a session (no end signal)
    # We keep 0-duration visits but flag them; filter happens after categorization
    df["duration_seconds"] = (df["chrome_duration"] / 1_000_000).fillna(0).astype(float)
    
    # Step 3: Filter out accidental tab opens (< 3 seconds)
    df = df[df["duration_seconds"] >= config.MIN_VISIT_DURATION_SECONDS].copy()
    logger.info(f"After duration filter (>= {config.MIN_VISIT_DURATION_SECONDS}s): {len(df):,} records")
    
    # Step 4: Basic domain extraction (strip protocol, www, path)
    df["domain"] = df["url"].str.extract(r"(?:https?://)?(?:www\.)?([^/?\#:]+)")[0]
    df["domain"] = df["domain"].str.lower().str.strip()
    
    # Step 5: Clean title (fill NaN, strip whitespace)
    df["title"] = df["title"].fillna("").str.strip()
    
    # Step 6: Select and order final columns
    df = df[["url", "domain", "title", "visit_time", "duration_seconds"]].copy()
    df = df.sort_values("visit_time").reset_index(drop=True)
    
    return df


def load_history(
    source_path: Path = None,
    lookback_days: int = None,
    use_existing_copy: bool = False,
) -> pd.DataFrame:
    """
    Main entry point: load, copy, parse, and return browser history as DataFrame.
    
    Args:
        source_path: Path to Chrome History file. Auto-detected if None.
        lookback_days: How many days back to load. Uses config default if None.
        use_existing_copy: If True, skip copying and use existing data/history_export.db.
                           Useful for re-running analysis without closing Chrome.
    
    Returns:
        pd.DataFrame with columns: url, domain, title, visit_time, duration_seconds
    """
    if lookback_days is None:
        lookback_days = config.HISTORY_LOOKBACK_DAYS
    
    if use_existing_copy and config.HISTORY_COPY_PATH.exists():
        db_path = config.HISTORY_COPY_PATH
        logger.info(f"Using existing history copy: {db_path}")
    else:
        db_path = copy_history_file(source_path)
    
    raw_df = _read_chrome_history(db_path, lookback_days)
    clean_df = _clean_and_transform(raw_df)
    
    logger.info(
        f"History loaded: {len(clean_df):,} visits | "
        f"{clean_df['visit_time'].min().date()} → {clean_df['visit_time'].max().date()}"
    )
    
    return clean_df


def generate_demo_history(days: int = 30, visits_per_day: int = 80) -> pd.DataFrame:
    """
    Generate realistic synthetic browser history for demo/testing purposes.
    
    Simulates realistic human browsing patterns:
    - Morning: mostly productive (email, docs, coding)
    - Afternoon: mixed with distraction spikes
    - Evening: distraction-heavy
    - Weekend: more distraction than weekdays
    
    This ensures FocusLens can be demonstrated without real browser history.
    """
    import numpy as np
    import random

    rng = np.random.default_rng(config.RANDOM_STATE)
    random.seed(config.RANDOM_STATE)

    # Realistic URL pools per category
    productive_urls = [
        ("github.com", "GitHub - Repository"),
        ("stackoverflow.com", "Stack Overflow - Question"),
        ("docs.python.org", "Python Documentation"),
        ("leetcode.com", "LeetCode - Problem"),
        ("kaggle.com", "Kaggle - Dataset"),
        ("coursera.org", "Coursera - Course"),
        ("notion.so", "Notion - Notes"),
        ("figma.com", "Figma - Design"),
        ("dev.to", "DEV Community - Article"),
        ("arxiv.org", "arXiv - Paper"),
    ]
    distraction_urls = [
        ("youtube.com", "YouTube - Video"),
        ("reddit.com", "Reddit - Thread"),
        ("instagram.com", "Instagram - Feed"),
        ("twitter.com", "Twitter - Timeline"),
        ("netflix.com", "Netflix - Show"),
        ("twitch.tv", "Twitch - Stream"),
        ("9gag.com", "9GAG - Memes"),
    ]
    neutral_urls = [
        ("google.com", "Google Search"),
        ("gmail.com", "Gmail - Inbox"),
        ("wikipedia.org", "Wikipedia - Article"),
        ("maps.google.com", "Google Maps"),
        ("slack.com", "Slack - Workspace"),
    ]

    records = []
    base_date = datetime.now() - timedelta(days=days)

    for day_offset in range(days):
        current_date = base_date + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5

        # Vary visits per day (weekends have fewer productive visits)
        daily_visits = int(visits_per_day * (0.6 if is_weekend else 1.0))
        daily_visits += rng.integers(-15, 15)

        for _ in range(max(daily_visits, 10)):
            # Pick hour with realistic distribution
            # Weekday: peaks at 10am and 3pm; weekend: peaks at 2pm
            if is_weekend:
                hour = int(rng.choice(range(24), p=_weekend_hour_probs()))
            else:
                hour = int(rng.choice(range(24), p=_weekday_hour_probs()))

            minute = rng.integers(0, 60)
            second = rng.integers(0, 60)
            visit_time = current_date.replace(
                hour=hour, minute=int(minute), second=int(second), microsecond=0
            )

            # Category probability varies by hour
            prod_prob = _productive_probability(hour, is_weekend)
            dist_prob = _distraction_probability(hour, is_weekend)
            neut_prob = 1.0 - prod_prob - dist_prob

            category_roll = rng.random()
            if category_roll < prod_prob:
                domain, title = random.choice(productive_urls)
            elif category_roll < prod_prob + dist_prob:
                domain, title = random.choice(distraction_urls)
            else:
                domain, title = random.choice(neutral_urls)

            # Duration: productive visits tend to be longer
            if "productive" in [domain]:
                duration = float(rng.exponential(180))  # avg 3 min
            else:
                duration = float(rng.exponential(90))   # avg 1.5 min

            duration = max(config.MIN_VISIT_DURATION_SECONDS, min(duration, 3600))

            records.append({
                "url": f"https://{domain}/page_{rng.integers(1000, 9999)}",
                "domain": domain,
                "title": title,
                "visit_time": visit_time,
                "duration_seconds": round(duration, 1),
            })

    df = pd.DataFrame(records).sort_values("visit_time").reset_index(drop=True)
    logger.info(f"Demo history generated: {len(df):,} visits over {days} days")
    return df


def _weekday_hour_probs() -> list:
    """Probability distribution over 24 hours for weekdays."""
    import numpy as np
    # Low at night, ramp up 8am, peak 10am and 3pm, taper evening
    raw = [0.1, 0.05, 0.03, 0.02, 0.02, 0.05,
           0.3,  0.8,  1.5,  2.0,  2.2,  1.8,
           1.5,  1.2,  1.8,  2.0,  1.5,  1.2,
           1.0,  0.8,  0.6,  0.4,  0.3,  0.2]
    arr = np.array(raw)
    return (arr / arr.sum()).tolist()


def _weekend_hour_probs() -> list:
    """Probability distribution over 24 hours for weekends."""
    import numpy as np
    raw = [0.2, 0.1, 0.05, 0.03, 0.02, 0.05,
           0.1,  0.3,  0.6,  0.9,  1.2,  1.4,
           1.5,  1.6,  2.0,  2.2,  2.0,  1.8,
           1.5,  1.2,  1.0,  0.8,  0.6,  0.4]
    arr = np.array(raw)
    return (arr / arr.sum()).tolist()


def _productive_probability(hour: int, is_weekend: bool) -> float:
    """Time-of-day probability of visiting a productive site."""
    if is_weekend:
        if 10 <= hour <= 14:
            return 0.35
        elif 9 <= hour <= 17:
            return 0.25
        else:
            return 0.10
    else:
        if 9 <= hour <= 12:
            return 0.55
        elif 13 <= hour <= 17:
            return 0.45
        elif 18 <= hour <= 20:
            return 0.25
        else:
            return 0.10


def _distraction_probability(hour: int, is_weekend: bool) -> float:
    """Time-of-day probability of visiting a distraction site."""
    if is_weekend:
        if 14 <= hour <= 22:
            return 0.50
        else:
            return 0.30
    else:
        if 12 <= hour <= 13:
            return 0.45  # lunch distraction spike
        elif 17 <= hour <= 22:
            return 0.55  # evening distraction
        elif 9 <= hour <= 11:
            return 0.15  # mostly focused morning
        else:
            return 0.30


if __name__ == "__main__":
    # Quick test: generate demo data and display summary
    df = generate_demo_history(days=30)
    print("\n=== FocusLens History Reader — Demo Data ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['visit_time'].min()} → {df['visit_time'].max()}")
    print(f"\nTop domains:\n{df['domain'].value_counts().head(10)}")
    print(f"\nSample rows:\n{df.head()}")
