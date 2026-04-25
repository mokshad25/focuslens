"""
FocusLens — database.py
========================
SQLite data persistence layer.

Schema design:
  - raw_visits: every individual browser visit (after cleaning)
  - hourly_features: engineered feature matrix per (date, hour)
  - daily_summary: per-day aggregated metrics + focus score
  - cluster_assignments: KMeans + DBSCAN labels per hourly bucket
  - distraction_spirals: detected spiral events
  - ai_reports: cached LLM-generated weekly reports
  - user_domain_overrides: custom category assignments from UI

All inserts use INSERT OR REPLACE to make the pipeline idempotent
(safe to re-run without duplicating data).
"""

import sys
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

DB_PATH = config.APP_DB_PATH


# ─── Schema Definitions ───────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Raw browser visits (cleaned, categorized)
CREATE TABLE IF NOT EXISTS raw_visits (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    url               TEXT NOT NULL,
    domain            TEXT,
    clean_domain      TEXT,
    title             TEXT,
    visit_time        DATETIME NOT NULL,
    duration_seconds  REAL DEFAULT 0,
    category          TEXT DEFAULT 'unknown',
    session_id        INTEGER DEFAULT 0,
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_visits_time ON raw_visits(visit_time);
CREATE INDEX IF NOT EXISTS idx_visits_domain ON raw_visits(clean_domain);
CREATE INDEX IF NOT EXISTS idx_visits_category ON raw_visits(category);

-- Hourly feature matrix (ML input)
CREATE TABLE IF NOT EXISTS hourly_features (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    date                  DATE NOT NULL,
    hour                  INTEGER NOT NULL CHECK(hour >= 0 AND hour <= 23),
    day_of_week           INTEGER,
    is_weekend            INTEGER,
    total_visits          INTEGER DEFAULT 0,
    total_minutes         REAL DEFAULT 0,
    productive_visits     INTEGER DEFAULT 0,
    distraction_visits    INTEGER DEFAULT 0,
    neutral_visits        INTEGER DEFAULT 0,
    productive_ratio      REAL DEFAULT 0,
    distraction_ratio     REAL DEFAULT 0,
    neutral_ratio         REAL DEFAULT 0,
    context_switches      INTEGER DEFAULT 0,
    context_switch_rate   REAL DEFAULT 0,
    distraction_streak    INTEGER DEFAULT 0,
    unique_domains        INTEGER DEFAULT 0,
    avg_session_minutes   REAL DEFAULT 0,
    UNIQUE(date, hour)
);
CREATE INDEX IF NOT EXISTS idx_hourly_date ON hourly_features(date);

-- Daily summary metrics
CREATE TABLE IF NOT EXISTS daily_summary (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    date                        DATE NOT NULL UNIQUE,
    productive_minutes          REAL DEFAULT 0,
    distraction_minutes         REAL DEFAULT 0,
    neutral_minutes             REAL DEFAULT 0,
    total_minutes               REAL DEFAULT 0,
    productive_ratio            REAL DEFAULT 0,
    distraction_ratio           REAL DEFAULT 0,
    context_switches            INTEGER DEFAULT 0,
    unique_domains              INTEGER DEFAULT 0,
    total_visits                INTEGER DEFAULT 0,
    focus_score                 REAL DEFAULT 0,
    distraction_limit_exceeded  INTEGER DEFAULT 0,
    top_productive_domain       TEXT,
    top_distraction_domain      TEXT
);

-- KMeans + DBSCAN cluster assignments per hourly bucket
CREATE TABLE IF NOT EXISTS cluster_assignments (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    date           DATE NOT NULL,
    hour           INTEGER NOT NULL,
    kmeans_label   INTEGER,
    kmeans_name    TEXT,
    dbscan_label   INTEGER,
    pca_x          REAL,
    pca_y          REAL,
    UNIQUE(date, hour)
);
CREATE INDEX IF NOT EXISTS idx_clusters_date ON cluster_assignments(date);

-- Detected distraction spirals
CREATE TABLE IF NOT EXISTS distraction_spirals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             DATE NOT NULL,
    start_time       DATETIME,
    end_time         DATETIME,
    duration_minutes REAL,
    visit_count      INTEGER,
    domains          TEXT
);
CREATE INDEX IF NOT EXISTS idx_spirals_date ON distraction_spirals(date);

-- Cached AI-generated weekly reports
CREATE TABLE IF NOT EXISTS ai_reports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start    DATE NOT NULL,
    week_end      DATE NOT NULL,
    report_text   TEXT,
    model_used    TEXT,
    generated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(week_start)
);

-- User-defined domain category overrides (from dashboard UI)
CREATE TABLE IF NOT EXISTS user_domain_overrides (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    domain    TEXT NOT NULL UNIQUE,
    category  TEXT NOT NULL CHECK(category IN ('productive', 'distraction', 'neutral')),
    added_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Metadata: track last pipeline run
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    visits_loaded INTEGER,
    date_from     DATE,
    date_to       DATE,
    status        TEXT DEFAULT 'success'
);
"""


# ─── Connection Manager ───────────────────────────────────────────────────────

@contextmanager
def get_connection(db_path: Path = None):
    """Context manager for SQLite connections with WAL mode for performance."""
    if db_path is None:
        db_path = DB_PATH
    
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row            # access columns by name
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


# ─── Schema Init ──────────────────────────────────────────────────────────────

def init_database(db_path: Path = None) -> None:
    """Create all tables if they don't exist. Safe to call multiple times."""
    if db_path is None:
        db_path = DB_PATH
    
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
    
    logger.info(f"Database initialized: {db_path}")


# ─── Insert Functions ─────────────────────────────────────────────────────────

def insert_raw_visits(df: pd.DataFrame, db_path: Path = None) -> int:
    """
    Insert cleaned, categorized visits into raw_visits table.
    Uses INSERT OR IGNORE to avoid duplicates on re-run.
    
    Returns number of rows inserted.
    """
    if df.empty:
        return 0
    
    if db_path is None:
        db_path = DB_PATH
    
    # Prepare columns (only insert columns that exist in schema)
    cols = ["url", "domain", "clean_domain", "title", "visit_time",
            "duration_seconds", "category", "session_id"]
    
    insert_df = df.copy()
    
    # Ensure all required columns exist
    for col in cols:
        if col not in insert_df.columns:
            insert_df[col] = None
    
    # Convert datetime to string for SQLite
    insert_df["visit_time"] = insert_df["visit_time"].astype(str)
    
    records = insert_df[cols].to_dict("records")
    
    with get_connection(db_path) as conn:
        # Clear and re-insert for idempotency (easier than upsert with datetime PKs)
        conn.execute("DELETE FROM raw_visits")
        conn.executemany(
            """INSERT OR IGNORE INTO raw_visits
               (url, domain, clean_domain, title, visit_time, duration_seconds, category, session_id)
               VALUES (:url, :domain, :clean_domain, :title, :visit_time, :duration_seconds, :category, :session_id)
            """,
            records
        )
    
    logger.info(f"Inserted {len(records)} visits into raw_visits")
    return len(records)


def insert_hourly_features(hourly_df: pd.DataFrame, db_path: Path = None) -> None:
    """Insert or replace hourly feature rows."""
    if hourly_df.empty:
        return
    
    if db_path is None:
        db_path = DB_PATH
    
    cols = [
        "date", "hour", "day_of_week", "is_weekend", "total_visits",
        "total_minutes", "productive_visits", "distraction_visits", "neutral_visits",
        "productive_ratio", "distraction_ratio", "neutral_ratio",
        "context_switches", "context_switch_rate", "distraction_streak",
        "unique_domains", "avg_session_minutes"
    ]
    
    df = hourly_df.copy()
    df["date"] = df["date"].astype(str)
    available = [c for c in cols if c in df.columns]
    records = df[available].to_dict("records")
    
    placeholders = ", ".join(f":{c}" for c in available)
    col_names = ", ".join(available)
    
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM hourly_features")
        conn.executemany(
            f"INSERT OR REPLACE INTO hourly_features ({col_names}) VALUES ({placeholders})",
            records
        )
    
    logger.info(f"Inserted {len(records)} hourly feature rows")


def insert_daily_summary(daily_df: pd.DataFrame, db_path: Path = None) -> None:
    """Insert or replace daily summary rows."""
    if daily_df.empty:
        return
    
    if db_path is None:
        db_path = DB_PATH
    
    cols = [
        "date", "productive_minutes", "distraction_minutes", "neutral_minutes",
        "total_minutes", "productive_ratio", "distraction_ratio", "context_switches",
        "unique_domains", "total_visits", "focus_score", "distraction_limit_exceeded",
        "top_productive_domain", "top_distraction_domain"
    ]
    
    df = daily_df.copy()
    df["date"] = df["date"].astype(str)
    df["distraction_limit_exceeded"] = df["distraction_limit_exceeded"].astype(int)
    
    available = [c for c in cols if c in df.columns]
    records = df[available].to_dict("records")
    placeholders = ", ".join(f":{c}" for c in available)
    col_names = ", ".join(available)
    
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM daily_summary")
        conn.executemany(
            f"INSERT OR REPLACE INTO daily_summary ({col_names}) VALUES ({placeholders})",
            records
        )
    
    logger.info(f"Inserted {len(records)} daily summary rows")


def insert_cluster_assignments(
    feature_df: pd.DataFrame,
    km_labels: list,
    cluster_names: dict,
    dbscan_labels: list,
    X_pca,
    db_path: Path = None,
) -> None:
    """Insert KMeans and DBSCAN cluster assignments with PCA coordinates."""
    if db_path is None:
        db_path = DB_PATH
    
    records = []
    for i, (_, row) in enumerate(feature_df.iterrows()):
        km_label = int(km_labels[i])
        records.append({
            "date": str(row["date"]),
            "hour": int(row["hour"]),
            "kmeans_label": km_label,
            "kmeans_name": cluster_names.get(km_label, "Unknown"),
            "dbscan_label": int(dbscan_labels[i]),
            "pca_x": float(X_pca[i, 0]),
            "pca_y": float(X_pca[i, 1]),
        })
    
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM cluster_assignments")
        conn.executemany(
            """INSERT OR REPLACE INTO cluster_assignments
               (date, hour, kmeans_label, kmeans_name, dbscan_label, pca_x, pca_y)
               VALUES (:date, :hour, :kmeans_label, :kmeans_name, :dbscan_label, :pca_x, :pca_y)
            """,
            records
        )
    
    logger.info(f"Inserted {len(records)} cluster assignment rows")


def insert_distraction_spirals(spirals_df: pd.DataFrame, db_path: Path = None) -> None:
    """Insert detected distraction spiral events."""
    if spirals_df.empty:
        return
    
    if db_path is None:
        db_path = DB_PATH
    
    df = spirals_df.copy()
    df["date"] = df["date"].astype(str)
    df["start_time"] = df["start_time"].astype(str)
    df["end_time"] = df["end_time"].astype(str)
    
    records = df.to_dict("records")
    
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM distraction_spirals")
        conn.executemany(
            """INSERT INTO distraction_spirals
               (date, start_time, end_time, duration_minutes, visit_count, domains)
               VALUES (:date, :start_time, :end_time, :duration_minutes, :visit_count, :domains)
            """,
            records
        )
    
    logger.info(f"Inserted {len(records)} distraction spiral records")


def save_ai_report(week_start: str, week_end: str, report_text: str, db_path: Path = None) -> None:
    """Cache an AI-generated report (upsert by week_start)."""
    if db_path is None:
        db_path = DB_PATH
    
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO ai_reports
               (week_start, week_end, report_text, model_used, generated_at)
               VALUES (?, ?, ?, ?, ?)
            """,
            (week_start, week_end, report_text, config.LLM_MODEL, datetime.now().isoformat())
        )
    
    logger.info(f"AI report saved for week {week_start} → {week_end}")


def get_cached_report(week_start: str, db_path: Path = None) -> Optional[str]:
    """Retrieve a cached AI report for the given week start date."""
    if db_path is None:
        db_path = DB_PATH
    
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT report_text FROM ai_reports WHERE week_start = ? ORDER BY generated_at DESC LIMIT 1",
            (week_start,)
        ).fetchone()
    
    return row["report_text"] if row else None


def save_domain_override(domain: str, category: str, db_path: Path = None) -> None:
    """Save a user-defined domain category override."""
    if db_path is None:
        db_path = DB_PATH
    
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_domain_overrides (domain, category) VALUES (?, ?)",
            (domain.lower(), category)
        )
    
    logger.info(f"Domain override saved: {domain} → {category}")


def get_domain_overrides(db_path: Path = None) -> dict:
    """Load all user-defined domain overrides as a dict {domain: category}."""
    if db_path is None:
        db_path = DB_PATH
    
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute("SELECT domain, category FROM user_domain_overrides").fetchall()
        return {row["domain"]: row["category"] for row in rows}
    except Exception:
        return {}


# ─── Query Functions ──────────────────────────────────────────────────────────

def query_daily_summary(db_path: Path = None) -> pd.DataFrame:
    """Load daily summary table as DataFrame."""
    if db_path is None:
        db_path = DB_PATH
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM daily_summary ORDER BY date", conn)


def query_hourly_features(db_path: Path = None) -> pd.DataFrame:
    if db_path is None:
        db_path = DB_PATH
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM hourly_features ORDER BY date, hour", conn)


def query_cluster_assignments(db_path: Path = None) -> pd.DataFrame:
    if db_path is None:
        db_path = DB_PATH
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM cluster_assignments ORDER BY date, hour", conn)


def query_raw_visits(db_path: Path = None) -> pd.DataFrame:
    if db_path is None:
        db_path = DB_PATH
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM raw_visits ORDER BY visit_time", conn)
    if not df.empty and "visit_time" in df.columns:
        df["visit_time"] = pd.to_datetime(df["visit_time"])
    return df


def query_distraction_spirals(db_path: Path = None) -> pd.DataFrame:
    if db_path is None:
        db_path = DB_PATH
    with get_connection(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM distraction_spirals ORDER BY start_time", conn)


def log_pipeline_run(visits_loaded: int, date_from, date_to, status: str = "success", db_path: Path = None) -> None:
    if db_path is None:
        db_path = DB_PATH
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO pipeline_runs (visits_loaded, date_from, date_to, status) VALUES (?, ?, ?, ?)",
            (visits_loaded, str(date_from), str(date_to), status)
        )


if __name__ == "__main__":
    init_database()
    print(f"Database created at: {DB_PATH}")
    
    # Verify schema
    with get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        print("Tables created:")
        for t in tables:
            print(f"  - {t['name']}")
