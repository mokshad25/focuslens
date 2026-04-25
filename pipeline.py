"""
FocusLens — pipeline.py
========================
End-to-end pipeline orchestrator. Runs all phases in sequence:
  1. Load browser history (or demo data)
  2. Categorize URLs
  3. Assign sessions + engineer features
  4. Run ML clustering
  5. Store everything to SQLite
  6. Detect distraction spirals

Run this before launching the dashboard:
    python pipeline.py            # uses real Chrome history
    python pipeline.py --demo     # uses generated demo data
    python pipeline.py --dry-run  # skips LLM API call
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.history_reader import load_history, generate_demo_history
from src.categorizer import categorize_dataframe
from src.feature_engineer import (
    assign_sessions,
    build_hourly_features,
    build_daily_summary,
    build_feature_matrix,
    detect_distraction_spirals,
)
from src.clusterer import run_full_clustering
from src.database import (
    init_database,
    insert_raw_visits,
    insert_hourly_features,
    insert_daily_summary,
    insert_cluster_assignments,
    insert_distraction_spirals,
    get_domain_overrides,
    log_pipeline_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("focuslens.pipeline")


def run_pipeline(use_demo: bool = False, dry_run: bool = False) -> dict:
    """
    Execute the full FocusLens data pipeline.

    Args:
        use_demo: Generate synthetic data instead of reading Chrome history.
        dry_run:  Skip the LLM API call for the insights report.

    Returns:
        dict with all computed artifacts (for dashboard or testing).
    """
    t0 = time.time()
    logger.info("=" * 55)
    logger.info("  FocusLens Pipeline Starting")
    logger.info("=" * 55)

    # ── Phase 0: Init DB ──────────────────────────────────────────────────
    logger.info("[0/6] Initialising database…")
    init_database()

    # ── Phase 1: Load history ─────────────────────────────────────────────
    logger.info("[1/6] Loading browser history…")
    try:
        if use_demo:
            raw_df = generate_demo_history(days=config.HISTORY_LOOKBACK_DAYS)
            logger.info("Demo data generated.")
        else:
            raw_df = load_history(lookback_days=config.HISTORY_LOOKBACK_DAYS)
    except (FileNotFoundError, PermissionError) as e:
        logger.warning(f"Could not load real history: {e}")
        logger.info("Falling back to demo data.")
        raw_df = generate_demo_history(days=config.HISTORY_LOOKBACK_DAYS)

    if raw_df.empty:
        logger.error("No history data loaded. Aborting pipeline.")
        return {}

    # ── Phase 2: Categorize ───────────────────────────────────────────────
    logger.info("[2/6] Categorising URLs…")
    # Load any user-defined overrides from DB
    overrides = get_domain_overrides()
    productive_custom  = [d for d, c in overrides.items() if c == "productive"]
    distraction_custom = [d for d, c in overrides.items() if c == "distraction"]
    neutral_custom     = [d for d, c in overrides.items() if c == "neutral"]

    df = categorize_dataframe(
        raw_df,
        custom_productive=productive_custom,
        custom_distraction=distraction_custom,
        custom_neutral=neutral_custom,
    )

    # ── Phase 3: Feature engineering ──────────────────────────────────────
    logger.info("[3/6] Engineering features…")
    df = assign_sessions(df)
    hourly_df = build_hourly_features(df)
    daily_df  = build_daily_summary(df)
    spirals_df = detect_distraction_spirals(df)

    logger.info(
        f"  Hourly buckets: {len(hourly_df)} | "
        f"Daily rows: {len(daily_df)} | "
        f"Spirals: {len(spirals_df)}"
    )

    # ── Phase 4: ML clustering ────────────────────────────────────────────
    logger.info("[4/6] Running ML clustering…")
    try:
        X, feat_df = build_feature_matrix(hourly_df)
        results = run_full_clustering(X, feat_df)
        logger.info(
            f"  KMeans silhouette: {results['silhouette']:.4f} | "
            f"DBSCAN anomalies: {len(results['anomaly_days'])}"
        )
    except ValueError as e:
        logger.warning(f"Clustering skipped: {e}")
        results = {}

    # ── Phase 5: Persist to SQLite ────────────────────────────────────────
    logger.info("[5/6] Storing data to SQLite…")
    insert_raw_visits(df)
    insert_hourly_features(hourly_df)
    insert_daily_summary(daily_df)
    insert_distraction_spirals(spirals_df)

    if results:
        insert_cluster_assignments(
            feat_df,
            results["km_labels"],
            results["cluster_names"],
            results["dbscan_labels"],
            results["X_pca"],
        )

    # ── Phase 6: Log run ──────────────────────────────────────────────────
    logger.info("[6/6] Logging pipeline run…")
    log_pipeline_run(
        visits_loaded=len(df),
        date_from=df["visit_time"].min().date(),
        date_to=df["visit_time"].max().date(),
    )

    elapsed = time.time() - t0
    logger.info(f"Pipeline complete in {elapsed:.1f}s ✓")
    logger.info("Run `streamlit run dashboard.py` to open the dashboard.")

    return {
        "df": df,
        "hourly_df": hourly_df,
        "daily_df": daily_df,
        "spirals_df": spirals_df,
        "cluster_results": results,
        "feat_df": feat_df if results else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FocusLens data pipeline")
    parser.add_argument("--demo",    action="store_true", help="Use generated demo data")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM API calls")
    args = parser.parse_args()

    run_pipeline(use_demo=args.demo, dry_run=args.dry_run)
