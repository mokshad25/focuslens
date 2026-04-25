# FocusLens Project Analysis

Generated on 2026-04-25 after reading the full repository.

## High-Level Purpose

FocusLens is a local-first personal productivity analytics app. It reads browser history, classifies visits into productivity categories, engineers behavioral features, runs clustering/anomaly detection, persists results to SQLite, and displays the analysis in a Streamlit dashboard. A small Chrome extension acts as a launcher for the local dashboard.

The codebase is compact and mostly organized around one pipeline:

1. Load Chrome history or generated demo history.
2. Categorize every visit by domain.
3. Assign sessions and build hourly/daily metrics.
4. Run KMeans, DBSCAN, and PCA on active hourly buckets.
5. Store artifacts in `data/focuslens.db`.
6. Render dashboards and optional Claude-generated productivity reports.

## Repository Shape

- `README.md`: product overview, setup instructions, architecture notes, dashboard page descriptions, privacy claims.
- `config.py`: central paths, domain category lists, ML parameters, session settings, focus score thresholds, Claude model config, chart colors.
- `pipeline.py`: end-to-end orchestration. This is the first command to run before opening the dashboard.
- `dashboard.py`: Streamlit app entrypoint and page routing.
- `requirements.txt`: Python runtime dependencies.
- `src/history_reader.py`: Chrome history copying, Chrome timestamp conversion, cleaning, and demo data generation.
- `src/categorizer.py`: URL/domain extraction and productive/distraction/neutral/unknown classification.
- `src/feature_engineer.py`: sessions, hourly feature rows, daily summaries, focus scores, distraction spiral detection.
- `src/clusterer.py`: StandardScaler, elbow analysis, KMeans, DBSCAN, PCA, cluster labeling.
- `src/database.py`: SQLite schema, inserts, queries, cached reports, domain overrides, pipeline run logs.
- `src/insights_generator.py`: privacy-reduced weekly summary construction and Claude/static report generation.
- `src/visualizer.py`: Plotly chart factory functions used by the Streamlit app.
- `chrome_extension/`: Manifest V3 popup that checks and opens `localhost:8501`.
- `data/`: generated local databases. This directory currently contains `history_export.db` and `focuslens.db`.

Approximate source size is 4,227 lines across the main tracked files.

## Runtime Flow

### Pipeline

`pipeline.py` exposes `run_pipeline(use_demo=False, dry_run=False)` and a CLI:

```bash
python3 pipeline.py
python3 pipeline.py --demo
python3 pipeline.py --demo --dry-run
```

Internally it:

1. Calls `init_database()`.
2. Loads real Chrome history through `load_history()` or synthetic data through `generate_demo_history()`.
3. Reads user domain overrides from SQLite and passes them into `categorize_dataframe()`.
4. Calls `assign_sessions()`, `build_hourly_features()`, `build_daily_summary()`, and `detect_distraction_spirals()`.
5. Builds an active-hour feature matrix with `build_feature_matrix()`.
6. Calls `run_full_clustering()` when there is enough data.
7. Replaces the main analysis tables in SQLite.
8. Logs the run in `pipeline_runs`.

The pipeline does not currently call the LLM. The `dry_run` parameter is parsed and documented, but it has no behavioral effect inside `pipeline.py`.

### Dashboard

`dashboard.py` is a Streamlit application:

```bash
streamlit run dashboard.py
```

It caches database reads for 5 minutes through `load_all_data()`. The sidebar provides navigation, lookback selection, distraction-limit selection, buttons to run real/demo pipelines, and domain override creation.

Currently routed pages:

- Overview
- Focus Patterns
- Distraction Analysis
- AI Report

The README describes an `ML Clusters` page, and `dashboard.py` imports the cluster chart functions, but there is no ML Clusters page in the sidebar or router right now.

## Data Model

SQLite database path: `data/focuslens.db`.

Tables:

- `raw_visits`: cleaned categorized visit rows.
- `hourly_features`: complete date/hour grid with activity and behavior features.
- `daily_summary`: daily totals, ratios, focus score, top domains.
- `cluster_assignments`: KMeans/DBSCAN/PCA results for active hours only.
- `distraction_spirals`: detected sequences of at least 3 distraction visits.
- `ai_reports`: cached weekly generated reports.
- `user_domain_overrides`: dashboard-created domain category overrides.
- `pipeline_runs`: run history and date coverage.

Current local database snapshot:

- `raw_visits`: 4,565 rows.
- `hourly_features`: 744 rows.
- `daily_summary`: 30 rows.
- `cluster_assignments`: 226 rows.
- `distraction_spirals`: 133 rows.
- `ai_reports`: 2 rows.
- `user_domain_overrides`: 1 row.
- `pipeline_runs`: 16 rows.
- Date range in `daily_summary`: 2026-03-26 to 2026-04-25.

## Feature Engineering

Sessions are created from time gaps greater than `SESSION_GAP_MINUTES`, default 30 minutes.

Hourly rows include:

- date, hour, day of week, weekend flag
- total visits and total minutes
- productive/distraction/neutral visits
- productive/distraction/neutral ratios
- context switches and context switch rate
- longest distraction streak
- unique domains
- average session minutes

The hourly grid is filled for every hour of every date in the observed range. Zero-activity hours are useful for dashboard continuity, but the ML feature matrix filters them out before clustering.

Daily summaries include:

- category minutes
- total minutes
- category ratios
- context switches
- unique domains
- total visits
- focus score
- whether the distraction limit was exceeded
- top productive/distraction domains

Focus score is a 0-100 weighted score using productive ratio, inverse distraction ratio, context-switch score, and a baseline for activity.

Distraction spirals are consecutive runs of 3 or more visits categorized as `distraction`.

## ML Design

`src/clusterer.py` uses:

- `StandardScaler` for all ML features.
- KMeans with `KMEANS_CLUSTERS = 4`.
- Rule-based cluster naming from inverse-transformed centroids.
- DBSCAN with configurable `DBSCAN_EPS` and `DBSCAN_MIN_SAMPLES`.
- PCA with 2 components for visualization only.
- Elbow analysis for k=2 through k=7.

Feature matrix columns:

- `productive_ratio`
- `distraction_ratio`
- `context_switch_rate`
- `avg_session_minutes`
- `domain_diversity`
- `streak_ratio`
- `is_weekend`
- `hour_sin`
- `hour_cos`

Cluster assignments are stored only for active hourly buckets, so `cluster_assignments` will usually have fewer rows than `hourly_features`.

## Domain Categorization

Default domains live in `config.py`:

- `PRODUCTIVE_DOMAINS`
- `DISTRACTION_DOMAINS`
- `NEUTRAL_DOMAINS`

`categorizer.py` prefers `tldextract` and falls back to simple string parsing. It builds a lookup where built-in productive domains override built-in distraction/neutral overlaps, and user overrides are applied after defaults. Unknown domains are tracked but not surfaced prominently in the current dashboard.

Important nuance: `extract_domain()` returns registered domains such as `python.org`, while the config includes entries like `docs.python.org`. Because `categorize_url()` checks exact registered domains first and then a subdomain fallback, some specific subdomain config entries may not match as intended after `tldextract` normalization.

## AI Report

`insights_generator.py` builds a one-week summary from daily, raw, and hourly data. It can call Anthropic Claude using:

- environment variable `ANTHROPIC_API_KEY`
- model from `config.LLM_MODEL`
- max tokens from `config.LLM_MAX_TOKENS`

If the key is missing, the SDK is missing, the API call fails, or dry-run mode is used, it returns a static rule-based report.

The README says no URLs are sent to the LLM. That is true for full URLs, but the current weekly summary may include top productive and distraction domain names.

## Chrome Extension

The extension popup:

- checks whether `http://localhost:<port>` responds
- opens the dashboard in a new tab
- lets the user save a custom port
- shows quick terminal commands

Potential issues:

- `manifest.json` currently has `"permissions": []`, but `popup.js` uses `chrome.storage.local` and `chrome.tabs.create()`. It likely needs at least `storage` and possibly `tabs`.
- The manifest references `icon16.png`, `icon48.png`, and `icon128.png`, but those files are not currently present in `chrome_extension/`.

## Verification Performed

I ran Python bytecode compilation across the project:

```bash
python3 -m compileall pipeline.py dashboard.py config.py src
```

It completed without syntax errors.

I also inspected the local SQLite database schema and counts with `sqlite3`.

## Notable Gaps And Risks

- README mentions `.env.example`, but the repo currently has `.env` and no `.env.example`.
- README mentions an ML Clusters dashboard page, but it is not routed in `dashboard.py`.
- `pipeline.py --dry-run` is accepted but unused because report generation happens only in the dashboard.
- `insert_distraction_spirals()` returns early when the new spiral DataFrame is empty, so old spiral rows could remain after a rerun with no spirals.
- Demo data duration logic checks `if "productive" in [domain]`, which is never true for domains like `github.com`; productive demo visits therefore do not get the intended longer duration distribution.
- `page_overview()` receives the sidebar `dist_limit`, but its warning uses `config.DAILY_DISTRACTION_LIMIT_MINUTES` instead of the current slider value.
- `dashboard.py` imports cluster visualizations but does not expose them in the UI.
- `chart_anomaly_calendar()` expects an anomaly-date DataFrame, but only raw cluster assignments are currently queried from SQLite; anomaly-day summaries are not persisted separately.
- `data/` contains local databases, so privacy-sensitive work should avoid committing generated data.
- The app is local-first, but AI report generation can send aggregate data and domain names to Anthropic when enabled.

## Good Places To Start For Future Tasks

- Pipeline changes: start in `pipeline.py`, then inspect the relevant `src/` module and `src/database.py`.
- Dashboard UI changes: start in `dashboard.py`, then use chart helpers from `src/visualizer.py`.
- Categorization behavior: start in `src/categorizer.py` and `config.py`.
- Feature or score changes: start in `src/feature_engineer.py`.
- ML behavior: start in `src/clusterer.py`; remember only active hours are clustered.
- Persistence changes: update `SCHEMA_SQL`, insert functions, and query functions together in `src/database.py`.
- AI report changes: start in `src/insights_generator.py`; keep privacy implications explicit.
- Extension fixes: start in `chrome_extension/manifest.json` and `chrome_extension/popup.js`.

