# 🔬 FocusLens

> **Personal Productivity Analytics** — a 100% local, privacy-first desktop dashboard that uses unsupervised ML to find your focus patterns and distraction spirals from your own browser history.

---

## What It Does

FocusLens reads your Chrome history, classifies every visit into **productive / neutral / distraction**, then uses **KMeans + DBSCAN clustering** to surface:

- 📊 Your 4 recurring behavior archetypes (Deep Focus / Distracted / Mixed / Offline)  
- 🕐 Your **peak focus window** (e.g., "Tuesday 9–11 AM")  
- 🌀 **Distraction spirals** — exact timestamps where you fell into rabbit holes  
- 📅 **Anomalous days** flagged by DBSCAN (unusual work patterns)  
- 🤖 An **AI-written weekly coaching report** via Claude API  
- 🔢 A **focus score (0–100)** for every day  

**Zero data leaves your machine.** No login. No cloud. No tracking.

---

## Project Structure

```
focuslens/
├── config.py                  # All constants, domain lists, ML params
├── pipeline.py                # End-to-end orchestrator (run this first)
├── dashboard.py               # Streamlit dashboard (5 pages)
├── requirements.txt
├── .env.example               # Copy to .env and add API key
├── .gitignore
│
├── src/
│   ├── history_reader.py      # Chrome SQLite → clean DataFrame
│   ├── categorizer.py         # URL → productive/distraction/neutral
│   ├── feature_engineer.py    # Hourly & daily behavioral features
│   ├── clusterer.py           # KMeans + DBSCAN + PCA + auto-labeling
│   ├── database.py            # SQLite persistence layer (8 tables)
│   ├── insights_generator.py  # LLM report generation
│   └── visualizer.py          # All Plotly chart functions
│
├── chrome_extension/          # One-click launcher extension
│   ├── manifest.json
│   ├── popup.html
│   └── popup.js
│
└── data/                      # Auto-created (gitignored)
    ├── history_export.db      # Copy of Chrome history (read-only)
    └── focuslens.db           # FocusLens application database
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key (optional — for AI reports)

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Run the pipeline

```bash
# Option A: Use your real Chrome history (close Chrome first!)
python pipeline.py

# Option B: Use 30 days of realistic demo data
python pipeline.py --demo

# Option C: Demo data + skip LLM API call
python pipeline.py --demo --dry-run
```

### 4. Launch the dashboard

```bash
streamlit run dashboard.py
```

Then open **http://localhost:8501** in your browser.

---

## Dashboard Pages

| Page | What It Shows |
|------|--------------|
| 📊 Overview | Today's split (pie), weekly stacked bar, focus score trend, week-vs-week comparison |
| 🎯 Focus Patterns | Hour×Day heatmap, activity-by-hour bar chart, peak focus window label |
| 😵 Distraction Analysis | Top distraction sites, spiral timeline, context switch rate by hour |
| 🤖 ML Clusters | PCA scatter, cluster distribution, elbow method chart, DBSCAN anomaly calendar |
| ✨ AI Report | LLM-generated weekly coaching report with regenerate button |

---

## ML Architecture

### KMeans (k=4)
Clusters hourly feature vectors into 4 **behavior archetypes**:
- 🎯 **Deep Focus** — high productive ratio, low context switches
- 😵 **Distracted** — high distraction ratio, long streaks
- ⚖️ **Mixed Mode** — alternating productive/distraction, many switches  
- 💤 **Low Activity** — few visits, low engagement

Labels are **derived from centroid characteristics** — not hardcoded.

### DBSCAN
Detects **anomalous hours** that don't fit any cluster (label = -1).  
These correspond to days with unusual browsing patterns — all-nighters, sick days, hyperfocus sessions.

### Feature Matrix (9 features)
```
productive_ratio    distraction_ratio   context_switch_rate
avg_session_minutes domain_diversity    streak_ratio
is_weekend          hour_sin            hour_cos  ← cyclical encoding
```

All features are **StandardScaler-normalized** before clustering.  
PCA (2D) is used **only for visualization**, not for clustering.

---

## Privacy Design

| What | How |
|------|-----|
| Browser history | Copied to `data/` — original never touched |
| LLM API | Only aggregated summary stats sent — no URLs |
| All data | Stored in local SQLite — never leaves machine |
| Git | `.gitignore` excludes `data/`, `.env`, `*.db` |

---

## Chrome Extension (Optional)

The extension adds a **one-click launcher** in your browser toolbar.

**Install:**
1. Open Chrome → `chrome://extensions`
2. Enable **Developer Mode** (top right)
3. Click **Load unpacked**
4. Select the `chrome_extension/` folder

**Then:** Click the 🔬 icon to open your dashboard instantly.

> **Note:** The extension is a launcher only — it opens `localhost:8501`.  
> The Python pipeline must be running separately.

---

## Configuration

Edit `config.py` to customize:

```python
KMEANS_CLUSTERS = 4                    # Number of behavior archetypes
DBSCAN_EPS = 1.2                       # Anomaly sensitivity (lower = more anomalies)
HISTORY_LOOKBACK_DAYS = 30             # How far back to analyze
DAILY_DISTRACTION_LIMIT_MINUTES = 60  # Warning threshold
LLM_MODEL = "claude-sonnet-4-20250514"

# Add your own domains
PRODUCTIVE_DOMAINS = ["github.com", ...]
DISTRACTION_DOMAINS = ["youtube.com", ...]
```

You can also add domains live from the **Streamlit sidebar** — no restart needed.

---

## Resume Highlights

This project demonstrates:

- **Real data engineering**: SQLite ingestion, Chrome timestamp conversion, session detection
- **Feature engineering**: 9 behavioral features with cyclical time encoding
- **Dual ML algorithms**: KMeans (archetypes) + DBSCAN (anomaly detection)
- **MLOps patterns**: Elbow method, silhouette scoring, auto-labeling from centroids
- **LLM integration**: Privacy-safe API usage with fallback and caching
- **Production Streamlit**: 5-page app with custom CSS, caching, state management
- **Software architecture**: Modular pipeline, separation of concerns, idempotent storage

---

## License

MIT — use freely for personal and educational purposes.
