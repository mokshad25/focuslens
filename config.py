"""
FocusLens Configuration
=======================
Central configuration for all constants, domain mappings, and ML parameters.
Edit this file to customize categorization and model behavior.
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

HISTORY_COPY_PATH = DATA_DIR / "history_export.db"
APP_DB_PATH = DATA_DIR / "focuslens.db"

# ─── Chrome History File Locations (by OS) ────────────────────────────────────
CHROME_HISTORY_PATHS = {
    "win32":  Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/User Data/Default/History",
    "darwin": Path.home() / "Library/Application Support/Google/Chrome/Default/History",
    "linux":  Path.home() / ".config/google-chrome/Default/History",
}

FIREFOX_HISTORY_PATHS = {
    "win32":  Path(os.environ.get("APPDATA", "")) / "Mozilla/Firefox/Profiles",
    "darwin": Path.home() / "Library/Application Support/Firefox/Profiles",
    "linux":  Path.home() / ".mozilla/firefox",
}

# ─── Domain Category Mappings ─────────────────────────────────────────────────
PRODUCTIVE_DOMAINS = [
    # Development
    "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com",
    "docs.python.org", "developer.mozilla.org", "devdocs.io",
    "replit.com", "codepen.io", "codesandbox.io", "jsfiddle.net",
    "leetcode.com", "hackerrank.com", "codeforces.com", "kaggle.com",
    # Learning
    "coursera.org", "udemy.com", "edx.org", "khanacademy.org",
    "pluralsight.com", "frontendmasters.com", "egghead.io",
    "brilliant.org", "duolingo.com", "codecademy.com",
    # Research & Reading
    "arxiv.org", "scholar.google.com", "semanticscholar.org",
    "medium.com", "dev.to", "hashnode.dev", "substack.com",
    "nature.com", "sciencedirect.com", "pubmed.ncbi.nlm.nih.gov",
    # Productivity & Work Tools
    "notion.so", "obsidian.md", "roamresearch.com",
    "linear.app", "jira.atlassian.com", "trello.com", "asana.com",
    "figma.com", "miro.com", "excalidraw.com",
    "docs.google.com", "sheets.google.com", "slides.google.com",
    "dropbox.com", "drive.google.com",
    "anthropic.com", "openai.com", "huggingface.co",
    "vercel.com", "netlify.com", "heroku.com", "render.com",
    "aws.amazon.com", "console.cloud.google.com", "portal.azure.com",
]

DISTRACTION_DOMAINS = [
    # Social Media
    "youtube.com", "instagram.com", "twitter.com", "x.com",
    "reddit.com", "facebook.com", "tiktok.com", "snapchat.com",
    "pinterest.com", "tumblr.com", "linkedin.com",
    # Entertainment
    "netflix.com", "primevideo.com", "hulu.com", "disneyplus.com",
    "twitch.tv", "crunchyroll.com", "funimation.com",
    "spotify.com", "soundcloud.com",
    # News & Clickbait
    "buzzfeed.com", "9gag.com", "ifunny.co", "imgur.com",
    "dailymail.co.uk", "tmz.com",
    # Gaming
    "store.steampowered.com", "epicgames.com", "roblox.com",
    "miniclip.com", "kongregate.com",
]

NEUTRAL_DOMAINS = [
    # Search & Navigation
    "google.com", "bing.com", "duckduckgo.com", "yahoo.com",
    "ecosia.org", "startpage.com",
    # Communication
    "gmail.com", "outlook.com", "mail.yahoo.com", "protonmail.com",
    "slack.com", "discord.com", "telegram.org", "whatsapp.com",
    "zoom.us", "meet.google.com", "teams.microsoft.com",
    # Reference
    "wikipedia.org", "wikimedia.org", "wiktionary.org",
    "maps.google.com", "weather.com", "timeanddate.com",
    # Shopping (neither productive nor distraction — context-dependent)
    "amazon.com", "ebay.com", "etsy.com",
]

# ─── ML Parameters ────────────────────────────────────────────────────────────
KMEANS_CLUSTERS = 4
PCA_COMPONENTS = 2
DBSCAN_EPS = 1.2
DBSCAN_MIN_SAMPLES = 5
RANDOM_STATE = 42

# ─── Time & Session Parameters ────────────────────────────────────────────────
HISTORY_LOOKBACK_DAYS = 30
MIN_VISIT_DURATION_SECONDS = 3       # filter out accidental tab opens
SESSION_GAP_MINUTES = 30             # gap that defines end of a browsing session
HOURLY_BUCKET_MINUTES = 60          # feature aggregation window

# ─── Focus Score Weights (must sum to 1.0) ────────────────────────────────────
FOCUS_SCORE_WEIGHTS = {
    "productive_ratio":     0.45,
    "distraction_ratio":    -0.35,   # negative = penalizes distraction
    "context_switches_inv": 0.10,    # inverse: fewer switches = better
    "session_length_norm":  0.10,
}

# ─── Dashboard Thresholds ─────────────────────────────────────────────────────
DAILY_DISTRACTION_LIMIT_MINUTES = 60   # warn if exceeded
FOCUS_SCORE_THRESHOLDS = {
    "excellent": 75,
    "good":      55,
    "average":   35,
    "poor":      0,
}

# ─── LLM Configuration ────────────────────────────────────────────────────────
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 1024

LLM_SYSTEM_PROMPT = """You are a personal productivity coach. You will receive a structured summary 
of someone's weekly browsing behavior. Write a warm, honest, actionable 3-paragraph productivity report.

Paragraph 1: What they did well this week — specific strengths, peak hours, best patterns.
Paragraph 2: Patterns of distraction — when they occur, what triggers them, how long they last.
Paragraph 3: 3 specific, practical suggestions to improve next week. 

Be direct but encouraging. Do not be generic. Reference the actual data points provided.
Keep the total response under 300 words. No bullet points — flowing prose only."""

# ─── Color Palette for Dashboard ──────────────────────────────────────────────
CLUSTER_COLORS = {
    0: "#00C9A7",   # Deep Focus — teal
    1: "#FF6B6B",   # Distracted — coral red
    2: "#FFD93D",   # Moderate Work — amber
    3: "#C4C4C4",   # Low Activity — grey
    -1: "#9B59B6",  # DBSCAN noise/anomaly — purple
}

CATEGORY_COLORS = {
    "productive":  "#00C9A7",
    "neutral":     "#74B9FF",
    "distraction": "#FF6B6B",
    "unknown":     "#DFE6E9",
}

CLUSTER_LABELS = {
    0: "🎯 Deep Focus",
    1: "😵 Distracted",
    2: "⚖️ Mixed Mode",
    3: "💤 Low Activity",
}
