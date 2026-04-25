"""
FocusLens — categorizer.py
===========================
Classifies browser visits into: productive, distraction, neutral, unknown.

Strategy:
1. Extract clean domain using tldextract (handles subdomains, ccTLDs, etc.)
2. Match against config domain lists
3. Unknown domains are flagged for user review via the dashboard
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import tldextract
    TLDEXTRACT_AVAILABLE = True
except ImportError:
    TLDEXTRACT_AVAILABLE = False
    logging.warning("tldextract not available — using basic domain extraction")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def extract_domain(url: str) -> str:
    """
    Extract the registered domain from a URL using tldextract.
    
    Examples:
        "https://www.github.com/user/repo" → "github.com"
        "https://docs.python.org/3/library" → "python.org"  (registered domain)
        "http://stackoverflow.com/questions/1" → "stackoverflow.com"
    
    Falls back to basic string splitting if tldextract is unavailable.
    """
    if not url or not isinstance(url, str):
        return "unknown"
    
    if TLDEXTRACT_AVAILABLE:
        extracted = tldextract.extract(url)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}".lower()
        elif extracted.domain:
            return extracted.domain.lower()
        return "unknown"
    else:
        # Basic fallback: strip protocol, www, and path
        domain = url.lower()
        for prefix in ("https://", "http://", "www."):
            domain = domain.replace(prefix, "")
        return domain.split("/")[0].split("?")[0].split("#")[0]


def _build_domain_lookup(custom_productive: list = None,
                          custom_distraction: list = None,
                          custom_neutral: list = None) -> dict:
    """
    Build a flat domain → category lookup dictionary.
    
    Priority: custom > config defaults.
    Custom domains allow users to add their own via the dashboard.
    """
    lookup = {}
    
    # Load defaults from config (distraction first, productive overrides if overlap)
    for domain in config.NEUTRAL_DOMAINS:
        lookup[domain.lower()] = "neutral"
    for domain in config.DISTRACTION_DOMAINS:
        lookup[domain.lower()] = "distraction"
    for domain in config.PRODUCTIVE_DOMAINS:
        lookup[domain.lower()] = "productive"
    
    # Apply custom overrides (user-defined in dashboard)
    for domain in (custom_neutral or []):
        lookup[domain.lower()] = "neutral"
    for domain in (custom_distraction or []):
        lookup[domain.lower()] = "distraction"
    for domain in (custom_productive or []):
        lookup[domain.lower()] = "productive"
    
    return lookup


def categorize_url(url: str, domain: str, lookup: dict) -> str:
    """
    Categorize a single URL based on its domain.
    
    Matching strategy (most to least specific):
    1. Exact registered domain match (e.g., "github.com")
    2. Subdomain match — check if any lookup key is a suffix of the domain
    3. Default to "unknown"
    """
    if not domain or domain == "unknown":
        return "unknown"
    
    domain_lower = domain.lower()
    
    # Exact match first
    if domain_lower in lookup:
        return lookup[domain_lower]
    
    # Subdomain fallback: "mail.google.com" → check "google.com"
    parts = domain_lower.split(".")
    if len(parts) > 2:
        registered = ".".join(parts[-2:])
        if registered in lookup:
            return lookup[registered]
    
    return "unknown"


def categorize_dataframe(
    df: pd.DataFrame,
    custom_productive: list = None,
    custom_distraction: list = None,
    custom_neutral: list = None,
) -> pd.DataFrame:
    """
    Add 'category' and 'clean_domain' columns to the history DataFrame.
    
    Args:
        df: DataFrame with at least 'url' and 'domain' columns.
        custom_productive: Additional domains to mark productive.
        custom_distraction: Additional domains to mark as distraction.
        custom_neutral: Additional domains to mark as neutral.
    
    Returns:
        DataFrame with added columns:
          - clean_domain: tldextract-cleaned domain
          - category: one of productive / distraction / neutral / unknown
    """
    result = df.copy()
    
    # Build lookup once for efficiency
    lookup = _build_domain_lookup(custom_productive, custom_distraction, custom_neutral)
    
    # Re-extract domains cleanly using tldextract (original extraction was basic)
    result["clean_domain"] = result["url"].apply(extract_domain)
    
    # Apply categorization row-by-row (vectorized via apply)
    result["category"] = result.apply(
        lambda row: categorize_url(row["url"], row["clean_domain"], lookup),
        axis=1
    )
    
    # Log category distribution
    counts = result["category"].value_counts()
    logger.info(f"Category distribution: {counts.to_dict()}")
    
    unknown_count = (result["category"] == "unknown").sum()
    if unknown_count > 0:
        unknown_domains = result[result["category"] == "unknown"]["clean_domain"].value_counts().head(10)
        logger.info(f"Top unknown domains (add to config):\n{unknown_domains}")
    
    return result


def get_unknown_domains(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Return the most frequently visited uncategorized domains.
    Used by the dashboard to let users manually assign categories.
    """
    if "category" not in df.columns:
        raise ValueError("DataFrame must be categorized first. Run categorize_dataframe().")
    
    unknown_df = df[df["category"] == "unknown"].copy()
    domain_counts = (
        unknown_df.groupby("clean_domain")
        .agg(visits=("url", "count"), total_seconds=("duration_seconds", "sum"))
        .sort_values("visits", ascending=False)
        .head(top_n)
        .reset_index()
    )
    domain_counts["total_minutes"] = (domain_counts["total_seconds"] / 60).round(1)
    return domain_counts[["clean_domain", "visits", "total_minutes"]]


def get_category_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict of time spent per category."""
    if "category" not in df.columns:
        return {}
    
    summary = {}
    for cat in ["productive", "distraction", "neutral", "unknown"]:
        subset = df[df["category"] == cat]
        summary[cat] = {
            "visits": len(subset),
            "minutes": round(subset["duration_seconds"].sum() / 60, 1),
            "top_domains": subset["clean_domain"].value_counts().head(5).to_dict()
        }
    return summary


if __name__ == "__main__":
    # Quick test
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.history_reader import generate_demo_history
    
    df = generate_demo_history(days=7)
    df_cat = categorize_dataframe(df)
    
    print("\n=== Categorizer Test ===")
    print(f"Total visits: {len(df_cat)}")
    print(f"\nCategory counts:\n{df_cat['category'].value_counts()}")
    print(f"\nSample:\n{df_cat[['domain', 'clean_domain', 'category', 'duration_seconds']].head(10)}")
    
    summary = get_category_summary(df_cat)
    print("\nTime per category (minutes):")
    for cat, data in summary.items():
        print(f"  {cat}: {data['minutes']} min ({data['visits']} visits)")
