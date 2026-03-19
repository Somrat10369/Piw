# file: config.py
"""
PIW — Personal Intelligence Workstation
Central configuration. Edit OLLAMA_MODEL and RSS_FEEDS to suit your setup.
"""

# ── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen3:14b-q4_K_M"   # change to mistral:7b if VRAM is tight
OLLAMA_TIMEOUT  = 120             # seconds per agent call

# ── Collector ───────────────────────────────────────────────────────────────
MAX_ARTICLES_PER_FEED = 5         # articles fetched per RSS source
MAX_ARTICLE_CHARS     = 6000      # truncate body to this length before embedding
SCRAPE_TIMEOUT        = 10        # HTTP timeout for article scraping

RSS_FEEDS = {
    # International
    "Reuters":          "https://feeds.reuters.com/reuters/topNews",
    "AP News":          "https://feeds.apnews.com/rss/apf-topnews",
    "BBC World":        "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera":       "https://www.aljazeera.com/xml/rss/all.xml",
    # Bangladesh
    "Daily Star":       "https://www.thedailystar.net/rss.xml",
    "Prothom Alo EN":   "https://en.prothomalo.com/feed",
    "bdnews24":         "https://bdnews24.com/feed/",
    # Fact-check
    "AFP Fact Check":   "https://factcheck.afp.com/list/all/feed",
}

# ── Clustering ───────────────────────────────────────────────────────────────
# Set True once you have sentence-transformers + faiss installed
USE_SEMANTIC_CLUSTERING = False
CLUSTER_SIMILARITY_THRESHOLD = 0.75   # cosine similarity for grouping

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "output"
SAVE_REPORTS = True   # write JSON + TXT reports to OUTPUT_DIR

# ── Memory (Phase 2) ─────────────────────────────────────────────────────────
MEMORY_DB_PATH      = "memory/piw_memory.db"   # SQLite file path
MEMORY_DEDUP_HOURS  = 48    # skip re-analyzing an article seen within N hours
ENTITY_MIN_MENTIONS = 2     # minimum mentions before entity is tracked

# ── Monitoring (Phase 2) ─────────────────────────────────────────────────────
MONITOR_INTERVAL_MIN  = 60     # minutes between feed polls in monitor mode
MONITOR_TOP_N         = 2      # clusters to analyze per monitor cycle
ALERT_NEW_ENTITY      = True   # alert when a brand-new major entity appears
ALERT_TOPIC_SPIKE     = True   # alert when a topic appears in 3+ sources suddenly

# ── Bangladesh Scoring (Phase 2) ─────────────────────────────────────────────
BD_LOCAL_SOURCES = {"Daily Star", "Prothom Alo EN", "bdnews24"}

# Keywords that raise Bangladesh relevance score
BD_RELEVANCE_KEYWORDS = [
    # Economy
    "bangladesh", "dhaka", "taka", "bdt", "imf", "world bank", "remittance",
    "garment", "rmg", "export", "ict", "software", "outsourcing",
    # Politics
    "awami", "bnp", "yunus", "interim government", "election",
    # Regional
    "india", "myanmar", "rohingya", "bay of bengal", "south asia", "saarc",
    # Global impact on BD
    "usd", "dollar", "inflation", "fuel", "gas", "electricity", "load shedding",
    "climate", "flood", "cyclone",
    # Tech / career relevant
    "ai", "artificial intelligence", "semiconductor", "visa", "h1b", "migration",
    "remote work", "freelance", "upwork",
]

# Score weights (0.0 – 1.0 additive per hit)
BD_SOURCE_WEIGHT   = 0.4   # bonus for being a BD local source
BD_KEYWORD_WEIGHT  = 0.1   # per keyword hit (capped at 0.6)
BD_SCORE_THRESHOLD = 0.3   # minimum score to inject BD context into personal agent