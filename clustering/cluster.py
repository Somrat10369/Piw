# file: collector/collector.py
"""
Collector Layer — fetches RSS feeds and scrapes article bodies.
Returns a list of Article dicts ready for the analysis pipeline.
"""

from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import feedparser
import requests
from bs4 import BeautifulSoup

try:
    from config import RSS_FEEDS, MAX_ARTICLES_PER_FEED, MAX_ARTICLE_CHARS, SCRAPE_TIMEOUT
except ImportError:
    from ..config import RSS_FEEDS, MAX_ARTICLES_PER_FEED, MAX_ARTICLE_CHARS, SCRAPE_TIMEOUT

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Article:
    id:       str
    source:   str
    title:    str
    url:      str
    summary:  str          # from RSS feed
    body:     str          # scraped full text (may be empty on failure)
    published: str
    tags:     list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Best available text for analysis."""
        return (self.body or self.summary or self.title)[:MAX_ARTICLE_CHARS]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "source": self.source, "title": self.title,
            "url": self.url, "summary": self.summary,
            "body_preview": self.body[:300] + "..." if len(self.body) > 300 else self.body,
            "published": self.published, "tags": self.tags,
        }

# ── Scraper ───────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

_CONTENT_SELECTORS = [
    "article", '[class*="article-body"]', '[class*="story-body"]',
    '[class*="post-content"]', '[class*="entry-content"]', "main", ".content",
]

def scrape_article_body(url: str) -> str:
    """Extract main text from a URL. Returns empty string on failure."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=SCRAPE_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        # Try targeted content selectors first
        for selector in _CONTENT_SELECTORS:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator=" ", strip=True)
                if len(text) > 200:
                    return text[:MAX_ARTICLE_CHARS]

        # Fallback: all paragraphs
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return text[:MAX_ARTICLE_CHARS]

    except Exception:
        return ""

# ── RSS reader ────────────────────────────────────────────────────────────────

def _make_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]

def fetch_feed(source_name: str, feed_url: str,
               scrape: bool = True, verbose: bool = True) -> list[Article]:
    """Fetch one RSS feed, optionally scraping each article."""
    articles: list[Article] = []
    try:
        feed = feedparser.parse(feed_url)
        entries = feed.entries[:MAX_ARTICLES_PER_FEED]
        if verbose:
            print(f"  [{source_name}] {len(entries)} entries found")

        for entry in entries:
            url     = getattr(entry, "link", "")
            title   = getattr(entry, "title", "No title")
            summary = getattr(entry, "summary", "")
            published = getattr(entry, "published", "")
            tags    = [t.term for t in getattr(entry, "tags", [])]

            # Strip HTML from summary
            summary = BeautifulSoup(summary, "html.parser").get_text(strip=True)

            body = ""
            if scrape and url:
                body = scrape_article_body(url)
                time.sleep(0.3)   # polite delay

            articles.append(Article(
                id=_make_id(url),
                source=source_name,
                title=title,
                url=url,
                summary=summary,
                body=body,
                published=published,
                tags=tags,
            ))
    except Exception as e:
        if verbose:
            print(f"  [{source_name}] ERROR: {e}")
    return articles


def collect_all(feeds: dict | None = None,
                scrape: bool = True,
                verbose: bool = True) -> list[Article]:
    """Collect from all configured RSS feeds."""
    if feeds is None:
        feeds = RSS_FEEDS
    all_articles: list[Article] = []
    print(f"\n[Collector] Fetching {len(feeds)} feeds...")
    for name, url in feeds.items():
        articles = fetch_feed(name, url, scrape=scrape, verbose=verbose)
        all_articles.extend(articles)
    print(f"[Collector] Total articles collected: {len(all_articles)}\n")
    return all_articles


def collect_from_url(url: str, source_name: str = "Manual") -> Article:
    """Scrape a single user-provided URL into an Article."""
    body = scrape_article_body(url)
    return Article(
        id=_make_id(url),
        source=source_name,
        title=url,
        url=url,
        summary="",
        body=body,
        published="",
        tags=[],
    )