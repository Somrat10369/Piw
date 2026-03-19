# PIW — Personal Intelligence Workstation

Local, Ollama-based multi-agent news intelligence system.  
No paid APIs. No cloud. Runs entirely on your machine.

---

## Quick Start (Windows 11 + PowerShell)

### 1. Install Ollama
Download from https://ollama.com/download and install.

Then pull the recommended model:
```powershell
ollama pull qwen3:14b-q4_K_M       # ~9GB download, fits in RTX 4070 12GB VRAM
# OR if VRAM is tight:
ollama pull mistral:7b
```

Edit `config.py` → set `OLLAMA_MODEL = "qwen3:14b-q4_K_M"` (or whichever you pulled).

### 2. Install Python dependencies
```powershell
cd path\to\piw
pip install -r requirements.txt
```

### 3. Run Ollama server (keep this terminal open)
```powershell
ollama serve
```

### 4. Run PIW
```powershell
# Interactive menu (default)
python piw.py

# OR subcommands:
python piw.py check                     # verify Ollama connection
python piw.py analyze --top 3           # fetch feeds, analyze top 3 topics
python piw.py topic "IMF Bangladesh"    # filter + analyze a specific topic
python piw.py url https://example.com/article   # deep-dive a single URL
```

---

## File Structure

```
piw/
├── piw.py                  ← main CLI entrypoint
├── config.py               ← all settings (model, feeds, thresholds)
├── requirements.txt
├── collector/
│   └── collector.py        ← RSS fetcher + BeautifulSoup scraper
├── clustering/
│   └── cluster.py          ← TF-IDF clustering (Phase 1), FAISS (Phase 2)
├── agents/
│   ├── ollama_client.py    ← Ollama HTTP wrapper
│   └── agents.py           ← 7-agent pipeline
└── output/
    ├── renderer.py         ← Rich terminal output + file saver
    └── [generated reports] ← saved here as .json + .txt
```

---

## Agent Pipeline

Each article cluster passes through 7 sequential agents:

| # | Agent | Purpose |
|---|-------|---------|
| 1 | Reality Extractor | Facts only, timeline, contradictions |
| 2 | Bias Detector | Framing, emotional language, manipulation tactics |
| 3 | Missing Info Finder | What's NOT being reported |
| 4 | Incentive Analyzer | Who benefits, who loses, hidden motivations |
| 5 | Trend Analyzer | Patterns, historical parallels, signals |
| 6 | Scenario Builder | Best/worst/likely outcomes + indicators |
| 7 | Personal Strategist | Your risks, opportunities, actions |

---

## Config Reference (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `qwen3:14b-q4_K_M` | Model to use for all agents |
| `OLLAMA_TIMEOUT` | `120` | Seconds per agent call |
| `MAX_ARTICLES_PER_FEED` | `5` | Articles fetched per RSS source |
| `MAX_ARTICLE_CHARS` | `6000` | Max chars per article before truncation |
| `USE_SEMANTIC_CLUSTERING` | `False` | Enable FAISS clustering (Phase 2) |
| `CLUSTER_SIMILARITY_THRESHOLD` | `0.75` | How similar articles must be to cluster |
| `SAVE_REPORTS` | `True` | Auto-save .json + .txt to output/ |

---

## Phase 2 Upgrade: Semantic Clustering

After installing extra deps:
```powershell
pip install sentence-transformers faiss-cpu
```

Set in `config.py`:
```python
USE_SEMANTIC_CLUSTERING = True
```

This replaces TF-IDF with dense embeddings (`all-MiniLM-L6-v2`) + FAISS for
much more accurate topic grouping, especially across multilingual sources.

---

## Adding More RSS Feeds

Edit `RSS_FEEDS` in `config.py`:
```python
RSS_FEEDS = {
    "My Source": "https://example.com/rss.xml",
    ...
}
```

Bangladesh sources pre-configured: Daily Star, Prothom Alo EN, bdnews24.

---

## Troubleshooting

**`Cannot connect to Ollama`** → Run `ollama serve` in a separate terminal.  
**`model not found`** → Run `ollama pull qwen3:14b-q4_K_M`.  
**Slow responses** → Try `mistral:7b` for faster but shallower analysis.  
**Empty article bodies** → Some sites block scrapers; summary fallback is used.  
**VRAM OOM** → Switch to `mistral:7b` or `qwen2.5:7b` in config.py.