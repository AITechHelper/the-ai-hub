#!/usr/bin/env python3
"""
The AI Hub Briefing Generator

Usage:
  python3 generate_ai_hub_briefing.py              # full run (writes markdown only)
  python3 generate_ai_hub_briefing.py --repo       # also write the issue into the site repo
  python3 generate_ai_hub_briefing.py --fresh      # ignore cache, re-fetch all feeds
  python3 generate_ai_hub_briefing.py --no-enrich  # skip full-text article scraping
  python3 generate_ai_hub_briefing.py --no-hn      # skip Hacker News signals

Model:
  Defaults to gpt-4.1. Set BRIEFING_MODEL=claude-opus-4-6 in .env to use Claude.

Publishing:
  The weekly issue is published by writing content/ai-hub/<slug>.json (+ image)
  into the aitechhelper-site repo (--repo). The GitHub Action commits that and
  the push deploys the issue. See the "PUBLISH TO SITE REPO" section below.
"""

import os
import re
import time
import json
import hashlib
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
load_dotenv()

import feedparser
import requests
from dateutil import parser as dateparser

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

# =========================
# CONFIG
# =========================

LOOKBACK_DAYS = 7
TARGET_ITEMS = 50
CACHE_FILE = Path("rss_cache.json")
CACHE_MAX_AGE_HOURS = 2

# Set BRIEFING_MODEL=claude-opus-4-6 in .env to use Claude instead
MODEL = os.getenv("BRIEFING_MODEL") or "gpt-4.1"

RSS_FEEDS = [
    # Company / lab blogs (highest signal — 100% AI content)
    "https://openai.com/news/rss.xml",
    "https://www.anthropic.com/rss.xml",
    "https://blog.google/technology/ai/rss/",    # covers Google AI + DeepMind
    "https://huggingface.co/blog/feed.xml",
    "https://engineering.fb.com/feed/",           # Meta / FAIR engineering
    "https://mistral.ai/feed.xml",
    "https://www.deeplearning.ai/feed/",

    # AI-only news and analysis
    "https://venturebeat.com/category/ai/feed/",  # AI section only (was full feed)
    "https://www.artificialintelligence-news.com/feed/",
    "https://www.thealgorithmicbridge.com/feed",  # AI newsletter / analysis
    "https://aisnakeoil.substack.com/feed",       # AI research / critique
    "https://www.interconnects.ai/feed",          # AI research explainers
    "https://www.import-ai.net/feed",             # Jack Clark's Import AI
    "https://simonwillison.net/atom/everything/", # AI tools / LLM practitioner

    # Reputable tech with strong AI-specific feeds
    "https://arstechnica.com/tag/artificial-intelligence/feed/",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "https://www.wired.com/feed/tag/ai/latest/rss",
    "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
]

SOURCE_WEIGHTS = {
    # First-party lab/company blogs
    "openai.com": 5,
    "anthropic.com": 5,
    "blog.google": 4,
    "huggingface.co": 4,
    "engineering.fb.com": 3,
    "mistral.ai": 3,
    "deeplearning.ai": 3,
    # AI-only publications
    "artificialintelligence-news.com": 3,
    "thealgorithmicbridge.com": 3,
    "interconnects.ai": 3,
    "simonwillison.net": 3,
    "import-ai.net": 3,
    # Reputable tech with AI focus
    "venturebeat.com": 3,
    "technologyreview.com": 3,
    "arstechnica.com": 3,
    "wired.com": 3,
    "theverge.com": 2,
}

STOP_WORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "are", "was", "were", "with", "by", "from", "how", "what", "why",
    "when", "this", "that", "its", "new", "now", "will", "can", "has",
    "have", "had", "be", "as", "it", "into", "not", "but", "about",
}

# =========================
# HELPERS
# =========================

def normalize_url(url: str) -> str:
    try:
        u = url.strip()
        if not u:
            return u
        parsed = urlparse(u)
        netloc = parsed.netloc.lower().replace("www.", "")
        parsed = parsed._replace(netloc=netloc, fragment="")
        if parsed.query:
            keep = [
                part for part in parsed.query.split("&")
                if not part.split("=")[0].lower().startswith("utm_")
                and part.split("=")[0].lower() not in {"utm", "gclid", "fbclid", "mc_cid", "mc_eid"}
            ]
            parsed = parsed._replace(query="&".join(keep))
        return urlunparse(parsed)
    except Exception:
        return url.strip()

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def parse_entry_date(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        if key in entry and entry[key]:
            try:
                return datetime.fromtimestamp(time.mktime(entry[key]), tz=timezone.utc)
            except Exception:
                pass
    for key in ("published", "updated", "created"):
        if key in entry and entry[key]:
            try:
                dt = dateparser.parse(entry[key])
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                continue
    return None

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""

def score_item(item: dict) -> float:
    dom = domain_of(item["url"])
    weight = SOURCE_WEIGHTS.get(dom, 1)
    age_hours = (datetime.now(timezone.utc) - item["published_at"]).total_seconds() / 3600.0
    recency = max(0.0, 72.0 - age_hours) / 72.0   # boost last 3 days
    hn_boost = min(item.get("hn_points", 0) / 100.0, 3.0)  # cap HN boost at 3
    # Boost stories about new tools or product launches readers can actually use
    tool_signals = {
        "launch", "launches", "release", "releases", "released", "now available",
        "new feature", "new model", "new tool", "new app", "open source", "open-source",
        "update", "upgrade", "announces", "announced", "just dropped",
    }
    title_lower = item.get("title", "").lower()
    tool_boost = 0.5 if any(s in title_lower for s in tool_signals) else 0.0
    return weight + recency + hn_boost + tool_boost

def stable_id(title: str, url: str) -> str:
    return hashlib.sha256((title + "||" + url).encode()).hexdigest()[:16]

def title_words(title: str) -> set:
    words = re.sub(r"[^\w\s]", "", title.lower()).split()
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# =========================
# CACHE
# =========================

def load_cache() -> list[dict] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
        saved_at = datetime.fromisoformat(data["saved_at"])
        age_hours = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600
        if age_hours > CACHE_MAX_AGE_HOURS:
            print(f"  Cache is {age_hours:.1f}h old (max {CACHE_MAX_AGE_HOURS}h) — fetching fresh")
            return None
        items = []
        for it in data["items"]:
            it["published_at"] = datetime.fromisoformat(it["published_at"])
            items.append(it)
        print(f"  Using cache ({age_hours:.1f}h old, {len(items)} items)")
        return items
    except Exception as e:
        print(f"  Cache error: {e}")
        return None

def save_cache(items: list[dict]) -> None:
    serializable = [
        {**it, "published_at": it["published_at"].isoformat()}
        for it in items
    ]
    CACHE_FILE.write_text(
        json.dumps({"saved_at": datetime.now(timezone.utc).isoformat(), "items": serializable}, indent=2)
    )
    print(f"  Saved {len(items)} items to cache")

# =========================
# FETCH
# =========================

STRONG_KEYWORDS = [
    "chatgpt", "openai", "gpt", "gemini", "claude", "copilot", "llm",
    "machine learning", "generative", "deep learning", "mistral", "meta ai",
]
WEAK_KEYWORDS = [
    "ai", "artificial intelligence", "agent", "agents", "model", "automation",
    "prompt", "inference", "training", "fine-tuning",
]

def fetch_rss_items() -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    items: list[dict] = []

    for feed_url in RSS_FEEDS:
        fp = feedparser.parse(feed_url)
        if fp.bozo and not fp.entries:
            print(f"  [skip] {domain_of(feed_url)} — parse error")
            continue

        count_before = len(items)
        for e in fp.entries:
            url = normalize_url(getattr(e, "link", "") or "")
            title = clean_text(getattr(e, "title", "") or "")
            if not url or not title:
                continue
            dt = parse_entry_date(e)
            if dt is None or dt < cutoff:
                continue
            summary = clean_text(getattr(e, "summary", "") or getattr(e, "description", "") or "")
            title_lower = title.lower()
            text_blob = f"{title} {summary}".lower()
            title_strong = sum(1 for k in STRONG_KEYWORDS if k in title_lower)
            title_weak = sum(1 for k in WEAK_KEYWORDS if k in title_lower)
            body_strong = sum(1 for k in STRONG_KEYWORDS if k in text_blob)
            body_weak = sum(1 for k in WEAK_KEYWORDS if k in text_blob)
            # Title must have at least one AI keyword hit; body-only matches are too noisy
            title_match = title_strong >= 1 or title_weak >= 2
            # Allow body-only if the signal is very strong (2+ strong keywords)
            body_strong_match = body_strong >= 2
            if not (title_match or body_strong_match):
                continue
            items.append({
                "id": stable_id(title, url),
                "title": title,
                "url": url,
                "summary": summary[:500],
                "published_at": dt,
                "source": domain_of(url),
                "hn_points": 0,
            })

        added = len(items) - count_before
        print(f"  {domain_of(feed_url)}: {added} items")

    return items

def fetch_hn_signals() -> dict[str, int]:
    """Return {normalized_url: hn_points} for trending AI stories on HN this week."""
    print("Fetching Hacker News signals...")
    signals: dict[str, int] = {}
    seen_ids: set[str] = set()
    week_ago = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp())

    for query in ["AI LLM", "ChatGPT", "OpenAI", "Claude Anthropic", "Gemini Google"]:
        try:
            resp = requests.get(
                "https://hn.algolia.com/api/v1/search_by_date",
                params={
                    "tags": "story",
                    "query": query,
                    "numericFilters": f"created_at_i>{week_ago},points>30",
                    "hitsPerPage": 30,
                },
                timeout=10,
            )
            for hit in resp.json().get("hits", []):
                hit_id = hit.get("objectID", "")
                if hit_id in seen_ids:
                    continue
                seen_ids.add(hit_id)
                url = normalize_url(hit.get("url") or "")
                if url:
                    signals[url] = max(signals.get(url, 0), hit.get("points", 0))
        except Exception as e:
            print(f"  HN query '{query}' failed: {e}")

    print(f"  HN: {len(signals)} unique story signals")
    return signals

def apply_hn_signals(items: list[dict], signals: dict[str, int]) -> list[dict]:
    boosted = 0
    for item in items:
        pts = signals.get(item["url"], 0)
        if pts:
            item["hn_points"] = pts
            boosted += 1
    if boosted:
        print(f"  Applied HN signal to {boosted} items")
    return items

def enrich_with_full_text(items: list[dict]) -> list[dict]:
    """Scrape full article body for selected items to give the LLM better context."""
    if not HAS_TRAFILATURA:
        print("  [skip] Full text enrichment — run: pip install trafilatura")
        return items
    print(f"  Enriching {len(items)} items with full article text...")
    for item in items:
        try:
            downloaded = trafilatura.fetch_url(item["url"])
            if downloaded:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False,
                )
                if text and len(text) > len(item["summary"]):
                    item["full_text"] = text[:2000]
        except Exception:
            pass
    enriched = sum(1 for it in items if "full_text" in it)
    print(f"  Full text enriched: {enriched}/{len(items)} items")
    return items

# =========================
# DEDUP
# =========================

def dedupe_items(items: list[dict], fuzzy_threshold: float = 0.55) -> list[dict]:
    """Exact URL/title dedup + fuzzy title overlap to catch cross-outlet rephrases."""
    seen_urls: set[str] = set()
    seen_title_words: list[set] = []
    out: list[dict] = []

    for it in items:
        if it["url"] in seen_urls:
            continue
        words = title_words(it["title"])
        if any(jaccard(words, prev) >= fuzzy_threshold for prev in seen_title_words):
            continue
        seen_urls.add(it["url"])
        seen_title_words.append(words)
        out.append(it)

    return out

def pick_top_items(items: list[dict], n: int) -> list[dict]:
    return sorted(items, key=score_item, reverse=True)[:n]

# =========================
# PROMPT + GENERATION
# =========================

def week_range_str() -> str:
    today = datetime.now()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    return f"{monday.strftime('%B %-d')}–{sunday.strftime('%-d, %Y')}"

def build_prompt(selected: list[dict]) -> str:
    date_range = week_range_str()
    month_year = f"{date_range.split('–')[0].rsplit(' ', 1)[0].split()[-1]} {date_range.split(', ')[-1]}"

    # Split items: top stories = direct company/lab sources OR stories
    # that prominently name a major AI player in the title
    top_sources = {
        "openai.com", "anthropic.com", "blog.google", "huggingface.co",
        "engineering.fb.com", "deepmind.google",
    }
    # These keywords in the TITLE are enough to make any story a top story
    top_title_keywords = [
        # Major AI labs / models
        "anthropic", "claude", "openai", "chatgpt", "gpt-", "gemini",
        "meta ai", "llama", "deepmind", "mistral", "copilot",
        # Popular AI tools readers actually use
        "perplexity", "midjourney", "sora", "dall-e", "dalle", "cursor",
        "github copilot", "notion ai", "grammarly", "runway", "elevenlabs",
        "adobe firefly", "canva ai", "stable diffusion", "flux",
        "google ai", "microsoft ai", "apple intelligence",
        # Product launch / release language — any source
        "launches", "releases", "announces", "now available", "just launched",
        "new feature", "new model", "major update",
    ]

    # Phrases that signal "a new AI tool or feature you can use" — get a score boost
    tool_launch_phrases = [
        "launch", "release", "available", "update", "feature", "new tool",
        "new app", "new ai", "free", "pro plan", "api access", "open source",
        "open-source", "just dropped",
    ]

    def is_top_story(it: dict) -> bool:
        if it["source"] in top_sources:
            return True
        if it.get("hn_points", 0) >= 100:
            return True
        title_lower = it["title"].lower()
        if any(k in title_lower for k in top_title_keywords):
            return True
        # Stories about new tools/launches that readers can act on
        if any(p in title_lower for p in tool_launch_phrases):
            return True
        return False

    # Cap top stories at 12 — score-sorted, most important first
    top_candidates = [it for it in selected if is_top_story(it)]
    top = sorted(top_candidates, key=score_item, reverse=True)[:12]
    top_ids = {it["id"] for it in top}
    rest = [it for it in selected if it["id"] not in top_ids]

    lines = [
        f"You are writing the weekly AI briefing for The AI Hub — week of {date_range}.",
        "",
        "AUDIENCE: Curious, intelligent readers who follow AI but are NOT developers or researchers.",
        "They know ChatGPT exists. They may not know what a 'system card', 'TPU', or 'inference endpoint' is.",
        "",
        "TONE: Knowledgeable, direct, and engaging. Write like a sharp tech journalist who also uses these tools daily.",
        "You can have opinions. You can challenge the reader. You can react to what's happening.",
        "Do NOT be promotional. Do NOT use marketing language. Be honest and specific.",
        "",
        "CONTENT FILTER — non-negotiable:",
        "- ONLY include stories that are directly and primarily about AI, machine learning, or AI-powered products.",
        "- If a story is fundamentally about politics, sports, crime, finance, or any non-AI topic — even if 'AI' appears",
        "  somewhere in the article — DO NOT include it. Omit it entirely.",
        "",
        "STORY PRIORITY — this is critical:",
        "- PRIORITIZE stories about new AI tools, apps, features, or models that readers can actually try or use.",
        "- PRIORITIZE stories about product launches, major updates, pricing changes, or availability milestones.",
        "- PRIORITIZE stories about new technology that changes what AI can do for everyday people.",
        "- DEPRIORITIZE pure business/corporate stories (funding rounds, org charts, executive moves, lawsuits)",
        "  UNLESS they directly affect a product readers use or signal a major shift in the AI landscape.",
        "- When in doubt: ask 'can a reader do something differently because of this story?' — if yes, it ranks higher.",
        "",
        "HEADLINE RULES — this is critical:",
        "- Write headlines a smart non-technical reader would immediately understand.",
        "- Never use jargon or product names as the entire headline.",
        "- BAD: 'GPT-5.5 System Card Released'",
        "- GOOD: 'OpenAI Publishes Behind-the-Scenes Safety Report on Its Newest AI Model'",
        "- BAD: 'WebSockets Added to Responses API'",
        "- GOOD: 'OpenAI Makes Its AI Tools Dramatically Faster for Businesses That Build With Them'",
        "- The headline should tell you WHAT happened and give enough context to care.",
        "",
        "OUTPUT FORMAT — follow this structure exactly, in this order:",
        "",
        "---",
        "",
        f"# The AI Hub Weekly: [Punchy subtitle naming the single biggest story of the week in plain English. No jargon. Max 10 words. Example: 'OpenAI Drops GPT-5.5, Redefines Agents and Automation']",
        "",
        "[INTRO — 2 to 3 sentences. Set the scene for the week. What was the dominant theme or tension? ",
        "Make it feel like an opening to a good article, not a table of contents.]",
        "",
        "## TOP STORIES",
        "",
        "Write the FIRST 6 items from the TOP STORY ITEMS list as headlines + bullets.",
        "",
        "Use this format for each story:",
        "",
        "### [Readable, context-friendly headline — see HEADLINE RULES above]",
        "- [Bullet: key fact — what specifically happened, in plain English]",
        "- [Bullet: context — who it affects, what changed, timeline, pricing, or how it compares to before]",
        "- [Bullet: one more relevant detail — availability, caveats, or what to watch for]",
        "- [Source](url)",
        "",
        "Rules for bullets:",
        "- Be specific. Include numbers, dates, prices where available.",
        "- If it's a new AI model: what it can do, how it compares, when available, what it costs.",
        "- If it's a pricing or plan change: what changed, who is affected, when it takes effect.",
        "- If it's a business story: what happened, who is involved, what it signals.",
        "- 3–4 bullets per story. No filler. Save opinions for the article.",
        "",
        "---",
        "",
        "## THIS WEEK IN AI",
        "",
        "Write a 4–6 paragraph article reacting to the week's top stories as a whole.",
        "This is NOT a summary. The reader already read the headlines above.",
        "Instead: connect the dots, find the bigger pattern, challenge assumptions, ask hard questions.",
        "Give readers something to think about or try. Make it worth reading.",
        "Write in first-person plural ('we', 'us') or direct second person ('you').",
        "End with a specific call to action or a question that makes the reader think.",
        "",
        "---",
        "",
        "## MORE TOP STORIES",
        "",
        "Write the REMAINING 6 items from the TOP STORY ITEMS list using the same headline + bullets format.",
        "",
        "---",
        "",
        "## ALSO THIS WEEK",
        "",
        "List every item from the REST ITEMS list below. One line each:",
        "- [Plain-English title or description] — [one sentence summary] ([Source](url))",
        "",
        "Include every item. Do not skip any.",
        "",
        "---",
        "",
        "TOP STORY ITEMS (first 6 go above the article, remaining 6 go below it):",
    ]

    for it in top:
        text = it.get("full_text") or it["summary"]
        hn = f" | HN: {it['hn_points']} pts" if it.get("hn_points", 0) > 0 else ""
        lines.append(f"- {it['title']} | {text[:400]} | {it['url']}{hn}")

    lines.append("")
    lines.append("REST ITEMS (list these in the ALSO THIS WEEK section):")

    for it in rest:
        text = it.get("full_text") or it["summary"]
        lines.append(f"- {it['title']} | {text[:200]} | {it['url']}")

    return "\n".join(lines)

def generate_briefing(prompt: str) -> str:
    if "claude" in MODEL.lower():
        from anthropic import Anthropic
        client = Anthropic()
        msg = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    else:
        from openai import OpenAI
        client = OpenAI()
        resp = client.responses.create(model=MODEL, input=prompt)
        return resp.output_text.strip()

def output_filename() -> str:
    monday = datetime.now() - timedelta(days=datetime.now().weekday())
    return f"ai_hub_briefing_{monday.strftime('%Y-%m-%d')}.md"

# =========================
# PUBLISH TO SITE REPO (git-as-CMS)
# =========================
# Writes each issue as a JSON file the aitechhelper-site Next app reads
# directly (content/ai-hub/<slug>.json) plus its image (public/ai-hub/<slug>.jpg).
# The workflow commits these files to the site repo, that push deploys, and the
# issue is live. This is the only publish path. See lib/posts.ts there.

def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text).lower()
    text = text.replace("’", "").replace("'", "")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:96] or "issue"

def md_to_site_html(md: str) -> tuple[str, str]:
    """Markdown briefing -> (title, clean semantic HTML) for the site.

    Clean markup with no inline spacing hacks — the site's stylesheet owns
    spacing.
    """
    import markdown as md_lib

    lines = md.strip().splitlines()
    title = ""
    body_lines = []
    for line in lines:
        if not title and line.startswith("# "):
            title = line.lstrip("#").strip()
        else:
            body_lines.append(line)

    html = md_lib.markdown("\n".join(body_lines), extensions=["extra"])
    # Open source links in a new tab (the site styles these as citation chips).
    html = re.sub(r'<a href="([^"]+)">', r'<a href="\1" target="_blank" rel="noopener">', html)
    return title, html.strip()

def excerpt_from_html(html: str, limit: int = 200) -> str:
    m = re.search(r"<p>(.*?)</p>", html, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text

def write_site_post(
    title: str,
    html: str,
    image_bytes: bytes | None,
    content_dir: str,
    public_dir: str,
) -> str:
    """Write <slug>.json (+ image) into the site repo. Returns the slug."""
    slug = slugify(title)
    Path(content_dir).mkdir(parents=True, exist_ok=True)
    Path(public_dir).mkdir(parents=True, exist_ok=True)

    image_path = None
    if image_bytes:
        img_file = Path(public_dir) / f"{slug}.jpg"
        img_file.write_bytes(image_bytes)
        image_path = f"/ai-hub/{slug}.jpg"

    record = {
        "slug": slug,
        "title": title,
        "date": datetime.now().replace(microsecond=0).isoformat(),
        "excerpt": excerpt_from_html(html),
        "image": image_path,
        "imageAlt": "",
        "contentHtml": html,
    }
    out = Path(content_dir) / f"{slug}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return slug


def fetch_featured_image(query: str = "artificial intelligence technology") -> bytes | None:
    """
    Fetch a landscape photo for the featured image.
    Priority: Pexels (PEXELS_API_KEY) → Unsplash (UNSPLASH_ACCESS_KEY) → skip.
    Get a free Pexels key at: https://www.pexels.com/api/
    Get a free Unsplash key at: https://unsplash.com/developers
    """
    import random

    pexels_key = os.getenv("PEXELS_API_KEY", "")
    if pexels_key:
        try:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                params={"query": query, "per_page": 15, "orientation": "landscape"},
                headers={"Authorization": pexels_key},
                timeout=10,
            )
            photos = resp.json().get("photos", [])
            if photos:
                img_url = random.choice(photos[:8])["src"]["large2x"]
                img_resp = requests.get(img_url, timeout=30)
                if img_resp.status_code == 200:
                    return img_resp.content
        except Exception as e:
            print(f"  [warn] Pexels error: {e}")

    unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY", "")
    if unsplash_key:
        try:
            resp = requests.get(
                "https://api.unsplash.com/search/photos",
                params={"query": query, "per_page": 10, "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {unsplash_key}"},
                timeout=10,
            )
            results = resp.json().get("results", [])
            if results:
                img_url = random.choice(results[:5])["urls"]["regular"]
                img_resp = requests.get(img_url, timeout=30)
                if img_resp.status_code == 200:
                    return img_resp.content
        except Exception as e:
            print(f"  [warn] Unsplash error: {e}")

    return None


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Generate The AI Hub briefing")
    parser.add_argument("--repo",      action="store_true", help="Write the issue into the site repo (content/ai-hub + public/ai-hub)")
    parser.add_argument("--content-dir", default=os.getenv("SITE_CONTENT_DIR", "site/content/ai-hub"), help="Where to write the issue JSON")
    parser.add_argument("--public-dir",  default=os.getenv("SITE_PUBLIC_DIR", "site/public/ai-hub"), help="Where to write the issue image")
    parser.add_argument("--fresh",     action="store_true", help="Ignore cache and re-fetch all feeds")
    parser.add_argument("--no-enrich", action="store_true", help="Skip full-text article scraping")
    parser.add_argument("--no-hn",     action="store_true", help="Skip Hacker News signals")
    args = parser.parse_args()

    # 1. Fetch items (from cache or live)
    items = None
    if not args.fresh:
        print("Checking cache...")
        items = load_cache()

    if items is None:
        print(f"Fetching {len(RSS_FEEDS)} RSS feeds...")
        items = fetch_rss_items()
        print(f"Raw total: {len(items)} items")
        save_cache(items)

    # 2. Apply Hacker News social signals
    if not args.no_hn:
        signals = fetch_hn_signals()
        items = apply_hn_signals(items, signals)

    # 3. Dedupe (exact URL/title + fuzzy title overlap)
    before = len(items)
    items = dedupe_items(items)
    print(f"After dedup: {len(items)} items (removed {before - len(items)} duplicates)")

    if not items:
        raise SystemExit("No items found — check your feeds or widen LOOKBACK_DAYS.")

    # 4. Score and select top items
    selected = pick_top_items(items, TARGET_ITEMS)
    print(f"Selected top {len(selected)} items")

    # 5. Enrich selected items with full article text
    if not args.no_enrich:
        selected = enrich_with_full_text(selected)

    # 6. Generate briefing
    print(f"Generating briefing with {MODEL}...")
    prompt = build_prompt(selected)
    briefing = generate_briefing(prompt)

    # 7. Save markdown file
    filename = output_filename()
    Path(filename).write_text(briefing + "\n", encoding="utf-8")
    print(f"Saved: {filename}")

    # 8. Publish. --repo writes the issue into the site repo, which the GitHub
    #    Action then commits and pushes to deploy. Without it, only the local
    #    markdown file is produced.
    if args.repo:
        print("Writing issue into the site repo...")
        title, html = md_to_site_html(briefing)
        image = fetch_featured_image()
        slug = write_site_post(title, html, image, args.content_dir, args.public_dir)
        print(f"Wrote site post ✅  {args.content_dir}/{slug}.json")
    else:
        print("Done ✅  (run with --repo to publish to the site)")

if __name__ == "__main__":
    main()
