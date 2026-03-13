#!/usr/bin/env python3
"""
AI Tools & Capabilities Weekly Briefing Generator

Usage:
  python3 generate_weekly_briefing.py              # full run
  python3 generate_weekly_briefing.py --post       # generate + post to WordPress
  python3 generate_weekly_briefing.py --fresh      # ignore cache, re-fetch all feeds
  python3 generate_weekly_briefing.py --no-enrich  # skip full-text article scraping
  python3 generate_weekly_briefing.py --no-hn      # skip Hacker News signals

Model:
  Defaults to gpt-4.1. Set BRIEFING_MODEL=claude-opus-4-6 in .env to use Claude.

WordPress setup (one-time):
  Add to .env:
    WP_SITE_URL=https://aitechhelper.com
    WP_USERNAME=your_wp_username
    WP_APP_PASSWORD=xxxx xxxx xxxx xxxx xxxx xxxx
  Generate the app password at: WP Admin → Users → Profile → Application Passwords
"""

import os
import re
import time
import json
import hashlib
import base64
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
    # Company / lab blogs (highest signal)
    "https://openai.com/news/rss.xml",
    "https://blog.google/technology/ai/rss/",   # covers Google AI + DeepMind
    "https://huggingface.co/blog/feed.xml",
    "https://engineering.fb.com/feed/",          # Meta / FAIR engineering

    # AI-focused coverage
    "https://venturebeat.com/feed/",
    "https://www.artificialintelligence-news.com/feed/",
    "https://www.wired.com/feed/tag/ai/latest/rss",

    # Reputable tech coverage
    "https://arstechnica.com/tag/artificial-intelligence/feed/",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "https://www.theverge.com/rss/index.xml",
    "https://www.zdnet.com/topic/artificial-intelligence/rss.xml",
    "https://api.axios.com/feed/",
    "https://www.fastcompany.com/section/tech/rss",
]

SOURCE_WEIGHTS = {
    "openai.com": 5,
    "blog.google": 4,
    "huggingface.co": 4,
    "engineering.fb.com": 3,
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
    return weight + recency + hn_boost

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
            text_blob = f"{title} {summary}".lower()
            strong_hits = sum(1 for k in STRONG_KEYWORDS if k in text_blob)
            weak_hits = sum(1 for k in WEAK_KEYWORDS if k in text_blob)
            if not (strong_hits >= 1 or weak_hits >= 2):
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
    lines = [
        "You are writing the weekly blog post for aitechhelper.com.",
        "The site helps non-technical people stay up to date on AI — specifically what tools and features are AVAILABLE RIGHT NOW.",
        "",
        "PRIMARY FOCUS: new AI tools, new features, new capabilities — what users can actually do today.",
        "Secondary: organize the most useful items by audience (Business / Creators / Entrepreneurs).",
        "",
        "HARD RULES:",
        "- Avoid general news unless it directly changes what users can do with AI tools this week.",
        "- Every bullet must name a specific tool/feature, describe what it does, and give a concrete use case.",
        "- Use ONLY the provided source items. Do NOT invent features, pricing, benchmarks, dates, or partnerships.",
        "- No rumors, leaks, or unconfirmed claims.",
        "- Every bullet MUST include a markdown link: [Source](url) — must be one of the provided URLs.",
        "- Output in clean Markdown. Use ## for section headers, **bold** for tool/feature names.",
        "- Prefer product/feature releases over opinion pieces or general commentary.",
        "- Where full_text is available for an item, use it to write more accurate, specific bullets.",
        "- Items with high HN points are community-validated as important — prioritize them.",
        "",
        f"# AI Tools & Capabilities Weekly — Week of {date_range}",
        "(The line above is the post title — use a single # for it, then ## for all section headers below.)",
        "",
        "Write a 2-sentence intro paragraph summarizing the biggest theme or story of the week before the first section.",
        "",
        "## WHAT'S NEW — 12 items",
        "- **Tool/Feature**: what it does (1 sentence). Best for: [who benefits]. [Source](url)",
        "",
        "## AI FOR BUSINESS — 8 items",
        "- **Tool/Feature**: capability + business use case (1 sentence). [Source](url)",
        "",
        "## AI FOR CREATORS — 8 items",
        "- **Tool/Feature**: capability + creator use case (1 sentence). [Source](url)",
        "",
        "## AI FOR ENTREPRENEURS — 8 items",
        "- **Tool/Feature**: capability + entrepreneur use case (1 sentence). [Source](url)",
        "",
        "## HOW TO USE THIS WEEK — 5 bullets",
        "- **Action**: specific thing to try in under 10 minutes (1 sentence).",
        "",
        "## QUICK HITS — 8 items",
        "- One sharp sentence. [Source](url)",
        "",
        "## SOURCES",
        "- Deduplicated list of every source used, as markdown links.",
        "",
        "---",
        "SOURCE ITEMS (title | text | url | hn_points if notable):",
    ]
    for it in selected:
        text = it.get("full_text") or it["summary"]
        hn = f" | HN: {it['hn_points']} pts" if it.get("hn_points", 0) > 0 else ""
        lines.append(f"- {it['title']} | {text} | {it['url']}{hn}")

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
    return f"ai_weekly_briefing_{monday.strftime('%Y-%m-%d')}.md"

# =========================
# WORDPRESS
# =========================

# Section config: (accent_color, icon, anchor_id)
SECTION_STYLES = {
    "WHAT'S NEW":           ("#8EF2FE", "🔥", "whats-new"),
    "AI FOR BUSINESS":      ("#a78bfa", "💼", "business"),
    "AI FOR CREATORS":      ("#f472b6", "🎨", "creators"),
    "AI FOR ENTREPRENEURS": ("#fbbf24", "🚀", "entrepreneurs"),
    "HOW TO USE":           ("#34d399", "⚡", "how-to"),
    "QUICK HITS":           ("#8EF2FE", "📌", "quick-hits"),
    "SOURCES":              ("#64748b", "🔗", "sources"),
}


def markdown_to_html(md: str) -> tuple[str, str]:
    """Convert markdown briefing to styled HTML. Returns (title, body_html)."""
    import markdown as md_lib
    import re as _re

    # ── Extract title line ──────────────────────────────────────────────────
    lines = md.strip().splitlines()
    title = ""
    body_lines = []
    for line in lines:
        # Accept # or ## for the title line (LLM sometimes uses either)
        if not title and (line.startswith("# ") or line.startswith("## ")) and "AI Tools" in line:
            title = line.lstrip("#").strip()
        else:
            body_lines.append(line)

    body_md = "\n".join(body_lines)
    raw_html = md_lib.markdown(body_md, extensions=["extra"])

    # ── 1. Style <p> tags ───────────────────────────────────────────────────
    raw_html = raw_html.replace(
        "<p>", '<p style="margin-bottom:16px; line-height:1.7;">'
    )

    # ── 2. Section headers: large dividers with section anchors ────────────
    def style_h2(match):
        text = match.group(1)
        color, icon, anchor = "#8EF2FE", "", _re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        section_key = "OTHER"
        for key, (c, i, a) in SECTION_STYLES.items():
            if key in text.upper():
                color, icon, anchor = c, i + " ", a
                section_key = key
                break
        clean_text = _re.sub(r"<[^>]+>", "", text).strip()
        return (
            f'<h2 id="{anchor}" data-section="{section_key}" style="margin:0; padding:0;">'
            f'<span style="font-size:0.78em; font-weight:800; text-transform:uppercase; '
            f'letter-spacing:0.16em; color:{color};">{icon}{clean_text}</span>'
            f'</h2>'
        )

    raw_html = _re.sub(r"<h2>(.*?)</h2>", style_h2, raw_html)

    # ── 3. Split into intro + sections; apply per-section list styling ───────
    h2_pat = _re.compile(
        r'(<h2 id="[^"]*" data-section="([^"]*)"[^>]*>.*?</h2>)', _re.DOTALL
    )
    parts = h2_pat.split(raw_html)
    # parts layout: [intro, full_h2, key, content, full_h2, key, content, ...]

    def style_section_lists(content: str, key: str) -> str:
        """Apply section-specific list item styles."""
        if "HOW TO USE" in key:
            # Numbered step circles
            counter = [0]
            def step_li(m):
                counter[0] += 1
                inner = m.group(1)
                return (
                    f'<li style="list-style:none; display:flex; align-items:flex-start; '
                    f'gap:18px; padding:20px 0; margin-bottom:0; '
                    f'border-bottom:1px solid rgba(142,242,254,0.08);">'
                    f'<span style="flex-shrink:0; width:38px; height:38px; min-width:38px; '
                    f'border-radius:50%; background:rgba(52,211,153,0.12); '
                    f'border:2px solid #34d399; color:#34d399; font-weight:800; '
                    f'font-size:0.85em; display:inline-flex; align-items:center; '
                    f'justify-content:center;">{counter[0]}</span>'
                    f'<div style="line-height:1.7; padding-top:8px; flex:1;">{inner}</div>'
                    f'</li>'
                )
            content = _re.sub(r"<li>(.*?)</li>", step_li, content, flags=_re.DOTALL)
            content = content.replace("<ul>", '<ul style="list-style:none; padding:0; margin:0;">')
            content = content.replace("<ol>", '<ol style="list-style:none; padding:0; margin:0;">')

        elif "QUICK HITS" in key:
            # Flash bullets with › icon
            def flash_li(m):
                inner = m.group(1)
                return (
                    f'<li style="list-style:none; display:flex; align-items:flex-start; '
                    f'gap:10px; padding:12px 0; margin-bottom:0; line-height:1.7; '
                    f'border-bottom:1px solid rgba(128,128,128,0.1);">'
                    f'<span style="color:#8EF2FE; flex-shrink:0; font-size:1.3em; '
                    f'line-height:1.3; margin-top:1px;">›</span>'
                    f'<span style="flex:1;">{inner}</span>'
                    f'</li>'
                )
            content = _re.sub(r"<li>(.*?)</li>", flash_li, content, flags=_re.DOTALL)
            content = content.replace("<ul>", '<ul style="list-style:none; padding:0; margin:0;">')

        elif "SOURCES" in key:
            content = content.replace(
                "<li>",
                '<li style="list-style:none; padding:5px 0; line-height:1.7; '
                'opacity:0.55; font-size:0.85em;">',
            )
            content = content.replace("<ul>", '<ul style="list-style:none; padding:0; margin:0;">')

        else:
            # Card blocks: What's New, Business, Creators, Entrepreneurs
            content = content.replace(
                "<li>",
                '<li style="list-style:none; padding:20px 24px; margin-bottom:16px; '
                'border-radius:10px; border:1px solid rgba(142,242,254,0.13); '
                'border-left:4px solid rgba(142,242,254,0.45); '
                'line-height:1.7; background:rgba(142,242,254,0.03);">',
            )
            content = content.replace("<ul>", '<ul style="list-style:none; padding:0; margin:0;">')

        return content

    # Intro → "This Week in AI" highlight card
    intro_html = parts[0] if parts else ""
    intro_block = (
        f'<div style="background:rgba(142,242,254,0.05); '
        f'border:1px solid rgba(142,242,254,0.22); border-left:4px solid #8EF2FE; '
        f'border-radius:10px; padding:22px 28px 16px; margin-bottom:44px;">'
        f'<div style="font-size:0.68em; font-weight:800; text-transform:uppercase; '
        f'letter-spacing:0.15em; color:#8EF2FE; margin-bottom:14px;">⚡ This Week in AI</div>'
        f'{intro_html}'
        f'</div>'
    )

    # Build each section block with top divider
    section_blocks = []
    i = 1
    while i + 1 < len(parts):
        h2_html  = parts[i]
        sec_key  = parts[i + 1]
        content  = parts[i + 2] if i + 2 < len(parts) else ""
        i += 3
        styled = style_section_lists(content, sec_key)
        section_blocks.append(
            f'<div style="margin-top:72px;">'
            f'<div style="border-top:1px solid rgba(142,242,254,0.1); '
            f'padding-top:32px; margin-bottom:28px;">'
            f'{h2_html}'
            f'</div>'
            f'{styled}'
            f'</div>'
        )

    raw_html = intro_block + "\n".join(section_blocks)

    # ── 4. Source badges ────────────────────────────────────────────────────
    raw_html = _re.sub(
        r'<a href="([^"]+)">Source</a>',
        r'<a href="\1" target="_blank" rel="noopener" '
        r'style="font-size:0.7em; text-decoration:none; color:#8EF2FE; opacity:0.75; '
        r'border:1px solid rgba(142,242,254,0.3); border-radius:4px; '
        r'padding:2px 8px; margin-left:8px; white-space:nowrap; vertical-align:middle;">↗ source</a>',
        raw_html,
    )

    # ── 5. Jump-to-section nav ──────────────────────────────────────────────
    nav_links = [
        ("#whats-new",     "🔥 Releases"),
        ("#business",      "💼 Business"),
        ("#creators",      "🎨 Creators"),
        ("#entrepreneurs", "🚀 Entrepreneurs"),
        ("#how-to",        "⚡ Try This Week"),
        ("#quick-hits",    "📌 Quick Hits"),
    ]
    nav_items = "".join(
        f'<a href="{href}" style="color:#8EF2FE; text-decoration:none; '
        f'font-size:0.76em; font-weight:600; white-space:nowrap; '
        f'border:1px solid rgba(142,242,254,0.28); border-radius:20px; '
        f'padding:6px 14px; opacity:0.8;">{label}</a>'
        for href, label in nav_links
    )
    jump_nav = (
        f'<div style="margin-bottom:44px; padding:16px 20px; '
        f'background:rgba(0,3,56,0.4); border-radius:10px; '
        f'border:1px solid rgba(142,242,254,0.1);">'
        f'<div style="font-size:0.68em; font-weight:700; text-transform:uppercase; '
        f'letter-spacing:0.12em; opacity:0.4; margin-bottom:12px;">Jump to:</div>'
        f'<div style="display:flex; flex-wrap:wrap; gap:8px;">{nav_items}</div>'
        f'</div>'
    )

    # ── 6. Final wrapper ────────────────────────────────────────────────────
    week_of = title.split("Week of")[-1].strip() if "Week of" in title else ""

    html = f"""<div style="max-width:760px; margin:0 auto; font-family:inherit; line-height:1.7;">

  <!-- Dateline bar -->
  <div style="text-align:center; padding:28px 0 36px; border-bottom:1px solid rgba(142,242,254,0.15); margin-bottom:36px;">
    <div style="font-size:0.68em; font-weight:700; text-transform:uppercase; letter-spacing:0.18em; opacity:0.4; margin-bottom:8px;">Week of {week_of}</div>
    <div style="opacity:0.35; font-size:0.8em;">📡 Curated from 13 sources &nbsp;·&nbsp; Ranked by community signal</div>
  </div>

  <!-- Jump nav -->
  {jump_nav}

  <!-- Content -->
  {raw_html}

</div>""".strip()

    return title, html


def wp_auth_token() -> tuple[str, str, str]:
    """Return (site_url, username, base64_token). Raises SystemExit if missing."""
    site_url = os.getenv("WP_SITE_URL", "").rstrip("/")
    username  = os.getenv("WP_USERNAME", "")
    app_pass  = os.getenv("WP_APP_PASSWORD", "")
    if not all([site_url, username, app_pass]):
        raise SystemExit(
            "WordPress credentials missing. Add to .env:\n"
            "  WP_SITE_URL=https://aitechhelper.com\n"
            "  WP_USERNAME=your_username\n"
            "  WP_APP_PASSWORD=xxxx xxxx xxxx xxxx xxxx xxxx"
        )
    token = base64.b64encode(f"{username}:{app_pass}".encode()).decode()
    return site_url, username, token


def get_or_create_category(name: str, site_url: str, token: str) -> int | None:
    """Find existing WP category by name or create it. Returns category ID."""
    headers = {"Authorization": f"Basic {token}"}
    try:
        # Fetch all categories (not just search) to avoid search API quirks
        resp = requests.get(
            f"{site_url}/wp-json/wp/v2/categories",
            params={"per_page": 100},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        for cat in resp.json():
            if cat.get("name", "").lower() == name.lower():
                print(f"  Category '{name}' found (id={cat['id']})")
                return cat["id"]
        # Not found — create it
        resp = requests.post(
            f"{site_url}/wp-json/wp/v2/categories",
            headers={**headers, "Content-Type": "application/json"},
            json={"name": name},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            new_id = resp.json().get("id")
            print(f"  Category '{name}' created (id={new_id})")
            return new_id
    except Exception as e:
        print(f"  [warn] Category lookup failed: {e}")
    return None


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


def upload_featured_image(image_bytes: bytes, filename: str, site_url: str, token: str) -> int | None:
    """Upload image bytes to WP media library. Returns media ID or None."""
    try:
        resp = requests.post(
            f"{site_url}/wp-json/wp/v2/media",
            headers={
                "Authorization": f"Basic {token}",
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "image/jpeg",
            },
            data=image_bytes,
            timeout=60,
        )
        if resp.status_code in (200, 201):
            return resp.json().get("id")
        print(f"  [warn] Image upload failed ({resp.status_code})")
    except Exception as e:
        print(f"  [warn] Image upload error: {e}")
    return None


def post_to_wordpress(title: str, html: str) -> str:
    """Post to WordPress with category + featured image. Returns published URL."""
    site_url, _, token = wp_auth_token()

    # Category
    print("  Resolving category...")
    cat_id = get_or_create_category("AI Tools", site_url, token)

    # Featured image from Unsplash
    print("  Fetching featured image...")
    image_bytes = fetch_featured_image("artificial intelligence machine learning technology")
    media_id = None
    if image_bytes:
        monday = datetime.now() - timedelta(days=datetime.now().weekday())
        filename = f"ai-briefing-{monday.strftime('%Y-%m-%d')}.jpg"
        media_id = upload_featured_image(image_bytes, filename, site_url, token)
        if media_id:
            print(f"  Featured image uploaded (id={media_id})")
        else:
            print("  [warn] Image upload failed — posting without featured image")
    else:
        print("  [warn] Could not fetch Unsplash image — posting without featured image")

    # Build post payload
    payload: dict = {
        "title":   title,
        "content": html,
        "status":  "publish",
        "format":  "standard",
    }
    if cat_id:
        payload["categories"] = [cat_id]
    if media_id:
        payload["featured_media"] = media_id

    resp = requests.post(
        f"{site_url}/wp-json/wp/v2/posts",
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type":  "application/json",
        },
        json=payload,
        timeout=30,
    )

    if resp.status_code not in (200, 201):
        raise SystemExit(f"WordPress post failed ({resp.status_code}):\n{resp.text[:400]}")

    return resp.json().get("link", f"{site_url}/wp-json/wp/v2/posts")

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Generate AI weekly briefing")
    parser.add_argument("--post",      action="store_true", help="Post to WordPress after generating")
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

    # 8. Post to WordPress (if requested)
    if args.post:
        print("Posting to WordPress...")
        title, html = markdown_to_html(briefing)
        post_url = post_to_wordpress(title, html)
        print(f"Done ✅  Published: {post_url}")
    else:
        print("Done ✅  (run with --post to publish to WordPress)")

if __name__ == "__main__":
    main()
