#!/usr/bin/env python3
"""
The AI Hub Daily — Instagram Post Generator

Usage:
  python3 generate_instagram_posts.py          # use cached RSS data
  python3 generate_instagram_posts.py --fresh  # re-fetch RSS feeds first

Output:
  instagram_posts/YYYY-MM-DD.txt  — all 7 posts in one file, ready to copy/paste
"""

import os
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT_DIR    = Path(__file__).parent
CACHE_FILE     = PROJECT_DIR / "rss_cache.json"
OUTPUT_ROOT    = PROJECT_DIR / "instagram_posts"
NEWS_POSTS = 7
TIP_POSTS  = 7
TEXT_MODEL     = os.getenv("INSTAGRAM_MODEL") or "gpt-4.1"
NEWSLETTER_URL = "aitechhelper.com/ai-news"

# =========================
# LOAD CACHE
# =========================

def load_cached_items() -> list[dict]:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            "No RSS cache found. Run generate_ai_hub_briefing.py first, "
            "or use --fresh to fetch now."
        )
    data = json.loads(CACHE_FILE.read_text())
    items = data["items"]
    print(f"Loaded {len(items)} items from RSS cache.")
    return items

# =========================
# GENERATE CONTENT
# =========================

PROMPT = """You are writing content for The AI Hub Daily Instagram account.

AUDIENCE: Everyday people curious about AI — small business owners, creators, professionals, curious consumers. Not developers.

TASK: From the news items below, generate two sets of Instagram posts as a single JSON array of exactly {total} objects.

--- SET 1: {news} NEWS POSTS ---
Select the {news} most engaging stories. For each, set "type": "news" and:

"headline": 6-12 words, ALL CAPS, emotionally engaging, curiosity-driven.
  BAD: "NVIDIA RELEASES COSMOS 3 FOR PHYSICAL AI"
  GOOD: "ROBOTS JUST GOT A BRAIN"

"image_direction": One sentence. Who or what to feature. No style instructions.

"caption":
  - Punchy hook sentence
  - Blank line
  - 3-4 short conversational paragraphs
  - Blank line
  - Engagement question
  - Blank line
  - "Follow for more AI news and tips."
  - "Subscribe to our free AI newsletter for daily AI news and insights:"
  - "{newsletter}"
  - Blank line
  - 5 hashtags

--- SET 2: {tips} TIP POSTS ---
Take {tips} of the same stories and reframe them as practical AI tips for business owners, professionals, or everyday users. For each, set "type": "tip" and:

"headline": 6-12 words, ALL CAPS, tip-focused and actionable.
  Example: "DITCH CHATGPT AND RUN YOUR OWN PRIVATE AI"
  Example: "USE AI TO RESPOND TO CUSTOMER EMAILS IN SECONDS"

"image_direction": One sentence. Who or what to feature. No style instructions.

"caption":
  - Start with "AI Tip #[1-{tips}]:" followed by a one-line summary of the tip
  - Blank line
  - 3-4 short paragraphs explaining HOW to use or apply this tip in plain English
  - Make it actionable — tell the reader exactly what to do or try
  - Blank line
  - Engagement question
  - Blank line
  - "Follow for more AI news and tips."
  - "Subscribe to our free AI newsletter for daily AI news and insights:"
  - "{newsletter}"
  - Blank line
  - 5 hashtags

Return ONLY a valid JSON array. No markdown, no code fences.

NEWS ITEMS:
{items}"""

def build_items_text(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items[:60], 1):
        text = it.get("full_text") or it.get("summary", "")
        lines.append(f"{i}. TITLE: {it['title']}\n   SUMMARY: {text[:300]}\n   URL: {it['url']}\n")
    return "\n".join(lines)

def generate_posts(items: list[dict]) -> list[dict]:
    from openai import OpenAI
    client = OpenAI()

    total = NEWS_POSTS + TIP_POSTS
    prompt = PROMPT.format(total=total, news=NEWS_POSTS, tips=TIP_POSTS, newsletter=NEWSLETTER_URL, items=build_items_text(items))

    print(f"Generating {NEWS_POSTS} news posts + {TIP_POSTS} tip posts with {TEXT_MODEL}...")
    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    posts = json.loads(raw.strip())
    print(f"  Done — {len(posts)} posts generated.")
    return posts

# =========================
# SAVE OUTPUT
# =========================

def save_output(posts: list[dict], out_path: Path) -> None:
    lines = [
        f"THE AI HUB DAILY — Instagram Posts",
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        "=" * 60,
        "",
    ]

    news_posts = [p for p in posts if p.get("type") == "news"]
    tip_posts  = [p for p in posts if p.get("type") == "tip"]

    for section, section_posts in [("NEWS POSTS", news_posts), ("TIP POSTS", tip_posts)]:
        lines += [f"── {section} ──", ""]
        for i, post in enumerate(section_posts, 1):
            lines += [
                f"POST {i} OF {len(section_posts)}",
                "-" * 40,
                "",
                f"HEADLINE: {post['headline']}",
                "",
                f"IMAGE DIRECTION: {post['image_direction']}",
                "",
                "CAPTION:",
                post["caption"],
                "",
                "=" * 60,
                "",
            ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to: {out_path}")

# =========================
# MAIN
# =========================

def send_email(content: str, date_str: str) -> None:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    gmail_address  = os.getenv("GMAIL_ADDRESS", "")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")
    if not gmail_address or not gmail_password:
        print("  Gmail credentials missing — skipping email.")
        return

    msg = MIMEMultipart()
    msg["From"]    = gmail_address
    msg["To"]      = gmail_address
    msg["Subject"] = f"AI Hub Weekly — Instagram Posts {date_str}"
    msg.attach(MIMEText(content, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_address, gmail_password)
            server.sendmail(gmail_address, gmail_address, msg.as_string())
        print(f"  Email sent to {gmail_address}")
    except Exception as e:
        print(f"  Email failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Re-fetch RSS feeds first")
    parser.add_argument("--post",  action="store_true", help="Also email the posts to yourself")
    args = parser.parse_args()

    if args.fresh:
        import subprocess, sys
        print("Refreshing RSS cache...")
        subprocess.run([sys.executable, str(PROJECT_DIR / "generate_ai_hub_briefing.py"), "--fresh"], check=False)

    items = load_cached_items()
    if not items:
        raise SystemExit("No items in cache. Run generate_ai_hub_briefing.py first.")

    posts = generate_posts(items)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = OUTPUT_ROOT / f"{date_str}.txt"
    save_output(posts, out_path)

    if args.post:
        print("Emailing Instagram posts...")
        send_email(out_path.read_text(), date_str)

if __name__ == "__main__":
    main()
