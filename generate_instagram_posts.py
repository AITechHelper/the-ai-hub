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
POSTS_PER_RUN  = 14
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

AUDIENCE: Everyday people curious about AI — not developers. Think small business owners, creators, professionals, curious consumers.

TASK: From the news items below, select the {n} stories BEST suited for Instagram. Prioritize:
- New AI tools or features everyday people can actually use
- Big product launches (ChatGPT updates, new models, major apps)
- Surprising or counterintuitive AI stories
- Stories with broad human interest
- Avoid: purely technical research, niche developer tools, corporate/legal/funding stories unless major

For each story output a JSON array with exactly {n} objects. Each object must have:

"headline": 6-12 words, ALL CAPS, emotionally engaging, curiosity-driven. NOT a news article title.
  BAD: "NVIDIA RELEASES COSMOS 3 FOR PHYSICAL AI"
  GOOD: "ROBOTS JUST GOT A BRAIN"
  BAD: "OPENAI EXPANDS CODEX USAGE"
  GOOD: "5 MILLION PEOPLE NOW USE CODEX"

"image_direction": One sentence. Just say who or what to feature in the image. No style instructions.
  Example: "Sam Altman, ChatGPT interface"
  Example: "Jensen Huang, robot"
  Example: "Small business owner at laptop"

"caption": Full Instagram caption:
  - Hook sentence (1 line, punchy)
  - Blank line
  - 3-4 short paragraphs in plain conversational English
  - Blank line
  - Engagement question
  - Blank line
  - "Subscribe to our free AI newsletter for daily AI news and insights:"
  - "{newsletter}"
  - Blank line
  - 5 hashtags (mix of broad and story-specific)

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

    prompt = PROMPT.format(n=POSTS_PER_RUN, newsletter=NEWSLETTER_URL, items=build_items_text(items))

    print(f"Generating {POSTS_PER_RUN} posts with {TEXT_MODEL}...")
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

    for i, post in enumerate(posts, 1):
        lines += [
            f"POST {i} OF {len(posts)}",
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
    msg["Subject"] = f"AI Hub Daily — Instagram Posts {date_str}"
    msg.attach(MIMEText(content, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_address, gmail_password)
            server.sendmail(gmail_address, gmail_address, msg.as_string())
        print(f"  Email sent to {gmail_address}")
    except Exception as e:
        print(f"  Email failed: {e}")


def post_to_wordpress(content: str, date_str: str) -> None:
    import base64, requests as req
    site_url  = os.getenv("WP_SITE_URL", "").rstrip("/")
    username  = os.getenv("WP_USERNAME", "")
    password  = os.getenv("WP_APP_PASSWORD", "")
    if not all([site_url, username, password]):
        print("  WordPress credentials missing — skipping WP draft.")
        return

    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    resp = req.post(
        f"{site_url}/wp-json/wp/v2/posts",
        headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
        json={
            "title": f"Instagram Posts — {date_str}",
            "content": f"<pre>{content}</pre>",
            "status": "draft",
        },
        timeout=30,
    )
    if resp.status_code in (200, 201):
        print(f"  Saved as WordPress draft: {resp.json().get('link', '')}")
    else:
        print(f"  WordPress draft failed ({resp.status_code}): {resp.text[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Re-fetch RSS feeds first")
    parser.add_argument("--post",  action="store_true", help="Also save as WordPress draft")
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
