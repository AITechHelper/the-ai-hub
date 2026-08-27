#!/usr/bin/env python3
"""
The AI Hub — Weekly Video (Reel/Short) Script Generator

Writes ready-to-shoot 30-45s vertical video scripts for the two Instagram brands,
from the same weekly news cache the briefing/Instagram generators use.

Two brands, two jobs:
  - AI Hub Daily   → pure AI news. Goal: funnel viewers to the free AI newsletter
                     and capture business-owner emails. CTA = newsletter signup.
  - AI Tech Helper → teach business owners how to leverage the latest AI, and
                     promote the tools/services on aitechhelper.com. Goal: get
                     people to reach out for AI implementation help. CTA = DM / book.

Usage:
  python3 generate_video_scripts.py           # use cached RSS data, write file
  python3 generate_video_scripts.py --fresh   # re-fetch RSS feeds first
  python3 generate_video_scripts.py --post    # also email the scripts to yourself

Output:
  video_scripts/YYYY-MM-DD.txt  — all scripts for both brands, ready to shoot.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PROJECT_DIR = Path(__file__).parent
CACHE_FILE  = PROJECT_DIR / "rss_cache.json"
OUTPUT_ROOT = PROJECT_DIR / "video_scripts"

TEXT_MODEL       = os.getenv("SCRIPTS_MODEL") or "gpt-4.1"
SCRIPTS_PER_BRAND = int(os.getenv("SCRIPTS_PER_BRAND") or 5)

NEWSLETTER_URL = "aitechhelper.com/ai-news"
WEBSITE_URL    = "aitechhelper.com"

# Fill this with the ACTUAL tools/features on your site so AI Tech Helper scripts
# name-drop them specifically instead of pitching generically. One per line, e.g.
#   "AI email responder — drafts replies to customer emails in seconds"
#   "Private AI setup — run your own ChatGPT that never leaks your data"
# Leave the list empty to have scripts pitch your services generically.
WEBSITE_TOOLS = [
    "AI Receptionist ($200/mo) — 24/7 AI voice agent that answers every call, qualifies the lead, books the appointment, handles FAQs, syncs to your calendar, and texts/emails you each new lead instantly",
    "AI Messaging Agent ($300/mo) — one AI agent that answers every inbound message across SMS, WhatsApp, Instagram DMs, Facebook Messenger, and website chat, all from one dashboard",
    "Bundle ($400/mo, best value) — AI Receptionist + AI Messaging Agent together; calls and messages fully covered for one flat rate, saves $100/mo",
    "Add-on automations (quoted per business) — Missed-Call Text-Back, Appointment Reminders, Review & Referral, Invoice Follow-Up, Estimate Follow-Up, Contract & Waiver, New-Client Onboarding, and Inbox Management",
]

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

def build_items_text(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items[:60], 1):
        text = it.get("full_text") or it.get("summary", "")
        lines.append(f"{i}. TITLE: {it['title']}\n   SUMMARY: {text[:300]}\n   URL: {it['url']}\n")
    return "\n".join(lines)

# =========================
# BRAND PROMPTS
# =========================
# Every script, for both brands, returns the SAME structure so the output is
# consistent and ready to shoot:
#   headline, hook, script, on_screen_text[], broll, caption, cta

_SHARED_STRUCTURE = """For each video, return an object with EXACTLY these keys:

"headline": 4-9 words naming the story/topic (for your reference, not spoken).

"hook": The first 3 seconds, spoken. This is the whole game — it must stop the
  scroll. Use a contrarian take, a direct callout, a withheld-information framing,
  or a stakes-raising claim. NEVER a flat headline restatement.
  BAD: "OpenAI released a new model today."
  GOOD: "Your job just got a deadline and nobody told you."

"script": The full 30-45 second voiceover, written to be read aloud. 80-110 words,
  2-3 short punchy beats. Opinionated and direct, no corporate tone, no filler.
  The hook is beat one; do not repeat it verbatim, continue from it.

"on_screen_text": An array of 3-5 SHORT text overlays (2-6 words each) that punch
  up the key beats as captions on screen.

"broll": One line of concrete b-roll / visual direction — what to show on screen
  while the voiceover plays. Specific and shootable.

"caption": The Instagram caption:
  - A punchy first line (the hook, reworded)
  - Blank line
  - 2-3 short conversational lines
  - Blank line
  - Engagement question
  - Blank line
  - The CTA line(s) below for this brand
  - Blank line
  - 5 relevant hashtags

"cta": The single call-to-action for this brand (see BRAND GOAL), one short line."""

AI_HUB_DAILY_PROMPT = """You are the scriptwriter for THE AI HUB DAILY — an AI-news Instagram/Reels account.

BRAND GOAL: This is a NEWS account. Every video reports the latest AI news in a
fast, engaging, slightly provocative way, then drives the viewer to sign up for
the free AI newsletter. The audience is business owners and professionals — the
newsletter is how we capture their email. So the CTA is ALWAYS newsletter signup.

AUDIENCE: Business owners, professionals, and curious non-technical people. Not developers.

CTA (use for every "cta" and end every caption with this):
  "Follow for daily AI news 🗞️  Get it first — free newsletter at {newsletter}"

TASK: From the news items below, pick the {n} most engaging/surprising/controversial
stories and write {n} vertical video scripts (Reels/Shorts, 30-45 seconds each).

{structure}

Return ONLY a valid JSON array of {n} objects. No markdown, no code fences.

NEWS ITEMS:
{items}"""

AI_TECH_HELPER_PROMPT = """You are the scriptwriter for AI TECH HELPER — the company Instagram/Reels account
for an AI-implementation consultancy (website: {website}).

BRAND GOAL: This account TEACHES business owners how to use the latest AI to their
advantage — take a piece of recent AI news or a new feature and explain, concretely,
how a business owner can leverage it. Every video must also promote the company's
tools/services and push viewers to reach out for hands-on help implementing AI.
This is a lead-generation account for the consultancy. So the CTA is ALWAYS
"reach out / DM / book help," and the script should naturally plug the company's
tools where it fits.

AUDIENCE: Small and mid-size business owners who know AI matters but don't know how
to actually apply it. Speak to their time, money, and competition.

{tools_block}

CTA (use for every "cta" and end every caption with this):
  "Want this running in your business? Book a free consult at {website}/contact-us or DM me 👉"

TASK: From the news items below, pick the {n} stories/features with the clearest
business leverage and write {n} vertical video scripts (Reels/Shorts, 30-45 seconds
each). Each must translate the news into "here's exactly how a business owner uses
this," and work in a plug for the company's help/tools.

{structure}

Return ONLY a valid JSON array of {n} objects. No markdown, no code fences.

NEWS ITEMS:
{items}"""

def _tools_block() -> str:
    if not WEBSITE_TOOLS:
        return ("OUR TOOLS/SERVICES: (none listed yet — pitch the consultancy's hands-on "
                "AI implementation help generically, e.g. 'we set this up for you').")
    listed = "\n".join(f"  - {t}" for t in WEBSITE_TOOLS)
    return ("OUR TOOLS/SERVICES (name-drop the relevant one when it fits the story):\n"
            + listed)

# =========================
# GENERATE
# =========================

def _call_model(prompt: str) -> list[dict]:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

def generate_scripts(items: list[dict]) -> dict[str, list[dict]]:
    items_text = build_items_text(items)

    print(f"Generating {SCRIPTS_PER_BRAND} AI Hub Daily scripts with {TEXT_MODEL}...")
    hub = _call_model(AI_HUB_DAILY_PROMPT.format(
        n=SCRIPTS_PER_BRAND, newsletter=NEWSLETTER_URL,
        structure=_SHARED_STRUCTURE, items=items_text,
    ))
    print(f"  Done — {len(hub)} scripts.")

    print(f"Generating {SCRIPTS_PER_BRAND} AI Tech Helper scripts with {TEXT_MODEL}...")
    helper = _call_model(AI_TECH_HELPER_PROMPT.format(
        n=SCRIPTS_PER_BRAND, website=WEBSITE_URL, tools_block=_tools_block(),
        structure=_SHARED_STRUCTURE, items=items_text,
    ))
    print(f"  Done — {len(helper)} scripts.")

    return {"AI Hub Daily": hub, "AI Tech Helper": helper}

# =========================
# SAVE OUTPUT
# =========================

def _format_script(s: dict, i: int, total: int) -> list[str]:
    ost = s.get("on_screen_text") or []
    if isinstance(ost, str):
        ost = [ost]
    return [
        f"SCRIPT {i} OF {total}",
        "-" * 40,
        "",
        f"TOPIC: {s.get('headline', '')}",
        "",
        f"HOOK (0-3s): {s.get('hook', '')}",
        "",
        "VOICEOVER (30-45s):",
        s.get("script", ""),
        "",
        "ON-SCREEN TEXT:",
        *[f"  • {t}" for t in ost],
        "",
        f"B-ROLL / VISUAL: {s.get('broll', '')}",
        "",
        f"CTA: {s.get('cta', '')}",
        "",
        "CAPTION:",
        s.get("caption", ""),
        "",
        "=" * 60,
        "",
    ]

def save_output(brands: dict[str, list[dict]], out_path: Path) -> None:
    lines = [
        "THE AI HUB — Weekly Video Scripts (Reels/Shorts, 30-45s)",
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        "=" * 60,
        "",
    ]
    for brand, scripts in brands.items():
        lines += [f"════ {brand.upper()} ════", ""]
        for i, s in enumerate(scripts, 1):
            lines += _format_script(s, i, len(scripts))
        lines += [""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to: {out_path}")

# =========================
# EMAIL
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
    msg["Subject"] = f"AI Hub Weekly — Video Scripts {date_str}"
    msg.attach(MIMEText(content, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_address, gmail_password)
            server.sendmail(gmail_address, gmail_address, msg.as_string())
        print(f"  Email sent to {gmail_address}")
    except Exception as e:
        print(f"  Email failed: {e}")

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Generate weekly video scripts for both brands")
    parser.add_argument("--fresh", action="store_true", help="Re-fetch RSS feeds first")
    parser.add_argument("--post",  action="store_true", help="Also email the scripts to yourself")
    args = parser.parse_args()

    if args.fresh:
        import subprocess, sys
        print("Refreshing RSS cache...")
        subprocess.run([sys.executable, str(PROJECT_DIR / "generate_ai_hub_briefing.py"), "--fresh"], check=False)

    items = load_cached_items()
    if not items:
        raise SystemExit("No items in cache. Run generate_ai_hub_briefing.py first.")

    brands = generate_scripts(items)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_path = OUTPUT_ROOT / f"{date_str}.txt"
    save_output(brands, out_path)

    if args.post:
        print("Emailing video scripts...")
        send_email(out_path.read_text(), date_str)

if __name__ == "__main__":
    main()
