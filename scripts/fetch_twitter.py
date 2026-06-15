#!/usr/bin/env python3
"""
World Cup 2026 — Twitter Data Fetcher (TikHub)
================================================
One-time batch fetch of real football analysis from Twitter/X via TikHub API.

Fetches:
  1. Team strength / prediction tweets for all 48 teams
  2. Injury news for World Cup squads
  3. World Cup 2026 power rankings from football journalists
  4. Starting XI / squad depth discussions

Output: data/twitter_data.json  →  consumed by signals.py

Usage:
  set TIKHUB_API_KEY=your_key
  python scripts/fetch_twitter.py              # fetch all teams
  python scripts/fetch_twitter.py --team ARG   # single team
  python scripts/fetch_twitter.py --quick      # top 16 teams only (faster)
"""

import json
import os
import re
import sys
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import requests

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("TIKHUB_API_KEY", "")
BASE_URL = "https://api.tikhub.io/api/v1/twitter/web"
TEAMS_FILE = Path(__file__).parent.parent / "data" / "teams.json"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "twitter_data.json"

# Delay between API calls (be nice to the API)
REQUEST_DELAY = 0.8
TWEETS_PER_QUERY = 20

# Football analysis accounts (prioritize tweets from these)
TRUSTED_ACCOUNTS = [
    "FabrizioRomano", "ESPNFC", "brfootball", "Squawka", "OptaJoe",
    "WhoScored", "TheAthleticFC", "CBSSportsGolazo", "FIFAWorldCup",
    "ESPNDeportes", "FOXSoccer", "NBCSportsSoccer", "BBCSport",
    "SkySportsPL", "433", "goal", "OneFootball",
]

# ---------------------------------------------------------------------------
# NLP: keyword-based football analysis extractor
# ---------------------------------------------------------------------------

# Strength indicators
STRENGTH_POSITIVE = [
    r'\b(favorites?\b|contenders?\b|dark\s*horses?\b|unbeaten\b|dominant\b)',
    r'\b(world\s*class\b|elite\b|stacked\b|deep\s*squad\b|generational\b)',
    r'\b(champion\w*\b|title\b|trophy\b|winner\b|unstoppable\b)',
    r'\b(on\s*fire\b|in\s*form\b|peaking\b|momentum\b|confident\b)',
    r'\b(power\s*house\b|juggernaut\b|machine\b|wagon\b|serious\b)',
]

STRENGTH_NEGATIVE = [
    r'\b(overrated\b|flop|flopping|bottlers?\b|chokers?\b|frauds?\b)',
    r'\b(struggling\b|weak\b|vulnerable\b|exposed\b|leaky\b)',
    r'\b(injured\b|injuries\b|absent\b|missing\b|unavailable\b|doubt\b)',
    r'\b(out\s*of\s*form\b|declining\b|aging\b|past\s*it\b)',
    r'\b(disappointing\b|underwhelming\b|questionable\b|concerning\b)',
]

# Injury-related keywords
INJURY_PATTERNS = [
    r'(injured|injury|out of|ruled out|doubt|doubtful|unavailable|missing|absent|setback|fitness|recovery|rehab|scan|MRI|surgery|ACL|hamstring|ankle|knee|muscle|strain|fracture|concussion)',
    r'(hurt|knock|blow|sidelined|stretchered|crutches|walking boot)',
]

# Player name extraction (basic)
PLAYER_NAME_PATTERN = re.compile(
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b'
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeamTwitterData:
    """Processed Twitter data for one team."""
    code: str
    strength_score: float = 0.0        # -1.0 to 1.0 (from analysis tweets)
    confidence: float = 0.0             # 0.0 to 1.0 (based on tweet volume)
    tweet_volume: int = 0
    injuries: List[dict] = field(default_factory=list)  # [{player, status, source}]
    top_topics: List[str] = field(default_factory=list)
    key_quotes: List[str] = field(default_factory=list)  # actual tweet texts
    ranking_mentions: List[dict] = field(default_factory=list)  # [{source, rank}]


@dataclass
class WorldCupTwitterData:
    """Aggregated Twitter data for the whole tournament."""
    last_updated: str
    teams: Dict[str, TeamTwitterData] = field(default_factory=dict)
    global_power_rankings: List[dict] = field(default_factory=list)
    global_topics: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TikHub API Client
# ---------------------------------------------------------------------------

class TikHubTwitter:
    """Minimal TikHub Twitter API wrapper."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })

    def search_tweets(self, keyword: str, search_type: str = "Top",
                       count: int = 20) -> List[dict]:
        """Search Twitter for tweets matching keyword."""
        url = f"{BASE_URL}/fetch_search_timeline"
        params = {"keyword": keyword, "search_type": search_type}

        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"  ⚠️ API error {resp.status_code}: {resp.text[:200]}")
                return []

            data = resp.json()
            # TikHub response structure varies; drill down to tweets
            tweets = self._extract_tweets(data)
            return tweets[:count]

        except requests.RequestException as e:
            print(f"  ❌ Request failed: {e}")
            return []
        except json.JSONDecodeError:
            print(f"  ❌ Invalid JSON response")
            return []

    def _extract_tweets(self, data: dict) -> List[dict]:
        """Extract tweet list from TikHub response (handles multiple formats)."""
        # TikHub Twitter format: {"data": {"timeline": [...]}}
        if isinstance(data, list):
            return data
        if "data" in data:
            inner = data["data"]
            if isinstance(inner, list):
                return inner
            if isinstance(inner, dict):
                # Primary format: data.timeline
                if "timeline" in inner:
                    return inner["timeline"]
                if "tweets" in inner:
                    return inner["tweets"]
                if "items" in inner:
                    return inner["items"]
                if "entries" in inner:
                    return self._parse_entries(inner["entries"])
        if "tweets" in data:
            return data["tweets"]
        if "timeline" in data:
            return data["timeline"]
        if "results" in data:
            return data["results"]
        return []

    def _parse_entries(self, entries: list) -> List[dict]:
        """Parse Twitter timeline entries."""
        tweets = []
        for entry in entries:
            if isinstance(entry, dict):
                content = entry.get("content", {})
                tweet = content.get("tweet", content.get("item", {}))
                if tweet:
                    tweets.append(tweet)
        return tweets


# ---------------------------------------------------------------------------
# Tweet text analysis
# ---------------------------------------------------------------------------

def analyze_tweet_text(text: str, team_names: List[str]) -> dict:
    """Extract structured signals from a tweet about a team.

    Returns: {strength_signal: float, has_injury: bool, injury_player: str|None,
               topics: [str], is_trusted: bool}
    """
    text_lower = text.lower()
    result = {
        "strength_signal": 0.0,
        "has_injury": False,
        "injury_player": None,
        "topics": [],
        "is_analysis": False,
    }

    # Count positive/negative strength signals
    pos_count = sum(len(re.findall(p, text_lower)) for p in STRENGTH_POSITIVE)
    neg_count = sum(len(re.findall(p, text_lower)) for p in STRENGTH_NEGATIVE)

    total = pos_count + neg_count
    if total > 0:
        result["strength_signal"] = (pos_count - neg_count) / (total + 2)
        result["is_analysis"] = total >= 2

    # Check for injuries
    injury_hits = sum(len(re.findall(p, text_lower)) for p in INJURY_PATTERNS)
    if injury_hits >= 2:
        result["has_injury"] = True
        # Try to find player name near injury mention
        for name in team_names:
            if name.lower() in text_lower:
                result["injury_player"] = name
                break

    # Extract topics (capitalized phrases, hashtags)
    hashtags = re.findall(r'#(\w+)', text)
    result["topics"] = [h for h in hashtags if len(h) > 3][:5]

    return result


def compute_team_strength(tweets: List[dict], team: dict) -> TeamTwitterData:
    """Aggregate tweets about a team into a TeamTwitterData record."""
    code = team["code"]
    team_names = [team["name_en"], team["name_zh"]] + team.get("key_players", [])
    # Add common nicknames
    nicknames = {
        "ARG": ["La Albiceleste", "Messi"],
        "BRA": ["Seleção", "Canarinho"],
        "ENG": ["Three Lions"],
        "FRA": ["Les Bleus", "Mbappé"],
        "GER": ["Die Mannschaft"],
        "ESP": ["La Roja", "La Furia"],
        "NED": ["Oranje"],
        "POR": ["Seleção das Quinas", "CR7", "Ronaldo"],
        "USA": ["USMNT"],
        "MEX": ["El Tri"],
        "JPN": ["Samurai Blue"],
        "KOR": ["Taegeuk Warriors"],
        "SEN": ["Lions of Teranga"],
        "GHA": ["Black Stars"],
        "NGA": ["Super Eagles"],
        "EGY": ["Pharaohs"],
        "MAR": ["Atlas Lions"],
        "CRO": ["Vatreni"],
        "URU": ["La Celeste"],
        "COL": ["Los Cafeteros"],
    }
    if code in nicknames:
        team_names.extend(nicknames[code])

    td = TeamTwitterData(code=code)
    strength_signals = []
    injury_found = []

    for tweet in tweets:
        # Handle TikHub tweet format: {text, screen_name, tweet_id, favorites, retweets...}
        text = ""
        if "text" in tweet:
            text = tweet["text"]
        elif "full_text" in tweet:
            text = tweet["full_text"]
        elif "content" in tweet:
            text = str(tweet["content"])

        if not text:
            continue

        # Get author
        author = tweet.get("screen_name") or tweet.get("user", {}).get("screen_name", "")

        analysis = analyze_tweet_text(text, team_names)

        if analysis["is_analysis"]:
            strength_signals.append(analysis["strength_signal"])
            td.top_topics.extend(analysis["topics"])

            # Save notable tweets as quotes
            if analysis["strength_signal"] > 0.3 or analysis["strength_signal"] < -0.3:
                clean_text = text[:200].replace("\n", " ")
                td.key_quotes.append(f"@{author}: {clean_text}")

        if analysis["has_injury"]:
            injury_found.append({
                "player": analysis["injury_player"] or "unknown",
                "status": "uncertain",
                "source": author or "twitter",
            })

    # Aggregate
    td.tweet_volume = len(tweets)
    td.confidence = min(1.0, math.log2(len(strength_signals) + 1) / 6) if strength_signals else 0.0

    if strength_signals:
        # Weighted: more emphasis on extreme signals, dampen neutral ones
        td.strength_score = round(sum(strength_signals) / len(strength_signals), 3)

    td.injuries = injury_found[:5]
    td.top_topics = list(dict.fromkeys(td.top_topics))[:8]  # dedupe, keep order
    td.key_quotes = td.key_quotes[:6]

    return td


# ---------------------------------------------------------------------------
# Main fetch orchestration
# ---------------------------------------------------------------------------

def build_search_queries(team: dict) -> List[str]:
    """Build Twitter search queries for a team."""
    name_en = team["name_en"]
    code = team["code"]

    queries = [
        f'"{name_en}" "World Cup 2026" (squad OR prediction OR analysis OR preview)',
        f'"{name_en}" World Cup 2026 (strength OR favorite OR contender OR dark horse)',
        f'"{name_en}" World Cup 2026 (injury OR injured OR fitness OR absence)',
    ]

    # For top teams, add specific queries
    top_teams = ["Argentina", "Brazil", "France", "England", "Spain", "Portugal",
                 "Germany", "Netherlands", "Belgium", "Italy"]
    if name_en in top_teams:
        queries.insert(0, f'"{name_en}" World Cup 2026 champion')

    return queries


def fetch_all_teams(teams: dict, api_key: str,
                     quick: bool = False,
                     callback=None) -> WorldCupTwitterData:
    """Fetch Twitter data for all 48 teams (or top 16 if quick=True)."""
    api = TikHubTwitter(api_key)

    # Determine teams to fetch
    all_teams = []
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            all_teams.append(t)

    if quick:
        # Top 16 by Elo + hosts
        all_teams.sort(key=lambda t: t["elo"], reverse=True)
        hosts = {"USA", "CAN", "MEX"}
        top = [t for t in all_teams[:14] if t["code"] not in hosts]
        hosts_teams = [t for t in all_teams if t["code"] in hosts]
        all_teams = top[:13] + hosts_teams

    result = WorldCupTwitterData(last_updated="")

    print(f"\n{'='*60}")
    print(f"  🐦 Fetching Twitter data for {len(all_teams)} teams via TikHub")
    print(f"{'='*60}\n")

    for i, team in enumerate(all_teams):
        code = team["code"]
        name = team["name_en"]
        print(f"  [{i+1}/{len(all_teams)}] {name} ({code})...", end=" ", flush=True)

        all_tweets = []
        queries = build_search_queries(team)

        for query in queries:
            tweets = api.search_tweets(query, search_type="Top",
                                        count=TWEETS_PER_QUERY)
            all_tweets.extend(tweets)
            time.sleep(REQUEST_DELAY)

        # Deduplicate by tweet ID (TikHub uses tweet_id)
        seen_ids = set()
        unique_tweets = []
        for t in all_tweets:
            tid = t.get("tweet_id") or t.get("id") or t.get("id_str") or hash(str(t))
            if tid not in seen_ids:
                seen_ids.add(tid)
                unique_tweets.append(t)

        td = compute_team_strength(unique_tweets, team)
        result.teams[code] = td

        signal_icon = "🟢" if td.strength_score > 0.15 else ("🟡" if td.strength_score > -0.15 else "🔴")
        print(f"{signal_icon} score={td.strength_score:+.2f} vol={td.tweet_volume} conf={td.confidence:.1%}")

        if callback:
            callback(i + 1, len(all_teams))

        time.sleep(REQUEST_DELAY)

    # Fetch global power rankings
    print(f"\n  🌍 Fetching global World Cup power rankings...")
    ranking_tweets = api.search_tweets(
        '"World Cup 2026" "power rankings" OR "top 10" OR "ranking"',
        search_type="Top", count=30
    )
    # Extract ranking info
    for tweet in ranking_tweets[:10]:
        text = tweet.get("text", tweet.get("full_text", ""))
        if text:
            source = tweet.get("user", {}).get("screen_name", "unknown")
            result.global_power_rankings.append({
                "source": source,
                "text": text[:300],
            })
    time.sleep(REQUEST_DELAY)

    # Fetch injury roundup
    print(f"  🏥 Fetching injury news...")
    injury_tweets = api.search_tweets(
        '"World Cup 2026" (injury OR injured OR ruled out OR fitness)',
        search_type="Latest", count=30
    )
    if injury_tweets:
        # Check each team for new injuries
        for team in all_teams:
            code = team["code"]
            name = team["name_en"]
            for tweet in injury_tweets:
                text = tweet.get("text", tweet.get("full_text", ""))
                if name.lower() in text.lower():
                    analysis = analyze_tweet_text(text, team.get("key_players", []))
                    if analysis["has_injury"]:
                        existing = {(i["player"], i["source"])
                                     for i in result.teams[code].injuries}
                        new_injury = {
                            "player": analysis["injury_player"] or "unknown",
                            "status": "reported",
                            "source": tweet.get("screen_name", tweet.get("user", {}).get("screen_name", "twitter")),
                        }
                        key = (new_injury["player"], new_injury["source"])
                        if key not in existing:
                            result.teams[code].injuries.append(new_injury)

    from datetime import datetime
    result.last_updated = datetime.now().isoformat()

    return result


# ---------------------------------------------------------------------------
# Save & Load
# ---------------------------------------------------------------------------

def save_twitter_data(data: WorldCupTwitterData, path: Path = OUTPUT_FILE):
    """Save Twitter data to JSON cache file."""
    output = {
        "last_updated": data.last_updated,
        "teams": {code: asdict(td) for code, td in data.teams.items()},
        "global_power_rankings": data.global_power_rankings,
        "global_topics": data.global_topics,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Twitter data saved to {path}")
    print(f"   Teams with data: {len(data.teams)}")


def load_twitter_data(path: Path = OUTPUT_FILE) -> dict:
    """Load cached Twitter data."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(data: WorldCupTwitterData, teams: dict):
    """Print a summary of fetched Twitter data."""
    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = t

    print(f"\n{'='*60}")
    print(f"  📊 TWITTER SENTIMENT SUMMARY")
    print(f"{'='*60}")

    # Teams ranked by Twitter strength score
    ranked = sorted(data.teams.items(),
                    key=lambda x: x[1].strength_score, reverse=True)

    print(f"\n  Most bullish (positive Twitter sentiment):")
    for code, td in ranked[:8]:
        team = elo_lookup.get(code, {})
        name = team.get("name_en", code)
        bar = "█" * max(1, int((td.strength_score + 1) * 15))
        print(f"  🟢 {name:<20} {td.strength_score:+.2f}  vol={td.tweet_volume}  {bar}")

    print(f"\n  Most bearish (negative Twitter sentiment):")
    for code, td in ranked[-8:]:
        team = elo_lookup.get(code, {})
        name = team.get("name_en", code)
        bar = "█" * max(1, int(abs(td.strength_score) * 15))
        print(f"  🔴 {name:<20} {td.strength_score:+.2f}  vol={td.tweet_volume}  {bar}")

    # Injuries found
    injured_teams = [(code, td) for code, td in data.teams.items() if td.injuries]
    if injured_teams:
        print(f"\n  🏥 Injury mentions found:")
        for code, td in injured_teams:
            team = elo_lookup.get(code, {})
            name = team.get("name_en", code)
            for inj in td.injuries[:3]:
                print(f"  • {name}: {inj['player']} ({inj['status']}) — via {inj['source']}")

    # Top quotes
    print(f"\n  💬 Notable tweets:")
    for code, td in ranked[:5]:
        team = elo_lookup.get(code, {})
        name = team.get("name_en", code)
        for quote in td.key_quotes[:2]:
            print(f"  [{name}] {quote[:150]}...")


# ===================================================================
# CLI
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fetch World Cup 2026 Twitter data via TikHub API"
    )
    parser.add_argument("--team", help="Fetch single team (code, e.g. ARG)")
    parser.add_argument("--quick", action="store_true", help="Top 16 teams only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show queries without calling API")
    args = parser.parse_args()

    if not API_KEY:
        print("❌ TIKHUB_API_KEY environment variable not set.")
        print("   Run: set TIKHUB_API_KEY=your_key_here")
        print("   Then re-run this script.")
        return

    # Load teams
    with open(TEAMS_FILE, "r", encoding="utf-8") as f:
        teams = json.load(f)

    if args.dry_run:
        print("\n📋 Dry run — showing search queries:\n")
        for grp_teams in teams["groups"].values():
            for t in grp_teams[:2]:  # just 2 per group for brevity
                print(f"  {t['name_en']} ({t['code']}):")
                for q in build_search_queries(t):
                    print(f"    → {q}")
        return

    if args.team:
        # Single team fetch
        team = None
        for grp_teams in teams["groups"].values():
            for t in grp_teams:
                if t["code"].upper() == args.team.upper():
                    team = t
                    break
        if not team:
            print(f"❌ Team '{args.team}' not found.")
            return

        print(f"\n🐦 Fetching Twitter data for {team['name_en']} ({team['code']})...\n")
        api = TikHubTwitter(API_KEY)
        all_tweets = []
        for query in build_search_queries(team):
            tweets = api.search_tweets(query)
            all_tweets.extend(tweets)
            time.sleep(REQUEST_DELAY)

        td = compute_team_strength(all_tweets, team)
        print(f"\n  Strength score: {td.strength_score:+.2f}")
        print(f"  Tweet volume: {td.tweet_volume}")
        print(f"  Confidence: {td.confidence:.1%}")
        print(f"  Injuries: {td.injuries}")
        print(f"  Topics: {td.top_topics}")
        print(f"\n  Key quotes:")
        for q in td.key_quotes:
            print(f"  • {q}")
        return

    # Full fetch
    data = fetch_all_teams(teams, API_KEY, quick=args.quick)
    save_twitter_data(data)
    print_summary(data, teams)


if __name__ == "__main__":
    main()
