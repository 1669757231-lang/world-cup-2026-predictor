"""
World Cup 2026 — Reddit Sentiment Crawler
==========================================
Scrapes r/soccer and r/worldcup for community sentiment about each team.

Uses PRAW (Reddit API wrapper). Requires credentials.

Setup:
  1. Go to https://www.reddit.com/prefs/apps
  2. Create a "script" app
  3. Set env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Usage:
  python crawl_reddit.py                    # crawl recent posts
  python crawl_reddit.py --team ARG          # crawl for specific team
  python crawl_reddit.py --update-signals    # crawl and update signals.py data
"""

import json
import os
import re
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUBREDDITS = ["soccer", "worldcup"]
POST_LIMIT = 100          # posts to fetch per subreddit
COMMENT_LIMIT = 50        # top-level comments to analyze per post
TEAM_KEYWORDS_FILE = Path(__file__).parent.parent / "data" / "teams.json"

# Sentiment lexicon (simple keyword-based, can be upgraded to ML model)
POSITIVE_WORDS = [
    "win", "champion", "favorite", "strong", "dominate", "unstoppable",
    "amazing", "brilliant", "class", "world class", "elite", "best",
    "goat", "legendary", "incredible", "on fire", "🔥", "🏆", "⭐",
    "clinical", "dangerous", "lethal", "masterclass",
]

NEGATIVE_WORDS = [
    "lose", "overrated", "weak", "terrible", "awful", "embarrassing",
    "fraud", "finished", "washed", "bottled", "bottler", "choke",
    "trash", "garbage", "disaster", "pathetic", "💀", "😭",
    "out", "eliminated", "group stage exit", "fraudulent",
]

NEUTRALIZE_WORDS = ["not", "no", "never", "isn't", "aren't", "wasn't", "won't"]


# ---------------------------------------------------------------------------
# Sentiment Analyzer
# ---------------------------------------------------------------------------

class SimpleSentimentAnalyzer:
    """Keyword + rule-based sentiment analyzer.

    For production use, replace with a fine-tuned model or LLM-based analysis.
    """

    def analyze(self, text: str) -> float:
        """Return sentiment score from -1.0 (very negative) to 1.0 (very positive)."""
        text_lower = text.lower()
        words = text_lower.split()

        pos_score = 0
        neg_score = 0

        for i, word in enumerate(words):
            # Check for negation in previous 2 words
            negated = any(words[max(0, i - j)] in NEUTRALIZE_WORDS
                         for j in range(1, 3))

            if word in POSITIVE_WORDS:
                if negated:
                    neg_score += 1
                else:
                    pos_score += 1

            if word in NEGATIVE_WORDS:
                if negated:
                    pos_score += 1
                else:
                    neg_score += 1

        total = pos_score + neg_score
        if total == 0:
            return 0.0
        return (pos_score - neg_score) / total


# ---------------------------------------------------------------------------
# Team Mention Detector
# ---------------------------------------------------------------------------

def build_team_patterns(teams: dict) -> Dict[str, list]:
    """Build regex patterns to detect team mentions in text."""
    patterns = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            pats = [
                re.compile(r'\b' + re.escape(t["code"]) + r'\b', re.IGNORECASE),
                re.compile(r'\b' + re.escape(t["name_en"]) + r'\b', re.IGNORECASE),
            ]
            # Chinese name (partial match)
            if t["name_zh"]:
                pats.append(re.compile(re.escape(t["name_zh"])))
            # Key players
            for player in t.get("key_players", []):
                pats.append(re.compile(r'\b' + re.escape(player) + r'\b', re.IGNORECASE))
            patterns[t["code"]] = pats
    return patterns


def detect_teams(text: str, patterns: Dict[str, list]) -> List[str]:
    """Return list of team codes mentioned in text."""
    mentioned = []
    for code, pats in patterns.items():
        for pat in pats:
            if pat.search(text):
                mentioned.append(code)
                break
    return mentioned


# ---------------------------------------------------------------------------
# PRAW-based crawler
# ---------------------------------------------------------------------------

def crawl_reddit_praw(teams: dict, subreddits: list = None,
                       post_limit: int = 100) -> Dict[str, dict]:
    """Crawl Reddit using PRAW and return sentiment data per team.

    Returns: {team_code: {"sentiment_score": float, "volume": int, "topics": [str]}}
    """
    try:
        import praw
    except ImportError:
        print("❌ PRAW not installed. Run: pip install praw")
        print("   Also set env vars: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT")
        return {}

    if subreddits is None:
        subreddits = SUBREDDITS

    # Auth
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "world-cup-predictor/1.0")

    if not client_id or not client_secret:
        print("❌ Reddit API credentials not set.")
        print("   Export REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT")
        return {}

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    analyzer = SimpleSentimentAnalyzer()
    patterns = build_team_patterns(teams)

    # Accumulators
    team_sentiments = defaultdict(list)
    team_volumes = defaultdict(int)
    team_topics = defaultdict(lambda: defaultdict(int))

    for sub_name in subreddits:
        print(f"  Crawling r/{sub_name}...")
        subreddit = reddit.subreddit(sub_name)

        for post in subreddit.hot(limit=post_limit):
            # Analyze post title + body
            text = post.title
            if post.selftext:
                text += " " + post.selftext[:500]  # first 500 chars

            mentioned = detect_teams(text, patterns)
            if not mentioned:
                continue

            sentiment = analyzer.analyze(text)
            for team in mentioned:
                team_sentiments[team].append(sentiment)
                team_volumes[team] += 1

                # Extract topics (simple: noun phrases from title)
                words = post.title.lower().split()
                for w in words:
                    if len(w) > 4 and w.isalpha():
                        team_topics[team][w] += 1

            # Also analyze top comments
            try:
                post.comments.replace_more(limit=0)
                for comment in post.comments[:10]:
                    mentioned_c = detect_teams(comment.body, patterns)
                    if mentioned_c:
                        sent = analyzer.analyze(comment.body)
                        for team in mentioned_c:
                            team_sentiments[team].append(sent)
                            team_volumes[team] += 1
            except Exception:
                pass

    # Aggregate
    results = {}
    for code in patterns:
        sentiments = team_sentiments.get(code, [])
        volume = team_volumes.get(code, 0)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Top topics
        top_topics = sorted(team_topics.get(code, {}).items(),
                           key=lambda x: x[1], reverse=True)[:5]
        topics = [t for t, _ in top_topics]

        results[code] = {
            "sentiment_score": round(avg_sentiment, 3),
            "volume": volume,
            "topics": topics,
        }

    return results


# ---------------------------------------------------------------------------
# Offline / mock mode (for when Reddit API is unavailable)
# ---------------------------------------------------------------------------

def crawl_reddit_offline(teams: dict) -> Dict[str, dict]:
    """Generate mock Reddit sentiment data for testing / offline use.

    In production, this is replaced by the PRAW crawler.
    """
    import random
    rng = random.Random(42)

    results = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            # Higher Elo teams tend to have more positive sentiment
            elo_factor = (t["elo"] - 1500) / 500  # roughly -0.5 to 1.0
            base_sentiment = elo_factor * 0.4 + rng.uniform(-0.2, 0.2)
            sentiment = max(-1.0, min(1.0, base_sentiment))

            # Volume correlates with Elo and market value
            volume = int(50 + (t["elo"] - 1300) * 0.3 + rng.randint(0, 100))

            # Mock topics from key players
            topics = t.get("key_players", [])[:3]

            results[t["code"]] = {
                "sentiment_score": round(sentiment, 3),
                "volume": volume,
                "topics": topics,
            }

    return results


# ---------------------------------------------------------------------------
# Update signals
# ---------------------------------------------------------------------------

def update_signals(sentiment_data: Dict[str, dict]):
    """Write crawled sentiment data directly into signals.py runtime memory.

    In production, this would be done by importing and calling set_reddit_sentiment().
    Here we save to a JSON cache file that signals.py can load.
    """
    cache_path = Path(__file__).parent.parent / "data" / "reddit_sentiment.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sentiment_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Reddit sentiment cached to {cache_path}")
    print(f"   Teams with data: {len(sentiment_data)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reddit sentiment crawler for World Cup predictor")
    parser.add_argument("--team", help="Filter to specific team code (e.g. ARG)")
    parser.add_argument("--update-signals", action="store_true", help="Update signals cache")
    parser.add_argument("--offline", action="store_true", help="Use mock data (no API needed)")
    parser.add_argument("--posts", type=int, default=100, help="Posts per subreddit")
    args = parser.parse_args()

    # Load teams
    with open(TEAM_KEYWORDS_FILE, "r", encoding="utf-8") as f:
        teams = json.load(f)

    # Crawl
    if args.offline:
        print("📡 Using offline/mock sentiment data...")
        data = crawl_reddit_offline(teams)
    else:
        print("📡 Crawling Reddit for World Cup sentiment...")
        data = crawl_reddit_praw(teams, post_limit=args.posts)

    if not data:
        print("⚠️  No data retrieved. Use --offline for mock data.")
        return

    # Filter
    if args.team:
        if args.team in data:
            d = data[args.team]
            print(f"\n  {args.team}: sentiment={d['sentiment_score']:.2f}, "
                  f"volume={d['volume']}, topics={d['topics']}")
        else:
            print(f"  No data for {args.team}")
        return

    # Summary
    print(f"\n{'='*60}")
    print(f"  Reddit Sentiment Summary")
    print(f"{'='*60}")
    sorted_teams = sorted(data.items(), key=lambda x: x[1]["sentiment_score"], reverse=True)
    for code, d in sorted_teams[:20]:
        bar = "🟢" if d["sentiment_score"] > 0.2 else ("🟡" if d["sentiment_score"] > -0.2 else "🔴")
        print(f"  {bar} {code:<5} {d['sentiment_score']:+.3f}  (vol: {d['volume']})")

    if args.update_signals:
        update_signals(data)


if __name__ == "__main__":
    main()
