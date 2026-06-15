"""
World Cup 2026 — Dynamic Signal Layer
======================================
All signals that adjust Elo ratings BEFORE they enter the Poisson engine.

Signal sources:
  1. Injuries / squad changes — key player missing → Elo penalty
  2. Recent form — last 5 matches weighted Elo adjustment
  3. Reddit community sentiment — NLP score → small Elo modifier
  4. Betting market — odds-implied probability as calibration signal
  5. Home / confederation advantage — host nation & regional boost
  6. Market value depth — squad depth indicator for tournament endurance

Design: each signal is a PLUGGABLE function that takes a team dict and returns
an Elo delta. The main `apply_all_signals()` aggregates them and returns
adjusted Elo ratings for all teams.
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    """What a signal produced for one team."""
    team_code: str
    signal_name: str
    elo_delta: float
    reason: str


@dataclass
class AdjustedTeam:
    """A team with all signal adjustments applied."""
    code: str
    name_en: str
    name_zh: str
    base_elo: float
    adjusted_elo: float
    adjustments: List[SignalResult] = field(default_factory=list)

    @property
    def total_delta(self) -> float:
        return sum(a.elo_delta for a in self.adjustments)


# ---------------------------------------------------------------------------
# 1. Injury / Squad Signal
# ---------------------------------------------------------------------------

# Pre-defined injury penalties per key player (would be updated from news)
# Format: {team_code: {player_name: elo_penalty}}
# A star player missing = -20 to -50 Elo, role player = -5 to -15
INJURY_DB: Dict[str, Dict[str, float]] = {
    # Example entries — these get updated from live data
    # "BRA": {"Neymar": -30},
    # "ARG": {"Messi": -40},
    # "FRA": {"Mbappé": -35},
}

STAR_PLAYER_PENALTY = -35    # Ballon d'Or level player missing
KEY_PLAYER_PENALTY = -20     # Regular starter missing
ROLE_PLAYER_PENALTY = -8     # Rotation player missing


def injury_signal(team: dict) -> SignalResult:
    """Check if key players are injured/absent and apply Elo penalty."""
    code = team["code"]
    if code not in INJURY_DB or not INJURY_DB[code]:
        return SignalResult(code, "injury", 0.0, "no injuries reported")

    total_penalty = sum(INJURY_DB[code].values())
    missing = ", ".join(INJURY_DB[code].keys())
    return SignalResult(code, "injury", total_penalty,
                        f"missing: {missing} ({total_penalty:+d} Elo)")


# ---------------------------------------------------------------------------
# 2. Recent Form Signal
# ---------------------------------------------------------------------------

# Recent match results — {team_code: [(opponent, GF, GA, weight), ...]}
# Weight decays: most recent = 1.0, 5th recent = 0.4
RECENT_FORM: Dict[str, list] = {
    # Example format:
    # "ARG": [("BRA", 2, 1, 1.0), ("URU", 1, 0, 0.85), ("COL", 3, 2, 0.7)],
}

FORM_MAX_BOOST = 30     # Max Elo gain from great form
FORM_MAX_PENALTY = -30  # Max Elo loss from poor form
FORM_K = 20             # Sensitivity of form adjustment


def recent_form_signal(team: dict) -> SignalResult:
    """Adjust Elo based on recent match results (weighted)."""
    code = team["code"]
    if code not in RECENT_FORM or not RECENT_FORM[code]:
        return SignalResult(code, "form", 0.0, "no recent form data")

    matches = RECENT_FORM[code]
    weighted_perf = 0.0
    total_weight = 0.0

    for opponent, gf, ga, weight in matches:
        # Performance = goal diff scaled by opponent strength approximation
        margin = gf - ga
        perf = math.copysign(math.log2(abs(margin) + 1), margin)  # diminishing returns
        weighted_perf += perf * weight
        total_weight += weight

    avg_perf = weighted_perf / total_weight if total_weight > 0 else 0
    delta = max(FORM_MAX_PENALTY, min(FORM_MAX_BOOST, avg_perf * FORM_K))

    desc = "strong" if delta > 15 else ("solid" if delta > 5 else
           ("poor" if delta < -15 else ("weak" if delta < -5 else "neutral")))
    return SignalResult(code, "form", round(delta, 1),
                        f"recent form: {desc} ({delta:+.1f})")


# ---------------------------------------------------------------------------
# 3. Reddit Sentiment Signal
# ---------------------------------------------------------------------------

# This would be populated by crawl_reddit.py scraping r/soccer, r/worldcup
# Format: {team_code: {"sentiment_score": -1.0 to 1.0, "volume": post_count, "topics": [...]}}
REDDIT_SENTIMENT: Dict[str, dict] = {}

SENTIMENT_MAX_ELO = 15    # Max Elo adjustment from Reddit (intentionally small)


def reddit_sentiment_signal(team: dict) -> SignalResult:
    """Apply Reddit community sentiment as a small Elo modifier."""
    code = team["code"]
    if code not in REDDIT_SENTIMENT:
        return SignalResult(code, "reddit", 0.0, "no Reddit data")

    data = REDDIT_SENTIMENT[code]
    score = data.get("sentiment_score", 0.0)
    volume = data.get("volume", 0)

    # Confidence scales with volume (more posts = more reliable)
    volume_factor = min(1.0, math.log2(volume + 1) / 8) if volume > 0 else 0.3
    delta = score * SENTIMENT_MAX_ELO * volume_factor

    mood = "bullish" if score > 0.3 else ("bearish" if score < -0.3 else "neutral")
    return SignalResult(code, "reddit", round(delta, 1),
                        f"Reddit {mood} (score={score:.2f}, vol={volume})")


# ---------------------------------------------------------------------------
# 4. Betting Market Signal (Odds Calibration)
# ---------------------------------------------------------------------------

# Odds-implied probabilities. Format: {team_code: implied_win_prob}
# Derived from betting odds: 1/decimal_odds, then normalized
BETTING_IMPLIED: Dict[str, float] = {}

ODDS_ELO_WEIGHT = 0.15   # How much to blend market odds into Elo


def betting_market_signal(team: dict) -> SignalResult:
    """Use betting market odds as a calibration signal.

    Converts odds-implied probability to an equivalent Elo difference,
    then blends it with the team's current Elo.
    """
    code = team["code"]
    if code not in BETTING_IMPLIED:
        return SignalResult(code, "betting", 0.0, "no odds data")

    implied_p = BETTING_IMPLIED[code]
    # Very rough: implied_p → Elo. 50% = 0, 60% = ~70, 70% = ~150
    # Inverse of Elo expectation formula
    if 0.01 < implied_p < 0.99:
        implied_elo_diff = -400 * math.log10(1.0 / implied_p - 1.0)
        delta = implied_elo_diff * ODDS_ELO_WEIGHT
        return SignalResult(code, "betting", round(delta, 1),
                            f"market implied {implied_p:.1%} → {delta:+.1f} Elo")
    return SignalResult(code, "betting", 0.0, "odds out of range")


# ---------------------------------------------------------------------------
# 5. Home / Confederation Advantage
# ---------------------------------------------------------------------------

HOST_BONUS = 50           # Elo boost for host nations
CONCACAF_BONUS = 25       # Smaller boost for other CONCACAF teams (familiar conditions)


def host_advantage_signal(team: dict, hosts: list = None) -> SignalResult:
    """Apply home advantage boost for host nations and regional teams."""
    if hosts is None:
        hosts = ["USA", "CAN", "MEX"]
    code = team["code"]

    if code in hosts:
        return SignalResult(code, "host", HOST_BONUS,
                            f"host nation bonus (+{HOST_BONUS})")
    if team.get("confederation") == "CONCACAF":
        return SignalResult(code, "host", CONCACAF_BONUS,
                            f"CONCACAF regional bonus (+{CONCACAF_BONUS})")
    return SignalResult(code, "host", 0.0, "no home advantage")


# ---------------------------------------------------------------------------
# 6. Squad Depth (Market Value)
# ---------------------------------------------------------------------------

DEPTH_ELO_CAP = 20        # Max Elo bonus for deep squad
DEPTH_THRESHOLD_M = 500   # Market value (million €) threshold for max bonus


def squad_depth_signal(team: dict) -> SignalResult:
    """Deeper squads (higher market value) get a small bonus for tournament endurance.

    This primarily affects Monte Carlo simulations of later knockout rounds.
    """
    mv = team.get("market_value_m", 100)
    # Logarithmic scaling: €100M = ~10pts, €500M = ~20pts
    delta = min(DEPTH_ELO_CAP, math.log2(mv / 50 + 1) * 5) if mv > 0 else 0
    return SignalResult(team_code=team["code"], signal_name="depth",
                        elo_delta=round(delta, 1),
                        reason=f"squad value €{mv}M → +{delta:.1f} Elo")


# ---------------------------------------------------------------------------
# Signal Registry & Orchestration
# ---------------------------------------------------------------------------

# All active signals (add/remove to customize the model)
SIGNAL_REGISTRY: List[Callable] = [
    injury_signal,
    recent_form_signal,
    reddit_sentiment_signal,
    betting_market_signal,
    host_advantage_signal,
    squad_depth_signal,
]


def apply_all_signals(teams: dict,
                       active_signals: List[Callable] = None,
                       verbose: bool = False) -> Dict[str, AdjustedTeam]:
    """Apply all active signals to all teams, returning adjusted Elo ratings.

    Args:
        teams: loaded teams.json data
        active_signals: list of signal functions (default: SIGNAL_REGISTRY)
        verbose: print each adjustment

    Returns:
        {team_code: AdjustedTeam}
    """
    if active_signals is None:
        active_signals = SIGNAL_REGISTRY

    adjusted = {}

    for group_teams in teams["groups"].values():
        for team in group_teams:
            code = team["code"]
            at = AdjustedTeam(
                code=code,
                name_en=team["name_en"],
                name_zh=team["name_zh"],
                base_elo=team["elo"],
                adjusted_elo=team["elo"],
            )

            for signal_fn in active_signals:
                result = signal_fn(team)
                at.adjustments.append(result)
                at.adjusted_elo += result.elo_delta
                if verbose and result.elo_delta != 0:
                    print(f"  [{code}] {result.signal_name}: {result.reason}")

            adjusted[code] = at

    return adjusted


def get_adjusted_elo(team_code: str, adjusted: Dict[str, AdjustedTeam]) -> float:
    """Get the final adjusted Elo for a team."""
    return adjusted[team_code].adjusted_elo


def print_adjustments(adjusted: Dict[str, AdjustedTeam], top_n: int = 10):
    """Pretty-print the teams with largest Elo adjustments."""
    all_teams = sorted(adjusted.values(), key=lambda t: abs(t.total_delta), reverse=True)
    print(f"\n{'='*70}")
    print(f"  Top {top_n} Teams by Elo Adjustment Magnitude")
    print(f"{'='*70}")
    for t in all_teams[:top_n]:
        if t.total_delta == 0:
            continue
        direction = "↑" if t.total_delta > 0 else "↓"
        print(f"  {t.name_en:<20} {t.base_elo:.0f} → {t.adjusted_elo:.0f} "
              f"({t.total_delta:+.1f}) {direction}")
        for adj in t.adjustments:
            if adj.elo_delta != 0:
                print(f"    • {adj.signal_name}: {adj.reason}")


# ===================================================================
# Data update helpers (called by external scrapers)
# ===================================================================

def set_injury(team_code: str, player: str, penalty: float):
    """Register an injury/absence. Called by the news scraper."""
    if team_code not in INJURY_DB:
        INJURY_DB[team_code] = {}
    INJURY_DB[team_code][player] = penalty


def set_recent_form(team_code: str, results: list):
    """Set recent form data. results = [(opponent, gf, ga, weight), ...]."""
    RECENT_FORM[team_code] = results


def set_reddit_sentiment(team_code: str, score: float, volume: int, topics: list = None):
    """Set Reddit sentiment data."""
    REDDIT_SENTIMENT[team_code] = {
        "sentiment_score": score,
        "volume": volume,
        "topics": topics or [],
    }


def set_betting_odds(team_code: str, implied_prob: float):
    """Set betting market implied probability."""
    BETTING_IMPLIED[team_code] = implied_prob


def clear_all_signals():
    """Reset all signal data."""
    INJURY_DB.clear()
    RECENT_FORM.clear()
    REDDIT_SENTIMENT.clear()
    BETTING_IMPLIED.clear()


# ===================================================================
# Demo
# ===================================================================

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    with open(data_dir / "teams.json", "r", encoding="utf-8") as f:
        teams = json.load(f)

    # Inject some demo signals
    set_injury("ARG", "Messi", -40)
    set_injury("FRA", "Mbappé", -35)
    set_recent_form("ENG", [("GER", 3, 0, 1.0), ("ITA", 2, 1, 0.85), ("FRA", 1, 1, 0.7)])
    set_reddit_sentiment("BRA", 0.7, 500, ["favorites", "Vini Jr", "attack"])
    set_reddit_sentiment("GER", -0.3, 300, ["rebuilding", "goalkeeper concerns"])

    # Apply all signals
    adjusted = apply_all_signals(teams, verbose=True)
    print_adjustments(adjusted)

    # Show top adjusted Elo
    ranking = sorted(adjusted.values(), key=lambda t: t.adjusted_elo, reverse=True)
    print(f"\n{'='*70}")
    print(f"  Top 16 by Adjusted Elo")
    print(f"{'='*70}")
    for i, t in enumerate(ranking[:16], 1):
        delta_str = f"({t.total_delta:+.0f})" if t.total_delta != 0 else ""
        print(f"  {i:>2}. {t.name_en:<18} {t.adjusted_elo:.0f} {delta_str}")
