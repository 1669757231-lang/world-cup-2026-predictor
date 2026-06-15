"""
World Cup 2026 — Elo + Poisson Prediction Engine
================================================
Core math layer: Elo ratings → expected goals → Poisson distribution → match probabilities.

All dynamic adjustments happen OUTSIDE this module (see signals.py).
This module takes a final (adjusted) Elo for each team and computes probabilities.
"""

import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Constants (empirically calibrated from historical World Cup data)
# ---------------------------------------------------------------------------
ELO_GOAL_BASELINE = 1.0        # expected goals for equal-Elo teams (per side)
ELO_GOAL_SCALE = 0.004         # each Elo point difference adds this many expected goals
HOME_ADVANTAGE_ELO = 100       # Elo equivalent of home advantage

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MatchPrediction:
    """Full prediction for a single match."""
    home: str
    away: str
    home_elo: float
    away_elo: float
    home_expected_goals: float
    away_expected_goals: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    score_probs: Dict[str, float]   # "2:1" → probability (top 15 scores)
    most_likely_score: str


@dataclass
class GroupStanding:
    """Projected group standing."""
    team: str
    pts: float
    gf: float
    ga: float
    gd: float
    advance_prob: float
    win_group_prob: float


# ---------------------------------------------------------------------------
# Poisson helpers
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF: P(X = k) for lambda = lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    """Poisson CDF: P(X <= k)."""
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


# ---------------------------------------------------------------------------
# Goal expectation
# ---------------------------------------------------------------------------

def expected_goals(elo_attack: float, elo_defense: float,
                   home: bool = False, neutral: bool = True) -> float:
    """Compute expected goals for one side.

    Args:
        elo_attack: attacking team's adjusted Elo
        elo_defense: defending team's adjusted Elo
        home: is this team playing at home?
        neutral: is the venue neutral? (World Cup: most matches are neutral)
    """
    elo_diff = elo_attack - elo_defense
    if home and not neutral:
        elo_diff += HOME_ADVANTAGE_ELO
    goals = ELO_GOAL_BASELINE + ELO_GOAL_SCALE * elo_diff
    return max(0.15, goals)  # floor at 0.15 goals


# ---------------------------------------------------------------------------
# Match prediction
# ---------------------------------------------------------------------------

def predict_match(home_team: str, away_team: str,
                  home_elo: float, away_elo: float,
                  neutral: bool = True,
                  home_advantage: bool = False,
                  max_goals: int = 10) -> MatchPrediction:
    """Predict a single match outcome.

    Returns full probability breakdown: W/D/L and score distribution.
    """
    # Compute expected goals each side
    xg_home = expected_goals(home_elo, away_elo, home=home_advantage, neutral=neutral)
    xg_away = expected_goals(away_elo, home_elo, home=False, neutral=neutral)

    # Score probability matrix: P(home=i, away=j) = P(i|λ_h) × P(j|λ_a)
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0
    score_probs = {}

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson_pmf(i, xg_home) * poisson_pmf(j, xg_away)
            if i > j:
                home_win_prob += p
            elif i == j:
                draw_prob += p
            else:
                away_win_prob += p
            score_probs[f"{i}:{j}"] = p

    # Top scores by probability
    top_scores = dict(
        sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:15]
    )
    most_likely = max(top_scores, key=top_scores.get)

    return MatchPrediction(
        home=home_team, away=away_team,
        home_elo=home_elo, away_elo=away_elo,
        home_expected_goals=round(xg_home, 3),
        away_expected_goals=round(xg_away, 3),
        prob_home_win=round(home_win_prob, 4),
        prob_draw=round(draw_prob, 4),
        prob_away_win=round(away_win_prob, 4),
        score_probs={k: round(v, 4) for k, v in top_scores.items()},
        most_likely_score=most_likely,
    )


# ---------------------------------------------------------------------------
# Elo update after actual result
# ---------------------------------------------------------------------------

def update_elo(home_elo: float, away_elo: float,
               home_goals: int, away_goals: int,
               k: float = 32.0) -> Tuple[float, float]:
    """Update Elo ratings based on an actual match result.

    Returns (new_home_elo, new_away_elo).
    """
    # Expected result (0-1 scale)
    dr = home_elo - away_elo + (HOME_ADVANTAGE_ELO if False else 0)
    expected = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

    # Actual result
    if home_goals > away_goals:
        actual = 1.0
    elif home_goals == away_goals:
        actual = 0.5
    else:
        actual = 0.0

    # Goal difference multiplier (blowouts matter more, up to 2x)
    gd = abs(home_goals - away_goals)
    goal_mult = min(2.0, 1.0 + (gd - 1) * 0.25) if gd > 1 else 1.0

    delta = k * goal_mult * (actual - expected)
    return home_elo + delta, away_elo - delta


# ---------------------------------------------------------------------------
# Match prediction from team dicts (convenience)
# ---------------------------------------------------------------------------

def predict_match_from_teams(home_team: dict, away_team: dict,
                              neutral: bool = True) -> MatchPrediction:
    """Convenience wrapper: predict from team dicts (with 'elo' key)."""
    return predict_match(
        home_team=home_team["code"],
        away_team=away_team["code"],
        home_elo=home_team["elo"],
        away_elo=away_team["elo"],
        neutral=neutral,
    )


# ---------------------------------------------------------------------------
# Bulk: compute all group-stage predictions
# ---------------------------------------------------------------------------

def predict_group_stage(teams: dict, fixtures: list) -> list:
    """Predict all group-stage matches, returning (fixture, prediction) pairs."""
    results = []
    team_lookup = {}
    for group_teams in teams["groups"].values():
        for t in group_teams:
            team_lookup[t["code"]] = t

    for match in fixtures:
        home = team_lookup[match["home"]]
        away = team_lookup[match["away"]]
        pred = predict_match_from_teams(home, away)
        results.append({"fixture": match, "prediction": pred})
    return results


# ---------------------------------------------------------------------------
# Expected points from probabilities
# ---------------------------------------------------------------------------

def expected_points(prob_home: float, prob_draw: float, prob_away: float,
                    perspective: str = "home") -> float:
    """Convert W/D/L probabilities to expected points (3-1-0 system)."""
    if perspective == "home":
        return 3 * prob_home + 1 * prob_draw
    else:
        return 3 * prob_away + 1 * prob_draw


# ===================================================================
# Demo / CLI
# ===================================================================

if __name__ == "__main__":
    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    with open(data_dir / "teams.json", "r", encoding="utf-8") as f:
        teams = json.load(f)

    # Demo: predict Argentina vs Nigeria
    arg = teams["groups"]["C"][0]  # ARG
    nga = teams["groups"]["C"][1]  # NGA

    pred = predict_match_from_teams(arg, nga)
    print(f"\n{'='*60}")
    print(f"  {arg['name_en']} ({arg['elo']}) vs {nga['name_en']} ({nga['elo']})")
    print(f"{'='*60}")
    print(f"  xG: {arg['code']} {pred.home_expected_goals} — {pred.away_expected_goals} {nga['code']}")
    print(f"  Win: {pred.prob_home_win:.1%}  |  Draw: {pred.prob_draw:.1%}  |  Away: {pred.prob_away_win:.1%}")
    print(f"  Most likely score: {pred.most_likely_score}")
    print(f"  Top 5 scorelines:")
    for score, prob in list(pred.score_probs.items())[:5]:
        bar = "█" * int(prob * 200)
        print(f"    {score:>5}  {prob:.1%}  {bar}")

    # Demo: show all favourites
    print(f"\n{'='*60}")
    print(f"  Group Favourites (by base Elo)")
    print(f"{'='*60}")
    for grp, tms in teams["groups"].items():
        best = max(tms, key=lambda t: t["elo"])
        print(f"  Group {grp}: {best['name_zh']} ({best['name_en']}) — Elo {best['elo']}")
