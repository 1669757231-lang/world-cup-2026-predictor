"""
World Cup 2026 — Monte Carlo Tournament Simulator
==================================================
Simulates the entire tournament N times, tracking:
  - Each team's probability of reaching each round
  - Champion probability
  - Most likely final matchups
  - Expected goals / entertainment value

Uses the Elo → Poisson pipeline from elo_model.py for individual matches.
"""

import json
import math
import random
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from elo_model import predict_match, MatchPrediction, update_elo

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Result of a single tournament simulation."""
    champion: str
    runner_up: str
    third_place: str
    semifinalists: List[str]
    quarterfinalists: List[str]
    round_of_16: List[str]
    round_of_32: List[str]
    group_winners: Dict[str, str]      # group → team
    group_runners_up: Dict[str, str]   # group → team
    golden_boot: str                   # top scorer (simplified)
    surprise_team: str                 # lowest Elo to reach QF+
    match_results: Dict[int, dict]     # match_id → {home, away, score, round}


@dataclass
class TournamentReport:
    """Aggregated results from N simulations."""
    n_sims: int
    champion_probs: Dict[str, float]          # team → win probability
    final_probs: Dict[str, float]             # team → reach final probability
    semifinal_probs: Dict[str, float]
    quarterfinal_probs: Dict[str, float]
    round_of_16_probs: Dict[str, float]
    round_of_32_probs: Dict[str, float]
    group_advance_probs: Dict[str, float]
    most_likely_final: List[Tuple[str, str, float]]
    expected_champion_elo: float
    dark_horse: str                             # low Elo but high surprise potential


# ---------------------------------------------------------------------------
# Group stage simulation
# ---------------------------------------------------------------------------

def simulate_group(teams_list: List[dict], adjusted_elos: Dict[str, float],
                    rng: random.Random) -> Tuple[List[Tuple[str, float]], str, str]:
    """Simulate one group (3 teams, round-robin)."""
    standings = {t["code"]: {"pts": 0, "gf": 0, "ga": 0, "elo": adjusted_elos.get(t["code"], t["elo"])}
                 for t in teams_list}

    # Each pair plays once (3 matches total)
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        t1, t2 = teams_list[i], teams_list[j]
        elo1 = adjusted_elos.get(t1["code"], t1["elo"])
        elo2 = adjusted_elos.get(t2["code"], t2["elo"])

        pred = predict_match(t1["code"], t2["code"], elo1, elo2)

        # Sample a scoreline from the distribution
        scores = list(pred.score_probs.keys())
        probs = list(pred.score_probs.values())
        score_str = rng.choices(scores, weights=probs, k=1)[0]
        g1, g2 = map(int, score_str.split(":"))

        # Update standings
        standings[t1["code"]]["gf"] += g1
        standings[t1["code"]]["ga"] += g2
        standings[t2["code"]]["gf"] += g2
        standings[t2["code"]]["ga"] += g1

        if g1 > g2:
            standings[t1["code"]]["pts"] += 3
        elif g1 < g2:
            standings[t2["code"]]["pts"] += 3
        else:
            standings[t1["code"]]["pts"] += 1
            standings[t2["code"]]["pts"] += 1

    # Sort by points, then GD, then GF
    ranked = sorted(standings.items(),
                    key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]),
                    reverse=True)
    winner = ranked[0][0]
    runner_up = ranked[1][0]
    return ranked, winner, runner_up


def simulate_all_groups(teams: dict, adjusted_elos: Dict[str, float],
                         rng: random.Random) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Simulate all 16 groups. Returns (group_winners, group_runners_up)."""
    group_winners = {}
    group_runners_up = {}

    for grp_name, grp_teams in teams["groups"].items():
        _, winner, runner_up = simulate_group(grp_teams, adjusted_elos, rng)
        group_winners[grp_name] = winner
        group_runners_up[grp_name] = runner_up

    return group_winners, group_runners_up


# ---------------------------------------------------------------------------
# Knockout stage simulation
# ---------------------------------------------------------------------------

def simulate_knockout_match(team1_code: str, team2_code: str,
                             elo_lookup: Dict[str, float],
                             rng: random.Random,
                             extra_time: bool = True) -> Tuple[str, str]:
    """Simulate a knockout match. Returns (winner, score_string).

    For knockout, draws are resolved by sampling from the non-draw scores
    (or we simulate extra time as a second Poisson draw).
    """
    elo1 = elo_lookup.get(team1_code, 1500)
    elo2 = elo_lookup.get(team2_code, 1500)

    # Slight boost for "knockout experience" — higher Elo teams get a tiny edge
    pred = predict_match(team1_code, team2_code, elo1, elo2)

    # Sample score, re-roll draws for knockout
    scores = list(pred.score_probs.keys())
    probs = list(pred.score_probs.values())

    for _ in range(10):  # max 10 retries
        score_str = rng.choices(scores, weights=probs, k=1)[0]
        g1, g2 = map(int, score_str.split(":"))
        if g1 != g2:
            break
        # On draw: re-weight toward non-draw outcomes
        non_draw = [(s, p) for s, p in zip(scores, probs) if s.split(":")[0] != s.split(":")[1]]
        if non_draw:
            scores_nd, probs_nd = zip(*non_draw)
            score_str = rng.choices(scores_nd, weights=probs_nd, k=1)[0]
            g1, g2 = map(int, score_str.split(":"))
            break
    else:
        # Fallback: higher Elo wins on penalties
        if elo1 >= elo2:
            g1, g2 = 1, 0
        else:
            g1, g2 = 0, 1

    winner = team1_code if g1 > g2 else team2_code
    return winner, f"{g1}:{g2}"


def build_knockout_bracket(group_winners: Dict[str, str],
                            group_runners_up: Dict[str, str]) -> List[dict]:
    """Build the Round of 32 bracket from group results.

    2026 format: 16 groups → 32 teams advance.
    Bracket pairing follows a pre-defined pattern.
    """
    # Standard 2026 bracket pairing
    pairings = [
        # (winner_group, runner_up_group) for each R32 match
        ("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"),
        ("I", "J"), ("K", "L"), ("M", "N"), ("O", "P"),
        ("B", "A"), ("D", "C"), ("F", "E"), ("H", "G"),
        ("J", "I"), ("L", "K"), ("N", "M"), ("P", "O"),
    ]

    bracket = []
    for i, (wg, rg) in enumerate(pairings):
        bracket.append({
            "match_id": 49 + i,
            "team1": group_winners[wg],
            "team2": group_runners_up[rg],
            "round": "Round of 32",
        })
    return bracket


def simulate_knockout_round(matches: List[dict], elo_lookup: Dict[str, float],
                             rng: random.Random) -> List[dict]:
    """Simulate one round of knockout matches. Returns winners for next round."""
    winners = []
    for m in matches:
        winner, score = simulate_knockout_match(
            m["team1"], m["team2"], elo_lookup, rng
        )
        m["winner"] = winner
        m["score"] = score
        winners.append(winner)
    return winners


def simulate_full_tournament(teams: dict, adjusted_elos: Dict[str, float],
                              seed: int = None) -> SimResult:
    """Run one complete tournament simulation."""
    rng = random.Random(seed)

    # Build Elo lookup
    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = adjusted_elos.get(t["code"], t["elo"])

    # Group stage
    group_winners, group_runners_up = simulate_all_groups(teams, adjusted_elos, rng)

    # Round of 32
    r32_bracket = build_knockout_bracket(group_winners, group_runners_up)
    r32_winners = simulate_knockout_round(r32_bracket, elo_lookup, rng)
    r32_teams = [w for m in r32_bracket for w in [m["team1"], m["team2"]]]

    # Round of 16
    r16_matches = [
        {"match_id": 65 + i, "team1": r32_winners[i * 2], "team2": r32_winners[i * 2 + 1],
         "round": "Round of 16"}
        for i in range(8)
    ]
    r16_winners = simulate_knockout_round(r16_matches, elo_lookup, rng)
    r16_teams = r32_winners[:]

    # Quarter-finals
    qf_matches = [
        {"match_id": 73 + i, "team1": r16_winners[i * 2], "team2": r16_winners[i * 2 + 1],
         "round": "Quarter-final"}
        for i in range(4)
    ]
    qf_winners = simulate_knockout_round(qf_matches, elo_lookup, rng)
    qf_teams = r16_winners[:]

    # Semi-finals
    sf_matches = [
        {"match_id": 77, "team1": qf_winners[0], "team2": qf_winners[1], "round": "Semi-final"},
        {"match_id": 78, "team1": qf_winners[2], "team2": qf_winners[3], "round": "Semi-final"},
    ]
    sf_winners = simulate_knockout_round(sf_matches, elo_lookup, rng)
    sf_losers = [m["team1"] if m["winner"] == m["team2"] else m["team2"] for m in sf_matches]
    sf_teams = qf_winners[:]

    # Third place
    tp_winner, tp_score = simulate_knockout_match(sf_losers[0], sf_losers[1], elo_lookup, rng)
    third = tp_winner

    # Final
    final_winner, final_score = simulate_knockout_match(sf_winners[0], sf_winners[1], elo_lookup, rng)
    champion = final_winner
    runner_up = sf_winners[1] if final_winner == sf_winners[0] else sf_winners[0]

    # Surprise team: lowest Elo among QF+ teams
    qf_plus = set(qf_teams)
    surprise = min(qf_plus, key=lambda c: elo_lookup.get(c, 1500))

    return SimResult(
        champion=champion,
        runner_up=runner_up,
        third_place=third,
        semifinalists=sf_teams,
        quarterfinalists=qf_teams,
        round_of_16=r16_teams,
        round_of_32=r32_teams,
        group_winners=group_winners,
        group_runners_up=group_runners_up,
        golden_boot="",  # simplified
        surprise_team=surprise,
        match_results={},
    )


# ---------------------------------------------------------------------------
# Monte Carlo aggregation
# ---------------------------------------------------------------------------

def monte_carlo_simulate(teams: dict, adjusted_elos: Dict[str, float],
                          n_sims: int = 10_000,
                          progress_callback=None) -> TournamentReport:
    """Run N tournament simulations and aggregate results.

    Args:
        teams: loaded teams.json
        adjusted_elos: {team_code: adjusted_elo} from signals.py
        n_sims: number of simulations (10k recommended for stability)
        progress_callback: optional fn(sim_index) for progress reporting
    """
    # Counters
    champions = defaultdict(int)
    finalists = defaultdict(int)
    semifinalists = defaultdict(int)
    quarterfinalists = defaultdict(int)
    r16_count = defaultdict(int)
    r32_count = defaultdict(int)
    group_advance = defaultdict(int)
    final_matchups = defaultdict(int)

    all_teams = []
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            all_teams.append(t["code"])

    for sim_i in range(n_sims):
        result = simulate_full_tournament(teams, adjusted_elos, seed=sim_i)

        champions[result.champion] += 1
        finalists[result.champion] += 1
        finalists[result.runner_up] += 1
        for t in result.semifinalists:
            semifinalists[t] += 1
        for t in result.quarterfinalists:
            quarterfinalists[t] += 1
        for t in result.round_of_16:
            r16_count[t] += 1
        for t in result.round_of_32:
            r32_count[t] += 1
        for t in result.group_winners.values():
            group_advance[t] += 1
        for t in result.group_runners_up.values():
            group_advance[t] += 1

        # Track final matchups
        final_key = f"{result.champion} vs {result.runner_up}"
        final_matchups[final_key] += 1

        if progress_callback and (sim_i + 1) % 1000 == 0:
            progress_callback(sim_i + 1)

    n = n_sims

    # Normalize
    def norm(d): return {k: round(v / n, 4) for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)}

    # Most likely finals
    top_finals = sorted(final_matchups.items(), key=lambda x: x[1], reverse=True)[:10]
    top_finals_norm = [(k, round(v / n, 4)) for k, v in top_finals]

    # Expected champion Elo (Elo-weighted champion)
    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = adjusted_elos.get(t["code"], t["elo"])
    exp_champ_elo = sum(elo_lookup.get(c, 1500) * v / n for c, v in champions.items())

    # Dark horse: team with lowest Elo that still has >5% QF+ chance
    dark_horses = [(t, elo_lookup.get(t, 1500), quarterfinalists.get(t, 0) / n)
                   for t in all_teams if quarterfinalists.get(t, 0) / n > 0.05]
    dark_horse = min(dark_horses, key=lambda x: x[1])[0] if dark_horses else all_teams[0]

    return TournamentReport(
        n_sims=n,
        champion_probs=norm(champions),
        final_probs=norm(finalists),
        semifinal_probs=norm(semifinalists),
        quarterfinal_probs=norm(quarterfinalists),
        round_of_16_probs=norm(r16_count),
        round_of_32_probs=norm(r32_count),
        group_advance_probs=norm(group_advance),
        most_likely_final=top_finals_norm,
        expected_champion_elo=round(exp_champ_elo, 1),
        dark_horse=dark_horse,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: TournamentReport, teams: dict):
    """Pretty-print the Monte Carlo results."""
    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = t

    print(f"\n{'='*70}")
    print(f"  🏆 2026 WORLD CUP — MONTE CARLO PREDICTIONS ({report.n_sims:,} simulations)")
    print(f"{'='*70}")

    # Champion probabilities
    print(f"\n  🥇 CHAMPION PROBABILITIES (Top 16)")
    print(f"  {'─'*50}")
    champ_items = list(report.champion_probs.items())[:16]
    for i, (code, prob) in enumerate(champ_items, 1):
        team = elo_lookup.get(code, {})
        name = team.get("name_en", code)
        bar = "█" * int(prob * 200)
        print(f"  {i:>2}. {name:<18} {prob:>6.1%}  {bar}")

    # Most likely finals
    print(f"\n  🏟️  MOST LIKELY FINALS")
    print(f"  {'─'*50}")
    for matchup, prob in report.most_likely_final[:5]:
        print(f"  {matchup:<35} {prob:.1%}")

    # Dark horse
    print(f"\n  🐴 DARK HORSE: {elo_lookup.get(report.dark_horse, {}).get('name_en', report.dark_horse)}")
    print(f"  📊 Expected Champion Elo: {report.expected_champion_elo:.0f}")


# ===================================================================
# Demo
# ===================================================================

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    with open(data_dir / "teams.json", "r", encoding="utf-8") as f:
        teams = json.load(f)

    # Use base Elo (no adjustments for demo)
    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = t["elo"]

    print("Running 5,000 Monte Carlo simulations...")
    report = monte_carlo_simulate(
        teams, elo_lookup, n_sims=5000,
        progress_callback=lambda i: print(f"  ... {i} simulations done")
    )
    print_report(report, teams)
