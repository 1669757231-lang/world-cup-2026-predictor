"""
World Cup 2026 — Monte Carlo Tournament Simulator (v2)
=======================================================
Simulates the 48-team, 12-group × 4-team format.
Top 2 + 8 best 3rd place → Round of 32 (32 teams).
104 matches total.

Uses the Elo → Poisson pipeline from elo_model.py for individual matches.
"""

import json
import math
import random
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from elo_model import predict_match, MatchPrediction

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
    group_winners: Dict[str, str]
    group_runners_up: Dict[str, str]
    best_thirds: List[str]
    surprise_team: str


@dataclass
class TournamentReport:
    """Aggregated results from N simulations."""
    n_sims: int
    champion_probs: Dict[str, float]
    final_probs: Dict[str, float]
    semifinal_probs: Dict[str, float]
    quarterfinal_probs: Dict[str, float]
    round_of_16_probs: Dict[str, float]
    round_of_32_probs: Dict[str, float]
    group_advance_probs: Dict[str, float]
    most_likely_final: List[Tuple[str, str, float]]
    expected_champion_elo: float
    dark_horse: str


# ---------------------------------------------------------------------------
# Group stage simulation (4 teams, round-robin)
# ---------------------------------------------------------------------------

def simulate_group_4(teams_list: List[dict], adjusted_elos: Dict[str, float],
                      rng: random.Random) -> List[Tuple[str, dict]]:
    """Simulate one group (4 teams, round-robin = 6 matches)."""
    standings = {}
    for t in teams_list:
        standings[t["code"]] = {
            "pts": 0, "gf": 0, "ga": 0,
            "elo": adjusted_elos.get(t["code"], t["elo"]),
            "code": t["code"],
        }

    # All 6 pairings in a 4-team group
    pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]

    for i, j in pairs:
        t1, t2 = teams_list[i], teams_list[j]
        code1, code2 = t1["code"], t2["code"]
        elo1 = adjusted_elos.get(code1, t1["elo"])
        elo2 = adjusted_elos.get(code2, t2["elo"])

        pred = predict_match(code1, code2, elo1, elo2)

        scores = list(pred.score_probs.keys())
        probs = list(pred.score_probs.values())
        score_str = rng.choices(scores, weights=probs, k=1)[0]
        g1, g2 = map(int, score_str.split(":"))

        standings[code1]["gf"] += g1
        standings[code1]["ga"] += g2
        standings[code2]["gf"] += g2
        standings[code2]["ga"] += g1

        if g1 > g2:
            standings[code1]["pts"] += 3
        elif g1 < g2:
            standings[code2]["pts"] += 3
        else:
            standings[code1]["pts"] += 1
            standings[code2]["pts"] += 1

    # Sort: points → GD → GF → Elo (tiebreaker)
    ranked = sorted(standings.items(),
                    key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"],
                                   x[1]["gf"], x[1]["elo"]),
                    reverse=True)
    return [(code, data) for code, data in ranked]


def simulate_all_groups(teams: dict, adjusted_elos: Dict[str, float],
                         rng: random.Random) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, dict]]:
    """Simulate all 12 groups. Returns (winners, runners_up, third_place_standings)."""
    group_winners = {}
    group_runners_up = {}
    third_places = {}

    for grp_name, grp_teams in teams["groups"].items():
        ranked = simulate_group_4(grp_teams, adjusted_elos, rng)
        group_winners[grp_name] = ranked[0][0]
        group_runners_up[grp_name] = ranked[1][0]
        third_places[grp_name] = ranked[2][1]  # store stats for best 3rd selection

    return group_winners, group_runners_up, third_places


def select_best_thirds(third_places: Dict[str, dict]) -> List[str]:
    """Select the 8 best 3rd-place teams across all 12 groups."""
    ranked = sorted(third_places.items(),
                    key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"],
                                   x[1]["gf"], x[1]["elo"]),
                    reverse=True)
    return [code for code, _ in ranked[:8]]


# ---------------------------------------------------------------------------
# Knockout
# ---------------------------------------------------------------------------

def simulate_knockout_match(team1_code: str, team2_code: str,
                             elo_lookup: Dict[str, float],
                             rng: random.Random) -> Tuple[str, str]:
    """Simulate a knockout match. Returns (winner, score_string)."""
    elo1 = elo_lookup.get(team1_code, 1500)
    elo2 = elo_lookup.get(team2_code, 1500)

    pred = predict_match(team1_code, team2_code, elo1, elo2)
    scores = list(pred.score_probs.keys())
    probs = list(pred.score_probs.values())

    for _ in range(10):
        score_str = rng.choices(scores, weights=probs, k=1)[0]
        g1, g2 = map(int, score_str.split(":"))
        if g1 != g2:
            break
        non_draw = [(s, p) for s, p in zip(scores, probs) if s.split(":")[0] != s.split(":")[1]]
        if non_draw:
            scores_nd, probs_nd = zip(*non_draw)
            score_str = rng.choices(scores_nd, weights=probs_nd, k=1)[0]
            g1, g2 = map(int, score_str.split(":"))
            break
    else:
        if elo1 >= elo2:
            g1, g2 = 1, 0
        else:
            g1, g2 = 0, 1

    winner = team1_code if g1 > g2 else team2_code
    return winner, f"{g1}:{g2}"


def build_r32_bracket(group_winners: Dict[str, str],
                       group_runners_up: Dict[str, str],
                       best_thirds: List[str]) -> List[dict]:
    """Build the Round of 32 bracket.

    2026 format: 12 winners + 12 runners-up + 8 best 3rd = 32 teams.
    FIFA pre-defined bracket with best 3rd place paths.
    Simplified version: seed by group and points.
    """
    # Simplified bracket: pair based on group order
    # Group winners A-H play best 3rd place teams
    # Group winners I-L play specific runners-up
    # Remaining runners-up play each other

    grp_order = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

    # Winners of A-H play best 3rd place teams
    w_ah = [group_winners[g] for g in grp_order[:8]]
    # Winners of I-L play runners-up of I, K, J, L (mixed)
    w_il = [group_winners[g] for g in grp_order[8:]]
    ru_il = [group_runners_up["I"], group_runners_up["K"],
             group_runners_up["J"], group_runners_up["L"]]
    # Remaining runners-up: A vs B, C vs D, E vs F, G vs H (mixed)
    ru_pairs = [
        (group_runners_up["A"], group_runners_up["B"]),
        (group_runners_up["C"], group_runners_up["D"]),
        (group_runners_up["E"], group_runners_up["F"]),
        (group_runners_up["G"], group_runners_up["H"]),
    ]

    # 16 matches total
    matches = []

    # 8 matches: group winners A-H vs best 3rd place
    for i, w in enumerate(w_ah):
        matches.append({"match_id": 73 + i, "team1": w, "team2": best_thirds[i],
                        "round": "Round of 32"})

    # 4 matches: group winners I-L vs runners-up I,J,K,L
    for i, (w, ru) in enumerate(zip(w_il, ru_il)):
        matches.append({"match_id": 81 + i, "team1": w, "team2": ru,
                        "round": "Round of 32"})

    # 4 matches: runners-up pairs
    for i, (ru1, ru2) in enumerate(ru_pairs):
        matches.append({"match_id": 85 + i, "team1": ru1, "team2": ru2,
                        "round": "Round of 32"})

    return matches[:16]


def simulate_knockout_round(matches: List[dict], elo_lookup: Dict[str, float],
                             rng: random.Random) -> List[str]:
    """Simulate one knockout round. Returns list of winners."""
    winners = []
    for m in matches:
        winner, score = simulate_knockout_match(m["team1"], m["team2"], elo_lookup, rng)
        m["winner"] = winner
        m["score"] = score
        winners.append(winner)
    return winners


def simulate_full_tournament(teams: dict, adjusted_elos: Dict[str, float],
                              seed: int = None) -> SimResult:
    """Run one complete tournament simulation."""
    rng = random.Random(seed)

    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = adjusted_elos.get(t["code"], t["elo"])

    # Group stage
    group_winners, group_runners_up, third_places = simulate_all_groups(
        teams, adjusted_elos, rng)
    best_thirds = select_best_thirds(third_places)

    # Collect advancing teams
    advancing = set()
    for w in group_winners.values():
        advancing.add(w)
    for ru in group_runners_up.values():
        advancing.add(ru)
    for bt in best_thirds:
        advancing.add(bt)

    # Round of 32
    r32_bracket = build_r32_bracket(group_winners, group_runners_up, best_thirds)
    r32_winners = simulate_knockout_round(r32_bracket, elo_lookup, rng)
    r32_teams = list(advancing)

    # Round of 16
    r16_matches = [
        {"match_id": 89 + i, "team1": r32_winners[i * 2],
         "team2": r32_winners[i * 2 + 1], "round": "Round of 16"}
        for i in range(8)
    ]
    r16_winners = simulate_knockout_round(r16_matches, elo_lookup, rng)
    r16_teams = r32_winners[:]

    # Quarter-finals
    qf_matches = [
        {"match_id": 97 + i, "team1": r16_winners[i * 2],
         "team2": r16_winners[i * 2 + 1], "round": "Quarter-final"}
        for i in range(4)
    ]
    qf_winners = simulate_knockout_round(qf_matches, elo_lookup, rng)
    qf_teams = r16_winners[:]

    # Semi-finals
    sf_matches = [
        {"match_id": 101, "team1": qf_winners[0], "team2": qf_winners[1],
         "round": "Semi-final"},
        {"match_id": 102, "team1": qf_winners[2], "team2": qf_winners[3],
         "round": "Semi-final"},
    ]
    sf_winners = simulate_knockout_round(sf_matches, elo_lookup, rng)
    sf_losers = [m["team1"] if m["winner"] == m["team2"] else m["team2"]
                 for m in sf_matches]
    sf_teams = qf_winners[:]

    # Third place
    tp_winner, tp_score = simulate_knockout_match(
        sf_losers[0], sf_losers[1], elo_lookup, rng)
    third = tp_winner

    # Final
    final_winner, final_score = simulate_knockout_match(
        sf_winners[0], sf_winners[1], elo_lookup, rng)
    champion = final_winner
    runner_up = sf_winners[1] if final_winner == sf_winners[0] else sf_winners[0]

    # Surprise: lowest Elo among QF+
    qf_plus = set(qf_teams)
    surprise = min(qf_plus, key=lambda c: elo_lookup.get(c, 1500))

    return SimResult(
        champion=champion, runner_up=runner_up, third_place=third,
        semifinalists=sf_teams, quarterfinalists=qf_teams,
        round_of_16=r16_teams, round_of_32=r32_teams,
        group_winners=group_winners, group_runners_up=group_runners_up,
        best_thirds=best_thirds, surprise_team=surprise,
    )


# ---------------------------------------------------------------------------
# Monte Carlo aggregation
# ---------------------------------------------------------------------------

def monte_carlo_simulate(teams: dict, adjusted_elos: Dict[str, float],
                          n_sims: int = 10_000,
                          progress_callback=None) -> TournamentReport:
    """Run N tournament simulations and aggregate results."""
    champions = defaultdict(int)
    finalists = defaultdict(int)
    semifinalists = defaultdict(int)
    quarterfinalists = defaultdict(int)
    r16_count = defaultdict(int)
    r32_count = defaultdict(int)
    group_advance = defaultdict(int)
    final_matchups = defaultdict(int)

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
        for t in result.best_thirds:
            group_advance[t] += 1

        final_key = f"{result.champion} vs {result.runner_up}"
        final_matchups[final_key] += 1

        if progress_callback and (sim_i + 1) % 1000 == 0:
            progress_callback(sim_i + 1)

    n = n_sims

    def norm(d):
        return {k: round(v / n, 4) for k, v in
                sorted(d.items(), key=lambda x: x[1], reverse=True)}

    top_finals = sorted(final_matchups.items(), key=lambda x: x[1], reverse=True)[:10]
    top_finals_norm = [(k, round(v / n, 4)) for k, v in top_finals]

    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = adjusted_elos.get(t["code"], t["elo"])
    exp_champ_elo = sum(elo_lookup.get(c, 1500) * v / n
                        for c, v in champions.items())

    all_teams = [t["code"] for grp_teams in teams["groups"].values()
                 for t in grp_teams]
    dark_horses = [(t, elo_lookup.get(t, 1500), quarterfinalists.get(t, 0) / n)
                   for t in all_teams if quarterfinalists.get(t, 0) / n > 0.05]
    dark_horse = min(dark_horses, key=lambda x: x[1])[0] if dark_horses else all_teams[0]

    return TournamentReport(
        n_sims=n, champion_probs=norm(champions), final_probs=norm(finalists),
        semifinal_probs=norm(semifinalists),
        quarterfinal_probs=norm(quarterfinalists),
        round_of_16_probs=norm(r16_count), round_of_32_probs=norm(r32_count),
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
    print(f"  🏆 2026 WORLD CUP — MONTE CARLO ({report.n_sims:,} sims)")
    print(f"{'='*70}")

    print(f"\n  🥇 CHAMPION (Top 16)")
    print(f"  {'─'*50}")
    for i, (code, prob) in enumerate(list(report.champion_probs.items())[:16], 1):
        team = elo_lookup.get(code, {})
        name = team.get("name_en", code)
        bar = "█" * int(prob * 200)
        print(f"  {i:>2}. {name:<18} {prob:>6.1%}  {bar}")

    print(f"\n  🏟️  MOST LIKELY FINALS")
    print(f"  {'─'*50}")
    for matchup, prob in report.most_likely_final[:5]:
        print(f"  {matchup:<35} {prob:.1%}")

    dh = elo_lookup.get(report.dark_horse, {})
    print(f"\n  🐴 Dark Horse: {dh.get('name_en', report.dark_horse)}")
    print(f"  📊 Expected Champion Elo: {report.expected_champion_elo:.0f}")


# ===================================================================
# Demo
# ===================================================================

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    with open(data_dir / "teams.json", "r", encoding="utf-8") as f:
        teams = json.load(f)

    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = t["elo"]

    print("Running 3,000 Monte Carlo simulations (12 groups × 4 teams)...")
    report = monte_carlo_simulate(
        teams, elo_lookup, n_sims=3000,
        progress_callback=lambda i: print(f"  ... {i} simulations done")
    )
    print_report(report, teams)
