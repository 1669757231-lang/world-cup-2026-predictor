#!/usr/bin/env python3
"""
World Cup 2026 — Main Prediction CLI
=====================================
Unified entry point for all prediction tasks.

Usage:
  # Full tournament prediction (Monte Carlo)
  python predict.py full --sims 10000

  # Single match prediction
  python predict.py match ARG FRA

  # Single match (Chinese names)
  python predict.py match 阿根廷 法国

  # Group stage prediction
  python predict.py group A

  # All groups
  python predict.py group --all

  # Champions only (quick)
  python predict.py champions --top 10

  # Generate all charts
  python predict.py charts

  # With signals (apply dynamic adjustments)
  python predict.py full --signals --sims 5000

  # Bilingual output
  python predict.py full --lang zh
  python predict.py full --lang en
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from elo_model import predict_match, predict_group_stage, MatchPrediction
from signals import apply_all_signals, print_adjustments, get_adjusted_elo
from monte_carlo import monte_carlo_simulate, print_report
from visualize import (
    plot_champion_probs, plot_match_card, plot_radar_chart,
    plot_bracket_tree, plot_group_heatmap, generate_all_charts,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def load_teams():
    with open(DATA_DIR / "teams.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_fixtures():
    with open(DATA_DIR / "fixtures.json", "r", encoding="utf-8") as f:
        return json.load(f)


def find_team(query: str, teams: dict) -> dict:
    """Find a team by code (ARG), English name (Argentina), or Chinese name (阿根廷)."""
    query_lower = query.strip().lower()
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            if (query_lower == t["code"].lower() or
                query_lower == t["name_en"].lower() or
                query_lower in t["name_zh"]):
                return t
    return None


def build_elo_lookup(teams: dict, use_signals: bool = False) -> dict:
    """Build {team_code: elo} lookup, optionally with signals applied."""
    if use_signals:
        adjusted = apply_all_signals(teams)
        return {code: at.adjusted_elo for code, at in adjusted.items()}
    else:
        lookup = {}
        for grp_teams in teams["groups"].values():
            for t in grp_teams:
                lookup[t["code"]] = t["elo"]
        return lookup


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_match(args, teams: dict):
    """Predict a single match."""
    home = find_team(args.team1, teams)
    away = find_team(args.team2, teams)

    if not home:
        print(f"❌ Team not found: '{args.team1}'. Use code (ARG), English, or Chinese name.")
        return
    if not away:
        print(f"❌ Team not found: '{args.team2}'. Use code (ARG), English, or Chinese name.")
        return

    # Apply signals if requested
    if args.signals:
        adjusted = apply_all_signals(teams)
        home_elo = adjusted[home["code"]].adjusted_elo
        away_elo = adjusted[away["code"]].adjusted_elo
    else:
        home_elo = home["elo"]
        away_elo = away["elo"]

    pred = predict_match(home["code"], away["code"], home_elo, away_elo)
    lang = args.lang or "en"
    name_fn = "name_zh" if lang == "zh" else "name_en"

    W = "胜" if lang == "zh" else "Win"
    D = "平" if lang == "zh" else "Draw"
    L = "负" if lang == "zh" else "Lose"

    print(f"\n{'='*60}")
    print(f"  {home[name_fn]} vs {away[name_fn]}")
    print(f"{'='*60}")
    print(f"  Elo:     {home_elo:.0f}  vs  {away_elo:.0f}")
    print(f"  xG:      {pred.home_expected_goals:.2f}  vs  {pred.away_expected_goals:.2f}")
    print(f"  {W}:       {pred.prob_home_win:.1%}")
    print(f"  {D}:      {pred.prob_draw:.1%}")
    print(f"  {L}:      {pred.prob_away_win:.1%}")
    print(f"  Most likely: {pred.most_likely_score}")
    print(f"\n  Score probabilities:")
    for score, prob in list(pred.score_probs.items())[:8]:
        bar = "█" * int(prob * 100)
        print(f"    {score:>5}  {prob:>5.1%}  {bar}")


def cmd_group(args, teams: dict):
    """Predict group stage results."""
    fixtures = load_fixtures()
    elo_lookup = build_elo_lookup(teams, args.signals)

    lang = args.lang or "en"

    groups_to_show = [args.group.upper()] if args.group != "all" else list(teams["groups"].keys())

    for grp_name in groups_to_show:
        if grp_name not in teams["groups"]:
            print(f"❌ Group {grp_name} not found.")
            continue

        grp_teams = teams["groups"][grp_name]
        grp_matches = [m for m in fixtures["group_stage"] if m["group"] == grp_name]

        print(f"\n{'='*60}")
        print(f"  Group {grp_name}")
        print(f"{'='*60}")

        # Team list
        for t in sorted(grp_teams, key=lambda t: elo_lookup.get(t["code"], t["elo"]), reverse=True):
            name = t[f"name_{lang}"]
            elo = elo_lookup.get(t["code"], t["elo"])
            print(f"  {name:<20} Elo: {elo:.0f}")

        print(f"\n  Predicted matches:")
        for m in grp_matches:
            home = next(t for t in grp_teams if t["code"] == m["home"])
            away = next(t for t in grp_teams if t["code"] == m["away"])
            h_elo = elo_lookup.get(home["code"], home["elo"])
            a_elo = elo_lookup.get(away["code"], away["elo"])
            pred = predict_match(home["code"], away["code"], h_elo, a_elo)

            h_name = home[f"name_{lang}"]
            a_name = away[f"name_{lang}"]
            print(f"  {h_name:<18} vs {a_name:<18}  →  {pred.prob_home_win:.0%} / {pred.prob_draw:.0%} / {pred.prob_away_win:.0%}  ({pred.most_likely_score})")


def cmd_full(args, teams: dict):
    """Run full Monte Carlo simulation."""
    elo_lookup = build_elo_lookup(teams, args.signals)
    lang = args.lang or "en"

    if args.signals:
        adjusted = apply_all_signals(teams, verbose=args.verbose)
        print_adjustments(adjusted)

    n_sims = args.sims or 10000
    print(f"\n🔄 Running {n_sims:,} Monte Carlo simulations...")
    print(f"   (This may take a minute for 10k+ sims)\n")

    report = monte_carlo_simulate(
        teams, elo_lookup, n_sims=n_sims,
        progress_callback=lambda i: print(f"   ... {i:,} / {n_sims:,} done") if i % 2000 == 0 else None
    )
    print_report(report, teams)

    # Generate charts
    if args.charts:
        print(f"\n📊 Generating charts...")
        paths = generate_all_charts(report, teams, lang=lang)
        for p in paths:
            print(f"   ✅ {p}")


def cmd_champions(args, teams: dict):
    """Quick: show top champion contenders."""
    elo_lookup = build_elo_lookup(teams, args.signals)
    lang = args.lang or "en"

    n_sims = args.sims or 5000
    print(f"\n🔄 Quick simulation: {n_sims:,} runs...")

    report = monte_carlo_simulate(teams, elo_lookup, n_sims=n_sims)

    top_n = args.top or 10
    items = list(report.champion_probs.items())[:top_n]
    elo_lookup_names = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup_names[t["code"]] = t

    print(f"\n{'='*50}")
    print(f"  🏆 TOP {top_n} CHAMPION CONTENDERS")
    print(f"{'='*50}")
    for i, (code, prob) in enumerate(items, 1):
        team = elo_lookup_names.get(code, {})
        name = team.get(f"name_{lang}", code)
        bar = "█" * int(prob * 100)
        print(f"  {i:>2}. {name:<18} {prob:>6.1%}  {bar}")

    # Dark horse
    print(f"\n  🐴 Dark horse: {elo_lookup_names.get(report.dark_horse, {}).get(f'name_{lang}', report.dark_horse)}")


def cmd_charts(args, teams: dict):
    """Generate all charts from a full simulation."""
    elo_lookup = build_elo_lookup(teams, args.signals)
    lang = args.lang or "en"

    n_sims = args.sims or 10000
    print(f"\n🔄 Running {n_sims:,} simulations for charts...")
    report = monte_carlo_simulate(teams, elo_lookup, n_sims=n_sims,
                                   progress_callback=lambda i: print(f"   ... {i:,} done") if i % 2000 == 0 else None)

    paths = generate_all_charts(report, teams, lang=lang)
    print(f"\n✅ Charts saved:")
    for p in paths:
        print(f"   📊 {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="🏆 2026 World Cup Football Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Prediction mode")

    def add_common_args(p):
        p.add_argument("--lang", choices=["en", "zh"], default="zh",
                       help="Output language (default: zh)")

    # match
    p_match = subparsers.add_parser("match", help="Predict a single match")
    add_common_args(p_match)
    p_match.add_argument("team1", help="Home team (code, English, or Chinese name)")
    p_match.add_argument("team2", help="Away team (code, English, or Chinese name)")
    p_match.add_argument("--signals", action="store_true", help="Apply dynamic signals")
    p_match.set_defaults(func=cmd_match)

    # group
    p_group = subparsers.add_parser("group", help="Predict group stage")
    add_common_args(p_group)
    p_group.add_argument("group", help="Group letter (A-P) or 'all'")
    p_group.add_argument("--signals", action="store_true", help="Apply dynamic signals")
    p_group.set_defaults(func=cmd_group)

    # full
    p_full = subparsers.add_parser("full", help="Full Monte Carlo tournament simulation")
    add_common_args(p_full)
    p_full.add_argument("--sims", type=int, default=10000, help="Number of simulations")
    p_full.add_argument("--signals", action="store_true", help="Apply dynamic signals")
    p_full.add_argument("--charts", action="store_true", help="Generate charts")
    p_full.add_argument("--verbose", action="store_true", help="Show signal details")
    p_full.set_defaults(func=cmd_full)

    # champions
    p_champ = subparsers.add_parser("champions", help="Quick champion probability")
    add_common_args(p_champ)
    p_champ.add_argument("--top", type=int, default=10, help="Show top N")
    p_champ.add_argument("--sims", type=int, default=5000, help="Number of simulations")
    p_champ.add_argument("--signals", action="store_true", help="Apply dynamic signals")
    p_champ.set_defaults(func=cmd_champions)

    # charts
    p_charts = subparsers.add_parser("charts", help="Generate all visualizations")
    add_common_args(p_charts)
    p_charts.add_argument("--sims", type=int, default=10000, help="Simulations for chart data")
    p_charts.add_argument("--signals", action="store_true", help="Apply dynamic signals")
    p_charts.set_defaults(func=cmd_charts)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    teams = load_teams()
    args.func(args, teams)


if __name__ == "__main__":
    main()
