"""
World Cup 2026 — Visualization Module
======================================
Generates charts from prediction data.

Chart types:
  1. Champion probability bar chart
  2. Match prediction card (W/D/L pie + score distribution)
  3. Team strength radar chart (multi-dimensional)
  4. Tournament bracket tree with probabilities
  5. Group stage heatmap
  6. Knockout path probability flow

Uses matplotlib + seaborn. All charts saved as PNG to output/charts/.
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import numpy as np

from elo_model import predict_match, MatchPrediction
from monte_carlo import TournamentReport

# ---------------------------------------------------------------------------
# CJK font setup (for Chinese labels)
# ---------------------------------------------------------------------------

def _find_cjk_font() -> str:
    """Find a CJK-capable font available on this system."""
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "FangSong", "KaiTi",
        "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei",
        "Source Han Sans SC", "PingFang SC", "Hiragino Sans GB",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

_CJK_FONT = _find_cjk_font()

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})
if _CJK_FONT:
    plt.rcParams["font.family"] = _CJK_FONT
    plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color palette (World Cup themed)
COLORS = {
    "gold": "#FFD700",
    "silver": "#C0C0C0",
    "bronze": "#CD7F32",
    "primary": "#1a1a2e",
    "secondary": "#16213e",
    "accent": "#0f3460",
    "highlight": "#e94560",
    "grass": "#2d8a4e",
    "top16": ["#e94560", "#0f3460", "#2d8a4e", "#FFD700", "#533483",
              "#E94560", "#16213e", "#CD7F32", "#1a1a2e", "#533483",
              "#e94560", "#0f3460", "#2d8a4e", "#FFD700", "#533483", "#CD7F32"],
}


# ---------------------------------------------------------------------------
# 1. Champion Probability Bar Chart (Horizontal)
# ---------------------------------------------------------------------------

def plot_champion_probs(report: TournamentReport, teams: dict,
                         top_n: int = 16, lang: str = "en",
                         save: bool = True) -> str:
    """Horizontal bar chart of champion probabilities."""
    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = t

    items = list(report.champion_probs.items())[:top_n]
    n = min(top_n, len(items))
    items = items[:n]
    codes = [c for c, _ in items]
    probs = [p * 100 for _, p in items]

    if lang == "zh":
        names = [elo_lookup.get(c, {}).get("name_zh", c) for c in codes]
        title = f"2026 世界杯 — 夺冠概率 Top {n}"
        xlabel = "夺冠概率 (%)"
    else:
        names = [elo_lookup.get(c, {}).get("name_en", c) for c in codes]
        title = f"2026 World Cup — Champion Probability (Top {n})"
        xlabel = "Champion Probability (%)"

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = COLORS["top16"][:n][::-1]
    bars = ax.barh(range(n), probs[::-1], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", pad=15)

    # Add value labels on bars
    for bar, prob in zip(bars, probs[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{prob:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(probs) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = str(OUTPUT_DIR / "champion_probs.png")
    if save:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    return path


# ---------------------------------------------------------------------------
# 2. Single Match Prediction Card
# ---------------------------------------------------------------------------

def plot_match_card(pred: MatchPrediction, teams: dict,
                     lang: str = "en", save: bool = True) -> str:
    """Visual card for a single match prediction."""
    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = t

    home_name = elo_lookup.get(pred.home, {}).get(f"name_{lang}", pred.home)
    away_name = elo_lookup.get(pred.away, {}).get(f"name_{lang}", pred.away)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.2])

    # --- W/D/L pie chart (top left) ---
    ax_pie = fig.add_subplot(gs[0, 0])
    labels = ["Win" if lang == "en" else "胜",
              "Draw" if lang == "en" else "平",
              "Win" if lang == "en" else "负"]
    sizes = [pred.prob_home_win * 100, pred.prob_draw * 100, pred.prob_away_win * 100]
    pie_colors = ["#2d8a4e", "#C0C0C0", "#e94560"]
    explode = (0.05, 0, 0.05)
    wedges, texts, autotexts = ax_pie.pie(
        sizes, explode=explode, labels=labels, colors=pie_colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10}
    )
    ax_pie.set_title(f"{home_name}\nvs\n{away_name}", fontweight="bold", fontsize=13)

    # --- Score distribution (top right) ---
    ax_scores = fig.add_subplot(gs[0, 1])
    top_scores = list(pred.score_probs.items())[:8]
    score_labels = [s for s, _ in top_scores]
    score_probs = [p * 100 for _, p in top_scores]
    bars = ax_scores.bar(range(len(score_labels)), score_probs,
                          color=COLORS["accent"], edgecolor="white")
    ax_scores.set_xticks(range(len(score_labels)))
    ax_scores.set_xticklabels(score_labels, rotation=45, ha="right", fontsize=9)
    ax_scores.set_ylabel("Probability (%)")
    ax_scores.set_title("Most Likely Scores" if lang == "en" else "最可能比分",
                        fontweight="bold")
    for bar, prob in zip(bars, score_probs):
        ax_scores.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                       f"{prob:.1f}%", ha="center", fontsize=8)
    ax_scores.spines["top"].set_visible(False)
    ax_scores.spines["right"].set_visible(False)

    # --- Info table (bottom) ---
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis("off")
    info_text = (
        f"{'═'*50}\n"
        f"  {home_name:<20} {away_name}\n"
        f"  {'─'*40}\n"
        f"  Elo:  {pred.home_elo:.0f}{' ':>18}Elo:  {pred.away_elo:.0f}\n"
        f"  xG:   {pred.home_expected_goals:.2f}{' ':>18}xG:   {pred.away_expected_goals:.2f}\n"
        f"  {'─'*40}\n"
        f"  Most Likely Score: {pred.most_likely_score}"
        f"{'═'*50}"
    )
    ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment="center", horizontalalignment="center",
                 fontfamily="monospace")

    plt.tight_layout()
    safe_name = f"{pred.home}_vs_{pred.away}".replace(" ", "_")
    path = str(OUTPUT_DIR / f"match_{safe_name}.png")
    if save:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    return path


# ---------------------------------------------------------------------------
# 3. Team Strength Radar Chart
# ---------------------------------------------------------------------------

def plot_radar_chart(team1_code: str, team2_code: str, teams: dict,
                      lang: str = "en", save: bool = True) -> str:
    """Multi-dimensional comparison radar chart for two teams."""
    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = t

    t1 = elo_lookup.get(team1_code)
    t2 = elo_lookup.get(team2_code)
    if not t1 or not t2:
        raise ValueError(f"Team not found: {team1_code if not t1 else team2_code}")

    # Dimensions (normalized 0-100)
    def normalize(value, min_v, max_v):
        return (value - min_v) / (max_v - min_v) * 100 if max_v > min_v else 50

    # Compute dimension scores for each team
    def compute_dims(team):
        elo = team["elo"]
        mv = team.get("market_value_m", 100)
        return [
            normalize(elo, 1250, 2050),            # Overall Strength
            normalize(elo * 0.7, 900, 1500),       # Attack (proxy)
            normalize(elo * 0.65, 850, 1400),      # Defense (proxy)
            normalize(mv, 5, 1400),                 # Squad Depth
            normalize(elo * 0.5, 650, 1050),       # Experience (proxy)
            normalize(elo * 0.3 + (50 if team["confederation"] in ["CONCACAF"] else 0),
                      400, 700),                    # Home/Region bonus
        ]

    dims1 = compute_dims(t1)
    dims2 = compute_dims(t2)

    categories = (["Overall", "Attack", "Defense", "Depth", "Experience", "Region"]
                  if lang == "en" else
                  ["综合实力", "进攻", "防守", "阵容深度", "大赛经验", "地区优势"])

    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    values1 = dims1 + dims1[:1]
    values2 = dims2 + dims2[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values1, alpha=0.25, color=COLORS["highlight"])
    ax.plot(angles, values1, linewidth=2, color=COLORS["highlight"],
            label=t1.get(f"name_{lang}", team1_code))
    ax.fill(angles, values2, alpha=0.25, color=COLORS["accent"])
    ax.plot(angles, values2, linewidth=2, color=COLORS["accent"],
            label=t2.get(f"name_{lang}", team2_code))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="grey")
    ax.set_title(f"{t1.get(f'name_{lang}', team1_code)} vs {t2.get(f'name_{lang}', team2_code)}",
                 fontweight="bold", pad=25, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    safe_name = f"{team1_code}_vs_{team2_code}_radar"
    path = str(OUTPUT_DIR / f"{safe_name}.png")
    if save:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    return path


# ---------------------------------------------------------------------------
# 4. Tournament Bracket (Sankey-style probability tree)
# ---------------------------------------------------------------------------

def plot_bracket_tree(report: TournamentReport, teams: dict,
                       lang: str = "en", save: bool = True) -> str:
    """Vertical probability tree showing each team's advancement probability
    through each knockout round."""
    elo_lookup = {}
    for grp_teams in teams["groups"].values():
        for t in grp_teams:
            elo_lookup[t["code"]] = t

    # Top 12 teams by champion probability
    top_teams = list(report.champion_probs.keys())[:12]
    names = [elo_lookup.get(c, {}).get(f"name_{lang}", c) for c in top_teams]

    rounds = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    if lang == "zh":
        rounds = ["32强", "16强", "8强", "半决赛", "决赛", "冠军"]

    round_probs = {
        "R32": report.round_of_32_probs,
        "R16": report.round_of_16_probs,
        "QF": report.quarterfinal_probs,
        "SF": report.semifinal_probs,
        "Final": report.final_probs,
        "Champion": report.champion_probs,
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    x = range(len(rounds))
    for i, (team_code, name) in enumerate(zip(top_teams, names)):
        probs = [round_probs[r].get(team_code, 0) * 100 for r in
                 ["R32", "R16", "QF", "SF", "Final", "Champion"]]
        line, = ax.plot(x, probs, marker="o", linewidth=2, markersize=6,
                        color=COLORS["top16"][i % 16], label=name, alpha=0.85)

        # Annotate champion probability
        if probs[-1] > 2:
            ax.annotate(f"{probs[-1]:.1f}%", (x[-1], probs[-1]),
                        textcoords="offset points", xytext=(8, 0),
                        fontsize=8, fontweight="bold", color=line.get_color())

    ax.set_xticks(x)
    ax.set_xticklabels(rounds, fontsize=10)
    ax.set_ylabel("Advancement Probability (%)" if lang == "en" else "晋级概率 (%)")
    ax.set_title("Tournament Advancement Probability" if lang == "en" else "淘汰赛晋级概率",
                 fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, ncol=1)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    path = str(OUTPUT_DIR / "bracket_tree.png")
    if save:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    return path


# ---------------------------------------------------------------------------
# 5. Group Stage Difficulty Heatmap
# ---------------------------------------------------------------------------

def plot_group_heatmap(teams: dict, lang: str = "en", save: bool = True) -> str:
    """Heatmap showing group strength (average Elo per group)."""
    groups = list(teams["groups"].keys())
    group_data = {}
    all_teams = []

    for grp, grp_teams in teams["groups"].items():
        elos = [t["elo"] for t in grp_teams]
        group_data[grp] = {
            "avg_elo": np.mean(elos),
            "max_elo": max(elos),
            "min_elo": min(elos),
            "range": max(elos) - min(elos),
        }
        for t in grp_teams:
            all_teams.append({"group": grp, "code": t["code"], "elo": t["elo"]})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Group average Elo bar chart ---
    ax1 = axes[0]
    avg_elos = [group_data[g]["avg_elo"] for g in groups]
    colors_bar = ["#e94560" if v == max(avg_elos) else
                  ("#2d8a4e" if v == min(avg_elos) else "#0f3460")
                  for v in avg_elos]
    ax1.bar(groups, avg_elos, color=colors_bar, edgecolor="white")
    ax1.axhline(y=np.mean(avg_elos), color="red", linestyle="--", alpha=0.5,
                label=f"Average: {np.mean(avg_elos):.0f}")
    ax1.set_ylabel("Average Elo")
    ax1.set_title("Group Strength (Avg Elo)" if lang == "en" else "小组实力 (平均Elo)",
                  fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Right: Group competitive balance (Elo range) ---
    ax2 = axes[1]
    ranges = [group_data[g]["range"] for g in groups]
    colors_range = ["#2d8a4e" if v == min(ranges) else
                    ("#e94560" if v == max(ranges) else "#0f3460")
                    for v in ranges]
    ax2.bar(groups, ranges, color=colors_range, edgecolor="white")
    ax2.set_ylabel("Elo Range (Max - Min)")
    ax2.set_title("Group Competitiveness (Elo Spread)" if lang == "en" else "小组竞争平衡 (Elo差距)",
                  fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    path = str(OUTPUT_DIR / "group_heatmap.png")
    if save:
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    return path


# ---------------------------------------------------------------------------
# 6. Generate all charts
# ---------------------------------------------------------------------------

def generate_all_charts(report: TournamentReport, teams: dict,
                         lang: str = "en") -> List[str]:
    """Generate all standard charts. Returns list of file paths."""
    paths = []

    print(f"[viz] Generating champion probability chart...")
    paths.append(plot_champion_probs(report, teams, lang=lang))

    print(f"[viz] Generating bracket tree...")
    paths.append(plot_bracket_tree(report, teams, lang=lang))

    print(f"[viz] Generating group strength heatmap...")
    paths.append(plot_group_heatmap(teams, lang=lang))

    return paths


# ===================================================================
# Demo
# ===================================================================

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    with open(data_dir / "teams.json", "r", encoding="utf-8") as f:
        teams = json.load(f)

    # Demo match card
    arg = teams["groups"]["C"][0]
    nga = teams["groups"]["C"][1]
    pred = predict_match(arg["code"], nga["code"], arg["elo"], nga["elo"])
    path = plot_match_card(pred, teams, lang="zh")
    print(f"Match card saved: {path}")

    # Demo radar
    path = plot_radar_chart("ARG", "FRA", teams, lang="zh")
    print(f"Radar chart saved: {path}")

    # Demo group heatmap
    path = plot_group_heatmap(teams, lang="zh")
    print(f"Group heatmap saved: {path}")

    print("\n✅ All demo charts generated in output/charts/")
