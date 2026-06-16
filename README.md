# 🏆 World Cup 2026 Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

> **Real Elo (eloratings.net) + Poisson + Monte Carlo + Twitter/X Football Analysis**
> 一个完全基于真实数据的开源世界杯预测引擎。

🔗 **GitHub**: https://github.com/1669757231-lang/world-cup-2026-predictor

---

## ✨ Features

- 🔢 **Real Elo** — 48 队真实 Elo 评分，来自 [eloratings.net](https://eloratings.net)
- 📊 **Poisson Model** — 泊松分布比分概率，参数基于 eloratings.net 量级校准
- 🎲 **Monte Carlo** — 10000+ 次完整锦标赛模拟（104 场比赛）
- 🐦 **Twitter Analysis** — TikHub API 实时抓取足球记者实力分析和伤病信息
- 📈 **可视化** — matplotlib 图表：比赛卡、冠军概率、晋级树
- 🌐 **Bilingual** — 中文 / English 输出
- 🔌 **Pluggable Signals** — 6 种可插拔信号（Twitter分析/伤病/状态/赔率/主场/深度）

---

## 🚀 Quick Start

```bash
git clone https://github.com/1669757231-lang/world-cup-2026-predictor.git
cd world-cup-2026-predictor
pip install -r requirements.txt
```

### 第一个预测

```bash
# 比利时 vs 埃及（G 组）
python scripts/predict.py match 比利时 埃及 --signals --lang zh

# 冠军概率 Top 16
python scripts/predict.py champions --top 16 --signals --lang zh

# 完整模拟 + 图表
python scripts/predict.py full --sims 5000 --signals --charts
```

---

## 📖 Usage

### 比赛预测

```bash
python scripts/predict.py match <队1> <队2> [--signals] [--lang zh|en]
```

示例：

```bash
python scripts/predict.py match 阿根廷 法国 --signals      # 中文
python scripts/predict.py match Brazil Germany --signals --lang en  # 英文
python scripts/predict.py match ESP CPV --signals           # FIFA 代码
```

输出：

```
============================================================
  比利时 vs 埃及
============================================================
  Elo:     1858  vs  1792
  xG:      1.37  vs  1.03
  胜:       44.5%
  平:       27.2%
  负:       28.3%
  Most likely: 1:1
```

### 冠军预测

```bash
python scripts/predict.py champions --top 16 --signals --lang zh
```

### 小组赛分析

```bash
python scripts/predict.py group G --signals --lang zh
python scripts/predict.py group all --signals
```

### 完整模拟 + 图表

```bash
python scripts/predict.py full --sims 10000 --signals --charts
```

### 刷新 Twitter 数据

```bash
export TIKHUB_API_KEY="your_key"
python scripts/fetch_twitter.py --quick    # Top 16 队
python scripts/fetch_twitter.py             # 全部 48 队
```

---

## 🔬 How It Works

```
┌─────────────────────────────────────────────────────┐
│  1. Data Layer                                        │
│                                                      │
│  eloratings.net ──→ 48 teams real Elo               │
│  Twitter/X ────────→ team analysis + injuries         │
│  Transfermarkt ───→ squad depth signal                │
│                                                      │
├─────────────────────────────────────────────────────┤
│  2. Signal Layer (6 pluggable signals)               │
│                                                      │
│  🐦 Twitter analysis  ±25 Elo                        │
│  🏥 Twitter injuries  -15 Elo per player              │
│  📊 Recent form       ±30 Elo                        │
│  💰 Betting odds      ×0.15 calibration               │
│  🏟️ Home advantage    +50 Elo (hosts)                │
│  💎 Squad depth       +20 Elo (market value)          │
│                                                      │
│  Base Elo + Signals = Adjusted Elo                    │
│                                                      │
├─────────────────────────────────────────────────────┤
│  3. Prediction Engine                                 │
│                                                      │
│  Elo diff → xG = 1.2 + 0.0025 × elo_diff             │
│  xG → Poisson distribution → scoreline probabilities │
│  → Win / Draw / Loss probabilities                    │
│                                                      │
├─────────────────────────────────────────────────────┤
│  4. Monte Carlo Simulation                           │
│                                                      │
│  12 groups × 4 teams → 72 group matches              │
│  Top 2 + 8 best 3rd → Round of 32                    │
│  R32 → R16 → QF → SF → Final                        │
│  Run 10,000 times → advancement probabilities         │
│                                                      │
├─────────────────────────────────────────────────────┤
│  5. Output                                           │
│                                                      │
│  📝 Bilingual text reports                           │
│  📊 PNG charts (champion bars, match cards, bracket) │
└─────────────────────────────────────────────────────┘
```

---

## 📡 Data Sources

| Source | What | Real? |
|--------|------|-------|
| [eloratings.net](https://eloratings.net) | Real World Football Elo ratings (48 teams) | ✅ Real |
| TikHub Twitter API | Football journalist analysis, injury news | ✅ Real-time |
| FIFA | Official group draw & match schedule (104 matches) | ✅ Official |
| Transfermarkt | Squad market values | ✅ |

---

## 🏗️ Project Structure

```
world-cup-2026-predictor/
├── data/
│   ├── teams.json              ← 48 teams: real Elo, players, market value
│   ├── fixtures.json           ← 104 matches: official FIFA schedule
│   ├── results.json            ← played match results tracking
│   └── twitter_data.json       ← Twitter football analysis cache
├── scripts/
│   ├── elo_model.py            ← Core: Elo → xG → Poisson → probabilities
│   ├── signals.py              ← 6 pluggable signal functions
│   ├── monte_carlo.py          ← Monte Carlo tournament simulator
│   ├── visualize.py            ← matplotlib chart generation
│   ├── predict.py              ← Main CLI entry point
│   └── fetch_twitter.py        ← TikHub Twitter data scraper
├── output/charts/              ← Generated PNG charts
├── skill.md                    ← Claude Code skill definition
└── README.md
```

---

## 🏆 Results Example

Belgium vs Egypt (Group G, June 15):

```
Belgium 1858 Elo vs Egypt 1792 Elo
Gap: only 66 points
Win: 44.5% | Draw: 27.2% | Lose: 28.3%
Most likely: 1:1
```

> With real eloratings.net data, this is a tight match — not a blowout.

---

## ⚠️ Limitations

1. Poisson assumes independent goals (first goal affects game rhythm)
2. No tactical model (can't simulate parking the bus, counter-attacks, rotation)
3. Elo parameters are empirically calibrated, not ML-trained
4. Twitter coverage limited to top 16 teams
5. "Best 3rd place" advancement rule is simplified in simulation

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <b>⚽ May the best model win.</b>
</p>
