# 🏆 World Cup 2026 Predictor / 2026 世界杯预测器

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

> **Real Elo + Poisson + Monte Carlo + Twitter/X Analysis** — A data-driven, open-source World Cup prediction engine with real eloratings.net data.
> 数据驱动的开源世界杯预测引擎，融合真实 Elo 评分（eloratings.net）、泊松分布、蒙特卡洛模拟和 Twitter/X 足球分析。

---

## 📖 Table of Contents / 目录

- [Features / 功能](#-features--功能)
- [Quick Start / 快速开始](#-quick-start--快速开始)
- [Usage / 使用方法](#-usage--使用方法)
- [How It Works / 原理](#-how-it-works--原理)
- [Architecture / 架构](#-architecture--架构)
- [Data Sources / 数据来源](#-data-sources--数据来源)
- [Contributing / 贡献](#-contributing--贡献)
- [License / 许可证](#-license--许可证)

---

## ✨ Features / 功能

- 🔢 **Elo Rating System** — Dynamic team strength ratings, adjusted by multiple real-world signals
- 📊 **Poisson Distribution** — Scoreline probability modeling for every match
- 🎲 **Monte Carlo Simulation** — 10,000+ tournament runs to estimate advancement probabilities
- 📱 **Reddit Sentiment** — Community mood from r/soccer & r/worldcup as a prediction signal
- ⚽ **Match Prediction** — W/D/L probabilities + exact scoreline distribution
- 🏟️ **Tournament Simulation** — Full 48-team bracket with round-by-round probabilities
- 📈 **Rich Visualization** — Champion bars, radar charts, bracket trees, group heatmaps
- 🌐 **Bilingual** — 中文 / English output
- 🔌 **Pluggable Signals** — Add/remove data sources without touching the core engine
- 🆓 **Open Source** — MIT license, hackable and transparent

## 🚀 Quick Start / 快速开始

### Prerequisites / 环境要求

- Python 3.10+
- pip

### Installation / 安装

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/world-cup-2026-predictor.git
cd world-cup-2026-predictor

# Install dependencies
pip install -r requirements.txt
```

### Your First Prediction / 第一个预测

```bash
# Single match / 单场比赛
python scripts/predict.py match 阿根廷 法国

# Champion probabilities (quick) / 冠军概率
python scripts/predict.py champions --top 10

# Full tournament with charts / 完整模拟 + 图表
python scripts/predict.py full --sims 10000 --signals --charts
```

---

## 📖 Usage / 使用方法

### Match Prediction / 比赛预测

```bash
# By Chinese name / 中文名
python scripts/predict.py match 巴西 德国 --lang zh

# By English name / 英文名
python scripts/predict.py match Brazil Germany --lang en

# By FIFA code / 代码
python scripts/predict.py match BRA GER

# With dynamic signals / 应用动态信号
python scripts/predict.py match BRA GER --signals
```

**Output / 输出:**
```
============================================================
  Brazil vs Germany
============================================================
  Elo:     1965  vs  1940
  xG:      1.65  vs  1.42
  胜:       38.5%
  平:       27.2%
  负:       34.3%
  Most likely: 1:1

  Score probabilities:
     1:1  12.4%  ████████████
     2:1   9.8%  █████████
     1:0   8.5%  ████████
     ...
```

### Group Stage / 小组赛

```bash
# Specific group / 指定小组
python scripts/predict.py group C

# All 16 groups / 所有小组
python scripts/predict.py group all --lang zh
```

### Tournament Simulation / 锦标赛模拟

```bash
# Full simulation / 完整模拟
python scripts/predict.py full --sims 10000 --signals

# With charts / 含图表
python scripts/predict.py full --sims 10000 --signals --charts
```

### Charts Only / 仅图表

```bash
python scripts/predict.py charts --sims 10000 --lang zh
```

### Reddit Sentiment / 社区情绪

```bash
# Offline mock data / 离线模拟数据
python scripts/crawl_reddit.py --offline

# Live crawl (requires Reddit API) / 在线爬取
python scripts/crawl_reddit.py --update-signals
```

---

## 🧠 How It Works / 原理

```
                    ┌──────────────────────────┐
                    │     Signal Layer          │
                    │                           │
                    │  🩹 Injuries / 伤病       │
                    │  📊 Recent Form / 近期状态 │
                    │  📱 Reddit / 社区舆论      │
                    │  💰 Betting Odds / 赔率    │
                    │  🏟️ Home Advantage / 主场 │
                    │  💎 Squad Depth / 阵容深度 │
                    └─────────────┬─────────────┘
                                  │ Elo ±Δ
                                  ▼
Base Elo (from teams.json) ──► Adjusted Elo ──► Expected Goals (xG)
                                                       │
                                                       ▼
                                               Poisson Distribution
                                               P(score = i:j)
                                                       │
                                                       ▼
                                              Match W/D/L + Score Probs
                                                       │
                                                       ▼
                                              Monte Carlo × 10,000
                                              Simulate full tournament
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Output          │
                                              │  🏆 Champion %   │
                                              │  🎯 Round probs  │
                                              │  📊 Charts (PNG) │
                                              │  📝 Text report  │
                                              └─────────────────┘
```

### Elo Rating / Elo 评分

Each team has a base Elo rating (in `data/teams.json`). Elo difference between two teams predicts the expected outcome. After each match, Elo ratings are updated based on actual results.

### Poisson Distribution / 泊松分布

Expected goals (xG) is derived from the Elo difference. A Poisson distribution models the probability of each possible scoreline. This gives us the full match outcome distribution.

### Monte Carlo Simulation / 蒙特卡洛模拟

We simulate the entire tournament 10,000+ times. Each simulation:
1. Play all 48 group matches (sampling scores from Poisson)
2. Determine group standings
3. Simulate the knockout bracket (R32 → R16 → QF → SF → Final)
4. Track who wins, reaches each round, etc.
5. Aggregate across all simulations for stable probabilities

### Dynamic Signals / 动态信号

| Signal | Source | Effect |
|--------|--------|--------|
| Injuries | News / manual input | Star out = -20 to -50 Elo |
| Recent Form | Last 5 matches | Up to ±30 Elo |
| Reddit | r/soccer sentiment NLP | Up to ±15 Elo |
| Betting | Odds-implied probability | Calibration blend |
| Home | Host nations (USA/CAN/MEX) | +50 Elo bonus |
| Depth | Squad market value | Up to +20 Elo for deep squads |

---

## 🏗️ Architecture / 架构

```
world-cup-2026-predictor/
├── data/
│   ├── teams.json              ← 48 teams: Elo, players, market value
│   ├── fixtures.json           ← 80 matches schedule
│   └── reddit_sentiment.json   ← Crawled sentiment cache
├── scripts/
│   ├── elo_model.py            ← Core: Elo + Poisson math
│   ├── signals.py              ← Dynamic signal layer (pluggable)
│   ├── monte_carlo.py          ← Tournament simulator
│   ├── visualize.py            ← Chart generation (matplotlib)
│   ├── crawl_reddit.py         ← Reddit crawler (PRAW)
│   └── predict.py              ← Main CLI entry point
├── output/
│   └── charts/                 ← Generated PNG charts
├── skill.md                    ← Claude Code skill definition
├── requirements.txt
├── LICENSE
└── README.md
```

### Module Dependencies / 模块依赖

```
predict.py  ──►  signals.py  ──►  elo_model.py
    │                                 │
    ├──►  monte_carlo.py  ────────────┘
    │
    └──►  visualize.py
```

---

## 📊 Data Sources / 数据来源

| Source | What | Status |
|--------|------|--------|
| **eloratings.net** | Real World Football Elo ratings (all 48 teams) | ✅ Real |
| **Twitter/X (TikHub)** | Football journalist analysis, injury news | ✅ Real-time |
| **FIFA** | Official group draw & match schedule | ✅ Official |
| **Transfermarkt** | Squad market values | ✅ Included |
| **fixtures.json** | 104 matches (72 group + 32 knockout) | ✅ Official |

---

## 🤝 Contributing / 贡献

Contributions welcome! Areas where help is especially valuable:

- 📊 **Live data scrapers** — FIFA rankings, injury news, betting odds
- 🧠 **Better sentiment model** — Replace keyword-based with ML/LLM
- 📈 **Interactive charts** — Plotly/D3 visualizations
- 🌍 **More languages** — Beyond Chinese and English
- 🔬 **Model improvements** — Better xG model, Dixon-Coles, Bayesian

### Setup for Development

```bash
pip install -r requirements.txt
python scripts/predict.py match BRA GER  # quick sanity check
```

---

## 📄 License / 许可证

MIT — see [LICENSE](LICENSE) for details.

**Note on data:** Elo ratings sourced from [eloratings.net](https://eloratings.net) (World Football Elo Ratings). Fixtures from FIFA official schedule. Twitter data via TikHub API.

---

<p align="center">
  <b>⚽ May the best model win. / 让最好的模型预测胜负。🏆</b>
</p>
