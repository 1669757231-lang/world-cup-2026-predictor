# world-cup-predictor

> 🏆 2026 FIFA World Cup prediction skill for Claude Code — Elo + Poisson + Monte Carlo + Reddit sentiment.
> 2026 世界杯预测技能 —— 综合 Elo 评分、泊松分布、蒙特卡洛模拟、Reddit 社区情绪。

## Trigger Keywords / 触发词

When the user mentions any of these, invoke this skill:
当用户提到以下任一关键词时，调用此技能：

- **预测世界杯** / **世界杯预测** / **世界杯分析** / **world cup prediction**
- **预测 [国家名]** — e.g. "预测巴西", "predict Brazil"
- **[国家A] vs [国家B]** — e.g. "阿根廷 vs 法国预测", "ARG vs FRA prediction"
- **冠军预测** / **夺冠概率** / **champion probability**
- **小组赛预测** / **group stage prediction**
- **[国家] 能走多远** / **how far can [team] go**
- **世界杯形势** / **world cup outlook**

## Workflow / 工作流程

### 1. Identify the user's intent / 识别意图

| User says | Command to run |
|-----------|---------------|
| "预测巴西 vs 德国" | `python scripts/predict.py match 巴西 德国 --signals` |
| "世界杯冠军预测" | `python scripts/predict.py full --sims 10000 --signals --charts` |
| "阿根廷能走多远" | `python scripts/predict.py full --sims 10000 --signals` → extract Argentina's probabilities |
| "分析D组" | `python scripts/predict.py group D --signals` |
| "看看现在舆论怎么说" | `python scripts/crawl_reddit.py --offline` (or `--update-signals`) |
| "生成可视化图表" | `python scripts/predict.py charts --sims 10000 --signals` |

### 2. Run the prediction / 运行预测

Always run from the project root: `d:\桌面\预测文件夹\`

```bash
cd "d:\桌面\预测文件夹"
pip install -r requirements.txt  # first time only
python scripts/predict.py <command>
```

### 3. Present results / 展示结果

Show the user:
1. **Key numbers**: W/D/L probabilities, most likely score, champion %
2. **Charts**: If `--charts` was used, display the PNGs from `output/charts/`
3. **Context**: Mention what signals are affecting the prediction (injuries, form, sentiment)
4. **Caveats**: Remind the user this is a probabilistic model, not certainty

### 4. Bilingual output / 双语输出

Default to Chinese (`--lang zh`). Use `--lang en` for English output.
Match the user's language in the response.

## Available Commands / 可用命令

| Command | Description |
|---------|-------------|
| `python scripts/predict.py match <队1> <队2>` | Single match prediction |
| `python scripts/predict.py group <A-P\|all>` | Group stage prediction |
| `python scripts/predict.py full --sims 10000` | Full tournament Monte Carlo |
| `python scripts/predict.py champions --top 10` | Quick champion probabilities |
| `python scripts/predict.py charts --sims 10000` | Generate all charts |
| `python scripts/crawl_reddit.py --offline` | Get Reddit sentiment (mock) |
| `python scripts/crawl_reddit.py --update-signals` | Live Reddit crawl |

Add `--signals` to any predict command to apply dynamic adjustments (injuries, form, Reddit sentiment, betting odds, home advantage).

## How It Works / 原理

```
                  ┌─────────────────────┐
                  │   Dynamic Signals    │
                  │  • Injuries          │
                  │  • Recent Form       │
                  │  • Reddit Sentiment  │
                  │  • Betting Market    │
                  │  • Home Advantage    │
                  │  • Squad Depth       │
                  └────────┬────────────┘
                           │ Elo Adjustment
                           ▼
Base Elo ──────────► Adjusted Elo ──────────► Expected Goals (xG)
                                                    │
                                                    ▼
                                             Poisson Distribution
                                                    │
                                                    ▼
                                          Match Probabilities
                                          (W / D / L + Scores)
                                                    │
                                                    ▼
                                          Monte Carlo Simulation
                                          (10,000+ tournaments)
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │  Output           │
                                          │  • Champion %     │
                                          │  • Round-by-round │
                                          │  • Charts (PNG)   │
                                          │  • Text Report    │
                                          └──────────────────┘
```

## Project Structure / 项目结构

```
预测文件夹/
├── skill.md                  ← This file
├── README.md                 ← Project overview (bilingual)
├── requirements.txt          ← Python dependencies
├── data/
│   ├── teams.json            ← 48 teams with Elo, players, market value
│   ├── fixtures.json          ← 80 matches schedule
│   └── reddit_sentiment.json ← Crawled sentiment cache
├── scripts/
│   ├── elo_model.py          ← Core: Elo + Poisson engine
│   ├── signals.py            ← Dynamic signal layer
│   ├── monte_carlo.py        ← Monte Carlo tournament simulator
│   ├── visualize.py          ← Chart generation (matplotlib)
│   ├── crawl_reddit.py       ← Reddit sentiment scraper
│   └── predict.py            ← Main CLI entry point
└── output/
    └── charts/               ← Generated PNG charts
```

## Notes for the Agent

1. **First run**: Check if `numpy` and `matplotlib` are installed. If not, run `pip install -r requirements.txt`.
2. **Data freshness**: Remind the user that `teams.json` has base Elo ratings and squad data that may need updating as the tournament progresses.
3. **Signal freshness**: After actual matches are played, call `signals.update_elo()` to feed results back into the model.
4. **Chart display**: When charts are generated, point the user to `output/charts/` or describe what's in them.
5. **Confidence calibration**: The model gives probabilities, not certainties. A 65% favorite still loses 35% of the time. Communicate this.
