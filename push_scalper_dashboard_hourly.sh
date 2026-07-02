#!/usr/bin/env bash
set -e

cd ~/Bitcoin

python cohesivx_scalp_paper_bot.py

git add -f \
  btc-swing-strategy/cohesivx_scalp_report.json \
  btc-swing-strategy/cohesivx_scalp_price_series.jsonl \
  btc-swing-strategy/cohesivx_scalp_trades.jsonl \
  cohesivx-scalper-dashboard.html

git commit -m "Update scalper dashboard data" || true
git push || true
