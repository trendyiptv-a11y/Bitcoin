#!/data/data/com.termux/files/usr/bin/bash
cd ~/Bitcoin || exit 1

while true; do
  echo "=== SCALPER 5M $(date -u) ==="

  python cohesivx_scalp_paper_bot.py
  python cohesivx_live_cohesion_guard_v0322.py
  python cohesivx_guarded_paper_exit_v0325.py

  # ./push_scalper_dashboard_hourly.sh  # disabled: local-only bot mode

  sleep 300
done
