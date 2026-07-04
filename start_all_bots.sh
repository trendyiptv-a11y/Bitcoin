#!/data/data/com.termux/files/usr/bin/bash
cd ~/Bitcoin || exit 1

termux-wake-lock 2>/dev/null || true

echo "=== CohesivX START ALL BOTS ==="

if ! pgrep -af "while true; do ./run_cohesivx_15m.sh" >/dev/null; then
  nohup bash -c 'while true; do ./run_cohesivx_15m.sh; sleep 900; done' > micro_live_loop.log 2>&1 &
  echo "started micro live loop"
else
  echo "micro live loop already running"
fi

if ! pgrep -af "push_scalper_dashboard_hourly.sh" >/dev/null; then
  nohup bash -c 'while true; do ./push_scalper_dashboard_hourly.sh; sleep 3600; done' > push_scalper_dashboard_hourly.log 2>&1 &
  echo "started scalper dashboard loop"
else
  echo "scalper dashboard loop already running"
fi

if ! pgrep -af "cohesivx_guarded_paper_loop_v0324.sh" >/dev/null; then
  nohup bash cohesivx_guarded_paper_loop_v0324.sh > cohesivx_guarded_paper_loop.log 2>&1 &
  echo "started guarded paper loop"
else
  echo "guarded paper loop already running"
fi

echo
echo "=== CHECK ==="
pgrep -af "run_cohesivx_15m|push_scalper_dashboard_hourly|cohesivx_guarded_paper_loop" || true
