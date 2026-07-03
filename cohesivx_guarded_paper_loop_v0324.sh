#!/usr/bin/env bash
# CohesivX Guarded Paper Loop v0.3.24
# PAPER-ONLY automation loop:
#   1) run live cohesion guard
#   2) run guarded paper exit consumer
#   3) sleep
#
# Safety:
#   - No real orders.
#   - Does not touch Micro-Live.
#   - Uses only:
#       cohesivx_live_cohesion_guard_v0322.py
#       cohesivx_guarded_paper_exit_v0325.py
#
# Usage:
#   cd ~/Bitcoin
#   bash cohesivx_guarded_paper_loop_v0324.sh
#
# Optional interval:
#   GUARD_SLEEP_SECONDS=60 bash cohesivx_guarded_paper_loop_v0324.sh
#
# Background:
#   nohup bash cohesivx_guarded_paper_loop_v0324.sh > cohesivx_guarded_paper_loop.log 2>&1 &
#   echo $! > cohesivx_guarded_paper_loop.pid

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1

SLEEP_SECONDS="${GUARD_SLEEP_SECONDS:-60}"

GUARD_SCRIPT="cohesivx_live_cohesion_guard_v0322.py"
EXIT_SCRIPT="cohesivx_guarded_paper_exit_v0325.py"

LOOP_REPORT="btc-swing-strategy/cohesivx_guarded_paper_loop_report.json"
LOOP_LOG="btc-swing-strategy/cohesivx_guarded_paper_loop_events.jsonl"

mkdir -p btc-swing-strategy

now_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

json_escape() {
  python - <<'PY' "$1"
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

write_report() {
  local status="$1"
  local result="$2"
  local reason="$3"
  local guard_status="$4"
  local exit_status="$5"

  python - <<'PY' "$LOOP_REPORT" "$status" "$result" "$reason" "$guard_status" "$exit_status" "$SLEEP_SECONDS"
import json, sys
from datetime import datetime, timezone

path, status, result, reason, guard_status, exit_status, sleep_seconds = sys.argv[1:]
data = {
    "bot": "CohesivX Guarded Paper Loop v0.3.24",
    "mode": "PAPER_ONLY_LOOP",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "result": result,
    "reason": reason,
    "guard_status": guard_status,
    "exit_status": exit_status,
    "sleep_seconds": int(float(sleep_seconds)),
    "real_orders": False,
    "micro_live_touched": False,
    "scripts": {
        "guard": "cohesivx_live_cohesion_guard_v0322.py",
        "exit_consumer": "cohesivx_guarded_paper_exit_v0325.py"
    }
}
open(path, "w").write(json.dumps(data, indent=2, sort_keys=True))
print(json.dumps(data, sort_keys=True))
PY
}

append_event() {
  local event_json="$1"
  printf '%s\n' "$event_json" >> "$LOOP_LOG"
}

if [ ! -f "$GUARD_SCRIPT" ]; then
  write_report "ERROR" "MISSING_GUARD_SCRIPT" "$GUARD_SCRIPT not found" "NOT_RUN" "NOT_RUN"
  exit 1
fi

if [ ! -f "$EXIT_SCRIPT" ]; then
  write_report "ERROR" "MISSING_EXIT_SCRIPT" "$EXIT_SCRIPT not found" "NOT_RUN" "NOT_RUN"
  exit 1
fi

python -m py_compile "$GUARD_SCRIPT" "$EXIT_SCRIPT" || {
  write_report "ERROR" "PY_COMPILE_FAILED" "Guard or exit script failed py_compile" "NOT_RUN" "NOT_RUN"
  exit 1
}

echo "=== CohesivX Guarded Paper Loop v0.3.24 ==="
echo "Root: $ROOT"
echo "Sleep: ${SLEEP_SECONDS}s"
echo "Safety: PAPER ONLY, no Micro-Live, no real orders"
echo

while true; do
  TS="$(now_utc)"
  echo "[$TS] running live guard..."

  GUARD_OUTPUT="$(python "$GUARD_SCRIPT" 2>&1)"
  GUARD_RC=$?

  if [ "$GUARD_RC" -ne 0 ]; then
    echo "$GUARD_OUTPUT"
    EVENT="$(python - <<'PY' "$TS" "$GUARD_RC" "$GUARD_OUTPUT"
import json, sys
ts, rc, output = sys.argv[1:]
print(json.dumps({
  "timestamp_utc": ts,
  "event": "GUARD_FAILED",
  "return_code": int(rc),
  "output": output[-2000:],
}, sort_keys=True))
PY
)"
    append_event "$EVENT"
    write_report "ERROR" "GUARD_FAILED" "Live guard returned non-zero" "FAILED" "NOT_RUN"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  echo "$GUARD_OUTPUT"

  GUARD_ACTION="$(python - <<'PY'
import json
p="btc-swing-strategy/cohesivx_live_cohesion_guard_report.json"
try:
    d=json.load(open(p))
    print(d.get("guard_action") or "")
except Exception:
    print("")
PY
)"

  echo "[$(now_utc)] guard_action: ${GUARD_ACTION:-UNKNOWN}"
  echo "[$(now_utc)] running guarded paper exit consumer..."

  EXIT_OUTPUT="$(python "$EXIT_SCRIPT" 2>&1)"
  EXIT_RC=$?

  if [ "$EXIT_RC" -ne 0 ]; then
    echo "$EXIT_OUTPUT"
    EVENT="$(python - <<'PY' "$TS" "$GUARD_ACTION" "$EXIT_RC" "$EXIT_OUTPUT"
import json, sys
ts, action, rc, output = sys.argv[1:]
print(json.dumps({
  "timestamp_utc": ts,
  "event": "EXIT_CONSUMER_FAILED",
  "guard_action": action,
  "return_code": int(rc),
  "output": output[-2000:],
}, sort_keys=True))
PY
)"
    append_event "$EVENT"
    write_report "ERROR" "EXIT_CONSUMER_FAILED" "Guarded exit consumer returned non-zero" "PASS" "FAILED"
    sleep "$SLEEP_SECONDS"
    continue
  fi

  echo "$EXIT_OUTPUT"

  EXIT_RESULT="$(python - <<'PY'
import json
p="btc-swing-strategy/cohesivx_guarded_paper_exit_report.json"
try:
    d=json.load(open(p))
    print(d.get("result") or "")
except Exception:
    print("")
PY
)"

  EVENT="$(python - <<'PY' "$TS" "$GUARD_ACTION" "$EXIT_RESULT"
import json, sys
ts, action, result = sys.argv[1:]
print(json.dumps({
  "timestamp_utc": ts,
  "event": "LOOP_CYCLE",
  "guard_action": action,
  "exit_result": result,
}, sort_keys=True))
PY
)"
  append_event "$EVENT"

  write_report "PASS" "LOOP_CYCLE_DONE" "Guard + paper exit consumer completed" "PASS" "PASS" >/dev/null

  echo "[$(now_utc)] exit_result: ${EXIT_RESULT:-UNKNOWN}"
  echo "[$(now_utc)] sleeping ${SLEEP_SECONDS}s"
  echo

  sleep "$SLEEP_SECONDS"
done
