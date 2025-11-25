#!/usr/bin/env python
# run_cohesiv_pipeline.py

"""
Rulează întreg pipeline-ul Coeziv pentru BTC + macro:

1. update_global_coeziv_state.py    → data_global/*.csv
2. build_global_coeziv_state.py     → data/global_coeziv_state.json
3. update_btc_state_latest_from_daily.py → data/btc_state_latest.json
4. export_ic_btc_series.py          → data/ic_btc_series.json
5. build_ic_btc_mega_state.py       → data/ic_btc_mega_latest.json
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n[PIPELINE] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[PIPELINE] Eroare la comanda: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    # 1. macro global
    run([sys.executable, "update_global_coeziv_state.py"])
    run([sys.executable, "build_global_coeziv_state.py"])

    # 2. BTC state + serie + mega
    run([sys.executable, "update_btc_state_latest_from_daily.py"])
    run([sys.executable, "export_ic_btc_series.py"])
    run([sys.executable, "build_ic_btc_mega_state.py"])

    print("\n[PIPELINE] ✅ Pipeline Coeziv BTC + Global complet.")


if __name__ == "__main__":
    main()
