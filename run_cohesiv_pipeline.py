#!/usr/bin/env python
# run_cohesiv_pipeline.py

"""
Pipeline Coeziv BTC-only (fără Yahoo / macro global).

Generează:
1. data/btc_state_latest.json
2. data/ic_btc_series.json
3. data/ic_btc_mega_latest.json
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
    # BTC state + serie + mega (fără macro/yahoo)
    run([sys.executable, "update_btc_state_latest_from_daily.py"])
    run([sys.executable, "export_ic_btc_series.py"])
    run([sys.executable, "build_ic_btc_mega_state.py"])

    print("\n[PIPELINE] ✅ Pipeline Coeziv BTC complet (fără Yahoo).")


if __name__ == "__main__":
    main()
