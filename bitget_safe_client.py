#!/usr/bin/env python3
import base64, hashlib, hmac, json, os, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "btc-swing-strategy"
OUT_DIR.mkdir(exist_ok=True)

ENV_PATH = ROOT / ".env"
SAFE_PATH = OUT_DIR / "bitget_safe_status.json"

BASE_URL = "https://api.bitget.com"
SYMBOL = "BTCUSDT"

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def load_env():
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True))

def sign(ts, method, path_with_query, body, secret):
    prehash = f"{ts}{method.upper()}{path_with_query}{body}"
    return base64.b64encode(
        hmac.new(secret.encode(), prehash.encode(), hashlib.sha256).digest()
    ).decode()

def get_public_ticker():
    path = f"/api/v2/spot/market/tickers?symbol={SYMBOL}"
    req = urllib.request.Request(BASE_URL + path, method="GET")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())

def get_account_assets(api_key, api_secret, passphrase):
    path = "/api/v2/spot/account/assets"
    ts = str(int(time.time() * 1000))
    sig = sign(ts, "GET", path, "", api_secret)

    req = urllib.request.Request(
        BASE_URL + path,
        method="GET",
        headers={
            "ACCESS-KEY": api_key,
            "ACCESS-SIGN": sig,
            "ACCESS-PASSPHRASE": passphrase,
            "ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
            "locale": "en-US",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())

def main():
    load_env()
    api_key = os.getenv("BITGET_API_KEY", "")
    api_secret = os.getenv("BITGET_API_SECRET", "")
    passphrase = os.getenv("BITGET_API_PASSPHRASE", "")

    out = {
        "timestamp_utc": now_iso(),
        "mode": "bitget_safe_read_only",
        "symbol": SYMBOL,
        "public_ticker": {"available": False, "raw": {}},
        "account_assets": {"available": False, "ok": False, "raw": {}},
        "errors": [],
    }

    try:
        out["public_ticker"] = {"available": True, "raw": get_public_ticker()}
    except Exception as e:
        out["errors"].append({"public_ticker": str(e)})

    if api_key and api_secret and passphrase:
        try:
            raw = get_account_assets(api_key, api_secret, passphrase)
            ok = str(raw.get("code")) == "00000"
            out["account_assets"] = {"available": ok, "ok": ok, "raw": raw}
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            out["errors"].append({"account_assets_http": e.code, "raw": raw})
        except Exception as e:
            out["errors"].append({"account_assets": str(e)})
    else:
        out["errors"].append({"credentials": "missing Bitget credentials"})

    write_json(SAFE_PATH, out)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
