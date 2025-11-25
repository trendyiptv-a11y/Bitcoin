# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="CohesivX BTC API",
    version="1.0.0",
    description="IC_BTC / ICD_BTC / regim coeziv + context macro global",
)

# 🔓 Poți restrânge la domeniul tău de landing
origins = [
    "*",  # sau "https://btc.cohesivx.com", "https://cohesivx.com" etc.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_json(path: Path):
    if not path.exists():
        return {"error": f"file_not_found", "path": str(path)}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/v1/ic_btc/latest")
def ic_btc_latest():
    """
    Snapshot simplu BTC – folosit în HERO (cardul de pe landing).
    Provine din data/btc_state_latest.json (scris de update_btc_state_latest_from_daily.py)
    """
    return load_json(DATA_DIR / "btc_state_latest.json")


@app.get("/v1/ic_btc/mega")
def ic_btc_mega():
    """
    Stare "mega" BTC – fază, subcicluri, etichete conceptuale.
    Provine din data/ic_btc_mega_latest.json (scris de build_ic_btc_mega_state.py)
    """
    return load_json(DATA_DIR / "ic_btc_mega_latest.json")


@app.get("/v1/ic_btc/history")
def ic_btc_history():
    """
    Seria istorică IC_BTC / ICD_BTC – pentru grafice.
    Provine din data/ic_btc_series.json (scris de export_ic_btc_series.py)
    """
    return load_json(DATA_DIR / "ic_btc_series.json")


@app.get("/v1/global/latest")
def global_latest():
    """
    Context macro coeziv global (IC_GLOBAL, ICD_GLOBAL, risk_score, macro_signal).
    Provine din data/global_coeziv_state.json (scris de build_global_coeziv_state.py)
    """
    return load_json(DATA_DIR / "global_coeziv_state.json")


@app.get("/health")
def health():
    return {"status": "ok"}
