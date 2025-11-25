import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SERIES_FILE = DATA_DIR / "ic_btc_series.json"
OUT_FILE = DATA_DIR / "ic_btc_mega_latest.json"


def clamp(x, a, b):
    return max(a, min(b, x))


def safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except:
        return None


def classify_phase(icc):
    if icc is None:
        return "unknown", "Date insuficiente"
    if icc < 15:
        return "base", "Bază structurală profundă"
    if icc < 30:
        return "early_bull", "Bull incipient / acumulare"
    if icc < 50:
        return "mid_bull", "Bull intermediar"
    if icc < 70:
        return "late_bull", "Bull târziu / top structural posibil"
    return "extreme", "Extensie / climax de ciclu"


def classify_structure(ic_struct):
    if ic_struct is None:
        return "Date insuficiente"
    if ic_struct < 20:
        return "Structură foarte slăbită"
    if ic_struct < 40:
        return "Structură în răcire"
    if ic_struct < 60:
        return "Structură neutră"
    if ic_struct < 80:
        return "Structură ridicată, tensionată"
    return "Structură extrem de ridicată"


def classify_direction(ic_flux):
    if ic_flux is None:
        return "Date insuficiente"
    if ic_flux < 20:
        return "Presiune bear dominantă"
    if ic_flux < 40:
        return "Direcție instabilă / laterală"
    if ic_flux < 60:
        return "Bias slab direcțional"
    if ic_flux < 80:
        return "Trend direcțional stabil"
    return "Extensie direcțională"


def classify_subcycles(icc, ic_flux):
    if icc is None or ic_flux is None:
        return "Date insuficiente"
    if icc > 70 and ic_flux < 30:
        return "Sub-cicluri bull/bear dese"
    if icc < 30 and ic_flux > 70:
        return "Compressie + acumulare"
    return "Structură policentrică activă"


def build_mega_score(icc, ic_struct, ic_flux):
    icc_v = icc or 0
    struct_v = ic_struct or 0
    flux_v = ic_flux or 0

    raw = (
        0.45 * icc_v +
        0.35 * (100 - struct_v) +
        0.20 * flux_v
    )
    return round(clamp(raw, 0, 100), 1)


def main():
    if not SERIES_FILE.exists():
        raise RuntimeError("Lipsă ic_btc_series.json")

    with open(SERIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    series = data.get("series", [])
    if not series:
        raise RuntimeError("Seria IC BTC este goală")

    last = series[-1]

    icc = safe_float(last.get("icc") or last.get("ic_cycle"))
    ic_struct = safe_float(last.get("ic_struct") or last.get("ic_btc"))
    ic_flux = safe_float(last.get("ic_flux") or last.get("icf_btc"))

    phase_code, phase_label = classify_phase(icc)
    struct_label = classify_structure(ic_struct)
    dir_label = classify_direction(ic_flux)
    subcycles_label = classify_subcycles(icc, ic_flux)
    mega_score = build_mega_score(icc, ic_struct, ic_flux)

    out = {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "icc": icc,
        "ic_struct": ic_struct,
        "ic_flux": ic_flux,
        "mega_score": mega_score,
        "mega_phase_code": phase_code,
        "mega_phase_label": phase_label,
        "structure_label": struct_label,
        "direction_label": dir_label,
        "subcycles_label": subcycles_label,
        "show_base_card": (phase_code == "base"),
    }

    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Mega Coeziv state salvat în {OUT_FILE}")


if __name__ == "__main__":
    main()
