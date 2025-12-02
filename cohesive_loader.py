"""
cohesive_loader.py

Încărcare indicatori coezivi (CT, CC, CS, CN) din fișierele reale din data/.
Adaptează numele coloanelor și câmpurilor la structura ta reală.

Presupunere de lucru:
- ic_btc_series.json sau j_btc_series.csv conține o serie temporală,
  cu cel puțin:
    - un câmp pentru dată (ex: "timestamp" sau "date")
    - patru câmpuri pentru indicatori (ex: "CT", "CC", "CS", "CN")
"""

from __future__ import annotations
import os
import pandas as pd
from typing import Optional


DATA_DIR = "data"
JSON_SERIES_PATH = os.path.join(DATA_DIR, "ic_btc_series.json")
CSV_SERIES_PATH = os.path.join(DATA_DIR, "j_btc_series.csv")


def _normalize_timestamp(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    df.index.name = "timestamp"
    return df


def load_cohesive_series() -> pd.DataFrame:
    """
    Încearcă să încarce serii de indicatori coezivi din:
    - ic_btc_series.json (dacă există)
    - altfel j_btc_series.csv

    Returnează un DataFrame indexat pe timestamp, cu coloanele:
    ['CT', 'CC', 'CS', 'CN'].

    IMPORTANT:
    - adaptează mapping-ul `columns_map` la numele reale din fișierele tale.
    """

    if os.path.exists(JSON_SERIES_PATH):
        df = pd.read_json(JSON_SERIES_PATH)
        source = JSON_SERIES_PATH
    elif os.path.exists(CSV_SERIES_PATH):
        df = pd.read_csv(CSV_SERIES_PATH)
        source = CSV_SERIES_PATH
    else:
        raise FileNotFoundError(
            f"Nu am găsit nici {JSON_SERIES_PATH}, nici {CSV_SERIES_PATH}. "
            "Verifică ce fișier conține seriile tale CT/CC/CS/CN."
        )

    print(f"[INFO] Încarc indicatori coezivi din {source}")

    # TODO: adaptează aceste nume la structura reală a fișierului tău
    # Exemplu: poate câmpurile se numesc "ct_value", "cycle_pos", etc.
    columns_map = {
        "timestamp": "timestamp",  # sau "date", "ts" etc.
        "CT": "CT",
        "CC": "CC",
        "CS": "CS",
        "CN": "CN",
    }

    # Dacă datele tale nu au exact aceste nume, modifică cheile din columns_map
    missing = [src for src in columns_map.keys() if src not in df.columns]
    if missing:
        raise KeyError(
            f"Lipsesc coloanele așteptate în fișierul de indicatori: {missing}. "
            "Verifică structura JSON/CSV și ajustează `columns_map` în cohesive_loader.py."
        )

    df = df[list(columns_map.keys())].rename(columns=columns_map)
    df = _normalize_timestamp(df, "timestamp")

    return df[["CT", "CC", "CS", "CN"]]
