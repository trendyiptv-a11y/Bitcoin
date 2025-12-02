import json
import pandas as pd

def load_ic_series(path="ic_btc_series.json"):
    with open(path, "r") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw["series"])
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.set_index("date").sort_index()

    return df[[
        "close",
        "ic_struct",
        "ic_dir",
        "ic_flux",
        "ic_cycle",
        "regime"
    ]]
