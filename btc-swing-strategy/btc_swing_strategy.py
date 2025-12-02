import pandas as pd

def generate_signals(df):

    df = df.copy()

    df["signal"] = "flat"

    # LONG
    long_cond = (
        (df["regime"].isin(["bull_early", "bull_mid", "bull_late"])) &
        (df["ic_struct"] > 50) &
        (df["ic_dir"] > 50) &
        (df["ic_flux"] < 20)
    )

    df.loc[long_cond, "signal"] = "long"

    # SHORT
    short_cond = (
        (df["regime"].str.contains("bear")) &
        (df["ic_struct"] < 50) &
        (df["ic_dir"] < 50) &
        (df["ic_flux"] > 20)
    )

    df.loc[short_cond, "signal"] = "short"

    # MIXED => FLAT
    df.loc[df["regime"] == "mixed", "signal"] = "flat"

    return df
