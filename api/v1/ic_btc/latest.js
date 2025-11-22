const fs = require("fs").promises;
const path = require("path");

module.exports = async (req, res) => {
  // CORS minim, să poți consuma și din alte domenii
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  try {
    // Citește btc_state_latest.json din folderul data/
    const filePath = path.join(process.cwd(), "data", "btc_state_latest.json");
    const raw = await fs.readFile(filePath, "utf8");
    const state = JSON.parse(raw);

    // Fallback-uri inteligente pe câmpuri, ca să nu crape dacă redenumești ceva
    const icStruct = Number(state.ic_struct ?? state.ic_btc_struct);
    const icDir = Number(state.ic_dir ?? state.ic_btc_dir);

    const probUp =
      state.prob_up !== undefined && state.prob_up !== null
        ? Number(state.prob_up)
        : null;
    const probDown =
      state.prob_down !== undefined && state.prob_down !== null
        ? Number(state.prob_down)
        : null;
    const confidence =
      state.confidence !== undefined && state.confidence !== null
        ? Number(state.confidence)
        : null;

    const regimeRaw = state.regime_coeziv ?? state.regime ?? "";
    const regimeLabel =
      state.regime_coeziv_label ?? state.regime_label ?? regimeRaw;

    const macroScore =
      state.macro_score !== undefined && state.macro_score !== null
        ? Number(state.macro_score)
        : null;
    const macroLabel = state.macro_label ?? null;

    const asOf = state.as_of ?? state.date ?? state.last_date ?? null;
    const timeframe = state.timeframe ?? "1D";
    const window = state.window ?? 260;

    res.setHeader("Content-Type", "application/json");

    return res.status(200).json({
      symbol: "BTCUSDT",
      as_of: asOf,
      ic_btc_struct: icStruct,
      ic_btc_dir: icDir,
      prob_up: probUp,
      prob_down: probDown,
      confidence,
      regime: regimeRaw,
      regime_label: regimeLabel,
      macro_score: macroScore,
      macro_label: macroLabel,
      meta: {
        source: "CohesivX BTC v1",
        timeframe,
        window,
      },
    });
  } catch (err) {
    console.error("[CohesivX API] Eroare:", err);
    return res.status(500).json({
      error: "failed_to_load_btc_state_latest",
      message:
        "Nu am putut citi data/btc_state_latest.json. Verifică dacă există și are JSON valid.",
    });
  }
};
