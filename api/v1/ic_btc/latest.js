const fs = require("fs").promises;
const path = require("path");

// Helpers
function clamp(x, min, max) {
  return Math.min(max, Math.max(min, x));
}

function safeNumber(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

async function readJson(relPath) {
  const filePath = path.join(process.cwd(), relPath);
  const raw = await fs.readFile(filePath, "utf8");
  return JSON.parse(raw);
}

async function readCsvLastTwo(relPath) {
  const filePath = path.join(process.cwd(), relPath);
  const text = await fs.readFile(filePath, "utf8");
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 3) {
    throw new Error(`Prea puține linii în ${relPath}`);
  }

  const header = lines[0].split(",");
  const last = lines[lines.length - 1].split(",");
  const prev = lines[lines.length - 2].split(",");

  const objLast = {};
  const objPrev = {};
  header.forEach((h, i) => {
    objLast[h.trim()] = last[i] !== undefined ? last[i].trim() : "";
    objPrev[h.trim()] = prev[i] !== undefined ? prev[i].trim() : "";
  });

  return { last: objLast, prev: objPrev };
}

function pctChange(lastObj, prevObj) {
  const keys = ["close", "Adj Close", "adj_close", "value"];
  let last = null;
  let prev = null;
  for (const k of keys) {
    if (last === null && lastObj[k] !== undefined) {
      last = safeNumber(lastObj[k]);
    }
    if (prev === null && prevObj[k] !== undefined) {
      prev = safeNumber(prevObj[k]);
    }
  }
  if (last == null || prev == null || prev === 0) return 0;
  return (last - prev) / prev;
}

function computeMacroScore(seriesReturns) {
  const spx = seriesReturns.spx ?? 0;
  const dxy = seriesReturns.dxy ?? 0;
  const gold = seriesReturns.gold ?? 0;
  const vix = seriesReturns.vix ?? 0;
  const oil = seriesReturns.oil ?? 0;

  // aceeași combinație ca în front-end
  const riskOn = spx * 1.3 + gold * 0.6 + oil * 0.7 - dxy * 0.7 - vix * 1.0;
  const scaled = clamp(riskOn * 4, -1, 1);

  let label = "echilibrat";
  if (scaled > 0.25) label = "risk-on (favorabil risc)";
  else if (scaled < -0.25) label = "risk-off (aversiune la risc)";

  return { score: scaled, label };
}

function computeCoezivProb(btcState, macroState) {
  const icStruct = safeNumber(btcState.ic_struct ?? btcState.ic_btc_struct) ?? 0;
  const icDir = safeNumber(btcState.ic_dir ?? btcState.ic_btc_dir) ?? 50;
  const vol30 = safeNumber(btcState.vol30);
  const close = safeNumber(btcState.close);
  const ema50 = safeNumber(btcState.ema50);
  const ema200 = safeNumber(btcState.ema200);
  const regime = btcState.regime_coeziv_label || btcState.regime_label || btcState.regime || null;

  const trendStrength = clamp(icStruct / 100, 0, 1);
  const directionBias = clamp((icDir - 50) / 50, -1, 1);

  let maBias = 0;
  if (close != null && ema50 != null && ema200 != null) {
    const dist50 = (close - ema50) / ema50;
    const dist200 = (close - ema200) / ema200;
    const s50 = sigmoid(dist50 * 8) * 2 - 1;
    const s200 = sigmoid(dist200 * 6) * 2 - 1;
    maBias = clamp(0.6 * s50 + 0.4 * s200, -1, 1);
  }

  let volFactor = 1.0;
  if (vol30 != null) {
    if (vol30 > 140) volFactor = 0.65;
    else if (vol30 > 110) volFactor = 0.8;
    else if (vol30 < 30) volFactor = 0.85;
  }

  let macroBias = 0;
  let macroConf = 0;
  if (macroState && Number.isFinite(macroState.score)) {
    macroBias = clamp(macroState.score, -1, 1);
    macroConf = clamp(Math.abs(macroBias) * 1.3, 0, 1);
  }

  const combinedBias = clamp(
    0.5 * directionBias + 0.25 * maBias + 0.25 * macroBias,
    -1,
    1
  );

  const probUpRaw = 0.5 + 0.5 * combinedBias;

  const baseConf =
    macroConf > 0 ? trendStrength * 0.6 + macroConf * 0.4 : trendStrength;
  const confidence = clamp(baseConf * volFactor, 0, 1);

  const probUp = clamp(
    probUpRaw * confidence + 0.5 * (1 - confidence),
    0,
    1
  );
  const probDown = 1 - probUp;

  let biasLabel = "neutru";
  if (probUp > 0.6) biasLabel = "bullish";
  else if (probUp < 0.4) biasLabel = "bearish";

  return {
    probUp,
    probDown,
    confidence,
    regime,
    icStruct,
    icDir,
    close,
    ema50,
    ema200,
    vol30,
    biasLabel
  };
}

module.exports = async (req, res) => {
  // CORS simplu
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  try {
    // 1) date BTC coeziv
    const btcState = await readJson("data/btc_state_latest.json");

    // 2) macro (ultimele 2 rânduri din fiecare CSV)
    const [spx, dxy, gold, vix, oil] = await Promise.all([
      readCsvLastTwo("data_global/spx.csv"),
      readCsvLastTwo("data_global/dxy.csv"),
      readCsvLastTwo("data_global/gold.csv"),
      readCsvLastTwo("data_global/vix.csv"),
      readCsvLastTwo("data_global/oil.csv")
    ]);

    const macroReturns = {
      spx: pctChange(spx.last, spx.prev),
      dxy: pctChange(dxy.last, dxy.prev),
      gold: pctChange(gold.last, gold.prev),
      vix: pctChange(vix.last, vix.prev),
      oil: pctChange(oil.last, oil.prev)
    };

    const macro = computeMacroScore(macroReturns);
    const model = computeCoezivProb(btcState, macro);

    const asOf =
      btcState.as_of ||
      btcState.date ||
      btcState.last_date ||
      null;

    const timeframe = btcState.timeframe || "1D";
    const window = btcState.window || 260;

    const probUp = model.probUp;
    const probDown = model.probDown;
    const confidence = model.confidence;

    res.setHeader("Content-Type", "application/json");

    // răspuns principal + obiect compact "tradingview-style"
    return res.status(200).json({
      symbol: "BTCUSDT",
      as_of: asOf,
      ic_btc_struct: model.icStruct,
      ic_btc_dir: model.icDir,
      prob_up: probUp,
      prob_down: probDown,
      confidence,
      regime: model.regime,
      regime_label: model.regime || btcState.regime_coeziv_label || btcState.regime_label || "",
      macro_score: macro.score,
      macro_label: macro.label,
      meta: {
        source: "CohesivX BTC v1",
        timeframe,
        window
      },
      tradingview: {
        s: "ok",
        symbol: "BTCUSDT",
        time: asOf,
        p_up: Number(probUp.toFixed(4)),
        p_down: Number(probDown.toFixed(4)),
        confidence: Number(confidence.toFixed(4)),
        ic_struct: Number(model.icStruct.toFixed(2)),
        ic_dir: Number(model.icDir.toFixed(2)),
        regime: model.regime
      }
    });
  } catch (err) {
    console.error("[CohesivX API] Eroare:", err);
    return res.status(500).json({
      error: "failed_to_compute_coeziv_btc",
      message:
        "Nu am putut citi datele sau nu am putut calcula probabilitățile coezive.",
    });
  }
};
