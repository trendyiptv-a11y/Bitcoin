#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "btc-swing-strategy" / "mecanism.html"

CSS_BLOCK = """
    /* MODEL_PRICE_EXPLANATION_CSS_START */
    .model-price-explainer {
      margin-top: 7px;
      padding: 8px 10px;
      border-radius: 14px;
      border: 1px solid rgba(56,189,248,0.22);
      background: rgba(56,189,248,0.055);
      color: var(--text-muted);
      font-size: 11px;
      line-height: 1.45;
      text-align: center;
    }
    body.light-mode .model-price-explainer {
      background: rgba(14,165,233,0.07);
      border-color: rgba(14,165,233,0.22);
      color: #475569;
    }
    .model-price-explainer strong {
      color: var(--text-main);
      font-weight: 750;
    }
    /* MODEL_PRICE_EXPLANATION_CSS_END */
"""

HTML_BLOCK = """
          <!-- MODEL_PRICE_EXPLANATION_UI_START -->
          <div id="model-price-explanation" class="model-price-explainer">
            Prețul mecanismului va fi explicat după următorul snapshot coeziv.
          </div>
          <!-- MODEL_PRICE_EXPLANATION_UI_END -->
"""

CONST_BLOCK = """
    // MODEL_PRICE_EXPLANATION_CONST_START
    const MODEL_PRICE_EXPLANATION_EL = document.getElementById("model-price-explanation");
    // MODEL_PRICE_EXPLANATION_CONST_END
"""

CONFIRMATION_URLS_BLOCK = """
    // STRUCTURAL_CONFIRMATION_URLS_START
    const STRUCTURAL_CONFIRMATION_URLS = [
      "comparative_backtest_summary.json",
      "./comparative_backtest_summary.json",
      "/btc-swing-strategy/comparative_backtest_summary.json"
    ];
    // STRUCTURAL_CONFIRMATION_URLS_END
"""

FUNCTION_BLOCK = """
    // MODEL_PRICE_EXPLANATION_JS_START
    function formatModelPriceExplanation(state) {
      if (!state || typeof state !== "object") return "";
      if (typeof state.model_price_explanation === "string" && state.model_price_explanation.trim()) {
        return state.model_price_explanation.trim();
      }

      const fmtUsd = (v) => {
        const n = Number(v);
        if (!Number.isFinite(n)) return null;
        return `${USD_FORMATTER.format(n)} USD`;
      };
      const fmtPct = (v) => {
        const n = Number(v);
        if (!Number.isFinite(n)) return null;
        const sign = n > 0 ? "+" : n < 0 ? "−" : "";
        return `${sign}${Math.abs(n * 100).toFixed(1)}%`;
      };

      const comps = state.model_price_components || {};
      const bands = state.model_price_bands || {};
      const model = fmtUsd(state.model_price_usd);
      const cost = fmtUsd(comps.production_cost_anchor_usd);
      const samples = Number(comps.similar_context_samples);
      const mult = Number(comps.historical_multiplier_p50);
      const dev = fmtPct(state.model_price_deviation);
      const p10 = fmtUsd(bands.p10);
      const p90 = fmtUsd(bands.p90);

      if (!model) return "";

      let text = `Preț coeziv central: ${model}`;
      if (cost) text += `, ancorat în costul de producție actual (${cost})`;
      if (Number.isFinite(samples) && samples > 0 && Number.isFinite(mult)) {
        text += ` și ${samples.toFixed(0)} contexte istorice similare, cu multiplicator median preț/cost ${mult.toFixed(3)}×`;
      } else if (Number.isFinite(samples) && samples > 0) {
        text += ` și ${samples.toFixed(0)} contexte istorice similare`;
      }
      text += ".";
      if (dev) text += ` Prețul live este ${dev} față de acest reper.`;
      if (p10 && p90) {
        text += ` Banda statistică este ${p10} – ${p90}; banda superioară nu este target, ci limită statistică a multiplicatorului istoric.`;
      }
      return text;
    }

    function updateModelPriceExplanation(state) {
      if (!MODEL_PRICE_EXPLANATION_EL) return;
      const text = formatModelPriceExplanation(state);
      if (!text) {
        MODEL_PRICE_EXPLANATION_EL.textContent = "Prețul mecanismului va fi explicat după următorul snapshot coeziv.";
        return;
      }
      MODEL_PRICE_EXPLANATION_EL.textContent = text;
    }
    // MODEL_PRICE_EXPLANATION_JS_END
"""

STRUCTURAL_CONFIRMATION_BLOCK = """
    // STRUCTURAL_CONFIRMATION_JS_START
    function pctDisplay(value, digits = 0) {
      const n = Number(value);
      if (!Number.isFinite(n)) return null;
      return `~${(n * 100).toFixed(digits)}%`;
    }

    function currentRegimeFromState(state) {
      const ctxRegime = state && state.model_price_context && state.model_price_context.regime;
      if (ctxRegime) return String(ctxRegime);
      const market = state && state.market_regime;
      if (market && typeof market.code === "string") return market.code;
      if (market && typeof market.label === "string") return market.label;
      return "n/a";
    }

    function confirmationStats(summary, horizonDays, thresholdName = "threshold_10pct") {
      const model = summary && summary.models && summary.models.cohesive_v2;
      const horizon = model && model[`horizon_${horizonDays}d`];
      const block = horizon && horizon[thresholdName];
      if (!block) return null;
      const rate = Number(block.directional_hit_rate);
      const events = Number(block.events);
      if (!Number.isFinite(rate) || !Number.isFinite(events)) return null;
      return { rate, events };
    }

    function updateStructuralConfirmation(state, summary) {
      if (!SIGNAL_PROB_EL && !SIGNAL_PROB_BREAKDOWN_EL && !DRIFT_EL) return;

      const h7 = confirmationStats(summary, 7);
      const h30 = confirmationStats(summary, 30);
      const samples = state && state.model_price_components
        ? Number(state.model_price_components.similar_context_samples)
        : null;
      const regime = currentRegimeFromState(state);
      const dev = pctDisplay(state && state.model_price_deviation, 1);

      if (SIGNAL_PROB_EL) {
        if (h7 && h30) {
          SIGNAL_PROB_EL.textContent =
            `Confirmare istorică: ${pctDisplay(h7.rate, 0)} pe 7 zile și ${pctDisplay(h30.rate, 0)} pe 30 zile în situații similare.`;
        } else {
          SIGNAL_PROB_EL.textContent =
            "Confirmare istorică: se actualizează după următorul backtest al mecanismului.";
        }
      }

      if (SIGNAL_PROB_BREAKDOWN_EL) {
        if (h30) {
          const sampleText = Number.isFinite(samples) && samples > 0
            ? ` Contextul curent folosește ${samples.toFixed(0)} contexte istorice similare.`
            : "";
          SIGNAL_PROB_BREAKDOWN_EL.textContent =
            `Bază statistică: ${h30.events.toFixed(0)} evenimente istorice cu deviații ample față de reperul coeziv.${sampleText}`;
        } else {
          SIGNAL_PROB_BREAKDOWN_EL.textContent =
            "Bază statistică: așteptăm fișierul de confirmare istorică.";
        }
      }

      if (DRIFT_EL) {
        const devText = dev ? ` Deviația curentă este ${dev} față de prețul coeziv.` : "";
        DRIFT_EL.textContent =
          `Semnal structural, nu intraday. Regim curent: ${regime}.${devText}`;
      }
    }
    // STRUCTURAL_CONFIRMATION_JS_END
"""


def replace_once(text: str, needle: str, replacement: str, label: str) -> str:
    if needle not in text:
        raise RuntimeError(f"Nu am găsit ancora pentru {label}.")
    return text.replace(needle, replacement, 1)


def main() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    original = html

    if "MODEL_PRICE_EXPLANATION_CSS_START" not in html:
        css_anchor = "    .deviation-status.dev-extreme { color: #f97373; }\n"
        html = replace_once(html, css_anchor, css_anchor + CSS_BLOCK, "CSS explicație preț mecanism")

    if "MODEL_PRICE_EXPLANATION_UI_START" not in html:
        ui_anchor = "          <div id=\"deviation-status\" class=\"deviation-status\">\n            Status deviație: n/a (așteptăm un snapshot al modelului).\n          </div>\n"
        html = replace_once(html, ui_anchor, ui_anchor + HTML_BLOCK, "UI explicație preț mecanism")

    if "MODEL_PRICE_EXPLANATION_CONST_START" not in html:
        const_anchor = "    const DEV_STATUS_EL = document.getElementById(\"deviation-status\");\n"
        html = replace_once(html, const_anchor, const_anchor + CONST_BLOCK, "const explicație preț mecanism")

    if "MODEL_PRICE_EXPLANATION_JS_START" not in html:
        fn_anchor = "    function resetDeviationStatus(text) {\n"
        html = replace_once(html, fn_anchor, FUNCTION_BLOCK + "\n" + fn_anchor, "funcții explicație preț mecanism")

    if "updateModelPriceExplanation(state);" not in html:
        call_anchor = "        MSG_EL.textContent = message || \"Nu există mesaj coeziv pentru acest moment.\";\n"
        html = replace_once(
            html,
            call_anchor,
            call_anchor + "        updateModelPriceExplanation(state);\n",
            "apel explicație preț mecanism",
        )

    if "STRUCTURAL_CONFIRMATION_URLS_START" not in html:
        urls_anchor = '    const STATE_URLS = ["coeziv_state.json", "./coeziv_state.json", "/btc-swing-strategy/coeziv_state.json", "/coeziv_state.json"];\n'
        html = replace_once(
            html,
            urls_anchor,
            urls_anchor + CONFIRMATION_URLS_BLOCK,
            "URL-uri confirmare structurală",
        )

    if "STRUCTURAL_CONFIRMATION_JS_START" not in html:
        fn_anchor = "    function resetDeviationStatus(text) {\n"
        html = replace_once(
            html,
            fn_anchor,
            STRUCTURAL_CONFIRMATION_BLOCK + "\n" + fn_anchor,
            "funcții confirmare structurală",
        )

    if "let structuralConfirmationSummary = null;" not in html:
        state_anchor = "        const state = await fetchJsonFallback(STATE_URLS);\n"
        html = replace_once(
            html,
            state_anchor,
            state_anchor
            + "        let structuralConfirmationSummary = null;\n"
            + "        try {\n"
            + "          structuralConfirmationSummary = await fetchJsonFallback(STRUCTURAL_CONFIRMATION_URLS);\n"
            + "        } catch (_) {\n"
            + "          structuralConfirmationSummary = null;\n"
            + "        }\n",
            "încărcare confirmare structurală",
        )

    if "updateStructuralConfirmation(state, structuralConfirmationSummary);" not in html:
        update_anchor = "        updateFGCard(fgBlock);\n"
        html = replace_once(
            html,
            update_anchor,
            "        updateStructuralConfirmation(state, structuralConfirmationSummary);\n" + update_anchor,
            "apel confirmare structurală",
        )

    if html != original:
        HTML_PATH.write_text(html, encoding="utf-8")
        print(f"Patch aplicat: {HTML_PATH}")
    else:
        print("Patch deja prezent; nu s-a modificat nimic.")


if __name__ == "__main__":
    main()
