#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "btc-swing-strategy" / "mecanism.html"

URLS_BLOCK = '''
    // STRUCTURAL_CONFIRMATION_URLS_START
    const STRUCTURAL_CONFIRMATION_URLS = [
      "comparative_backtest_summary.json",
      "./comparative_backtest_summary.json",
      "/btc-swing-strategy/comparative_backtest_summary.json"
    ];
    // STRUCTURAL_CONFIRMATION_URLS_END
'''

JS_BLOCK = r'''
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
'''

LOAD_BLOCK = '''        let structuralConfirmationSummary = null;
        try {
          structuralConfirmationSummary = await fetchJsonFallback(STRUCTURAL_CONFIRMATION_URLS);
        } catch (_) {
          structuralConfirmationSummary = null;
        }
'''


def replace_once(text: str, needle: str, replacement: str, label: str) -> str:
    if needle not in text:
        raise RuntimeError(f"Nu am găsit ancora pentru {label}.")
    return text.replace(needle, replacement, 1)


def insert_after_once(text: str, anchor: str, addition: str, label: str) -> str:
    return replace_once(text, anchor, anchor + addition, label)


def insert_before_once(text: str, anchor: str, addition: str, label: str) -> str:
    return replace_once(text, anchor, addition + "\n" + anchor, label)


def remove_between(text: str, start: str, end: str, replacement: str) -> str:
    i = text.find(start)
    j = text.find(end, i + len(start)) if i >= 0 else -1
    if i < 0 or j < 0:
        return text
    return text[:i] + replacement + text[j:]


def main() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    original = html

    if "STRUCTURAL_CONFIRMATION_URLS_START" not in html:
        html = insert_after_once(
            html,
            '    const STATE_URLS = ["coeziv_state.json", "./coeziv_state.json", "/btc-swing-strategy/coeziv_state.json", "/coeziv_state.json"];\n',
            URLS_BLOCK,
            "URL-uri confirmare structurală",
        )

    if "STRUCTURAL_CONFIRMATION_JS_START" not in html:
        html = insert_before_once(
            html,
            "    function resetDeviationStatus(text) {\n",
            JS_BLOCK,
            "funcții confirmare structurală",
        )

    if "let structuralConfirmationSummary = null;" not in html:
        html = insert_after_once(
            html,
            "        const state = await fetchJsonFallback(STATE_URLS);\n",
            LOAD_BLOCK,
            "încărcare confirmare structurală",
        )

    if "updateStructuralConfirmation(state, structuralConfirmationSummary);" not in html:
        html = insert_before_once(
            html,
            "        updateFGCard(fgBlock);\n",
            "        updateStructuralConfirmation(state, structuralConfirmationSummary);",
            "apel confirmare structurală",
        )

    # Elimină complet vechiul bloc tactic 72h, ca să nu mai poată apărea în UI.
    # Confirmarea structurală 7/30 zile este randată din JSON-ul de backtest.
    html = remove_between(
        html,
        "\n        // actualizăm textul probabilității istorice\n",
        "\n        updateStructuralConfirmation(state, structuralConfirmationSummary);\n",
        "\n        // Confirmarea structurală 7/30 zile înlocuiește vechiul bloc tactic 72h.\n",
    )

    if html != original:
        HTML_PATH.write_text(html, encoding="utf-8")
        print(f"Patch aplicat: {HTML_PATH}")
    else:
        print("Patch deja prezent; nu s-a modificat nimic.")


if __name__ == "__main__":
    main()
