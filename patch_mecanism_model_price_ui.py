#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "btc-swing-strategy" / "mecanism.html"

CSS_BLOCK = r'''
    /* STRUCTURAL_CONFIRMATION_CARD_CSS_START */
    .structural-confirmation-card {
      margin-top: 18px;
    }

    .structural-confirmation-title {
      font-size: 12px;
      letter-spacing: .18em;
      text-transform: uppercase;
      color: var(--text-soft);
      margin-bottom: 10px;
    }

    .structural-confirmation-main {
      padding: 12px 13px;
      border-radius: 16px;
      border: 1px solid rgba(56, 189, 248, .24);
      background: rgba(56, 189, 248, .07);
      font-size: 13px;
      line-height: 1.58;
      font-weight: 650;
      color: var(--text-main);
    }

    .structural-confirmation-base {
      margin-top: 10px;
      font-size: 11px;
      line-height: 1.45;
      color: var(--text-soft);
    }

    .structural-confirmation-context {
      margin-top: 10px;
      font-size: 13px;
      line-height: 1.58;
      color: var(--text-main);
    }

    body.light-mode .structural-confirmation-main {
      background: rgba(14, 165, 233, .08);
      border-color: rgba(14, 165, 233, .26);
    }
    /* STRUCTURAL_CONFIRMATION_CARD_CSS_END */
'''

HTML_BLOCK = r'''
          <!-- STRUCTURAL_CONFIRMATION_CARD_START -->
          <div class="card card-secondary structural-confirmation-card" id="structural-confirmation-card">
            <div class="card-inner">
              <div class="structural-confirmation-title">Confirmare structurală</div>

              <div id="structural-confirmation-main" class="structural-confirmation-main">
                Se încarcă confirmarea structurală...
              </div>

              <div id="structural-confirmation-base" class="structural-confirmation-base">
                Bază statistică: se actualizează.
              </div>

              <div id="structural-confirmation-context" class="structural-confirmation-context">
                Context structural curent: se actualizează.
              </div>
            </div>
          </div>
          <!-- STRUCTURAL_CONFIRMATION_CARD_END -->
'''

JS_BLOCK = r'''
    // STRUCTURAL_CONFIRMATION_JS_START
    function structuralPctDisplay(value, digits = 0) {
      const n = Number(value);
      if (!Number.isFinite(n)) return null;
      const normalized = Math.abs(n) <= 1 ? n * 100 : n;
      return `~${normalized.toFixed(digits)}%`;
    }

    function structuralWhole(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return null;
      return n.toFixed(0);
    }

    function structuralContextLabel(regime) {
      const key = String(regime || "").toLowerCase();
      const labels = {
        bear_late: "presiune descendentă aflată în fază matură, cu semne de stabilizare",
        bear_early: "presiune descendentă aflată în fază activă",
        bull_early: "reluare pozitivă timpurie, încă în confirmare",
        bull_late: "impuls pozitiv matur, cu risc de epuizare",
        range: "zonă de echilibru relativ, fără direcție structurală dominantă",
        neutral: "zonă neutră, cu direcție structurală insuficient confirmată"
      };
      return labels[key] || "context coeziv curent în evaluare structurală";
    }

    function getStructuralConfirmation(state) {
      return state && state.structural_confirmation ? state.structural_confirmation : null;
    }

    function updateStructuralConfirmation(state) {
      const mainEl = document.getElementById("structural-confirmation-main");
      const baseEl = document.getElementById("structural-confirmation-base");
      const contextEl = document.getElementById("structural-confirmation-context");
      if (!mainEl || !baseEl || !contextEl) return;

      const structural = getStructuralConfirmation(state);
      if (!structural) {
        mainEl.textContent = "Confirmarea structurală va apărea după următorul snapshot coeziv.";
        baseEl.textContent = "Bază statistică: așteptăm state.structural_confirmation din coeziv_state.json.";
        contextEl.textContent = "Context structural curent: indisponibil momentan. Semnalul este un reper de structură, nu o recomandare de tranzacționare.";
        return;
      }

      const h7 = structural.horizon_7d || {};
      const h30 = structural.horizon_30d || {};
      const pct7 = structuralPctDisplay(h7.directional_hit_rate, 0);
      const pct30 = structuralPctDisplay(h30.directional_hit_rate, 0);
      const events = structuralWhole(h30.events ?? h7.events);
      const samples = structuralWhole(
        structural.similar_context_samples ??
        (state && state.model_price_components && state.model_price_components.similar_context_samples)
      );
      const thresholdLabel = structural.threshold_label || "deviații ample față de reperul coeziv";
      const regime = structural.regime || (state && state.model_price_context && state.model_price_context.regime);
      const contextLabel = structural.context_label || structuralContextLabel(regime);

      if (pct7 && pct30) {
        mainEl.textContent =
          `În contexte istorice similare, mecanismul a confirmat direcția structurală în ${pct7} din cazuri pe 7 zile și ${pct30} din cazuri pe 30 zile.`;
      } else {
        mainEl.textContent = "Confirmarea structurală se actualizează după următorul backtest al mecanismului.";
      }

      if (events) {
        baseEl.textContent =
          `Bază statistică: ${events} evenimente istorice cu ${thresholdLabel}.`;
      } else {
        baseEl.textContent = "Bază statistică: evenimentele istorice se actualizează.";
      }

      const sampleSentence = samples
        ? `Analiza folosește ${samples} contexte istorice similare.`
        : "Analiza folosește contexte istorice similare.";
      contextEl.textContent =
        `Context structural curent: ${contextLabel}. ${sampleSentence} Semnalul este un reper de structură, nu o recomandare de tranzacționare.`;
    }
    // STRUCTURAL_CONFIRMATION_JS_END
'''


def replace_once(text: str, needle: str, replacement: str, label: str) -> str:
    if needle not in text:
        raise RuntimeError(f"Nu am găsit ancora pentru {label}.")
    return text.replace(needle, replacement, 1)


def insert_before_once(text: str, anchor: str, addition: str, label: str) -> str:
    return replace_once(text, anchor, addition + "\n" + anchor, label)


def insert_after_once(text: str, anchor: str, addition: str, label: str) -> str:
    return replace_once(text, anchor, anchor + addition, label)


def remove_marked_block(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
      return text
    line_start = text.rfind("\n", 0, start)
    if line_start < 0:
      line_start = 0
    end = text.find(end_marker, start)
    if end < 0:
      return text
    line_end = text.find("\n", end + len(end_marker))
    if line_end < 0:
      line_end = end + len(end_marker)
    return text[:line_start] + text[line_end:]


def remove_legacy_summary_loader(text: str) -> str:
    marker = "let structuralConfirmationSummary = null;"
    start = text.find(marker)
    if start < 0:
        return text
    line_start = text.rfind("\n", 0, start)
    if line_start < 0:
        line_start = start
    end_marker = "structuralConfirmationSummary = null;"
    end = text.find(end_marker, start)
    if end < 0:
        return text
    brace_end = text.find("\n", end + len(end_marker))
    if brace_end < 0:
        brace_end = end + len(end_marker)
    return text[:line_start] + text[brace_end:]


def remove_legacy_tactical_block(text: str) -> str:
    start = text.find("\n        // actualizăm textul probabilității istorice\n")
    if start < 0:
        return text
    end_call = text.find("updateStructuralConfirmation", start)
    if end_call < 0:
        return text
    line_end = text.find("\n", end_call)
    if line_end < 0:
        line_end = end_call
    return text[:start] + "\n        // Confirmarea structurală 7/30 zile înlocuiește vechiul bloc tactic 72h.\n" + text[line_end:]


def main() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    original = html

    # Curățăm încercările vechi: fără fetch separat la comparative_backtest_summary.json în browser.
    html = remove_marked_block(html, "STRUCTURAL_CONFIRMATION_URLS_START", "STRUCTURAL_CONFIRMATION_URLS_END")
    html = remove_marked_block(html, "STRUCTURAL_CONFIRMATION_JS_START", "STRUCTURAL_CONFIRMATION_JS_END")
    html = remove_marked_block(html, "STRUCTURAL_CONFIRMATION_CARD_CSS_START", "STRUCTURAL_CONFIRMATION_CARD_CSS_END")
    html = remove_marked_block(html, "STRUCTURAL_CONFIRMATION_CARD_START", "STRUCTURAL_CONFIRMATION_CARD_END")
    html = remove_legacy_summary_loader(html)
    html = remove_legacy_tactical_block(html)
    html = html.replace("updateStructuralConfirmation(state, structuralConfirmationSummary);", "updateStructuralConfirmation(state);")

    if "STRUCTURAL_CONFIRMATION_CARD_CSS_START" not in html:
        html = insert_before_once(html, "  </style>", CSS_BLOCK, "CSS card confirmare structurală")

    if "STRUCTURAL_CONFIRMATION_CARD_START" not in html:
        anchors = [
            "          <!-- CARD FEAR & GREED COEZIV -->",
            "          <!-- FEAR_GREED_CARD_START -->",
            "          <!-- HISTORY_CARD_START -->",
        ]
        for anchor in anchors:
            if anchor in html:
                html = insert_before_once(html, anchor, HTML_BLOCK, "card confirmare structurală")
                break
        else:
            raise RuntimeError("Nu am găsit ancora pentru cardul Confirmare structurală.")

    if "STRUCTURAL_CONFIRMATION_JS_START" not in html:
        html = insert_before_once(
            html,
            "    function resetDeviationStatus(text) {\n",
            JS_BLOCK,
            "JS confirmare structurală",
        )

    if "updateStructuralConfirmation(state);" not in html:
        html = insert_before_once(
            html,
            "        updateFGCard(fgBlock);\n",
            "        updateStructuralConfirmation(state);",
            "apel confirmare structurală",
        )

    if html != original:
        HTML_PATH.write_text(html, encoding="utf-8")
        print(f"Patch aplicat: {HTML_PATH}")
        print("Cardul citește exclusiv state.structural_confirmation din coeziv_state.json.")
    else:
        print("Patch deja prezent; nu s-a modificat nimic.")


if __name__ == "__main__":
    main()
