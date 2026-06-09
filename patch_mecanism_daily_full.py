#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

HTML = Path('btc-swing-strategy/mecanism.html')

CSS_MARKER = '/* DAILY_FULL_INTERPRETATION_EXPANDER_CSS */'
HTML_MARKER_START = '<!-- DAILY_FULL_INTERPRETATION_EXPANDER_START -->'
HTML_MARKER_END = '<!-- DAILY_FULL_INTERPRETATION_EXPANDER_END -->'
JS_MARKER = '// DAILY_FULL_INTERPRETATION_EXPANDER_JS'

CSS = r'''

    /* DAILY_FULL_INTERPRETATION_EXPANDER_CSS */
    .daily-full-toggle {
      margin: 12px 0 0;
      width: 100%;
      border: 1px solid rgba(56,189,248,0.45);
      background: rgba(56,189,248,0.08);
      color: #bae6fd;
      border-radius: 999px;
      padding: 9px 12px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
      cursor: pointer;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .daily-full-toggle:hover {
      border-color: rgba(56,189,248,0.75);
      background: rgba(56,189,248,0.14);
    }
    .daily-full-panel {
      display: none;
      margin-top: 10px;
      border-radius: 16px;
      border: 1px solid rgba(56,189,248,0.26);
      background: rgba(15,23,42,0.62);
      padding: 12px 13px;
      font-size: 13px;
      line-height: 1.58;
      color: var(--text-main);
      white-space: pre-wrap;
    }
    .daily-full-panel.open {
      display: block;
    }
    .daily-full-panel strong,
    .daily-full-panel b {
      font-weight: 750;
    }
    .daily-full-panel h2,
    .daily-full-panel h3 {
      margin: 12px 0 7px;
      font-size: 13px;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--text-muted);
    }
    body.light-mode .daily-full-toggle {
      background: rgba(14,165,233,0.08);
      color: #075985;
      border-color: rgba(14,165,233,0.38);
    }
    body.light-mode .daily-full-panel {
      background: rgba(248,250,252,0.88);
      border-color: rgba(14,165,233,0.25);
    }
'''

HTML_BLOCK = r'''
        <!-- DAILY_FULL_INTERPRETATION_EXPANDER_START -->
        <button id="daily-ai-full-toggle" class="daily-full-toggle" type="button" aria-expanded="false">
          Vezi interpretarea completă
        </button>
        <div id="daily-ai-full-panel" class="daily-full-panel" aria-hidden="true"></div>
        <!-- DAILY_FULL_INTERPRETATION_EXPANDER_END -->
'''

JS = r'''

    // DAILY_FULL_INTERPRETATION_EXPANDER_JS
    function formatDailyFullInterpretation(raw) {
      let text = String(raw || '').trim();
      if (!text) return '';
      text = text.replace(/^##\s+/gm, '');
      text = text.replace(/^###\s+/gm, '');
      text = text.replace(/\*\*(.*?)\*\*/g, '$1');
      return text;
    }

    function updateDailyFullInterpretation(data) {
      const btn = document.getElementById('daily-ai-full-toggle');
      const panel = document.getElementById('daily-ai-full-panel');
      if (!btn || !panel) return;

      const fullText = formatDailyFullInterpretation(data && data.full_interpretation);
      if (!fullText) {
        btn.style.display = 'none';
        panel.style.display = 'none';
        panel.textContent = '';
        return;
      }

      btn.style.display = 'block';
      panel.style.display = '';
      panel.textContent = fullText;

      if (!btn.dataset.bound) {
        btn.dataset.bound = '1';
        btn.addEventListener('click', () => {
          const isOpen = panel.classList.toggle('open');
          btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
          panel.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
          btn.textContent = isOpen ? 'Ascunde interpretarea completă' : 'Vezi interpretarea completă';
        });
      }
    }
'''


def replace_between(text, start, end, replacement):
    if start in text and end in text:
        a = text.index(start)
        b = text.index(end, a) + len(end)
        return text[:a] + replacement.strip('\n') + text[b:]
    return text


def main():
    s = HTML.read_text(encoding='utf-8')

    if CSS_MARKER not in s:
        s = s.replace('    /* COHESIVX_THEME_TOGGLE_CSS */', CSS + '\n    /* COHESIVX_THEME_TOGGLE_CSS */')

    if HTML_MARKER_START in s and HTML_MARKER_END in s:
        s = replace_between(s, HTML_MARKER_START, HTML_MARKER_END, HTML_BLOCK)
    elif '<div id="daily-ai-footer" class="daily-ai-footer">' in s:
        footer_start = s.index('<div id="daily-ai-footer" class="daily-ai-footer">')
        footer_end = s.index('</div>', footer_start) + len('</div>')
        s = s[:footer_end] + '\n' + HTML_BLOCK.rstrip('\n') + s[footer_end:]
    else:
        s = s.replace('        <div class="daily-ai-grid">', HTML_BLOCK + '\n        <div class="daily-ai-grid">')

    if JS_MARKER not in s:
        s = s.replace('    // COMPACT_REGIME_CHIP_JS', JS + '\n    // COMPACT_REGIME_CHIP_JS')

    call = '      updateDailyFullInterpretation(data);'
    if call not in s:
        s = s.replace('      setText("daily-ai-footer", data.disclaimer || "Interpretare structurală experimentală, nu recomandare financiară.");',
                      '      setText("daily-ai-footer", data.disclaimer || "Interpretare structurală experimentală, nu recomandare financiară.");\n' + call)

    fallback_call = '        updateDailyFullInterpretation(null);'
    if fallback_call not in s:
        s = s.replace('        setText("daily-ai-summary", "Cardul va fi completat după prima analiză zilnică aprobată.");\n        return;',
                      '        setText("daily-ai-summary", "Cardul va fi completat după prima analiză zilnică aprobată.");\n' + fallback_call + '\n        return;')

    HTML.write_text(s, encoding='utf-8')
    print('[OK] mecanism.html patched with daily full interpretation expander')


if __name__ == '__main__':
    main()
