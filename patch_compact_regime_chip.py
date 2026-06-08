from pathlib import Path
import re

path = Path('btc-swing-strategy/mecanism.html')
s = path.read_text(encoding='utf-8')

css_marker = '/* COMPACT_REGIME_CHIP_CSS */'
css = '''
    /* COMPACT_REGIME_CHIP_CSS */
    #regime-line.regime-chip {
      width: auto;
      max-width: 82px;
      min-width: 58px;
      align-self: flex-start;
      margin-top: 0;
      padding: 6px 8px;
      border-radius: 999px;
      text-align: center;
      font-size: 10px;
      line-height: 1.15;
      font-weight: 750;
      letter-spacing: .08em;
      text-transform: uppercase;
      white-space: normal;
      overflow-wrap: normal;
      word-break: normal;
      color: #bae6fd;
      background: rgba(15,23,42,0.78);
      border: 1px solid rgba(56,189,248,0.28);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 0 18px rgba(56,189,248,0.08);
    }
    body.light-mode #regime-line.regime-chip {
      color: #075985;
      background: rgba(248,250,252,0.88);
      border-color: rgba(14,165,233,0.32);
    }
    @media (max-width: 480px) {
      #regime-line.regime-chip {
        max-width: 74px;
        min-width: 56px;
        padding: 5px 7px;
        font-size: 9px;
      }
    }
'''
if css_marker not in s:
    s = s.replace('  </style>', css + '\n  </style>')

html_old = '<div id="regime-line" class="live-delta">'
html_new = '<div id="regime-line" class="live-delta regime-chip" title="Regim de piață: n/a">'
s = s.replace(html_old, html_new)

js_marker = '// COMPACT_REGIME_CHIP_JS'
js = '''
    // COMPACT_REGIME_CHIP_JS
    function compactRegimeLabel(fullText) {
      const t = String(fullText || '').toLowerCase();
      if (t.includes('ascendent') && t.includes('susținut')) return 'TREND+';
      if (t.includes('ascendent')) return 'TREND+';
      if (t.includes('descendent') && t.includes('susținut')) return 'TREND−';
      if (t.includes('descendent')) return 'TREND−';
      if (t.includes('range') && t.includes('pozitiv')) return 'RANGE+';
      if (t.includes('range') && t.includes('negativ')) return 'RANGE−';
      if (t.includes('range')) return 'RANGE';
      if (t.includes('tranzi')) return 'TRANZIȚIE';
      if (t.includes('neutru')) return 'NEUTRU';
      if (t.includes('n/a')) return 'N/A';
      return 'REGIM';
    }
    function applyCompactRegimeChip() {
      const el = document.getElementById('regime-line');
      if (!el) return;
      el.classList.add('regime-chip');
      const full = el.getAttribute('data-full-regime') || el.textContent || '';
      el.title = full;
      el.textContent = compactRegimeLabel(full);
    }
'''
pattern = r"\n\s*// COMPACT_REGIME_CHIP_JS\n\s*function compactRegimeLabel\(fullText\) \{.*?\n\s*function applyCompactRegimeChip\(\) \{.*?\n\s*\}\n"
if js_marker in s:
    s = re.sub(pattern, '\n' + js, s, count=1, flags=re.S)
else:
    s = s.replace('    const STATE_URL = "coeziv_state.json";', js + '\n    const STATE_URL = "coeziv_state.json";')

# Add sync calls after known updates; safe if already applied.
s = s.replace('if (REGIME_EL) REGIME_EL.textContent = `Regim de piață: ${regime}`;', "if (REGIME_EL) { REGIME_EL.textContent = `Regim de piață: ${regime}`; REGIME_EL.setAttribute('data-full-regime', `Regim de piață: ${regime}`); applyCompactRegimeChip(); }")
s = s.replace('if (REGIME_EL) REGIME_EL.textContent = "Regim de piață: n/a (așteptăm date suficiente din mecanism).";', "if (REGIME_EL) { REGIME_EL.textContent = 'Regim de piață: n/a (așteptăm date suficiente din mecanism).'; REGIME_EL.setAttribute('data-full-regime', REGIME_EL.textContent); applyCompactRegimeChip(); }")

if 'initThemeToggle();\n    applyCompactRegimeChip();' not in s:
    s = s.replace('initThemeToggle();', 'initThemeToggle();\n    applyCompactRegimeChip();')

path.write_text(s, encoding='utf-8')
