from pathlib import Path
import re

path = Path('btc-swing-strategy/mecanism.html')
s = path.read_text(encoding='utf-8')

css_marker = '/* COMPACT_REGIME_CHIP_CSS */'
css = '''
    /* COMPACT_REGIME_CHIP_CSS */
    #regime-line.regime-chip {
      width: 74px;
      height: 74px;
      min-width: 74px;
      max-width: 74px;
      align-self: flex-start;
      margin-top: 0;
      padding: 0;
      border-radius: 999px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      font-size: 11px;
      line-height: 1.05;
      font-weight: 850;
      letter-spacing: .08em;
      text-transform: uppercase;
      white-space: pre-line;
      overflow: hidden;
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
        width: 66px;
        height: 66px;
        min-width: 66px;
        max-width: 66px;
        font-size: 10px;
      }
    }
'''
# Replace old compact CSS block if present, otherwise insert.
css_pattern = r"\n\s*/\* COMPACT_REGIME_CHIP_CSS \*/\n\s*#regime-line\.regime-chip \{.*?\n\s*\}\n\s*body\.light-mode #regime-line\.regime-chip \{.*?\n\s*\}\n\s*@media \(max-width: 480px\) \{\n\s*#regime-line\.regime-chip \{.*?\n\s*\}\n\s*\}\n"
if css_marker in s:
    s = re.sub(css_pattern, '\n' + css, s, count=1, flags=re.S)
else:
    s = s.replace('  </style>', css + '\n  </style>')

html_old = '<div id="regime-line" class="live-delta">'
html_new = '<div id="regime-line" class="live-delta regime-chip" title="Regim de piață: n/a">'
s = s.replace(html_old, html_new)

js_marker = '// COMPACT_REGIME_CHIP_JS'
js = '''
    // COMPACT_REGIME_CHIP_JS
    function compactRegimeLabel(fullText) {
      const t = String(fullText || '').toLowerCase();
      if (t.includes('ascendent')) return 'TREND+';
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
      const current = el.textContent || '';
      const full = el.getAttribute('data-full-regime') || current;
      const compact = compactRegimeLabel(full);
      if (current !== compact) {
        el.setAttribute('data-full-regime', full);
        el.title = full;
        el.textContent = compact;
      }
    }
    function startCompactRegimeChipSync() {
      applyCompactRegimeChip();
      const el = document.getElementById('regime-line');
      if (!el) return;
      const observer = new MutationObserver(function(){
        window.requestAnimationFrame(applyCompactRegimeChip);
      });
      observer.observe(el, { childList: true, characterData: true, subtree: true });
      setInterval(applyCompactRegimeChip, 700);
    }
'''
pattern = r"\n\s*// COMPACT_REGIME_CHIP_JS\n\s*function compactRegimeLabel\(fullText\) \{.*?\n\s*function applyCompactRegimeChip\(\) \{.*?\n\s*\}\n(?:\s*function startCompactRegimeChipSync\(\) \{.*?\n\s*\}\n)?"
if js_marker in s:
    s = re.sub(pattern, '\n' + js, s, count=1, flags=re.S)
else:
    s = s.replace('    const STATE_URL = "coeziv_state.json";', js + '\n    const STATE_URL = "coeziv_state.json";')

# Replace old direct call with sync call.
s = s.replace('initThemeToggle();\n    applyCompactRegimeChip();', 'initThemeToggle();\n    startCompactRegimeChipSync();')
if 'startCompactRegimeChipSync();' not in s:
    s = s.replace('initThemeToggle();', 'initThemeToggle();\n    startCompactRegimeChipSync();')

path.write_text(s, encoding='utf-8')
