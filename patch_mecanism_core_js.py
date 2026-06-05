from pathlib import Path

HTML = Path('btc-swing-strategy/mecanism.html')

CORE = r'''
    const USD_FORMATTER = new Intl.NumberFormat("en-US", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    });

    let SNAPSHOT_PRICE = null;
    let LIVE_PRICE = null;
    let LAST_SIGNAL_RAW = null;
    let LAST_SNAPSHOT_TIME = null;
    let SIGNAL_HISTORY = [];

    let STATE_PROB = null;
    let STATE_PROB_SAMPLES = 0;
    let STATE_PROB_HORIZON = 24;
    let LAST_DEV_ABS_PCT = null;
    let LAST_DEV_SIGN = 0;

    let FLOW_BIAS = null;
    let FLOW_STRENGTH = null;
    let LIQ_REGIME = null;
    let LIQ_STRENGTH = null;

    let STATE_PROB_IN_DIR = null;
    let STATE_PROB_OPPOSITE = null;
    let STATE_PROB_FLAT = null;

    let oTimes = [];
    let oOpen = [];
    let oHigh = [];
    let oLow = [];
    let oClose = [];
    let oVolume = [];

    function mapSignal(signal) {
      const s = (signal || "").toLowerCase();
      if (s === "long") {
        return {
          tone: "bullish",
          label: "Context: presiune de creștere",
          shortLabel: "Presiune de creștere"
        };
      }
      if (s === "short") {
        return {
          tone: "bearish",
          label: "Context: risc de scădere",
          shortLabel: "Risc de scădere"
        };
      }
      return {
        tone: "neutral",
        label: "Context: neutru",
        shortLabel: "Context neutru"
      };
    }
'''

s = HTML.read_text(encoding='utf-8')

if 'const USD_FORMATTER = new Intl.NumberFormat' not in s:
    marker = '    function formatDate(iso) {'
    if marker not in s:
        raise SystemExit('Nu am găsit markerul function formatDate')
    s = s.replace(marker, CORE + '\n' + marker)
    HTML.write_text(s, encoding='utf-8')
    print('[OK] restored mecanism core JS block')
else:
    print('[OK] core JS block already present')
