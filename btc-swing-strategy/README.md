
# BTC Swing Trading Strategy

Acest repo conține un framework modular pentru o strategie de swing trading pe Bitcoin, construită pe baza:

- identificării regimului macro al pieței (A / B / C / D),
- stabilirii unui bias direcțional (LONG_ONLY, SHORT_TACTIC etc.),
- detectării setup-urilor de intrare (impuls → retragere → reconfirmare),
- gestionării pozițiilor prin TP-uri parțiale și trailing stop,
- controlului riscului prin position sizing dinamic.

Strategia rulează pe timeframe 4h–1D.

## Structura repo-ului
