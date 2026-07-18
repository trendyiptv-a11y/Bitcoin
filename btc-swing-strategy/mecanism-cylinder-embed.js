(function () {
  'use strict';

  const STYLE_ID = 'cohesivx-daily-cylinder-style';
  const LANG_KEY = 'coeziv_btc_lang';
  let LAST_DATA = null;

  function lang() {
    try { return localStorage.getItem(LANG_KEY) === 'en' ? 'en' : 'ro'; }
    catch (_) { return 'ro'; }
  }

  const TRADER_T = {
    ro: {
      signal: 'SEMNAL', asset: 'BTC', flow: 'Flux', participation: 'Participare', liquidity: 'Lichiditate', growth: 'Confirmare creștere', yes: 'da', no: 'nu