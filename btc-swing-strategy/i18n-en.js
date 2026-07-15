/* CohesivX BTC — RO/EN UI layer. Energy card is owned by mecanism.html. */
(function(){
  "use strict";

  const LANG_KEY = "coeziv_btc_lang";
  const SCALE_KEY = "coeziv_btc_scale";
  const SCALE_STEPS = [1, 1.15, 1.30];
  const USD = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  let I18N_BUSY = false;
  let I18N_TIMER = null;

  const L = {
    ro: {
      app: "MECANISM COEZIV BTC",
      snap: "Actual