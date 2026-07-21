(function () {
  const STYLE_ID = "cohesivx-daily-cylinder-style";
  const LANG_KEY = "coeziv_btc_lang";
  let LAST_DATA = null;

  function lang() {
    try { return localStorage.getItem(LANG_KEY) === "en" ? "en" : "ro"; }
    catch (_) { return "ro"; }
  }

  const T = {
    ro: {
      context: "CONTEXT", price: "PREȚ", participation