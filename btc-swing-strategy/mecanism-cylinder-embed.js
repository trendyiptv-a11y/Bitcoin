(function () {
  const STYLE_ID = "cohesivx-daily-cylinder-style";
  const LANG_KEY = "coeziv_btc_lang";
  const STATE_URLS = [
    "coeziv_state.json",
    "./coeziv_state.json",
    "/btc-swing-strategy/coeziv_state.json",
    "/coeziv_state.json"
  ];
  let LAST_DATA = null;

  function lang() {
    try { return localStorage.getItem(LANG_KEY) === "en" ? "en" :