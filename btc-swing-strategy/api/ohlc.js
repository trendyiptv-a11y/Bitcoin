export default async function handler(req, res) {
  try {
    const r = await fetch(
      "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=288"
    );
    const data = await r.json();

    res.setHeader("Access-Control-Allow-Origin", "*");
    res.status(200).json(data);
  } catch (e) {
    res.status(500).json({ error: "Binance fetch error" });
  }
}
