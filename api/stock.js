// Vercel Serverless Function：/api/stock?ticker=2330.TW&range=5y
// 代理 Yahoo Finance 日線資料並轉成 CSV（瀏覽器不能直接跨網域讀 Yahoo，所以由伺服器端代抓）
const RANGES = new Set(["6mo", "1y", "2y", "5y", "10y", "max"]);

async function fetchStockCsv(ticker, range) {
  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}` +
    `?range=${range}&interval=1d&includeAdjustedClose=true`;
  const r = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0" } });
  const j = await r.json().catch(() => null);
  const result = j?.chart?.result?.[0];
  if (!r.ok || !result || !result.timestamp) {
    const err = new Error(j?.chart?.error?.description || `Yahoo Finance 回應 HTTP ${r.status}`);
    err.status = r.status === 404 ? 404 : 502;
    throw err;
  }
  const q = result.indicators?.quote?.[0] || {};
  const adj = result.indicators?.adjclose?.[0]?.adjclose || [];
  const offset = result.meta?.gmtoffset || 0;
  const lines = ["Date,Open,High,Low,Close,Adj Close,Volume"];
  result.timestamp.forEach((t, i) => {
    if (q.close?.[i] == null) return;
    const date = new Date((t + offset) * 1000).toISOString().slice(0, 10);
    lines.push([date, q.open[i], q.high[i], q.low[i], q.close[i], adj[i] ?? q.close[i], q.volume?.[i] ?? ""].join(","));
  });
  return {
    csv: lines.join("\n") + "\n",
    name: result.meta?.longName || result.meta?.shortName || ticker,
    currency: result.meta?.currency || "",
  };
}

module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Expose-Headers", "X-Stock-Name, X-Stock-Currency");
  const ticker = String(req.query.ticker || "2330.TW").trim().toUpperCase();
  const range = String(req.query.range || "5y");
  if (!/^[A-Z0-9.\-^=]{1,20}$/.test(ticker)) return res.status(400).json({ error: "股票代號格式錯誤" });
  if (!RANGES.has(range)) return res.status(400).json({ error: "range 參數錯誤" });
  try {
    const { csv, name, currency } = await fetchStockCsv(ticker, range);
    res.setHeader("Content-Type", "text/csv; charset=utf-8");
    res.setHeader("Cache-Control", "s-maxage=3600, stale-while-revalidate=86400");
    res.setHeader("X-Stock-Name", encodeURIComponent(name));
    res.setHeader("X-Stock-Currency", currency);
    res.status(200).send(csv);
  } catch (e) {
    res.status(e.status || 502).json({ error: e.message });
  }
};
module.exports.fetchStockCsv = fetchStockCsv;
