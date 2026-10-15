// 問題 ②：自迴歸 Auto Regression（股價時間序列）—— 計算流程 + CRISP-DM 六個分頁
// AR(p)：y_t = c + φ1·y_{t−1} + … + φp·y_{t−p} + ε_t，等於把「落後值」當特徵的多元線性回歸
import * as S from "./stats.js";
import { buildSeries, nextBusinessDays } from "./data.js";
import { esc, fmt, fmtP, pEq, pct, table, card, chart, badge, kpiRow, plot, band, theme, toCSV, download } from "./ui.js";

export const LAG_MODES = { aic: "自動：AIC 最小", bic: "自動：BIC 最小", manual: "手動指定 p" };

const cache = { key: null };

export function runAR(ds, st) {
  const key = [ds.key, st.dateCol, st.valueCol].join("§");
  if (cache.key !== key) {
    const series = buildSeries(ds.rows, st.dateCol, st.valueCol);
    if (series.y.length < 40) throw new Error(`「${st.valueCol}」的有效資料只有 ${series.y.length} 筆，時間序列至少需要 40 筆`);
    Object.assign(cache, { key, series, scan: {} });
  }
  const { series } = cache;
  const y = series.y;
  const n = y.length;
  // 有 0 或負值（例如氣溫）時無法取 log，自動退回原始尺度
  const useLog = st.scale === "log" && y.every((v) => v > 0);
  const logFallback = st.scale === "log" && !useLog;
  // z：實際建模的尺度（原始價格或 log 價格）；預測結果最後都轉回原始價格
  const z = useLog ? y.map(Math.log) : y;
  const inv = useLog ? Math.exp : (v) => v;
  const back = (o) => Object.fromEntries(Object.entries(o).map(([k, a]) => [k, k.startsWith("se") ? a : a.map(inv)]));
  const nTest = Math.max(5, Math.round(n * st.testRatio));
  const nTrain = n - nTest;
  const maxLag = Math.max(1, Math.min(st.maxLag, Math.floor(nTrain / 5)));
  const scan = (cache.scan[`${st.scale}|${nTrain}|${maxLag}`] ??= S.lagScan(z, nTrain, maxLag));
  const argmin = (k) => scan.reduce((b, r) => (r[k] < b[k] ? r : b)).p;
  const p = st.lagMode === "aic" ? argmin("aic") : st.lagMode === "bic" ? argmin("bic") : Math.min(Math.max(1, st.pManual), maxLag);
  const sym = useLog ? "log y" : "y";
  const names = Array.from({ length: p }, (_, i) => `${sym}(t−${i + 1})`);
  const lagCols = (from, to) => Array.from({ length: p }, (_, i) => z.subarray(from - i - 1, to - i - 1));
  const alpha = 1 - st.conf / 100;
  const model = S.ols(lagCols(p, nTrain), z.subarray(p, nTrain), names);
  const pred = back(model.predict(lagCols(nTrain, n), alpha));
  const yTest = y.subarray(nTrain);
  const prev = y.subarray(nTrain - 1, n - 1);
  let covered = 0;
  yTest.forEach((v, i) => {
    if (v >= pred.piLo[i] && v <= pred.piHi[i]) covered++;
  });
  // 部署用：以全部資料重新配適，再往後遞迴預測
  const full = S.ols(lagCols(p, n), z.subarray(p, n), names);
  const fc = back(S.arForecast(full, z, p, st.horizon, alpha));
  return {
    ds, st, series, y, n, nTrain, nTest, maxLag, scan, p, model, pred, alpha, yTest, prev, full, fc, useLog, logFallback, sym,
    metrics: arMetrics(yTest, pred.mean, prev, p),
    naive: arMetrics(yTest, prev, prev, 0),
    trainM: S.regMetrics(y.subarray(p, nTrain), model.fitted.map(inv), p),
    coverage: covered / nTest,
    futureDates: series.hasDates ? nextBusinessDays(series.times[n - 1], st.horizon) : Array.from({ length: st.horizon }, (_, i) => n + i + 1),
    lb: S.ljungBox(model.resid, Math.max(1, Math.min(10, Math.floor(model.n / 5)))),
  };
}

function arMetrics(y, yhat, prev, k) {
  const base = S.regMetrics(y, yhat, k);
  let ape = 0;
  let ok = 0;
  let cnt = 0;
  for (let i = 0; i < y.length; i++) {
    ape += Math.abs((y[i] - yhat[i]) / y[i]);
    const a = Math.sign(y[i] - prev[i]);
    const b = Math.sign(yhat[i] - prev[i]);
    if (a !== 0 && b !== 0) {
      cnt++;
      if (a === b) ok++;
    }
  }
  return { ...base, MAPE: ape / y.length, dirAcc: cnt ? ok / cnt : NaN };
}

// ---------- 畫面 ----------
export function arKpis(R) {
  const better = R.metrics.RMSE < R.naive.RMSE;
  return [
    { label: "測試期 RMSE", value: fmt(R.metrics.RMSE), sub: `Naive ${fmt(R.naive.RMSE)}${better ? "（AR 較佳）" : "（Naive 較佳）"}` },
    { label: "測試期 MAPE", value: pct(R.metrics.MAPE, 2), sub: `R² ${fmt(R.metrics.R2)}` },
    { label: "漲跌方向準確率", value: pct(R.metrics.dirAcc), sub: "預測漲跌與實際相同的比例" },
    { label: `下一交易日預測（${esc(R.futureDates[0])}）`, value: fmt(R.fc.mean[0]), sub: `${R.st.conf}% PI ${fmt(R.fc.lo[0])} ～ ${fmt(R.fc.hi[0])}` },
  ];
}

const link = (u) => (/^https?:\/\//.test(u) ? `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(u)}</a>` : esc(u));
const DISCLAIMER = `<p class="alert inline">股價接近隨機漫步，本頁模型僅供課堂學習線性回歸與時間序列，<b>不構成任何投資建議</b>。</p>`;

export function renderAR(tab, R, root) {
  ({ business: renderBusiness, data: renderData, prep: renderPrep, model: renderModel, eval: renderEval, deploy: renderDeploy })[tab](R, root);
}

function renderBusiness(R, root) {
  const v = esc(R.st.valueCol);
  root.innerHTML =
    card(
      "Business Understanding 商業理解",
      `<dl class="facts">
        <dt>要解決的問題</dt><dd>用過去 p 個交易日的「${v}」預測<b>下一個交易日</b>的「${v}」，並往後預測 ${R.st.horizon} 個交易日、附上預測區間。</dd>
        <dt>資料集來源</dt><dd>${esc(R.ds.name)}<br>${link(R.ds.source)}</dd>
        <dt>模型</dt><dd>自迴歸 AR(p)：<code>y<sub>t</sub> = c + φ<sub>1</sub>y<sub>t−1</sub> + … + φ<sub>p</sub>y<sub>t−p</sub> + ε<sub>t</sub></code>。本質上就是把「前幾天的價格」當成特徵的<b>多元線性回歸</b>，因此可以沿用 OLS、信賴區間與預測區間。</dd>
        <dt>特徵選擇</dt><dd>要用幾天的落後值（p）就是特徵選擇問題：以 AIC／BIC 在訓練期挑選，或手動指定。</dd>
        <dt>建模尺度</dt><dd>${R.useLog ? "對數價格 log(y)：誤差以百分比計，適合價格漲跌幅度大的期間" : "原始價格（可在左側改成對數價格）"}</dd>
        <dt>使用者與決策</dt><dd>學生／投資人了解價格的慣性與不確定性：預測區間可以用來評估合理價格範圍與風險，而不是當作買賣訊號。</dd>
        <dt>成功標準</dt><dd>測試期的 RMSE 要比 <b>Naive baseline（明天價格 = 今天價格）</b> 低；${R.st.conf}% 預測區間的涵蓋率接近 ${R.st.conf}%。</dd>
      </dl>${DISCLAIMER}`,
    ) +
    card(
      "CRISP-DM 六個步驟在本頁的對應",
      `<ol class="steps">
        <li><b>Business Understanding</b>：定義預測目標、資料來源與成功標準（本頁）</li>
        <li><b>Data Understanding</b>：價格走勢、敘述統計、ACF／PACF 自我相關</li>
        <li><b>Data Preparation</b>：依日期排序、去除缺失與重複、建立落後特徵、依時間切分</li>
        <li><b>Modeling</b>：AIC／BIC 選擇落後期數 p、OLS 估計 AR 係數</li>
        <li><b>Evaluation</b>：與 Naive 比較 MAE／RMSE／MAPE、預測圖＋信賴區間／預測區間、殘差檢定</li>
        <li><b>Deployment</b>：未來 ${R.st.horizon} 個交易日預測與下載</li>
      </ol>`,
    ) +
    card(
      "換成你自己的資料",
      `<p class="prose">左側「資料來源」可以輸入任何股票代號（例如 <code>2330.TW</code>、<code>0050.TW</code>、<code>AAPL</code>，Vercel 版提供），或貼上任何含有日期與數值欄位的 CSV 網址、上傳 CSV。不限股價，氣溫、用電量、銷售量等時間序列都可以。</p>`,
    );
}

function ohlc(R) {
  const prof = R.ds.profile.map((c) => c.name);
  const find = (re) => prof.find((n) => re.test(n));
  const cols = { open: find(/(^|\.)open$/i), high: find(/(^|\.)high$/i), low: find(/(^|\.)low$/i), close: find(/(^|\.)close$/i) };
  if (!R.series.hasDates || Object.values(cols).some((c) => !c)) return null;
  const s = Object.fromEntries(Object.entries(cols).map(([k, c]) => [k, buildSeries(R.ds.rows, R.st.dateCol, c)]));
  const len = s.close.y.length;
  if (Object.values(s).some((x) => x.y.length !== len)) return null;
  return { dates: s.close.dates, open: s.open.y, high: s.high.y, low: s.low.y, close: s.close.y };
}

function renderData(R, root) {
  const { series, y, st } = R;
  const ret = Array.from({ length: y.length - 1 }, (_, i) => y[i + 1] / y[i] - 1);
  const nl = Math.min(30, Math.floor(y.length / 4));
  const ac = S.acf(y, nl);
  const pac = S.pacf(y, nl);
  const bound = 1.96 / Math.sqrt(y.length);
  const candle = ohlc(R);
  root.innerHTML =
    `<div class="kpis mini">${kpiRow([
      { label: "資料筆數", value: y.length.toLocaleString() },
      { label: "期間", value: series.hasDates ? `${series.dates[0]}` : "—", sub: series.hasDates ? `至 ${series.dates[y.length - 1]}` : "依列順序" },
      { label: "最新值", value: fmt(y[y.length - 1]) },
      { label: "最高 / 最低", value: fmt(Math.max(...y)), sub: `最低 ${fmt(Math.min(...y))}` },
      { label: "日報酬標準差", value: pct(S.sd(ret), 2), sub: `年化約 ${pct(S.sd(ret) * Math.sqrt(252), 1)}` },
    ])}</div>` +
    card(candle ? "價格走勢（K 線）" : `「${esc(st.valueCol)}」走勢`, chart("c-price", "tall"), { sub: "可拖曳縮放；下方滑桿可選擇期間" }) +
    `<div class="grid-2">${card("ACF 自我相關函數", chart("c-acf"), { sub: "緩慢遞減 → 價格具有強烈慣性（非定態）" })}${card("PACF 偏自我相關函數", chart("c-pacf"), {
      sub: "在第 p 期之後落入虛線內 → 適合 AR(p)",
    })}</div>` +
    `<div class="grid-2">${card("日報酬率分布", chart("c-ret"))}${card(
      "最近 10 筆資料",
      table(["日期", esc(st.valueCol)], series.dates.slice(-10).map((d, i) => [esc(d), fmt(y[y.length - 10 + i])]).reverse(), { num: [1] }),
    )}</div>`;

  const t = theme();
  const rs = series.hasDates ? { rangeslider: { visible: true, thickness: 0.08 }, type: "date" } : {};
  plot(
    "c-price",
    candle
      ? [{ type: "candlestick", x: candle.dates, open: candle.open, high: candle.high, low: candle.low, close: candle.close, name: "K 線", increasing: { line: { color: t.neg } }, decreasing: { line: { color: t.s3 } } }]
      : [{ type: "scatter", mode: "lines", x: series.dates, y: Array.from(y), name: st.valueCol, line: { color: t.s1, width: 1.5 } }],
    { height: 460, showlegend: false, xaxis: { ...rs, title: series.hasDates ? "日期" : "序號" }, yaxis: { title: st.valueCol } },
  );
  const lags = Array.from({ length: nl }, (_, i) => i + 1);
  const corrBars = (id, vals, name) =>
    plot(
      id,
      [{ type: "bar", x: lags, y: vals.slice(1), name, marker: { color: t.s1 }, hovertemplate: "lag %{x}<br>%{y:.3f}<extra></extra>" }],
      {
        showlegend: false,
        xaxis: { title: "落後期數 lag" },
        yaxis: { title: name, range: [Math.min(-0.2, ...vals) - 0.05, 1.05] },
        shapes: [bound, -bound].map((b) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: b, y1: b, line: { color: t.muted, dash: "dash", width: 1 } })),
      },
    );
  corrBars("c-acf", ac, "ACF");
  corrBars("c-pacf", pac, "PACF");
  plot("c-ret", [{ type: "histogram", x: ret.map((r) => r * 100), nbinsx: 60, marker: { color: t.s1, line: { color: t.surface, width: 1 } }, hovertemplate: "%{x:.2f}%<br>%{y} 天<extra></extra>" }], {
    showlegend: false,
    xaxis: { title: "日報酬率（%）" },
    yaxis: { title: "天數" },
  });
}

function renderPrep(R, root) {
  const { series, y, p, nTrain, nTest, st } = R;
  const d = series.dates;
  const show = Math.min(p, 5);
  const head = ["日期", `y(t) ${esc(st.valueCol)}`, ...Array.from({ length: show }, (_, i) => `y(t−${i + 1})`)];
  const rows = Array.from({ length: 6 }, (_, r) => {
    const tIdx = p + r;
    return [esc(d[tIdx]), fmt(y[tIdx]), ...Array.from({ length: show }, (_, i) => fmt(y[tIdx - i - 1]))];
  });
  root.innerHTML =
    card(
      "Data Preparation 資料準備步驟",
      `<ol class="steps">
        <li>${series.hasDates ? `以日期欄位「${esc(st.dateCol)}」解析日期並<b>由舊到新排序</b>` : "沒有日期欄位，依資料列順序當作時間順序"}。</li>
        <li>移除「${esc(st.valueCol)}」缺失或無法解析的資料列：${series.dropped} 筆；重複日期（保留最後一筆）：${series.dup} 筆。剩下 <b>${y.length.toLocaleString()}</b> 筆。</li>
        <li>建立落後特徵：第 t 天的特徵為前 1～${p} 天的值 y(t−1)…y(t−${p})，因此最前面 ${p} 筆沒有完整特徵、不列入訓練。</li>
        <li><b>依時間切分、不打亂順序</b>（避免用未來資料預測過去）：訓練期 ${nTrain.toLocaleString()} 筆（${esc(d[0])} ～ ${esc(d[nTrain - 1])}），測試期 ${nTest.toLocaleString()} 筆（${esc(d[nTrain])} ～ ${esc(d[y.length - 1])}）。</li>
        <li>落後期數 p 的選擇（AIC／BIC）只使用訓練期；測試期採「一步預測」：每天都用真實的前 p 天價格預測隔天。</li>
        <li>${
          R.useLog
            ? "建模尺度：<b>對數價格 log(y)</b>。價格越高、每日波動的金額越大，取 log 後誤差變成「百分比」尺度，變異數較穩定；預測完再以 exp() 轉回價格（因此預測區間上下不對稱）。"
            : R.logFallback
              ? "建模尺度：資料含有 0 或負值，無法取對數，<b>自動改用原始尺度</b>。"
              : "建模尺度：<b>原始價格</b>。若價格在資料期間漲跌幅度很大，可在左側改用「對數價格」讓誤差變異數更穩定。"
        }</li>
      </ol>`,
    ) +
    card(`落後特徵矩陣（前 6 筆${p > show ? `，只顯示前 ${show} 個落後項` : ""}）`, table(head, rows, { num: head.map((_, i) => i).slice(1) })) +
    card("訓練期與測試期", chart("c-split"), { sub: "藍色 = 訓練期、橘色 = 測試期" });
  const t = theme();
  plot(
    "c-split",
    [
      { type: "scatter", mode: "lines", name: "訓練期", x: d.slice(0, nTrain), y: Array.from(y.subarray(0, nTrain)), line: { color: t.s1, width: 1.5 } },
      { type: "scatter", mode: "lines", name: "測試期", x: d.slice(nTrain), y: Array.from(y.subarray(nTrain)), line: { color: t.s2, width: 1.5 } },
    ],
    { hovermode: "x unified", xaxis: { title: series.hasDates ? "日期" : "序號" }, yaxis: { title: st.valueCol } },
  );
}

function renderModel(R, root) {
  const { scan, p, model, st, maxLag } = R;
  const ci = model.confInt(R.alpha);
  const phiSum = model.b.reduce((s, v) => s + v, 0);
  const v = R.useLog ? "log y" : "y";
  const terms = model.b
    .slice(0, 10)
    .map((c, i) => `${c < 0 ? "−" : "+"} ${fmt(Math.abs(c))}·${v}<sub>t−${i + 1}</sub>`)
    .join(" ");
  const bestA = scan.reduce((b, r) => (r.aic < b.aic ? r : b)).p;
  const bestB = scan.reduce((b, r) => (r.bic < b.bic ? r : b)).p;
  const bestT = scan.reduce((b, r) => (r.testRmse < b.testRmse ? r : b)).p;
  root.innerHTML =
    card(
      "特徵選擇：落後期數 p",
      `<p class="prose">在訓練期對 p = 1～${maxLag} 分別配適 AR(p)（使用相同的樣本區間，資訊準則才能比較）。AIC 最小：<b>p = ${bestA}</b>；BIC 最小（懲罰較重、偏好簡單模型）：<b>p = ${bestB}</b>。
      目前選擇「${LAG_MODES[st.lagMode]}」→ <b>p = ${p}</b>。</p>`,
    ) +
    `<div class="grid-2">${card("AIC／BIC vs 落後期數", chart("c-ic"), { sub: "越低越好" })}${card(`測試期 RMSE vs 落後期數${R.useLog ? "（log 尺度）" : ""}`, chart("c-prmse"), {
      sub: `僅供對照（測試期最佳為 p = ${bestT}），選擇 p 時不使用測試期`,
    })}</div>` +
    card(
      `AR(${p}) 迴歸結果（OLS）`,
      `<div class="kpis mini">${kpiRow([
        { label: "R²（訓練期）", value: fmt(model.r2) },
        { label: "殘差標準差 σ", value: fmt(model.sigma) },
        { label: "AIC / BIC", value: fmt(model.aic), sub: `BIC ${fmt(model.bic)}` },
        { label: "Σφ（係數總和）", value: fmt(phiSum), sub: Math.abs(phiSum - 1) < 0.02 ? "≈ 1：接近隨機漫步" : "" },
        { label: "觀測數", value: model.n.toLocaleString() },
      ])}</div>
      ${table(
        ["變數", "係數", "標準誤", "t 值", "p 值", `${st.conf}% CI 下限`, `${st.conf}% CI 上限`],
        model.names.map((nm, i) => [esc(nm), fmt(model.coef[i]), fmt(model.se[i]), fmt(model.t[i]), fmtP(model.pval[i]), fmt(ci[i][0]), fmt(ci[i][1])]),
        { num: [1, 2, 3, 4, 5, 6] },
      )}
      <p class="eq"><b>模型：</b>${v}<sub>t</sub> = ${fmt(model.b0)} ${terms}${p > 10 ? " + …" : ""}</p>
      <p class="hint">Σφ 接近 1 表示價格幾乎是「昨天價格 + 隨機誤差」，這也是股價很難贏過 Naive baseline 的原因。</p>`,
    );
  const t = theme();
  const ps = scan.map((r) => r.p);
  const mark = { type: "line", x0: p, x1: p, yref: "paper", y0: 0, y1: 1, line: { color: t.muted, dash: "dot", width: 1.5 } };
  plot(
    "c-ic",
    [
      { type: "scatter", mode: "lines+markers", name: "AIC", x: ps, y: scan.map((r) => r.aic), line: { color: t.s1, width: 2 }, marker: { size: 6 } },
      { type: "scatter", mode: "lines+markers", name: "BIC", x: ps, y: scan.map((r) => r.bic), line: { color: t.s2, width: 2 }, marker: { size: 6 } },
    ],
    { hovermode: "x unified", xaxis: { title: "落後期數 p" }, yaxis: { title: "資訊準則" }, shapes: [mark] },
  );
  plot(
    "c-prmse",
    [{ type: "scatter", mode: "lines+markers", name: "測試期 RMSE", x: ps, y: scan.map((r) => r.testRmse), line: { color: t.s1, width: 2 }, marker: { size: 6 } }],
    { showlegend: false, hovermode: "x unified", xaxis: { title: "落後期數 p" }, yaxis: { title: "RMSE" }, shapes: [mark] },
  );
}

function renderEval(R, root) {
  const { series, y, nTrain, pred, st, metrics: m, naive: nv, model, fc, futureDates, lb } = R;
  const d = series.dates;
  const dTest = d.slice(nTrain);
  const nr = Math.min(20, Math.floor(model.n / 4));
  const rac = S.acf(model.resid, nr);
  root.innerHTML =
    card(
      `評估指標：AR(${R.p}) vs Naive baseline（測試期 ${R.nTest} 天，一步預測）`,
      table(
        ["指標", `AR(${R.p})`, "Naive（明天 = 今天）", "說明"],
        [
          ["MAE", fmt(m.MAE), fmt(nv.MAE), "平均絕對誤差"],
          ["MSE", fmt(m.MSE), fmt(nv.MSE), "均方誤差"],
          ["RMSE", fmt(m.RMSE), fmt(nv.RMSE), "與價格同單位"],
          ["MAPE", pct(m.MAPE, 2), pct(nv.MAPE, 2), "平均絕對百分比誤差"],
          ["R²", fmt(m.R2), fmt(nv.R2), "價格水準的 R² 通常很高，需與 Naive 比較才有意義"],
          ["漲跌方向準確率", pct(m.dirAcc), "—", "50% 約等於擲硬幣"],
          [`${st.conf}% PI 涵蓋率`, pct(R.coverage), "—", `理想值約 ${st.conf}%`],
        ],
        { num: [1, 2] },
      ) +
        `<p class="hint">${
          m.RMSE < nv.RMSE
            ? `AR(${R.p}) 的 RMSE 比 Naive 低 ${pct(1 - m.RMSE / nv.RMSE, 2)}。`
            : `AR(${R.p}) 沒有贏過 Naive（RMSE 高 ${pct(m.RMSE / nv.RMSE - 1, 2)}），符合「股價接近隨機漫步」的理論。`
        }${
          R.coverage < st.conf / 100 - 0.1
            ? ` 預測區間涵蓋率（${pct(R.coverage)}）明顯低於 ${st.conf}%：測試期的波動比訓練期大${
                R.useLog ? "（市場波動度改變，AR 模型假設誤差變異數固定）" : "，可在左側把「建模尺度」改成對數價格試試"
              }。`
            : ""
        }</p>`,
    ) +
    card(`預測圖：歷史價格＋測試期一步預測＋未來 ${st.horizon} 天預測`, chart("c-over", "tall"), { sub: `色帶為 ${st.conf}% 預測區間；虛線為訓練／測試分界` }) +
    card(`測試期放大：${st.conf}% 信賴區間與預測區間`, chart("c-test", "tall"), { sub: "深色帶 = 信賴區間 CI（平均值）；淺色帶 = 預測區間 PI（單日實際價格）" }) +
    `<div class="grid-2">${card(`殘差時間序列（訓練期${R.useLog ? "，log 尺度" : ""}）`, chart("c-rt"))}${card(`殘差分布${R.useLog ? "（log 尺度）" : ""}`, chart("c-rh"))}</div>` +
    card(
      "殘差自我相關（白噪音檢定）",
      chart("c-racf") +
        `<p class="prose">Ljung-Box 檢定（lag ${lb.lags}）：Q = ${fmt(lb.q)}，${pEq(lb.p)} ${badge(lb.p > 0.05, "殘差近似白噪音", "殘差仍有自我相關")}</p>`,
    );

  const t = theme();
  const split = d[nTrain];
  const isDate = series.hasDates;
  plot(
    "c-over",
    [
      { type: "scatter", mode: "lines", name: "實際價格", x: d, y: Array.from(y), line: { color: t.s1, width: 1.5 } },
      band(dTest, pred.piLo, pred.piHi, t.s2, 0.2, `測試期 ${st.conf}% PI`),
      { type: "scatter", mode: "lines", name: "測試期一步預測", x: dTest, y: Array.from(pred.mean), line: { color: t.s2, width: 1.5 } },
      band(futureDates, fc.lo, fc.hi, t.s3, 0.22, `未來 ${st.conf}% PI`),
      { type: "scatter", mode: "lines", name: `未來 ${st.horizon} 天預測`, x: futureDates, y: fc.mean, line: { color: t.s3, width: 2.5, dash: "dash" } },
    ],
    {
      height: 500,
      hovermode: "x unified",
      legend: { y: -0.16, yanchor: "top" },
      margin: { t: 40, b: 40 },
      xaxis: isDate
        ? { type: "date", title: "日期", rangeselector: { buttons: [{ count: 3, label: "3 月", step: "month", stepmode: "backward" }, { count: 1, label: "1 年", step: "year", stepmode: "backward" }, { step: "all", label: "全部" }], bgcolor: t.surface, activecolor: t.grid, font: { color: t.text2 } } }
        : { title: "序號" },
      yaxis: { title: st.valueCol },
      shapes: [{ type: "line", x0: split, x1: split, yref: "paper", y0: 0, y1: 1, line: { color: t.muted, dash: "dot", width: 1.5 } }],
      annotations: [{ x: split, yref: "paper", y: 0.02, text: " 測試期 →", showarrow: false, xanchor: "left", font: { color: t.text2 } }],
    },
  );
  plot(
    "c-test",
    [
      band(dTest, pred.piLo, pred.piHi, t.s1, 0.14, `${st.conf}% 預測區間 PI`),
      band(dTest, pred.ciLo, pred.ciHi, t.s1, 0.4, `${st.conf}% 信賴區間 CI`),
      { type: "scatter", mode: "lines", name: "預測值", x: dTest, y: Array.from(pred.mean), line: { color: t.s1, width: 2 } },
      { type: "scatter", mode: "markers+lines", name: "實際值", x: dTest, y: Array.from(R.yTest), line: { color: t.s2, width: 1 }, marker: { color: t.s2, size: 5 } },
    ],
    { height: 440, hovermode: "x unified", xaxis: { title: isDate ? "日期" : "序號", ...(isDate ? { type: "date" } : {}) }, yaxis: { title: st.valueCol } },
  );
  const dTrainFit = d.slice(R.p, nTrain);
  plot("c-rt", [{ type: "scatter", mode: "lines", name: "殘差", x: dTrainFit, y: Array.from(model.resid), line: { color: t.s1, width: 1 } }], {
    showlegend: false,
    xaxis: { title: isDate ? "日期" : "序號" },
    yaxis: { title: "殘差", zeroline: true, zerolinecolor: t.axis },
  });
  plot("c-rh", [{ type: "histogram", x: Array.from(model.resid), nbinsx: 50, marker: { color: t.s1, line: { color: t.surface, width: 1 } } }], {
    showlegend: false,
    xaxis: { title: "殘差" },
    yaxis: { title: "天數" },
  });
  const bound = 1.96 / Math.sqrt(model.n);
  plot(
    "c-racf",
    [{ type: "bar", x: Array.from({ length: nr }, (_, i) => i + 1), y: rac.slice(1), marker: { color: t.s1 }, hovertemplate: "lag %{x}<br>%{y:.3f}<extra></extra>" }],
    {
      showlegend: false,
      height: 300,
      xaxis: { title: "落後期數 lag" },
      yaxis: { title: "殘差 ACF" },
      shapes: [bound, -bound].map((b) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: b, y1: b, line: { color: t.muted, dash: "dash", width: 1 } })),
    },
  );
}

function renderDeploy(R, root) {
  const { series, y, fc, futureDates, st, n } = R;
  const last = series.dates[n - 1];
  root.innerHTML =
    card(
      `Deployment 部署：未來 ${st.horizon} 個交易日預測`,
      `<div class="pred-result">
        <div><span class="kpi-label">最新資料（${esc(last)}）</span><span class="pred-value">${fmt(y[n - 1])}</span></div>
        <div><span class="kpi-label">下一交易日預測（${esc(futureDates[0])}）</span><span class="pred-value">${fmt(fc.mean[0])}</span></div>
        <div><span class="kpi-label">${st.conf}% 預測區間</span><span>${fmt(fc.lo[0])} ～ ${fmt(fc.hi[0])}</span></div>
      </div>
      ${chart("c-fc", "tall")}
      <p class="hint">以全部資料重新估計 AR(${R.p}) 後遞迴預測：第 2 天以後要用「預測值」當作落後值，所以誤差會累積、預測區間越來越寬，平均預測也會逐漸收斂。${
        series.hasDates ? "日期以週一到週五推算，未扣除國定假日。" : ""
      }</p>${DISCLAIMER}`,
    ) +
    card(
      "預測明細",
      `<div class="actions"><button class="btn" id="dl-fc" type="button">下載預測結果（CSV）</button></div>` +
        table(
          ["日期", "預測值", `${st.conf}% PI 下限`, `${st.conf}% PI 上限`, R.useLog ? "標準誤（log 尺度）" : "標準誤"],
          futureDates.map((d, i) => [esc(d), fmt(fc.mean[i]), fmt(fc.lo[i]), fmt(fc.hi[i]), fmt(fc.se[i])]),
          { num: [1, 2, 3, 4] },
        ),
    );
  const t = theme();
  const back = Math.min(n, 120);
  const hd = series.dates.slice(n - back);
  plot(
    "c-fc",
    [
      { type: "scatter", mode: "lines", name: `最近 ${back} 筆實際值`, x: hd, y: Array.from(y.subarray(n - back)), line: { color: t.s1, width: 1.8 } },
      band(futureDates, fc.lo, fc.hi, t.s3, 0.22, `${st.conf}% 預測區間`),
      { type: "scatter", mode: "lines+markers", name: "預測值", x: [last, ...futureDates], y: [y[n - 1], ...fc.mean], line: { color: t.s3, width: 2.5, dash: "dash" }, marker: { size: 6 } },
    ],
    { height: 440, hovermode: "x unified", xaxis: { title: series.hasDates ? "日期" : "序號", ...(series.hasDates ? { type: "date" } : {}) }, yaxis: { title: st.valueCol } },
  );
  root.querySelector("#dl-fc").addEventListener("click", () => {
    download(
      `hw2_ar_forecast_${R.ds.name.replace(/[^\w.-]+/g, "_")}.csv`,
      toCSV(["date", "forecast", "pi_lower", "pi_upper", "se"], futureDates.map((d, i) => [d, fc.mean[i], fc.lo[i], fc.hi[i], fc.se[i]])),
    );
  });
}
