// 問題 ①：多元線性回歸（表格資料）—— 計算流程 + CRISP-DM 六個分頁
import * as S from "./stats.js";
import { prepareMLR, sampleRows } from "./data.js";
import { esc, fmt, fmtP, pEq, pct, table, card, chart, badge, kpiRow, plot, band, theme, toCSV, download } from "./ui.js";

export const FS_METHODS = {
  kbest: {
    label: "SelectKBest（F 檢定）",
    type: "Filter 過濾法",
    desc: "分別計算每個特徵與 y 的 F 統計量（單變量回歸的顯著性，等同相關係數平方的排序），由大到小取前 k 個。速度最快，但不考慮特徵之間的重疊資訊。",
  },
  rfe: {
    label: "RFE 遞迴特徵消除",
    type: "Wrapper 包裝法",
    desc: "先用全部特徵配適線性回歸（特徵已標準化），每一輪刪掉 |係數| 最小的特徵再重新配適，越晚被刪除的特徵越重要。",
  },
  forward: {
    label: "Forward Selection 前向逐步選擇",
    type: "Wrapper 包裝法",
    desc: "從 0 個特徵開始，每一步嘗試加入每個剩下的特徵，保留讓 5-fold 交叉驗證 RMSE 最小的那一個，直到全部加入。",
  },
  lasso: {
    label: "Lasso（L1 正則化）",
    type: "Embedded 嵌入法",
    desc: "L1 懲罰會把不重要特徵的係數壓成 0。λ 由大到小時越早出現非零係數的特徵越重要；λ 以 5-fold 交叉驗證選擇。",
  },
};

const cache = { key: null };

export function runMLR(ds, st) {
  const key = [ds.key, st.target, [...st.excluded].sort().join("|"), st.testRatio, st.maxRows].join("§");
  if (cache.key !== key) {
    const rows = sampleRows(ds.rows, st.maxRows);
    const prep = prepareMLR({ rows, profile: ds.profile, target: st.target, excluded: st.excluded, testRatio: st.testRatio });
    const cv = S.makeCV(prep.Ztrain, prep.ycTrain, 5);
    const r = prep.Ztrain.map((z) => S.corr(z, prep.yTrain));
    Object.assign(cache, { key, prep, cv, r, used: rows.length, total: ds.rows.length, rank: {}, curve: {}, extra: {} });
  }
  const { prep, cv, r } = cache;
  const p = prep.features.length;
  const rank = (cache.rank[st.method] ??= rankFeatures(st.method, prep, cv, r));
  const curve = (cache.curve[st.method] ??= kCurve(prep, cv, rank.order));
  const finite = curve.filter((c) => Number.isFinite(c.cvRmse));
  const bestK = finite.length ? finite.reduce((b, c) => (c.cvRmse < b.cvRmse ? c : b)).k : p;
  const k = st.kAuto || !st.k ? bestK : Math.min(Math.max(1, st.k), p);
  const sel = rank.order.slice(0, k);
  const feats = sel.map((j) => prep.features[j]);
  const model = S.ols(
    sel.map((j) => prep.Xtrain[j]),
    prep.yTrain,
    feats.map((f) => f.name),
  );
  const alpha = 1 - st.conf / 100;
  const pred = model.predict(
    sel.map((j) => prep.Xtest[j]),
    alpha,
  );
  let covered = 0;
  prep.yTest.forEach((y, i) => {
    if (y >= pred.piLo[i] && y <= pred.piHi[i]) covered++;
  });
  const ek = `${st.method}|${k}`;
  const extra = (cache.extra[ek] ??= { compare: compareModels(prep, cv, sel, pred), tests: assumptionTests(model, prep, sel) });
  return {
    ds, st, prep, r, rank, curve, bestK, k, sel, feats, model, pred, alpha, ...extra,
    used: cache.used,
    total: cache.total,
    testM: S.regMetrics(prep.yTest, pred.mean, k),
    trainM: S.regMetrics(prep.yTrain, model.fitted, k),
    cvSel: S.cvScore(cv, sel),
    coverage: covered / prep.yTest.length,
  };
}

function rankFeatures(method, prep, cv, r) {
  const p = prep.features.length;
  const n = prep.yTrain.length;
  const all = Array.from({ length: p }, (_, j) => j);
  const byCorr = all.slice().sort((a, b) => Math.abs(r[b]) - Math.abs(r[a]));
  if (method === "kbest") {
    const F = r.map((v) => ((v * v) / Math.max(1e-12, 1 - v * v)) * (n - 2));
    return { order: byCorr, score: F, label: "F 統計量（越大越重要）" };
  }
  if (method === "rfe") {
    const order = S.rfeOrder(cv);
    const score = new Array(p);
    order.forEach((j, i) => (score[j] = p - i));
    return { order, score, label: "RFE 保留順位（越大越晚被淘汰）" };
  }
  if (method === "forward") {
    const { order, stepRmse } = S.forwardOrder(cv, byCorr, 40);
    const score = new Array(p).fill(NaN);
    stepRmse.forEach((v, i) => (score[order[i]] = v));
    return { order, score, label: "加入此特徵後的 CV RMSE（越小越好）", lowerBetter: true };
  }
  const las = S.lassoCV(cv, all);
  const score = new Array(p).fill(0);
  las.S.forEach((j, i) => (score[j] = Math.abs(las.coef[i])));
  return { order: las.order, score, label: "|Lasso 標準化係數|（CV 最佳 λ）", lasso: las };
}

function kCurve(prep, cv, order) {
  const out = [];
  for (let k = 1; k <= Math.min(order.length, 60); k++) {
    const sub = order.slice(0, k);
    const c = S.cvScore(cv, sub);
    const beta = S.gramSolve(cv, sub);
    let test = { RMSE: NaN, R2: NaN };
    if (beta) test = S.regMetrics(prep.yTest, S.predictZ(beta, sub.map((j) => prep.Ztest[j]), prep.ym), k);
    out.push({ k, cvRmse: c.rmse, cvR2: c.r2, testRmse: test.RMSE, testR2: test.R2 });
  }
  return out;
}

function compareModels(prep, cv, sel, pred) {
  const p = prep.features.length;
  const k = sel.length;
  const zt = (idx) => idx.map((j) => prep.Ztest[j]);
  const rows = [];
  const add = (name, desc, yhat, kk, cvs, main = false) =>
    rows.push({ name, desc, k: kk, main, test: S.regMetrics(prep.yTest, yhat, kk), cv: cvs });
  add(`MLR（OLS）· 選出的 ${k} 個特徵`, "本頁主要模型", pred.mean, k, S.cvScore(cv, sel), true);
  if (k < p) {
    const all = Array.from({ length: p }, (_, j) => j);
    const b = S.gramSolve(cv, all);
    if (b) add(`MLR（OLS）· 全部 ${p} 個特徵`, "不做特徵選擇", S.predictZ(b, zt(all), prep.ym), p, S.cvScore(cv, all));
  }
  const rd = S.ridgeCV(cv, sel);
  add("Ridge（L2 正則化）", `α = ${fmt(rd.alpha)}（CV 選出）`, S.predictZ(rd.beta, zt(sel), prep.ym), k, rd.cv);
  const ls = S.lassoCV(cv, sel);
  add("Lasso（L1 正則化）", `λ = ${fmt(ls.lam)}，非零係數 ${ls.nonzero} 個`, S.predictZ(ls.beta, zt(sel), prep.ym), ls.nonzero, ls.cv);
  add("Baseline（只猜平均值）", "比較基準", new Float64Array(prep.yTest.length).fill(prep.ym), 0, S.cvBaseline(cv));
  return rows;
}

function assumptionTests(model, prep, sel) {
  const X = sel.map((j) => prep.Xtrain[j]);
  const e = model.resid;
  const rows = [];
  try {
    const rs = S.resetTest(model, X, prep.yTrain);
    rows.push({ name: "線性 Linearity", test: "Ramsey RESET", stat: rs.F, p: rs.p, ok: rs.p > 0.05, note: "p > 0.05：加入 ŷ² 沒有顯著改善 → 線性關係成立" });
  } catch {
    /* ŷ² 與特徵共線時略過 */
  }
  const dw = S.durbinWatson(e);
  rows.push({ name: "獨立性 Independence", test: "Durbin-Watson", stat: dw, p: NaN, ok: dw > 1.5 && dw < 2.5, note: "DW 介於 1.5～2.5：殘差沒有明顯自我相關（依原始資料順序）" });
  try {
    const bp = S.breuschPagan(e, X);
    rows.push({ name: "同質變異 Homoscedasticity", test: "Breusch-Pagan", stat: bp.lm, p: bp.p, ok: bp.p > 0.05, note: "p > 0.05：殘差變異數固定，不隨特徵改變" });
  } catch {
    /* 略過 */
  }
  const jb = S.jarqueBera(e);
  rows.push({ name: "常態性 Normality", test: "Jarque-Bera", stat: jb.jb, p: jb.p, ok: jb.p > 0.05, note: `偏態 ${fmt(jb.skew)}、峰度 ${fmt(jb.kurt)}；p > 0.05：殘差近似常態` });
  const vif = sel.length >= 2 ? model.vif() : null;
  if (vif) {
    const mx = Math.max(...vif);
    rows.push({ name: "無多重共線性", test: "最大 VIF", stat: mx, p: NaN, ok: mx < 10, note: "VIF < 10（嚴格標準 < 5）：特徵之間沒有嚴重共線" });
  }
  return { rows, vif };
}

// ---------- 畫面 ----------
export function mlrKpis(R) {
  return [
    { label: "測試集 R²", value: fmt(R.testM.R2), sub: `訓練集 ${fmt(R.trainM.R2)}` },
    { label: "測試集 RMSE", value: fmt(R.testM.RMSE), sub: `MAE ${fmt(R.testM.MAE)}` },
    { label: "Adjusted R²", value: fmt(R.testM.AdjR2), sub: `使用 ${R.k} / ${R.prep.features.length} 個特徵` },
    { label: `${R.st.conf}% 預測區間涵蓋率`, value: pct(R.coverage), sub: `測試集 ${R.prep.yTest.length.toLocaleString()} 筆` },
  ];
}

const link = (u) => (/^https?:\/\//.test(u) ? `<a href="${esc(u)}" target="_blank" rel="noopener">${esc(u)}</a>` : esc(u));

export function renderMLR(tab, R, root) {
  ({ business: renderBusiness, data: renderData, prep: renderPrep, model: renderModel, eval: renderEval, deploy: renderDeploy })[tab](R, root);
}

function renderBusiness(R, root) {
  const b = R.ds.business || {};
  const t = esc(R.st.target);
  root.innerHTML =
    card(
      "Business Understanding 商業理解",
      `<dl class="facts">
        <dt>要解決的問題</dt><dd>${b.problem || `以資料中的其他欄位預測「<b>${t}</b>」。目標是連續數值，屬於監督式學習的<b>回歸問題</b>。`}</dd>
        <dt>資料集來源</dt><dd>${esc(R.ds.name)}<br>${link(R.ds.source)}</dd>
        <dt>預測目標 y</dt><dd><code>${t}</code></dd>
        <dt>候選特徵</dt><dd>${R.prep.features.length} 個（類別欄位 one-hot 之後）</dd>
        <dt>使用者與決策</dt><dd>${b.users || "需要根據現有資訊估計目標值的決策者：用模型找出影響最大的因素，並以預測區間評估風險。"}</dd>
        <dt>成功標準</dt><dd>${b.success || "測試集 R² 越高、RMSE 越低越好，且明顯優於「只猜平均值」的 baseline；預測區間的實際涵蓋率接近設定的信賴水準。"}</dd>
        <dt>為什麼用線性回歸</dt><dd>係數可以直接解釋（其他條件不變時，特徵每增加 1 單位，y 平均改變多少）、能計算信賴區間與預測區間、計算快速，適合當作基準模型。</dd>
      </dl>`,
    ) +
    card(
      "CRISP-DM 六個步驟在本頁的對應",
      `<ol class="steps">
        <li><b>Business Understanding</b>：定義問題、資料來源與成功標準（本頁）</li>
        <li><b>Data Understanding</b>：欄位型態、缺失值、敘述統計、目標分布、相關係數</li>
        <li><b>Data Preparation</b>：排除 ID 欄位、one-hot 編碼、補值、訓練／測試切分、標準化</li>
        <li><b>Modeling</b>：特徵選擇（4 種方法可切換）、OLS 係數與顯著性、與 Ridge／Lasso 比較</li>
        <li><b>Evaluation</b>：MAE／MSE／RMSE／R²／Adj. R²、預測圖＋信賴區間／預測區間、回歸假設檢定</li>
        <li><b>Deployment</b>：輸入新資料即時預測、下載預測結果與模型參數</li>
      </ol>`,
    ) +
    card(
      "換成你自己的資料",
      `<p class="prose">左側「資料來源」可以選擇內建範例、貼上任何公開的 CSV 網址（例如 <code>https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv</code>），或上傳自己電腦裡的 CSV。載入後在「欄位設定」選擇要預測的欄位，其餘數值欄位自動成為特徵、文字欄位自動做 one-hot 編碼。</p>`,
    );
}

function renderData(R, root) {
  const { ds, st, prep } = R;
  const prof = ds.profile;
  const role = (c) =>
    c.name === st.target
      ? '<span class="tag tgt">目標 y</span>'
      : st.excluded.has(c.name)
        ? '<span class="tag off">排除</span>'
        : c.type === "numeric"
          ? '<span class="tag num">數值特徵</span>'
          : '<span class="tag cat">類別特徵</span>';
  const nums = prep.features.map((f, j) => ({ f, j })).filter((o) => o.f.kind === "num").slice(0, 25);
  root.innerHTML =
    `<div class="kpis mini">${kpiRow([
      { label: "資料筆數", value: ds.rows.length.toLocaleString() },
      { label: "欄位數", value: prof.length },
      { label: "數值欄位", value: prof.filter((c) => c.type === "numeric").length },
      { label: "類別欄位", value: prof.filter((c) => c.type === "categorical").length },
      { label: "缺失值（格）", value: prof.reduce((s, c) => s + c.missing, 0).toLocaleString() },
    ])}</div>` +
    card("資料預覽（前 8 筆）", table(ds.fields.map(esc), ds.rows.slice(0, 8).map((r) => ds.fields.map((f) => esc(r[f] ?? ""))))) +
    card(
      "欄位摘要",
      table(
        ["欄位", "角色", "型態", "缺失", "不同值", "平均", "標準差", "最小", "最大"],
        prof.map((c) => [esc(c.name), role(c), c.type === "numeric" ? "數值" : "類別", c.missing, c.uniqueCapped ? "1000+" : c.unique, fmt(c.mean), fmt(c.sd), fmt(c.min), fmt(c.max)]),
        { num: [3, 4, 5, 6, 7, 8] },
      ),
    ) +
    `<div class="grid-2">${card(`目標 ${esc(st.target)} 的分布`, chart("c-hist"))}${card("特徵與目標的相關係數 r", chart("c-corr", "auto"), {
      sub: "訓練集；藍色正相關、紅色負相關，最多顯示 30 個",
    })}</div>` +
    (nums.length >= 2 ? card("數值欄位相關係數矩陣", chart("c-heat", "auto"), { sub: "檢查特徵之間是否高度相關（多重共線性）" }) : "");

  const t = theme();
  const yAll = Array.from(prep.yTrain).concat(Array.from(prep.yTest));
  plot("c-hist", [{ type: "histogram", x: yAll, nbinsx: 40, marker: { color: t.s1, line: { color: t.surface, width: 1 } }, hovertemplate: "%{x}<br>%{y} 筆<extra></extra>" }], {
    showlegend: false,
    xaxis: { title: st.target },
    yaxis: { title: "筆數" },
  });
  const pairs = prep.features
    .map((f, j) => ({ name: f.name, r: R.r[j] }))
    .sort((a, b) => Math.abs(b.r) - Math.abs(a.r))
    .slice(0, 30)
    .reverse();
  plot(
    "c-corr",
    [{ type: "bar", orientation: "h", y: pairs.map((p) => p.name), x: pairs.map((p) => p.r), marker: { color: pairs.map((p) => (p.r >= 0 ? t.s1 : t.neg)) }, hovertemplate: "%{y}<br>r = %{x:.4f}<extra></extra>" }],
    { showlegend: false, height: Math.max(300, pairs.length * 22 + 70), margin: { l: 10, r: 16, t: 16, b: 48 }, xaxis: { title: "相關係數 r", zeroline: true, zerolinecolor: t.axis } },
  );
  if (nums.length >= 2) {
    const cols = [...nums.map((o) => prep.Xtrain[o.j]), prep.yTrain];
    const names = [...nums.map((o) => o.f.name), st.target];
    const M = cols.map((a) => cols.map((b) => S.corr(a, b)));
    plot(
      "c-heat",
      [{ type: "heatmap", z: M, x: names, y: names, zmin: -1, zmax: 1, colorscale: [[0, t.neg], [0.5, t.mid], [1, t.s1]], hovertemplate: "%{y} × %{x}<br>r = %{z:.3f}<extra></extra>", colorbar: { thickness: 10, outlinewidth: 0 } }],
      { showlegend: false, height: Math.max(380, names.length * 26 + 140), margin: { l: 10, r: 10, t: 16, b: 10 }, yaxis: { autorange: "reversed" }, xaxis: { tickangle: -45 } },
    );
  }
}

function renderPrep(R, root) {
  const { prep, st } = R;
  const info = prep.info;
  const list = (a) => (a.length ? a.map((x) => `<code>${esc(x)}</code>`).join("、") : "無");
  const steps = [
    R.used < R.total
      ? `資料共 ${R.total.toLocaleString()} 筆，超過上限 ${st.maxRows.toLocaleString()} 筆，<b>隨機抽樣 ${R.used.toLocaleString()} 筆</b>以加快瀏覽器計算（可在左側調整上限）。`
      : `使用全部 ${R.total.toLocaleString()} 筆資料。`,
    `目標「${esc(st.target)}」缺失或不是數字的資料列：<b>${info.droppedTarget}</b> 筆，已刪除；剩下 ${prep.n.toLocaleString()} 筆。`,
    `排除的欄位（ID、名稱、高基數文字等，不適合當特徵）：${list([...st.excluded])}。`,
    info.categorical.length
      ? `類別欄位 one-hot 編碼（drop-first，避免虛擬變數陷阱）：${info.categorical
          .map((c) => `<code>${esc(c.name)}</code>（${c.levels.length} 類，基準類別「${esc(c.base)}」）`)
          .join("、")} → 新增 ${info.dummies.length} 個虛擬變數。`
      : "沒有類別欄位，不需要 one-hot 編碼。",
    `數值欄位缺失值以<b>訓練集中位數</b>補值，共 ${info.imputed} 格（避免用到測試集資訊）。`,
    `移除常數欄位（只有一種值，無法提供資訊）：${list(info.constant)}。`,
    `隨機切分（seed = 42）：<b>訓練集 ${prep.yTrain.length.toLocaleString()} 筆</b>、<b>測試集 ${prep.yTest.length.toLocaleString()} 筆</b>（${Math.round(st.testRatio * 100)}%）。特徵選擇與調參只用訓練集。`,
    "特徵選擇、Ridge、Lasso 使用訓練集的平均數與標準差做標準化；OLS 係數仍以原始單位呈現，方便解釋。",
  ];
  root.innerHTML =
    card("Data Preparation 資料準備步驟", `<ol class="steps">${steps.map((s) => `<li>${s}</li>`).join("")}</ol>`) +
    card(
      `建模用的特徵（共 ${prep.features.length} 個）`,
      table(
        ["特徵", "來源欄位", "類型", "訓練集平均", "標準差", "最小", "最大"],
        prep.features.map((f) => [esc(f.name), esc(f.source), f.kind === "num" ? "數值" : "虛擬變數 0/1", fmt(f.mean), fmt(f.sd), fmt(f.min), fmt(f.max)]),
        { num: [3, 4, 5, 6] },
      ),
    );
}

function renderModel(R, root) {
  const { st, prep, rank, curve, model, k, bestK, feats } = R;
  const m = FS_METHODS[st.method];
  const ci = model.confInt(R.alpha);
  const eqTerms = feats.slice(0, 12).map((f, i) => {
    const c = model.b[i];
    return `${c < 0 ? "−" : "+"} ${fmt(Math.abs(c))} × ${esc(f.name)}`;
  });
  const eq = `${esc(st.target)} = ${fmt(model.b0)} ${eqTerms.join(" ")}${feats.length > 12 ? " + …" : ""}`;
  root.innerHTML =
    card(
      `特徵選擇：${m.label}`,
      `<p class="prose"><span class="tag num">${m.type}</span> ${m.desc}</p>
       <p class="prose">5-fold 交叉驗證（只用訓練集）的最佳特徵數 <b>k = ${bestK}</b>；目前使用 <b>k = ${k}</b>${k !== bestK ? "（手動指定）" : ""}。
       選出的特徵：${feats.map((f) => `<code>${esc(f.name)}</code>`).join("、")}</p>`,
    ) +
    `<div class="grid-2">${card("特徵重要性排序", chart("c-rank", "auto"), { sub: rank.label })}${card("特徵數 k 與誤差", chart("c-kcurve"), {
      sub: "依上面的排序取前 k 個特徵，比較交叉驗證與測試集 RMSE",
    })}</div>` +
    card(
      "OLS 迴歸結果（statsmodels 風格）",
      `<div class="kpis mini">${kpiRow([
        { label: "R²（訓練集）", value: fmt(model.r2) },
        { label: "Adj. R²", value: fmt(model.adjR2) },
        { label: "F 統計量", value: fmt(model.F), sub: pEq(model.Fp) },
        { label: "AIC / BIC", value: `${fmt(model.aic)}`, sub: `BIC ${fmt(model.bic)}` },
        { label: "觀測數 / 自由度", value: model.n.toLocaleString(), sub: `df = ${model.df}` },
      ])}</div>
      ${table(
        ["變數", "係數", "標準誤", "t 值", "p 值", `${st.conf}% CI 下限`, `${st.conf}% CI 上限`, "顯著"],
        model.names.map((nm, i) => [
          esc(nm), fmt(model.coef[i]), fmt(model.se[i]), fmt(model.t[i]), fmtP(model.pval[i]), fmt(ci[i][0]), fmt(ci[i][1]),
          model.pval[i] < 0.05 ? '<span class="badge ok">p &lt; 0.05</span>' : '<span class="badge muted">不顯著</span>',
        ]),
        { num: [1, 2, 3, 4, 5, 6] },
      )}
      <p class="eq"><b>回歸方程式：</b>${eq}</p>`,
    ) +
    card("標準化係數 β*（比較各特徵影響力）", chart("c-coef", "auto"), { sub: "特徵與 y 都標準化後的係數：特徵增加 1 個標準差時，y 改變幾個標準差" }) +
    card(
      "模型比較",
      table(
        ["模型", "說明", "特徵數", "CV R²", "CV RMSE", "測試 R²", "測試 RMSE", "測試 MAE"],
        R.compare.map((c) => [`${c.main ? "★ " : ""}${esc(c.name)}`, esc(c.desc), c.k, fmt(c.cv?.r2), fmt(c.cv?.rmse), fmt(c.test.R2), fmt(c.test.RMSE), fmt(c.test.MAE)]),
        { num: [2, 3, 4, 5, 6, 7], rowCls: (r, i) => (R.compare[i].main ? "hl" : "") },
      ) + `<p class="hint">CV = 5-fold 交叉驗證（訓練集）。Ridge／Lasso 使用與主要模型相同的特徵；若正則化沒有帶來明顯改善，代表 OLS 已經沒有過度擬合。</p>`,
    );

  const t = theme();
  const ord = rank.order.slice(0, 40).slice().reverse();
  const selSet = new Set(R.sel);
  const val = (j) => (Number.isFinite(rank.score[j]) ? rank.score[j] : null);
  plot(
    "c-rank",
    [true, false].map((on) => {
      const js = ord.filter((j) => selSet.has(j) === on);
      return {
        type: "bar", orientation: "h", name: on ? "已選入" : "未選入",
        y: js.map((j) => prep.features[j].name), x: js.map(val),
        marker: { color: on ? t.s1 : t.muted },
        hovertemplate: "%{y}<br>%{x:.4g}<extra></extra>",
      };
    }),
    { height: Math.max(300, ord.length * 22 + 80), margin: { l: 10, r: 16, t: 36, b: 48 }, yaxis: { categoryorder: "array", categoryarray: ord.map((j) => prep.features[j].name) }, xaxis: { title: rank.label } },
  );
  const ks = curve.map((c) => c.k);
  plot(
    "c-kcurve",
    [
      { type: "scatter", mode: "lines+markers", name: "CV RMSE（訓練集 5-fold）", x: ks, y: curve.map((c) => c.cvRmse), line: { color: t.s1, width: 2 }, marker: { size: 6 } },
      { type: "scatter", mode: "lines+markers", name: "測試集 RMSE", x: ks, y: curve.map((c) => c.testRmse), line: { color: t.s2, width: 2 }, marker: { size: 6 } },
    ],
    {
      hovermode: "x unified",
      xaxis: { title: "保留特徵數 k", dtick: ks.length > 20 ? 5 : 1 },
      yaxis: { title: "RMSE" },
      shapes: [{ type: "line", x0: k, x1: k, yref: "paper", y0: 0, y1: 1, line: { color: t.muted, dash: "dot", width: 1.5 } }],
      annotations: [{ x: k, yref: "paper", y: 0.02, text: ` 目前 k = ${k}`, showarrow: false, xanchor: k > ks.length / 2 ? "right" : "left", yanchor: "bottom", font: { color: t.text2 } }],
    },
  );
  const coefs = feats.map((f, i) => ({ name: f.name, v: model.stdCoef[i] })).sort((a, b) => Math.abs(a.v) - Math.abs(b.v));
  plot(
    "c-coef",
    [{ type: "bar", orientation: "h", y: coefs.map((c) => c.name), x: coefs.map((c) => c.v), marker: { color: coefs.map((c) => (c.v >= 0 ? t.s1 : t.neg)) }, hovertemplate: "%{y}<br>β* = %{x:.4f}<extra></extra>" }],
    { showlegend: false, height: Math.max(260, coefs.length * 24 + 70), margin: { l: 10, r: 16, t: 16, b: 48 }, xaxis: { title: "標準化係數 β*", zeroline: true, zerolinecolor: t.axis } },
  );
}

function renderEval(R, root) {
  const { st, prep, model, pred, testM, trainM, cvSel, k, tests } = R;
  const tgt = esc(st.target);
  root.innerHTML =
    card(
      "評估指標",
      table(
        ["指標", "訓練集", "5-fold CV", "測試集", "說明"],
        [
          ["MAE", fmt(trainM.MAE), "—", fmt(testM.MAE), "平均絕對誤差"],
          ["MSE", fmt(trainM.MSE), fmt(cvSel.rmse ** 2), fmt(testM.MSE), "均方誤差"],
          ["RMSE", fmt(trainM.RMSE), fmt(cvSel.rmse), fmt(testM.RMSE), `與 ${tgt} 同單位的典型誤差`],
          ["R²", fmt(trainM.R2), fmt(cvSel.r2), fmt(testM.R2), "可解釋的變異比例（1 最好）"],
          ["Adj. R²", fmt(trainM.AdjR2), "—", fmt(testM.AdjR2), "依特徵數調整後的 R²"],
          [`${st.conf}% PI 涵蓋率`, "—", "—", pct(R.coverage), `理想值約 ${st.conf}%`],
        ],
        { num: [1, 2, 3] },
      ) + `<p class="hint">訓練集與測試集的 R² 接近，代表模型沒有過度擬合。</p>`,
    ) +
    card(`預測圖：測試集預測值＋${st.conf}% 信賴區間／預測區間`, chart("c-pi", "tall"), {
      sub: "依預測值由小到大排序。深色帶 = 平均值的信賴區間（CI）；淺色帶 = 單一新樣本的預測區間（PI）；點 = 實際值",
    }) +
    (k === 1 ? card(`簡單線性回歸：${esc(R.feats[0].name)} → ${tgt}`, chart("c-slr", "tall"), { sub: "回歸線與信賴區間、預測區間" }) : "") +
    `<div class="grid-2">${card("實際值 vs 預測值", chart("c-avp"), { sub: "越貼近對角線越準" })}${card("殘差 vs 預測值", chart("c-rvf"), { sub: "殘差應隨機散布在 0 附近、沒有漏斗形" })}</div>` +
    `<div class="grid-2">${card("殘差分布", chart("c-rhist"))}${card("常態 Q-Q 圖", chart("c-qq"), { sub: "點落在直線上表示殘差近似常態" })}</div>` +
    card(
      "回歸假設檢定（訓練集殘差）",
      table(
        ["假設", "檢定方法", "統計量", "p 值", "結果", "判讀"],
        tests.rows.map((r) => [r.name, r.test, fmt(r.stat), fmtP(r.p), badge(r.ok), r.note]),
        { num: [2, 3] },
      ) + `<p class="hint">樣本數很大時，極小的偏離也會讓 p 值很小；請同時參考上方殘差圖判斷實際影響。</p>`,
    ) +
    (tests.vif
      ? card("VIF 變異數膨脹因子", table(["特徵", "VIF", "判讀"], R.feats.map((f, i) => [esc(f.name), fmt(tests.vif[i]), tests.vif[i] < 5 ? badge(true, "低") : tests.vif[i] < 10 ? badge(true, "中") : badge(false, "", "高度共線")]), { num: [1] }))
      : "");

  const t = theme();
  const n = prep.yTest.length;
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => pred.mean[a] - pred.mean[b]);
  const x = idx.map((_, i) => i + 1);
  const pick = (arr) => idx.map((i) => arr[i]);
  const big = n > 1500;
  plot(
    "c-pi",
    [
      band(x, pick(pred.piLo), pick(pred.piHi), t.s1, 0.14, `${st.conf}% 預測區間 PI`),
      band(x, pick(pred.ciLo), pick(pred.ciHi), t.s1, 0.4, `${st.conf}% 信賴區間 CI`),
      { type: big ? "scattergl" : "scatter", mode: "markers", name: "實際值", x, y: pick(prep.yTest), marker: { color: t.s2, size: big ? 4 : 7, opacity: big ? 0.55 : 0.85 }, hovertemplate: "實際值 %{y:.4g}<extra></extra>" },
      {
        type: "scatter", mode: "lines", name: "預測值", x, y: pick(pred.mean), line: { color: t.s1, width: 2 },
        customdata: idx.map((i) => [pred.piLo[i], pred.piHi[i]]),
        hovertemplate: "預測值 %{y:.4g}<br>PI [%{customdata[0]:.4g}, %{customdata[1]:.4g}]<extra></extra>",
      },
    ],
    { height: 440, hovermode: "x unified", xaxis: { title: "測試集樣本（依預測值排序）" }, yaxis: { title: st.target } },
  );

  if (k === 1) {
    const f = R.feats[0];
    const j = R.sel[0];
    const xs = Float64Array.from({ length: 120 }, (_, i) => f.min + ((f.max - f.min) * i) / 119);
    const pr = model.predict([xs], R.alpha);
    plot(
      "c-slr",
      [
        band(xs, pr.piLo, pr.piHi, t.s1, 0.14, `${st.conf}% 預測區間 PI`),
        band(xs, pr.ciLo, pr.ciHi, t.s1, 0.4, `${st.conf}% 信賴區間 CI`),
        { type: "scattergl", mode: "markers", name: "訓練集", x: Array.from(prep.Xtrain[j]), y: Array.from(prep.yTrain), marker: { color: t.muted, size: 6, opacity: 0.5 } },
        { type: "scattergl", mode: "markers", name: "測試集", x: Array.from(prep.Xtest[j]), y: Array.from(prep.yTest), marker: { color: t.s2, size: 7, opacity: 0.85 } },
        { type: "scatter", mode: "lines", name: "回歸線", x: Array.from(xs), y: Array.from(pr.mean), line: { color: t.s1, width: 2.5 } },
      ],
      { height: 440, xaxis: { title: f.name }, yaxis: { title: st.target } },
    );
  }

  const yt = Array.from(prep.yTest);
  const pm = Array.from(pred.mean);
  const lo = Math.min(...yt, ...pm);
  const hi = Math.max(...yt, ...pm);
  plot(
    "c-avp",
    [
      { type: big ? "scattergl" : "scatter", mode: "markers", name: "測試集樣本", x: pm, y: yt, marker: { color: t.s1, size: big ? 4 : 7, opacity: big ? 0.5 : 0.8 }, hovertemplate: "預測 %{x:.4g}<br>實際 %{y:.4g}<extra></extra>" },
      { type: "scatter", mode: "lines", name: "y = x（完美預測）", x: [lo, hi], y: [lo, hi], line: { color: t.muted, dash: "dash", width: 1.5 }, hoverinfo: "skip" },
    ],
    { xaxis: { title: "預測值" }, yaxis: { title: "實際值" } },
  );

  const sampleIdx = S.shuffledIndex(model.n, 3).slice(0, 3000);
  plot(
    "c-rvf",
    [
      { type: "scattergl", mode: "markers", name: "訓練集殘差", x: sampleIdx.map((i) => model.fitted[i]), y: sampleIdx.map((i) => model.resid[i]), marker: { color: t.s1, size: 4, opacity: 0.5 }, hovertemplate: "預測 %{x:.4g}<br>殘差 %{y:.4g}<extra></extra>" },
    ],
    { showlegend: false, xaxis: { title: "預測值" }, yaxis: { title: "殘差", zeroline: true, zerolinecolor: t.axis } },
  );
  plot("c-rhist", [{ type: "histogram", x: Array.from(model.resid), nbinsx: 50, marker: { color: t.s1, line: { color: t.surface, width: 1 } }, hovertemplate: "%{x}<br>%{y} 筆<extra></extra>" }], {
    showlegend: false,
    xaxis: { title: "殘差" },
    yaxis: { title: "筆數" },
  });
  const sorted = Float64Array.from(model.resid).sort();
  const m = Math.min(sorted.length, 1500);
  const qi = Array.from({ length: m }, (_, i) => Math.round((i * (sorted.length - 1)) / Math.max(1, m - 1)));
  const theo = qi.map((i) => S.normInv((i + 0.5) / sorted.length));
  const samp = qi.map((i) => sorted[i] / model.sigma);
  const ext = [Math.min(theo[0], samp[0]), Math.max(theo[m - 1], samp[m - 1])];
  plot(
    "c-qq",
    [
      { type: "scattergl", mode: "markers", name: "標準化殘差", x: theo, y: samp, marker: { color: t.s1, size: 4, opacity: 0.6 } },
      { type: "scatter", mode: "lines", name: "常態參考線", x: ext, y: ext, line: { color: t.muted, dash: "dash", width: 1.5 }, hoverinfo: "skip" },
    ],
    { xaxis: { title: "理論分位數" }, yaxis: { title: "樣本分位數" } },
  );
}

function renderDeploy(R, root) {
  const { st, prep, model, feats } = R;
  const needs = [];
  const seen = new Set();
  for (const f of feats) {
    const key = f.kind === "num" ? f.name : f.source;
    if (seen.has(key)) continue;
    seen.add(key);
    needs.push(f.kind === "num" ? { kind: "num", name: f.name, f } : { kind: "cat", name: f.source, levels: prep.info.categorical.find((c) => c.name === f.source).levels });
  }
  const input = (o, i) => {
    if (o.kind === "cat") return `<label>${esc(o.name)}<select data-i="${i}">${o.levels.map((l) => `<option>${esc(l)}</option>`).join("")}</select></label>`;
    const f = o.f;
    const v = f.isInt ? Math.round(f.median) : Number(f.median.toPrecision(5));
    return `<label>${esc(o.name)}<input type="number" data-i="${i}" value="${v}" step="${f.isInt ? 1 : "any"}"><small>訓練範圍 ${fmt(f.min)} ～ ${fmt(f.max)}</small></label>`;
  };
  root.innerHTML =
    card(
      "Deployment 部署：輸入新資料即時預測",
      `<p class="prose">模型只需要 ${feats.length} 個特徵。預設值為訓練集中位數，修改後按「預測」。</p>
       <form id="pred-form" class="form-grid">${needs.map(input).join("")}</form>
       <div class="actions"><button class="btn primary" id="pred-btn" type="button">預測 ${esc(st.target)}</button></div>
       <div id="pred-out"></div>`,
    ) +
    card(
      "下載模型成果",
      `<div class="actions">
        <button class="btn" id="dl-pred" type="button">下載測試集預測結果（CSV）</button>
        <button class="btn" id="dl-model" type="button">下載模型參數（JSON）</button>
      </div>
      <p class="hint">模型參數包含截距、係數、特徵清單與評估指標，可以在 Excel 或其他程式中直接套用回歸方程式。</p>`,
    ) +
    card(
      "信賴區間 CI 與預測區間 PI 的差別",
      `<ul class="prose">
        <li><b>信賴區間（CI）</b>：對「所有具有相同特徵值的樣本，其<em>平均</em> ${esc(st.target)}」的估計範圍，只考慮係數估計的不確定性，較窄。</li>
        <li><b>預測區間（PI）</b>：對「<em>某一筆</em>新樣本的實際 ${esc(st.target)}」的範圍，另外加上個別誤差 σ²，所以較寬。單筆預測應該看 PI。</li>
      </ul>`,
    );

  const form = root.querySelector("#pred-form");
  const out = root.querySelector("#pred-out");
  const predict = () => {
    const vals = needs.map((o, i) => {
      const el = form.querySelector(`[data-i="${i}"]`);
      return o.kind === "num" ? Number(el.value) : el.value;
    });
    if (vals.some((v, i) => needs[i].kind === "num" && !Number.isFinite(v))) {
      out.innerHTML = `<p class="alert inline">請輸入有效的數字</p>`;
      return;
    }
    const byName = Object.fromEntries(needs.map((o, i) => [o.name, vals[i]]));
    const row = feats.map((f) => (f.kind === "num" ? byName[f.name] : byName[f.source] === f.level ? 1 : 0));
    const pr = model.predict(row.map((v) => Float64Array.of(v)), R.alpha);
    const extrap = needs.filter((o, i) => o.kind === "num" && (vals[i] < o.f.min || vals[i] > o.f.max)).map((o) => o.name);
    out.innerHTML = `<div class="pred-result">
        <div><span class="kpi-label">預測 ${esc(st.target)}</span><span class="pred-value">${fmt(pr.mean[0])}</span></div>
        <div><span class="kpi-label">${st.conf}% 信賴區間（平均值）</span><span>${fmt(pr.ciLo[0])} ～ ${fmt(pr.ciHi[0])}</span></div>
        <div><span class="kpi-label">${st.conf}% 預測區間（單一樣本）</span><span>${fmt(pr.piLo[0])} ～ ${fmt(pr.piHi[0])}</span></div>
      </div>
      ${extrap.length ? `<p class="alert inline">注意：${extrap.map(esc).join("、")} 超出訓練資料範圍，屬於外插，預測可信度較低。</p>` : ""}
      <div class="chart short" id="c-pred1"></div>`;
    const t = theme();
    plot(
      "c-pred1",
      [
        { type: "scatter", mode: "lines", name: `${st.conf}% 預測區間 PI`, x: [pr.piLo[0], pr.piHi[0]], y: [0, 0], line: { color: t.s1, width: 6 }, opacity: 0.3 },
        { type: "scatter", mode: "lines", name: `${st.conf}% 信賴區間 CI`, x: [pr.ciLo[0], pr.ciHi[0]], y: [0, 0], line: { color: t.s1, width: 14 }, opacity: 0.6 },
        { type: "scatter", mode: "markers", name: "預測值", x: [pr.mean[0]], y: [0], marker: { color: t.s2, size: 14, line: { color: t.surface, width: 2 } } },
      ],
      { height: 150, margin: { l: 16, r: 16, t: 36, b: 36 }, yaxis: { visible: false, range: [-1, 1] }, xaxis: { title: st.target, showgrid: true } },
    );
  };
  root.querySelector("#pred-btn").addEventListener("click", predict);
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    predict();
  });
  predict();

  root.querySelector("#dl-pred").addEventListener("click", () => {
    const head = ["row", ...feats.map((f) => f.name), `actual_${st.target}`, "predicted", "ci_lower", "ci_upper", "pi_lower", "pi_upper", "residual"];
    const rows = prep.testIdx.map((ri, i) => [
      ri + 1, ...R.sel.map((j) => prep.Xtest[j][i]), prep.yTest[i], R.pred.mean[i], R.pred.ciLo[i], R.pred.ciHi[i], R.pred.piLo[i], R.pred.piHi[i], prep.yTest[i] - R.pred.mean[i],
    ]);
    download(`hw2_mlr_predictions_${st.target}.csv`, toCSV(head, rows));
  });
  root.querySelector("#dl-model").addEventListener("click", () => {
    const card = {
      model: "Multiple Linear Regression (OLS)",
      dataset: R.ds.name,
      source: R.ds.source,
      target: st.target,
      feature_selection: FS_METHODS[st.method].label,
      features: feats.map((f) => f.name),
      intercept: model.b0,
      coefficients: Object.fromEntries(feats.map((f, i) => [f.name, model.b[i]])),
      p_values: Object.fromEntries(model.names.map((nm, i) => [nm, model.pval[i]])),
      residual_std: model.sigma,
      confidence_level: st.conf,
      metrics: { train: R.trainM, cv: R.cvSel, test: R.testM, pi_coverage: R.coverage },
      created_at: new Date().toISOString(),
    };
    download(`hw2_mlr_model_${st.target}.json`, JSON.stringify(card, null, 2), "application/json");
  });
}
