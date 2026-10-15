// 主程式：問題切換（下拉選單）、資料來源、側邊欄設定、CRISP-DM 分頁
import * as D from "./data.js";
import { runMLR, renderMLR, mlrKpis, FS_METHODS } from "./mlr.js";
import { runAR, renderAR, arKpis, LAG_MODES } from "./ar.js";
import { esc, kpiRow, card } from "./ui.js";
import { REMOTE_STOCK_API, REPO_URL } from "./config.js";

const URL_STARTUPS = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv";
const URL_SCORES = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/studentscores.csv";
const URL_AAPL = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv";

const MLR_SAMPLES = {
  flood: {
    name: "Kaggle：Predict the Flood（20 個特徵）",
    path: "data/flood_train_sample.csv",
    target: "FloodProbability",
    source: "https://www.kaggle.com/competitions/playground-series-s4e5",
    note: "Kaggle Playground S4E5 訓練集隨機抽樣 20,000 筆；20 個洪水風險因子 → 洪水機率。",
    business: {
      problem:
        "依 20 項洪水風險因子（季風強度、地形排水、河川管理、森林砍伐、都市化、氣候變遷、水壩品質、淤積、排水系統、海岸脆弱度、山崩、基礎設施老化、人口、濕地流失、規劃不足、政治因素等，每項 0～16 分）預測一個地區的<b>洪水發生機率 FloodProbability</b>（0～1 的連續值）。",
      users: "地方政府防災單位、保險公司、都市規劃者：找出高風險地區、排序防災預算、估算保費；模型必須可解釋，並提供不確定性範圍。",
      success: "測試集 R² ≥ 0.80，且 95% 預測區間的實際涵蓋率接近 95%；並與 Kaggle 主流解法（冠軍 R² ≈ 0.869）比較。",
    },
  },
  startups: {
    name: "50_Startups（老師範例網址，4 個特徵）",
    url: URL_STARTUPS,
    target: "Profit",
    source: URL_STARTUPS,
    note: "50 家新創公司：研發、管理、行銷支出與所在州 → 利潤。含類別欄位 State。",
    business: {
      problem: "依新創公司的<b>研發支出、管理費用、行銷支出與所在州</b>預測公司利潤 <b>Profit</b>。",
      users: "創投／投資人評估新創的獲利能力；經營者判斷預算投入哪個部門最有效。",
    },
  },
  scores: {
    name: "Student Scores（1 個特徵，簡單線性回歸）",
    url: URL_SCORES,
    target: "Scores",
    source: URL_SCORES,
    note: "25 位學生的讀書時數 → 考試分數。",
    business: {
      problem: "依<b>讀書時數 Hours</b> 預測<b>考試分數 Scores</b>；只有 1 個特徵，屬於簡單線性回歸。",
      users: "老師與學生估計達到目標分數需要投入多少讀書時間。",
    },
  },
};

const AR_SAMPLES = {
  tsmc: {
    name: "台積電 2330.TW（內建 5 年日資料）",
    path: "data/2330_TW.csv",
    date: "Date",
    value: "Close",
    source: "https://finance.yahoo.com/quote/2330.TW/history",
    note: "Yahoo Finance 日線資料（2021-10 ～ 2026-10）。",
  },
  aapl: {
    name: "Apple AAPL（plotly 公開資料網址）",
    url: URL_AAPL,
    date: "Date",
    value: "AAPL.Close",
    source: URL_AAPL,
    note: "2015-02 ～ 2017-02 的每日股價。",
  },
};

const TABS = ["business", "data", "prep", "model", "eval", "deploy"];
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const state = {
  problem: location.hash === "#ar" ? "ar" : "mlr",
  tab: "business",
  results: { mlr: null, ar: null },
  mlr: { mode: "sample", ds: null, target: "", excluded: new Set(), method: "kbest", k: null, kAuto: true, testRatio: 0.2, conf: 95, maxRows: 20000 },
  ar: { mode: "sample", ds: null, dateCol: "", valueCol: "", scale: "log", lagMode: "aic", pManual: 5, maxLag: 30, testRatio: 0.2, horizon: 20, conf: 95 },
};

// ---------- 共用 UI ----------
let busyCount = 0;
function busy(on, text = "計算中…") {
  busyCount = Math.max(0, busyCount + (on ? 1 : -1));
  $("#busy-text").textContent = text;
  $("#busy").hidden = busyCount === 0;
}
const nextFrame = () => new Promise((r) => setTimeout(r, 16));

function setStatus(problem, msg, kind = "ok") {
  const el = $(`#${problem}-status`);
  el.className = `status ${kind}`;
  el.innerHTML = msg;
}

function showAlert(msg) {
  const el = $("#alert");
  el.innerHTML = `<b>發生錯誤：</b>${esc(msg)}`;
  el.hidden = false;
}
const hideAlert = () => ($("#alert").hidden = true);

const fileName = (url) => decodeURIComponent(url.split("?")[0].split("/").pop() || url);

async function fetchLocal(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`找不到內建資料 ${path}`);
  return D.parseCsvBuffer(await r.arrayBuffer());
}

function segInit(id, problem, onChange) {
  const seg = $(`#${id}`);
  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-value]");
    if (!b) return;
    $$("button", seg).forEach((x) => x.classList.toggle("on", x === b));
    const group = seg.closest(".panel");
    $$(".pane", group).forEach((p) => (p.hidden = p.dataset.mode !== b.dataset.value));
    state[problem].mode = b.dataset.value;
    onChange?.(b.dataset.value);
  });
}

// ---------- 問題 ①：多元線性回歸 ----------
async function loadMLR(src) {
  const st = state.mlr;
  busy(true, "讀取資料中…");
  setStatus("mlr", "讀取中…", "info");
  try {
    let parsed;
    let meta;
    if (src.kind === "sample") {
      const s = MLR_SAMPLES[src.id];
      parsed = s.path ? await fetchLocal(s.path) : await D.fetchCsv(s.url);
      meta = { key: `sample:${src.id}`, name: s.name, source: s.source, note: s.note, target: s.target, business: s.business };
    } else if (src.kind === "url") {
      parsed = await D.fetchCsv(src.url);
      meta = { key: `url:${src.url}`, name: fileName(src.url), source: D.toRawUrl(src.url), note: "使用者輸入的 CSV 網址" };
    } else {
      parsed = D.parseCsvBuffer(await src.file.arrayBuffer());
      meta = { key: `file:${src.file.name}:${src.file.size}:${src.file.lastModified}`, name: src.file.name, source: `使用者上傳的檔案：${src.file.name}`, note: "使用者上傳" };
    }
    const profile = D.profileColumns(parsed.fields, parsed.rows);
    const numeric = profile.filter((c) => c.type === "numeric").map((c) => c.name);
    if (!numeric.length) throw new Error("這份資料沒有數值欄位，無法做回歸");
    st.ds = { ...meta, ...parsed, profile };
    st.target = meta.target && numeric.includes(meta.target) ? meta.target : numeric[numeric.length - 1];
    st.excluded = new Set(D.guessExcluded(profile, st.target));
    st.k = null;
    st.kAuto = true;
    setStatus("mlr", `✓ 已載入 <b>${esc(meta.name)}</b>：${parsed.rows.length.toLocaleString()} 筆 × ${parsed.fields.length} 欄`, "ok");
    fillMLRColumns();
    await compute();
  } catch (e) {
    console.error(e);
    setStatus("mlr", esc(e.message), "err");
  } finally {
    busy(false);
  }
}

function fillMLRColumns() {
  const st = state.mlr;
  const prof = st.ds.profile;
  $("#mlr-target").innerHTML = prof
    .filter((c) => c.type === "numeric")
    .map((c) => `<option value="${esc(c.name)}"${c.name === st.target ? " selected" : ""}>${esc(c.name)}</option>`)
    .join("");
  $("#mlr-exclude").innerHTML = prof
    .filter((c) => c.name !== st.target)
    .map(
      (c) => `<label class="check"><input type="checkbox" value="${esc(c.name)}"${st.excluded.has(c.name) ? " checked" : ""}>
        <span>${esc(c.name)}</span><em class="tag ${c.type === "numeric" ? "num" : "cat"}">${c.type === "numeric" ? "數值" : `類別 ${c.uniqueCapped ? "1000+" : c.unique}`}</em></label>`,
    )
    .join("");
  $("#mlr-kauto").checked = st.kAuto;
}

function bindMLR() {
  const st = state.mlr;
  $("#mlr-sample").innerHTML = Object.entries(MLR_SAMPLES)
    .map(([id, s]) => `<option value="${id}">${esc(s.name)}</option>`)
    .join("");
  const sampleNote = () => ($("#mlr-sample-note").textContent = MLR_SAMPLES[$("#mlr-sample").value].note);
  sampleNote();
  $("#mlr-method").innerHTML = Object.entries(FS_METHODS)
    .map(([id, m]) => `<option value="${id}">${esc(m.label)}｜${esc(m.type)}</option>`)
    .join("");
  segInit("mlr-mode", "mlr", (mode) => {
    if (mode === "sample" && st.ds?.key !== `sample:${$("#mlr-sample").value}`) loadMLR({ kind: "sample", id: $("#mlr-sample").value });
  });
  $("#mlr-sample").addEventListener("change", (e) => {
    sampleNote();
    loadMLR({ kind: "sample", id: e.target.value });
  });
  const loadUrl = () => {
    const url = $("#mlr-url").value.trim();
    if (url) loadMLR({ kind: "url", url });
  };
  $("#mlr-url-load").addEventListener("click", loadUrl);
  $("#mlr-url").addEventListener("keydown", (e) => e.key === "Enter" && loadUrl());
  $$("[data-url]", $("#mlr-url").closest(".pane")).forEach((b) =>
    b.addEventListener("click", () => {
      $("#mlr-url").value = b.dataset.url;
      loadUrl();
    }),
  );
  $("#mlr-file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) {
      $("#mlr-file-name").textContent = f.name;
      loadMLR({ kind: "file", file: f });
    }
  });
  $("#mlr-target").addEventListener("change", (e) => {
    st.target = e.target.value;
    st.excluded.delete(st.target);
    st.k = null;
    st.kAuto = true;
    fillMLRColumns();
    compute();
  });
  $("#mlr-exclude").addEventListener("change", (e) => {
    if (e.target.checked) st.excluded.add(e.target.value);
    else st.excluded.delete(e.target.value);
    st.k = null;
    compute();
  });
  $("#mlr-method").addEventListener("change", (e) => {
    st.method = e.target.value;
    compute();
  });
  $("#mlr-k").addEventListener("input", (e) => ($("#mlr-k-out").textContent = e.target.value));
  $("#mlr-k").addEventListener("change", (e) => {
    st.k = Number(e.target.value);
    st.kAuto = false;
    $("#mlr-kauto").checked = false;
    compute();
  });
  $("#mlr-kauto").addEventListener("change", (e) => {
    st.kAuto = e.target.checked;
    compute();
  });
  $("#mlr-test").addEventListener("input", (e) => ($("#mlr-test-out").textContent = `${Math.round(e.target.value * 100)}%`));
  $("#mlr-test").addEventListener("change", (e) => {
    st.testRatio = Number(e.target.value);
    compute();
  });
  $("#mlr-conf").addEventListener("change", (e) => {
    st.conf = Number(e.target.value);
    compute();
  });
  $("#mlr-max").addEventListener("change", (e) => {
    st.maxRows = Math.min(200000, Math.max(100, Number(e.target.value) || 20000));
    e.target.value = st.maxRows;
    compute();
  });
}

// ---------- 問題 ②：Auto Regression ----------
async function fetchStock(ticker, range) {
  const q = `?ticker=${encodeURIComponent(ticker)}&range=${range}`;
  let lastErr = null;
  let reached = false;
  for (const base of ["api/stock", REMOTE_STOCK_API].filter(Boolean)) {
    try {
      const r = await fetch(base + q);
      const type = r.headers.get("content-type") || "";
      if (r.ok && type.includes("text/csv")) {
        return { parsed: D.parseCsvBuffer(await r.arrayBuffer()), name: decodeURIComponent(r.headers.get("X-Stock-Name") || ticker) };
      }
      if (type.includes("application/json")) {
        reached = true;
        lastErr = new Error((await r.json()).error || `HTTP ${r.status}`);
        if (r.status === 400 || r.status === 404) break;
      } else lastErr = new Error(`股價 API 不存在（HTTP ${r.status}）`);
    } catch (e) {
      lastErr = e;
    }
  }
  throw new Error(
    `無法取得「${ticker}」的股價：${lastErr?.message || "未知錯誤"}。` +
      (reached
        ? "請確認代號是否正確（台股上市加 .TW、上櫃加 .TWO）。"
        : "股票代號功能需要 Vercel 版的 /api/stock；GitHub Pages 靜態版請改用「範例」、「CSV 網址」或「上傳檔案」。"),
  );
}

async function loadAR(src) {
  const st = state.ar;
  busy(true, "讀取資料中…");
  setStatus("ar", "讀取中…", "info");
  try {
    let parsed;
    let meta;
    if (src.kind === "sample") {
      const s = AR_SAMPLES[src.id];
      parsed = s.path ? await fetchLocal(s.path) : await D.fetchCsv(s.url);
      meta = { key: `sample:${src.id}`, name: s.name, source: s.source, date: s.date, value: s.value };
    } else if (src.kind === "ticker") {
      const r = await fetchStock(src.ticker, src.range);
      parsed = r.parsed;
      meta = { key: `yf:${src.ticker}:${src.range}`, name: `${src.ticker} ${r.name}`, source: `https://finance.yahoo.com/quote/${encodeURIComponent(src.ticker)}/history`, date: "Date", value: "Close" };
    } else if (src.kind === "url") {
      parsed = await D.fetchCsv(src.url);
      meta = { key: `url:${src.url}`, name: fileName(src.url), source: D.toRawUrl(src.url) };
    } else {
      parsed = D.parseCsvBuffer(await src.file.arrayBuffer());
      meta = { key: `file:${src.file.name}:${src.file.size}:${src.file.lastModified}`, name: src.file.name, source: `使用者上傳的檔案：${src.file.name}` };
    }
    const profile = D.profileColumns(parsed.fields, parsed.rows);
    if (!profile.some((c) => c.type === "numeric")) throw new Error("這份資料沒有數值欄位");
    st.ds = { ...meta, ...parsed, profile };
    st.dateCol = meta.date && parsed.fields.includes(meta.date) ? meta.date : D.guessDateColumn(parsed.fields, parsed.rows);
    st.valueCol = D.guessValueColumn(profile, meta.value);
    setStatus("ar", `✓ 已載入 <b>${esc(meta.name)}</b>：${parsed.rows.length.toLocaleString()} 筆`, "ok");
    fillARColumns();
    await compute();
  } catch (e) {
    console.error(e);
    setStatus("ar", esc(e.message), "err");
  } finally {
    busy(false);
  }
}

function fillARColumns() {
  const st = state.ar;
  const prof = st.ds.profile;
  $("#ar-date").innerHTML =
    `<option value="">（沒有日期，依列順序）</option>` +
    st.ds.fields.map((f) => `<option value="${esc(f)}"${f === st.dateCol ? " selected" : ""}>${esc(f)}</option>`).join("");
  $("#ar-value").innerHTML = prof
    .filter((c) => c.type === "numeric")
    .map((c) => `<option value="${esc(c.name)}"${c.name === st.valueCol ? " selected" : ""}>${esc(c.name)}</option>`)
    .join("");
}

function bindAR() {
  const st = state.ar;
  $("#ar-sample").innerHTML = Object.entries(AR_SAMPLES)
    .map(([id, s]) => `<option value="${id}">${esc(s.name)}</option>`)
    .join("");
  const sampleNote = () => ($("#ar-sample-note").textContent = AR_SAMPLES[$("#ar-sample").value].note);
  sampleNote();
  $("#ar-lagmode").innerHTML = Object.entries(LAG_MODES)
    .map(([id, l]) => `<option value="${id}">${l}</option>`)
    .join("");
  segInit("ar-mode", "ar", (mode) => {
    if (mode === "sample" && st.ds?.key !== `sample:${$("#ar-sample").value}`) loadAR({ kind: "sample", id: $("#ar-sample").value });
  });
  $("#ar-sample").addEventListener("change", (e) => {
    sampleNote();
    loadAR({ kind: "sample", id: e.target.value });
  });
  const loadTicker = () => {
    const ticker = $("#ar-ticker").value.trim().toUpperCase();
    if (ticker) loadAR({ kind: "ticker", ticker, range: $("#ar-range").value });
  };
  $("#ar-ticker-load").addEventListener("click", loadTicker);
  $("#ar-ticker").addEventListener("keydown", (e) => e.key === "Enter" && loadTicker());
  $$("[data-ticker]").forEach((b) =>
    b.addEventListener("click", () => {
      $("#ar-ticker").value = b.dataset.ticker;
      loadTicker();
    }),
  );
  const loadUrl = () => {
    const url = $("#ar-url").value.trim();
    if (url) loadAR({ kind: "url", url });
  };
  $("#ar-url-load").addEventListener("click", loadUrl);
  $("#ar-url").addEventListener("keydown", (e) => e.key === "Enter" && loadUrl());
  $("#ar-file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) {
      $("#ar-file-name").textContent = f.name;
      loadAR({ kind: "file", file: f });
    }
  });
  $("#ar-date").addEventListener("change", (e) => {
    st.dateCol = e.target.value;
    compute();
  });
  $("#ar-value").addEventListener("change", (e) => {
    st.valueCol = e.target.value;
    compute();
  });
  $("#ar-scale").addEventListener("change", (e) => {
    st.scale = e.target.value;
    compute();
  });
  $("#ar-lagmode").addEventListener("change", (e) => {
    st.lagMode = e.target.value;
    $("#ar-p-field").hidden = st.lagMode !== "manual";
    compute();
  });
  $("#ar-p").addEventListener("input", (e) => ($("#ar-p-out").textContent = e.target.value));
  $("#ar-p").addEventListener("change", (e) => {
    st.pManual = Number(e.target.value);
    compute();
  });
  $("#ar-maxlag").addEventListener("change", (e) => {
    st.maxLag = Math.min(60, Math.max(1, Number(e.target.value) || 30));
    e.target.value = st.maxLag;
    compute();
  });
  $("#ar-test").addEventListener("input", (e) => ($("#ar-test-out").textContent = `${Math.round(e.target.value * 100)}%`));
  $("#ar-test").addEventListener("change", (e) => {
    st.testRatio = Number(e.target.value);
    compute();
  });
  $("#ar-h").addEventListener("input", (e) => ($("#ar-h-out").textContent = e.target.value));
  $("#ar-h").addEventListener("change", (e) => {
    st.horizon = Number(e.target.value);
    compute();
  });
  $("#ar-conf").addEventListener("change", (e) => {
    st.conf = Number(e.target.value);
    compute();
  });
}

// ---------- 計算與畫面 ----------
async function compute() {
  const problem = state.problem;
  const st = state[problem];
  if (!st.ds) return;
  busy(true, "計算中…");
  await nextFrame();
  try {
    const R = problem === "mlr" ? runMLR(st.ds, st) : runAR(st.ds, st);
    state.results[problem] = R;
    hideAlert();
    if (problem === "mlr") {
      const k = $("#mlr-k");
      k.max = R.prep.features.length;
      k.value = R.k;
      k.disabled = R.prep.features.length < 2;
      $("#mlr-k-out").textContent = `${R.k}${st.kAuto ? "（CV 最佳）" : ""}`;
    } else {
      const p = $("#ar-p");
      p.max = R.maxLag;
      p.value = R.p;
      $("#ar-p-out").textContent = `${R.p}`;
    }
    render();
  } catch (e) {
    console.error(e);
    state.results[problem] = null;
    showAlert(e.message);
    render();
  } finally {
    busy(false);
  }
}

function render() {
  const R = state.results[state.problem];
  const root = $("#tab-content");
  if (!R) {
    $("#kpis").innerHTML = "";
    root.innerHTML = card("尚未有結果", `<p class="prose">請在左側選擇或修正資料來源與欄位設定。</p>`);
    return;
  }
  $("#kpis").innerHTML = kpiRow(state.problem === "mlr" ? mlrKpis(R) : arKpis(R));
  try {
    (state.problem === "mlr" ? renderMLR : renderAR)(state.tab, R, root);
  } catch (e) {
    console.error(e);
    root.innerHTML = card("畫面產生失敗", `<p class="alert inline">${esc(e.message)}</p>`);
  }
}

function switchProblem(problem) {
  state.problem = problem;
  $("#problem").value = problem;
  history.replaceState(null, "", problem === "ar" ? "#ar" : "#mlr");
  $$(".panel-group").forEach((g) => (g.hidden = g.dataset.problem !== problem));
  document.body.dataset.problem = problem;
  hideAlert();
  const st = state[problem];
  if (!st.ds) {
    render();
    if (problem === "mlr") loadMLR({ kind: "sample", id: $("#mlr-sample").value });
    else loadAR({ kind: "sample", id: $("#ar-sample").value });
  } else if (!state.results[problem]) compute();
  else render();
}

function init() {
  if (!window.Plotly || !window.Papa || !window.jStat) {
    showAlert("圖表或統計函式庫載入失敗，請確認網路連線後重新整理頁面。");
    return;
  }
  bindMLR();
  bindAR();
  $("#problem").addEventListener("change", (e) => switchProblem(e.target.value));
  $("#tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]");
    if (!b) return;
    state.tab = b.dataset.tab;
    $$("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
    render();
  });
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, select, textarea") || !/^[1-6]$/.test(e.key)) return;
    $(`#tabs button[data-tab="${TABS[Number(e.key) - 1]}"]`).click();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  window.addEventListener("hashchange", () => switchProblem(location.hash === "#ar" ? "ar" : "mlr"));
  if (REPO_URL) {
    const a = $("#repo-link");
    a.href = REPO_URL;
    a.hidden = false;
  }
  switchProblem(state.problem);
}

init();
