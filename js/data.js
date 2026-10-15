// 資料讀取與前處理：CSV 網址／上傳檔案、欄位型態判斷、one-hot、補值、切分訓練／測試集、時間序列整理
import { shuffledIndex, median, mean, sd } from "./stats.js";

// GitHub 頁面網址 → raw；Google 試算表 → 匯出 CSV
export function toRawUrl(input) {
  const url = input.trim();
  let m = url.match(/^https?:\/\/github\.com\/([^/]+)\/([^/]+)\/(?:blob|raw)\/(.+)$/i);
  if (m) return `https://raw.githubusercontent.com/${m[1]}/${m[2]}/${m[3]}`;
  m = url.match(/^https?:\/\/docs\.google\.com\/spreadsheets\/d\/([^/]+)/i);
  if (m) {
    const gid = (url.match(/[#&?]gid=(\d+)/) || [])[1];
    return `https://docs.google.com/spreadsheets/d/${m[1]}/export?format=csv${gid ? `&gid=${gid}` : ""}`;
  }
  return url;
}

export async function fetchCsv(input) {
  const url = toRawUrl(input);
  if (!/^https?:\/\//i.test(url)) throw new Error("網址必須以 http:// 或 https:// 開頭");
  let resp;
  try {
    resp = await fetch(url);
  } catch {
    throw new Error(
      "無法下載這個網址：可能網址錯誤，或該網站不允許瀏覽器跨網域讀取（CORS）。建議使用 GitHub raw 連結，或先下載再用「上傳檔案」。",
    );
  }
  if (!resp.ok) throw new Error(`下載失敗（HTTP ${resp.status}），請確認網址可以公開存取`);
  return parseCsvBuffer(await resp.arrayBuffer());
}

export function parseCsvBuffer(buf) {
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(buf);
  } catch {
    text = new TextDecoder("big5").decode(buf); // 台灣政府開放資料常見的 Big5 編碼
  }
  text = text.replace(/^﻿/, "");
  if (/^\s*<(!doctype|html)/i.test(text)) {
    throw new Error(
      "這個網址回傳的是網頁（HTML）而不是 CSV。GitHub 請按檔案頁面的「Raw」再複製網址；Kaggle 需要登入，請先下載後用「上傳檔案」。",
    );
  }
  const res = window.Papa.parse(text, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: "greedy",
    transformHeader: (h, i) => String(h).trim() || `欄位${i + 1}`,
  });
  const fields = res.meta.fields || [];
  if (fields.length < 2) throw new Error("讀到的資料少於 2 個欄位，請確認檔案是逗號（或 Tab／分號）分隔的 CSV");
  if (!res.data.length) throw new Error("CSV 沒有任何資料列");
  return { fields, rows: res.data };
}

export function isMissing(v) {
  if (v === null || v === undefined) return true;
  if (typeof v === "number") return Number.isNaN(v);
  return /^\s*(|na|n\/a|nan|null|none|-)\s*$/i.test(String(v));
}

export function toNumber(v) {
  if (typeof v === "number") return Number.isFinite(v) ? v : NaN;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (v === null || v === undefined) return NaN;
  const s = String(v).trim().replace(/,/g, "");
  return /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s) ? parseFloat(s) : NaN;
}

// 判斷每個欄位是數值或類別，並計算摘要統計
export function profileColumns(fields, rows) {
  return fields.map((name) => {
    let nonEmpty = 0;
    let nNum = 0;
    const uniq = new Set();
    const nums = [];
    let sample = "";
    for (const r of rows) {
      const v = r[name];
      if (isMissing(v)) continue;
      if (!nonEmpty) sample = v;
      nonEmpty++;
      const x = toNumber(v);
      if (Number.isFinite(x)) {
        nNum++;
        nums.push(x);
      }
      if (uniq.size <= 1000) uniq.add(String(v));
    }
    const type = nonEmpty > 0 && nNum / nonEmpty >= 0.95 ? "numeric" : "categorical";
    const col = { name, type, missing: rows.length - nonEmpty, unique: uniq.size, uniqueCapped: uniq.size > 1000, sample };
    if (type === "numeric" && nums.length) {
      let mn = Infinity;
      let mx = -Infinity;
      for (const x of nums) {
        if (x < mn) mn = x;
        if (x > mx) mx = x;
      }
      Object.assign(col, { mean: mean(nums), sd: nums.length > 1 ? sd(nums) : 0, min: mn, max: mx });
      col.isIdLike =
        uniq.size === rows.length && rows.length > 20 && nums.every(Number.isInteger) && nums.every((x, i) => !i || x > nums[i - 1]);
    }
    return col;
  });
}

// 預設排除：ID／流水號、名稱類的高基數文字欄位
export function guessExcluded(profile, target) {
  return profile
    .filter((c) => c.name !== target)
    .filter(
      (c) =>
        /^(id|index|no\.?|編號|序號|流水號)$/i.test(c.name) ||
        /(^|[_\s])id$/i.test(c.name) ||
        /^unnamed/i.test(c.name) ||
        (c.type === "categorical" && c.unique > 30) ||
        c.isIdLike,
    )
    .map((c) => c.name);
}

export function sampleRows(rows, max, seed = 7) {
  if (rows.length <= max) return rows;
  const idx = shuffledIndex(rows.length, seed).slice(0, max).sort((a, b) => a - b);
  return idx.map((i) => rows[i]);
}

// 多元線性回歸的資料準備
export function prepareMLR({ rows, profile, target, excluded, testRatio, seed = 42 }) {
  const info = { droppedTarget: 0, imputed: 0, constant: [], dummies: [], numeric: [], categorical: [] };
  const valid = [];
  const yAll = [];
  for (const r of rows) {
    const v = toNumber(r[target]);
    if (Number.isFinite(v)) {
      valid.push(r);
      yAll.push(v);
    } else info.droppedTarget++;
  }
  const n = valid.length;
  if (n < 10) throw new Error(`有效資料只有 ${n} 筆（目標欄位「${target}」必須是數值），至少需要 10 筆`);

  const feats = [];
  for (const col of profile) {
    if (col.name === target || excluded.has(col.name)) continue;
    if (col.type === "numeric") {
      feats.push({ name: col.name, kind: "num", source: col.name, values: Float64Array.from(valid, (r) => toNumber(r[col.name])) });
      info.numeric.push(col.name);
    } else {
      const vals = valid.map((r) => (isMissing(r[col.name]) ? "（缺失）" : String(r[col.name]).trim()));
      const levels = [...new Set(vals)].sort((a, b) => a.localeCompare(b, "zh-Hant"));
      if (levels.length < 2) {
        info.constant.push(col.name);
        continue;
      }
      info.categorical.push({ name: col.name, levels, base: levels[0] });
      for (const lv of levels.slice(1)) {
        const name = `${col.name}=${lv}`;
        feats.push({ name, kind: "dummy", source: col.name, level: lv, values: Float64Array.from(vals, (v) => (v === lv ? 1 : 0)) });
        info.dummies.push(name);
      }
    }
  }
  if (!feats.length) throw new Error("沒有可用的特徵欄位，請在左側取消排除部分欄位");
  if (feats.length > 120) throw new Error(`one-hot 之後有 ${feats.length} 個特徵，太多了，請在左側排除類別很多的欄位`);

  const order = shuffledIndex(n, seed);
  const nTest = Math.max(2, Math.round(n * testRatio));
  const testIdx = order.slice(0, nTest).sort((a, b) => a - b);
  const trainIdx = order.slice(nTest).sort((a, b) => a - b);
  const pick = (arr, idx) => Float64Array.from(idx, (i) => arr[i]);
  const yTrain = pick(yAll, trainIdx);
  const yTest = pick(yAll, testIdx);

  const kept = [];
  for (const f of feats) {
    const tr = pick(f.values, trainIdx);
    const te = pick(f.values, testIdx);
    if (f.kind === "num") {
      const med = median(tr); // 缺失值以「訓練集」中位數補
      f.median = med;
      for (const a of [tr, te])
        for (let i = 0; i < a.length; i++)
          if (Number.isNaN(a[i])) {
            a[i] = med;
            info.imputed++;
          }
    }
    let mn = Infinity;
    let mx = -Infinity;
    for (const v of tr) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (!(mx > mn)) {
      info.constant.push(f.name);
      continue;
    }
    Object.assign(f, { train: tr, test: te, mean: mean(tr), sd: sd(tr), min: mn, max: mx, isInt: tr.every(Number.isInteger) });
    delete f.values;
    kept.push(f);
  }
  if (!kept.length) throw new Error("所有特徵在訓練集都是常數，無法建模");
  const ym = mean(yTrain);
  const Z = (f, a) => Float64Array.from(a, (v) => (v - f.mean) / f.sd);
  return {
    n, info, trainIdx, testIdx, yTrain, yTest, ym,
    features: kept,
    Xtrain: kept.map((f) => f.train),
    Xtest: kept.map((f) => f.test),
    Ztrain: kept.map((f) => Z(f, f.train)),
    Ztest: kept.map((f) => Z(f, f.test)),
    ycTrain: Float64Array.from(yTrain, (v) => v - ym),
  };
}

// ---------- 時間序列 ----------
// 支援 2024-01-31、2024/1/31、20240131、民國 113/01/31
export function parseDate(v) {
  if (v === null || v === undefined || v === "") return null;
  const s = String(v).trim();
  let m = s.match(/^(\d{2,4})[-/.](\d{1,2})[-/.](\d{1,2})/);
  if (m) {
    let y = +m[1];
    if (y < 1000) y += 1911;
    return Date.UTC(y, +m[2] - 1, +m[3]);
  }
  m = s.match(/^(\d{4})(\d{2})(\d{2})$/);
  if (m) return Date.UTC(+m[1], +m[2] - 1, +m[3]);
  const t = Date.parse(s);
  if (Number.isNaN(t)) return null;
  const d = new Date(t);
  return Date.UTC(d.getFullYear(), d.getMonth(), d.getDate());
}

export const isoDate = (t) => new Date(t).toISOString().slice(0, 10);

export function guessDateColumn(fields, rows) {
  const byName = fields.find((f) => /date|time|日期|時間|年月日/i.test(f));
  if (byName) return byName;
  const head = rows.slice(0, 50);
  return fields.find((f) => head.filter((r) => typeof r[f] === "string" && parseDate(r[f]) !== null).length >= head.length * 0.9) || "";
}

export function guessValueColumn(profile, preferred) {
  const nums = profile.filter((c) => c.type === "numeric").map((c) => c.name);
  if (preferred && nums.includes(preferred)) return preferred;
  return nums.find((n) => /^close$/i.test(n)) || nums.find((n) => /close|收盤/i.test(n)) || nums[nums.length - 1] || "";
}

export function buildSeries(rows, dateCol, valueCol) {
  const pts = [];
  rows.forEach((r, i) => {
    const v = toNumber(r[valueCol]);
    if (!Number.isFinite(v)) return;
    let t = i;
    if (dateCol) {
      t = parseDate(r[dateCol]);
      if (t === null) return;
    }
    pts.push({ t, v });
  });
  let dropped = rows.length - pts.length;
  let dup = 0;
  let list = pts;
  if (dateCol) {
    list = [];
    pts.sort((a, b) => a.t - b.t).forEach((p) => {
      if (list.length && list[list.length - 1].t === p.t) {
        list[list.length - 1] = p;
        dup++;
      } else list.push(p);
    });
  }
  return {
    hasDates: !!dateCol,
    times: list.map((p) => p.t),
    dates: list.map((p, i) => (dateCol ? isoDate(p.t) : i + 1)),
    y: Float64Array.from(list, (p) => p.v),
    dropped,
    dup,
  };
}

// 從最後一天往後推算 h 個營業日（略過週六、週日）
export function nextBusinessDays(lastT, h) {
  const out = [];
  let t = lastT;
  while (out.length < h) {
    t += 86400000;
    const d = new Date(t).getUTCDay();
    if (d !== 0 && d !== 6) out.push(isoDate(t));
  }
  return out;
}
