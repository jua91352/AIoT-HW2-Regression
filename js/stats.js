// 統計與線性代數核心（純 JavaScript，全部在瀏覽器內計算）
// - OLS：係數、標準誤、t 檢定、信賴區間 CI、預測區間 PI
// - 以 Gram 矩陣加速的 K-fold 交叉驗證，供特徵選擇（前向選擇、RFE、Lasso）與 Ridge 使用
// - 回歸假設檢定、時間序列工具（ACF / PACF / Ljung-Box / AR 多步預測）

const jstat = () => {
  if (!window.jStat) throw new Error("jStat 函式庫載入失敗，請檢查網路連線後重新整理");
  return window.jStat;
};

// ---------- 亂數與基本統計 ----------
export function rng(seed = 42) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffledIndex(n, seed = 42) {
  const r = rng(seed);
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function mean(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

export function sd(a, ddof = 1) {
  const m = mean(a);
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2;
  return Math.sqrt(s / (a.length - ddof));
}

export function median(a) {
  const b = Float64Array.from(a).filter((v) => !Number.isNaN(v)).sort();
  const n = b.length;
  if (!n) return NaN;
  return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2;
}

export function quantileSorted(s, q) {
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

export function corr(x, y) {
  const mx = mean(x);
  const my = mean(y);
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < x.length; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0;
}

export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function rmse(y, yhat) {
  let s = 0;
  for (let i = 0; i < y.length; i++) s += (y[i] - yhat[i]) ** 2;
  return Math.sqrt(s / y.length);
}

export function regMetrics(y, yhat, k) {
  const n = y.length;
  const my = mean(y);
  let sse = 0;
  let sae = 0;
  let sst = 0;
  for (let i = 0; i < n; i++) {
    const e = y[i] - yhat[i];
    sse += e * e;
    sae += Math.abs(e);
    sst += (y[i] - my) ** 2;
  }
  const r2 = sst > 0 ? 1 - sse / sst : NaN;
  return {
    n,
    MAE: sae / n,
    MSE: sse / n,
    RMSE: Math.sqrt(sse / n),
    R2: r2,
    AdjR2: n - k - 1 > 0 ? 1 - ((1 - r2) * (n - 1)) / (n - k - 1) : NaN,
  };
}

// ---------- 線性代數 ----------
// 部分樞軸高斯消去法解 Ax = b；矩陣奇異時回傳 null
export function solve(A, b) {
  const n = b.length;
  const M = A.map((r, i) => {
    const row = new Float64Array(n + 1);
    for (let j = 0; j < n; j++) row[j] = r[j];
    row[n] = b[i];
    return row;
  });
  let scale = 0;
  for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(M[i][i]));
  const tol = 1e-11 * (scale || 1);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < tol) return null;
    if (piv !== c) [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      if (f === 0) continue;
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

// Gauss-Jordan 求反矩陣；奇異時回傳 null
export function invert(A) {
  const n = A.length;
  const M = A.map((r, i) => {
    const row = new Float64Array(2 * n);
    for (let j = 0; j < n; j++) row[j] = r[j];
    row[n + i] = 1;
    return row;
  });
  let scale = 0;
  for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(M[i][i]));
  const tol = 1e-11 * (scale || 1);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < tol) return null;
    if (piv !== c) [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c];
    for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((r) => r.slice(n));
}

// ---------- 機率分布（jStat） ----------
export const tInv = (p, df) => jstat().studentt.inv(p, df);
export const tSf2 = (t, df) => Math.max(0, 2 * (1 - jstat().studentt.cdf(Math.abs(t), df)));
export const chi2Sf = (x, df) => Math.max(0, 1 - jstat().chisquare.cdf(x, df));
export const fSf = (x, d1, d2) => Math.max(0, 1 - jstat().centralF.cdf(x, d1, d2));
export const normInv = (p) => jstat().normal.inv(p, 0, 1);

// ---------- 普通最小平方法 OLS ----------
// Xcols：特徵欄（每欄一個 Float64Array）；內部先標準化再解正規方程式，數值較穩定
export function ols(Xcols, y, names = []) {
  const n = y.length;
  const k = Xcols.length;
  const p = k + 1;
  const df = n - p;
  if (df < 1) throw new Error(`樣本數不足：訓練資料只有 ${n} 筆，但模型有 ${p} 個參數`);
  const mx = new Float64Array(k);
  const sx = new Float64Array(k);
  const Z = Xcols.map((c, j) => {
    mx[j] = mean(c);
    sx[j] = sd(c) || 1;
    const z = new Float64Array(n);
    for (let i = 0; i < n; i++) z[i] = (c[i] - mx[j]) / sx[j];
    return z;
  });
  const ym = mean(y);
  const C = Array.from({ length: k }, () => new Float64Array(k));
  const cy = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    const zj = Z[j];
    let s = 0;
    for (let i = 0; i < n; i++) s += zj[i] * (y[i] - ym);
    cy[j] = s;
    for (let l = 0; l <= j; l++) {
      C[j][l] = C[l][j] = dot(zj, Z[l]);
    }
  }
  const Cinv = invert(C);
  if (!Cinv) throw new Error("特徵之間完全共線（某欄位是其他欄位的線性組合），請排除重複或衍生的欄位");
  const bz = Cinv.map((row) => dot(row, cy));
  const b = Float64Array.from(bz, (v, j) => v / sx[j]);
  let b0 = ym;
  for (let j = 0; j < k; j++) b0 -= b[j] * mx[j];

  const fitted = new Float64Array(n);
  const resid = new Float64Array(n);
  let ssr = 0;
  let sst = 0;
  for (let i = 0; i < n; i++) {
    let f = b0;
    for (let j = 0; j < k; j++) f += b[j] * Xcols[j][i];
    fitted[i] = f;
    resid[i] = y[i] - f;
    ssr += resid[i] ** 2;
    sst += (y[i] - ym) ** 2;
  }
  const s2 = ssr / df;
  const cov = (j, l) => (s2 * Cinv[j][l]) / (sx[j] * sx[l]);
  const se = new Float64Array(p);
  let v0 = s2 / n;
  for (let j = 0; j < k; j++) for (let l = 0; l < k; l++) v0 += mx[j] * mx[l] * cov(j, l);
  se[0] = Math.sqrt(v0);
  for (let j = 0; j < k; j++) se[j + 1] = Math.sqrt(cov(j, j));
  const coef = Float64Array.from([b0, ...b]);
  const t = coef.map((c, i) => c / se[i]);
  const pval = Array.from(t, (v) => tSf2(v, df));
  const r2 = sst > 0 ? 1 - ssr / sst : NaN;
  const F = k > 0 ? (sst - ssr) / k / (ssr / df) : NaN;
  const llf = (-n / 2) * (Math.log(2 * Math.PI) + Math.log(ssr / n) + 1);
  const sdY = sd(y);

  return {
    n, k, p, df, coef, se, t, pval, b0, b, mx, sx, Cinv, s2, ssr, sst, r2, F, llf, fitted, resid,
    names: ["截距 const", ...names],
    sigma: Math.sqrt(s2),
    adjR2: 1 - ((1 - r2) * (n - 1)) / df,
    Fp: k > 0 ? fSf(F, k, df) : NaN,
    aic: -2 * llf + 2 * p,
    bic: -2 * llf + Math.log(n) * p,
    stdCoef: Array.from(bz, (v) => v / sdY), // 標準化係數 β*（可比較重要性）
    confInt(alpha = 0.05) {
      const tc = tInv(1 - alpha / 2, df);
      return Array.from(coef, (c, i) => [c - tc * se[i], c + tc * se[i]]);
    },
    // 回傳每一筆新資料的預測值、平均值的信賴區間 CI、單一觀測值的預測區間 PI
    predict(cols, alpha = 0.05) {
      const m = cols.length ? cols[0].length : 0;
      const tc = tInv(1 - alpha / 2, df);
      const out = {
        mean: new Float64Array(m), ciLo: new Float64Array(m), ciHi: new Float64Array(m),
        piLo: new Float64Array(m), piHi: new Float64Array(m), seMean: new Float64Array(m),
      };
      const z = new Float64Array(k);
      for (let i = 0; i < m; i++) {
        let f = b0;
        for (let j = 0; j < k; j++) {
          const x = cols[j][i];
          f += b[j] * x;
          z[j] = (x - mx[j]) / sx[j];
        }
        let q = 0;
        for (let j = 0; j < k; j++) q += z[j] * dot(Cinv[j], z);
        const vm = s2 * (1 / n + q);
        const sm = Math.sqrt(vm);
        const sp = Math.sqrt(s2 + vm);
        out.mean[i] = f;
        out.seMean[i] = sm;
        out.ciLo[i] = f - tc * sm;
        out.ciHi[i] = f + tc * sm;
        out.piLo[i] = f - tc * sp;
        out.piHi[i] = f + tc * sp;
      }
      return out;
    },
    vif() {
      return Array.from({ length: k }, (_, j) => Cinv[j][j] * (n - 1));
    },
  };
}

// ---------- K-fold 交叉驗證（Gram 矩陣） ----------
// Zcols：訓練集標準化後的特徵；yc：置中後的 y。每一折只需累加一次 [1, z]ᵀ[1, z]，
// 之後任何特徵子集的訓練／驗證誤差都能由子矩陣直接算出，前向選擇與 Lasso 路徑因此很快。
export function makeCV(Zcols, yc, nFolds = 5, seed = 42) {
  const n = yc.length;
  const k = Zcols.length;
  const P = k + 1;
  const F = Math.max(2, Math.min(nFolds, n));
  const fold = new Int32Array(n);
  shuffledIndex(n, seed).forEach((row, pos) => (fold[row] = pos % F));
  const mk = () => Array.from({ length: P }, () => new Float64Array(P));
  const G = Array.from({ length: F }, mk);
  const g = Array.from({ length: F }, () => new Float64Array(P));
  const yy = new Float64Array(F);
  const cnt = new Float64Array(F);
  const v = new Float64Array(P);
  v[0] = 1;
  for (let i = 0; i < n; i++) {
    const f = fold[i];
    for (let j = 0; j < k; j++) v[j + 1] = Zcols[j][i];
    const Gf = G[f];
    const gf = g[f];
    const yi = yc[i];
    for (let a = 0; a < P; a++) {
      const va = v[a];
      const row = Gf[a];
      for (let b = a; b < P; b++) row[b] += va * v[b];
      gf[a] += va * yi;
    }
    yy[f] += yi * yi;
    cnt[f]++;
  }
  const Gt = mk();
  const gt = new Float64Array(P);
  let yyt = 0;
  for (let f = 0; f < F; f++) {
    for (let a = 0; a < P; a++) {
      for (let b = a + 1; b < P; b++) G[f][b][a] = G[f][a][b];
      for (let b = 0; b < P; b++) Gt[a][b] += G[f][a][b];
      gt[a] += g[f][a];
    }
    yyt += yy[f];
  }
  return { F, k, P, n, G, g, yy, cnt, Gt, gt, yyt };
}

const sub = (M, idx) => idx.map((a) => Float64Array.from(idx, (b) => M[a][b]));
const subv = (v, idx) => Float64Array.from(idx, (a) => v[a]);

function trainSystem(cv, f, idx) {
  const A = idx.map((a) => Float64Array.from(idx, (b) => cv.Gt[a][b] - cv.G[f][a][b]));
  const r = Float64Array.from(idx, (a) => cv.gt[a] - cv.g[f][a]);
  return { A, r };
}

function valSSE(cv, f, idx, beta) {
  let s = cv.yy[f];
  for (let i = 0; i < idx.length; i++) {
    s -= 2 * beta[i] * cv.g[f][idx[i]];
    const row = cv.G[f][idx[i]];
    for (let j = 0; j < idx.length; j++) s += beta[i] * beta[j] * row[idx[j]];
  }
  return Math.max(0, s);
}

const foldSST = (cv, f) => Math.max(1e-300, cv.yy[f] - cv.g[f][0] ** 2 / cv.cnt[f]);

// 特徵子集 S（0-based 特徵索引）的 CV RMSE 與 CV R²（各折平均）
export function cvScore(cv, S) {
  const idx = [0, ...S.map((j) => j + 1)];
  let rm = 0;
  let r2 = 0;
  for (let f = 0; f < cv.F; f++) {
    const { A, r } = trainSystem(cv, f, idx);
    const beta = solve(A, r);
    if (!beta) return { rmse: Infinity, r2: NaN };
    const sse = valSSE(cv, f, idx, beta);
    rm += Math.sqrt(sse / cv.cnt[f]);
    r2 += 1 - sse / foldSST(cv, f);
  }
  return { rmse: rm / cv.F, r2: r2 / cv.F };
}

// 只預測平均值的 baseline 的 CV 分數
export function cvBaseline(cv) {
  let rm = 0;
  let r2 = 0;
  for (let f = 0; f < cv.F; f++) {
    const m = (cv.gt[0] - cv.g[f][0]) / (cv.n - cv.cnt[f]);
    const sse = cv.yy[f] - 2 * m * cv.g[f][0] + cv.cnt[f] * m * m;
    rm += Math.sqrt(Math.max(0, sse) / cv.cnt[f]);
    r2 += 1 - sse / foldSST(cv, f);
  }
  return { rmse: rm / cv.F, r2: r2 / cv.F };
}

// 用全部訓練資料解 OLS（標準化空間）：回傳 [b0, b_S...]
export function gramSolve(cv, S) {
  const idx = [0, ...S.map((j) => j + 1)];
  return solve(sub(cv.Gt, idx), subv(cv.gt, idx));
}

export function predictZ(beta, Zcols, ym) {
  const m = Zcols[0].length;
  const out = new Float64Array(m).fill(ym + beta[0]);
  Zcols.forEach((z, j) => {
    const bj = beta[j + 1];
    for (let i = 0; i < m; i++) out[i] += bj * z[i];
  });
  return out;
}

// 前向逐步選擇：每一步加入讓 CV RMSE 最小的特徵
export function forwardOrder(cv, fallback, maxSteps = 40) {
  const rem = fallback.slice();
  const chosen = [];
  const stepRmse = [];
  const steps = Math.min(cv.k, maxSteps);
  while (chosen.length < steps) {
    let best = -1;
    let bestR = Infinity;
    for (const j of rem) {
      const r = cvScore(cv, [...chosen, j]).rmse;
      if (r < bestR) {
        bestR = r;
        best = j;
      }
    }
    if (best < 0) break;
    chosen.push(best);
    stepRmse.push(bestR);
    rem.splice(rem.indexOf(best), 1);
  }
  return { order: [...chosen, ...rem], stepRmse };
}

// RFE：每輪用剩下的特徵配適 OLS（標準化後），刪掉 |係數| 最小者；越晚刪越重要
export function rfeOrder(cv) {
  const rem = Array.from({ length: cv.k }, (_, j) => j);
  const elim = [];
  while (rem.length > 1) {
    const beta = gramSolve(cv, rem);
    let w = rem.length - 1;
    if (beta) {
      w = 0;
      for (let i = 1; i < rem.length; i++) if (Math.abs(beta[i + 1]) < Math.abs(beta[w + 1])) w = i;
    }
    elim.push(rem[w]);
    rem.splice(w, 1);
  }
  return [...rem, ...elim.reverse()];
}

// 由 Gram 矩陣得到「置中」後的 ZᵀZ 與 Zᵀy（f = -1 代表全部訓練資料，否則代表扣掉第 f 折）
function centered(cv, f, S) {
  const idx = S.map((j) => j + 1);
  const get = (a, b) => (f < 0 ? cv.Gt[a][b] : cv.Gt[a][b] - cv.G[f][a][b]);
  const getg = (a) => (f < 0 ? cv.gt[a] : cv.gt[a] - cv.g[f][a]);
  const n = get(0, 0);
  const zm = Float64Array.from(idx, (a) => get(0, a) / n);
  const ym = getg(0) / n;
  const C = idx.map((a, i) => Float64Array.from(idx, (b, j) => get(a, b) - n * zm[i] * zm[j]));
  const c = Float64Array.from(idx, (a, i) => getg(a) - n * zm[i] * ym);
  return { C, c, n, zm, ym, idx: [0, ...idx] };
}

function withIntercept(cc, b) {
  let b0 = cc.ym;
  for (let i = 0; i < b.length; i++) b0 -= cc.zm[i] * b[i];
  return Float64Array.from([b0, ...b]);
}

// Lasso 座標下降：min (1/2n)‖y − Zb‖² + λ‖b‖₁（與 scikit-learn 相同的目標函數）
function lassoCD(cc, lam, b, maxIter = 3000, tol = 1e-9) {
  const { C, c, n } = cc;
  const m = c.length;
  for (let it = 0; it < maxIter; it++) {
    let maxD = 0;
    let maxB = 0;
    for (let j = 0; j < m; j++) {
      const cjj = C[j][j];
      if (cjj <= 0) continue;
      const rho = (c[j] - dot(C[j], b) + cjj * b[j]) / n;
      const nb = (Math.sign(rho) * Math.max(Math.abs(rho) - lam, 0)) / (cjj / n);
      maxD = Math.max(maxD, Math.abs(nb - b[j]));
      b[j] = nb;
      maxB = Math.max(maxB, Math.abs(nb));
    }
    if (maxD <= tol * Math.max(1, maxB)) break;
  }
  return b;
}

export function lassoCV(cv, S, nLam = 60) {
  const m = S.length;
  const full = centered(cv, -1, S);
  const lamMax = Math.max(...Array.from(full.c, Math.abs)) / full.n || 1;
  const lambdas = Array.from({ length: nLam }, (_, i) => lamMax * 10 ** ((-3 * i) / (nLam - 1)));
  const cvR = new Float64Array(nLam);
  const cvR2 = new Float64Array(nLam);
  for (let f = 0; f < cv.F; f++) {
    const cc = centered(cv, f, S);
    const b = new Float64Array(m);
    lambdas.forEach((lam, li) => {
      lassoCD(cc, lam, b);
      const sse = valSSE(cv, f, cc.idx, withIntercept(cc, b));
      cvR[li] += Math.sqrt(sse / cv.cnt[f]) / cv.F;
      cvR2[li] += (1 - sse / foldSST(cv, f)) / cv.F;
    });
  }
  let best = 0;
  for (let i = 1; i < nLam; i++) if (cvR[i] < cvR[best]) best = i;
  const b = new Float64Array(m);
  const entry = new Array(m).fill(Infinity);
  let coefBest = null;
  lambdas.forEach((lam, li) => {
    lassoCD(full, lam, b);
    for (let j = 0; j < m; j++) if (b[j] !== 0 && entry[j] === Infinity) entry[j] = li;
    if (li === best) coefBest = b.slice();
  });
  const local = Array.from({ length: m }, (_, j) => j).sort(
    (a, z) => entry[a] - entry[z] || Math.abs(full.c[z]) - Math.abs(full.c[a]),
  );
  return {
    lambdas, cvRmse: cvR, best, lam: lambdas[best], coef: coefBest, S,
    beta: withIntercept(full, coefBest),
    order: local.map((j) => S[j]),
    entry: local.map((j) => entry[j]),
    cv: { rmse: cvR[best], r2: cvR2[best] },
    nonzero: coefBest.filter((v) => v !== 0).length,
  };
}

export function ridgeCV(cv, S, alphas = Array.from({ length: 22 }, (_, i) => 10 ** (-3 + i / 3))) {
  const folds = Array.from({ length: cv.F }, (_, f) => centered(cv, f, S));
  const fitB = (cc, a) => solve(cc.C.map((row, i) => row.map((v, j) => (i === j ? v + a : v))), cc.c);
  let best = null;
  for (const a of alphas) {
    let rm = 0;
    let r2 = 0;
    folds.forEach((cc, f) => {
      const sse = valSSE(cv, f, cc.idx, withIntercept(cc, fitB(cc, a)));
      rm += Math.sqrt(sse / cv.cnt[f]) / cv.F;
      r2 += (1 - sse / foldSST(cv, f)) / cv.F;
    });
    if (!best || rm < best.cv.rmse) best = { alpha: a, cv: { rmse: rm, r2 } };
  }
  const full = centered(cv, -1, S);
  return { ...best, S, beta: withIntercept(full, fitB(full, best.alpha)) };
}

// ---------- 回歸假設檢定 ----------
export function durbinWatson(e) {
  let num = 0;
  let den = e[0] ** 2;
  for (let i = 1; i < e.length; i++) {
    num += (e[i] - e[i - 1]) ** 2;
    den += e[i] ** 2;
  }
  return num / den;
}

export function jarqueBera(e) {
  const n = e.length;
  const m = mean(e);
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (let i = 0; i < n; i++) {
    const d = e[i] - m;
    m2 += d * d;
    m3 += d ** 3;
    m4 += d ** 4;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  const skew = m3 / m2 ** 1.5;
  const kurt = m4 / m2 ** 2;
  const jb = (n / 6) * (skew ** 2 + (kurt - 3) ** 2 / 4);
  return { jb, p: chi2Sf(jb, 2), skew, kurt };
}

// Breusch-Pagan（Koenker 版）：殘差平方對 X 回歸，LM = n·R²
export function breuschPagan(e, Xcols) {
  const aux = ols(Xcols, Float64Array.from(e, (v) => v * v));
  const lm = e.length * aux.r2;
  return { lm, p: chi2Sf(lm, Xcols.length) };
}

// Ramsey RESET：加入 ŷ² 後是否顯著改善（檢查線性關係）
export function resetTest(model, Xcols, y) {
  const aux = ols([...Xcols, Float64Array.from(model.fitted, (v) => v * v)], y);
  const F = (model.ssr - aux.ssr) / (aux.ssr / aux.df);
  return { F, p: fSf(F, 1, aux.df) };
}

// ---------- 時間序列 ----------
export function acf(x, nlags) {
  const n = x.length;
  const m = mean(x);
  let c0 = 0;
  for (let i = 0; i < n; i++) c0 += (x[i] - m) ** 2;
  const r = [1];
  for (let k = 1; k <= nlags; k++) {
    let s = 0;
    for (let t = 0; t < n - k; t++) s += (x[t] - m) * (x[t + k] - m);
    r.push(s / c0);
  }
  return r;
}

// Durbin-Levinson 遞迴求 PACF
export function pacf(x, nlags) {
  const r = acf(x, nlags);
  const out = [1];
  let phi = [];
  for (let k = 1; k <= nlags; k++) {
    let num = r[k];
    let den = 1;
    for (let j = 1; j < k; j++) {
      num -= phi[j - 1] * r[k - j];
      den -= phi[j - 1] * r[j];
    }
    const pkk = num / den;
    const next = phi.map((v, j) => v - pkk * phi[k - j - 2]);
    next.push(pkk);
    phi = next;
    out.push(pkk);
  }
  return out;
}

export function ljungBox(e, lags = 10) {
  const n = e.length;
  const r = acf(e, lags);
  let q = 0;
  for (let k = 1; k <= lags; k++) q += r[k] ** 2 / (n - k);
  q *= n * (n + 2);
  return { q, p: chi2Sf(q, lags), lags };
}

// AR(p) 遞迴多步預測；預測誤差變異數 σ²·Σψⱼ²（ψ 為 MA(∞) 係數，σ² = SSR/n，與 statsmodels AutoReg 相同）
export function arForecast(model, y, p, h, alpha = 0.05) {
  const sigma2 = model.ssr / model.n;
  const phi = model.b;
  const hist = Array.from(y.slice(y.length - p));
  const meanF = [];
  for (let i = 0; i < h; i++) {
    let f = model.b0;
    for (let j = 0; j < p; j++) f += phi[j] * hist[hist.length - 1 - j];
    meanF.push(f);
    hist.push(f);
  }
  const psi = [1];
  for (let j = 1; j < h; j++) {
    let s = 0;
    for (let i = 1; i <= Math.min(j, p); i++) s += phi[i - 1] * psi[j - i];
    psi.push(s);
  }
  const z = normInv(1 - alpha / 2);
  let cum = 0;
  const se = psi.map((v) => Math.sqrt(sigma2 * (cum += v * v)));
  return { mean: meanF, se, lo: meanF.map((m, i) => m - z * se[i]), hi: meanF.map((m, i) => m + z * se[i]) };
}

// 為每個 p 配適 AR(p)（同一段樣本，AIC/BIC 才可比較），並計算測試集一步預測 RMSE
export function lagScan(y, nTrain, maxLag) {
  const rows = [];
  for (let p = 1; p <= maxLag; p++) {
    const cols = Array.from({ length: p }, (_, i) => y.subarray(maxLag - i - 1, nTrain - i - 1));
    const m = ols(cols, y.subarray(maxLag, nTrain));
    const tcols = Array.from({ length: p }, (_, i) => y.subarray(nTrain - i - 1, y.length - i - 1));
    const pred = m.predict(tcols).mean;
    rows.push({ p, aic: m.aic, bic: m.bic, testRmse: rmse(y.subarray(nTrain), pred) });
  }
  return rows;
}
