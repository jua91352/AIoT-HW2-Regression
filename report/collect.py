"""從網頁實際執行結果擷取報告用的數字（report/results.json）與截圖（report/figures/）。

用法（專案根目錄）：.venv/Scripts/python report/collect.py
需要：Node.js（啟動本機伺服器）、Google Chrome、pip install playwright
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
PORT = 5174
BASE = f"http://localhost:{PORT}"
URL_STARTUPS = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
URL_CARS = "https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv"
URL_SCORES = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/studentscores.csv"

# 在頁面中直接呼叫 js/ 模組，數字與畫面上顯示的完全相同
EXTRACT_JS = """
async (urls) => {
  const D = await import('/js/data.js');
  const M = await import('/js/mlr.js');
  const A = await import('/js/ar.js');
  const arr = (a) => Array.from(a);
  const load = async (url, key) => {
    const p = D.parseCsvBuffer(await (await fetch(url)).arrayBuffer());
    return { key, name: key, source: url, ...p, profile: D.profileColumns(p.fields, p.rows) };
  };
  const mlrSettings = (ds, target, extra = {}) => ({
    target, excluded: new Set(D.guessExcluded(ds.profile, target)), method: 'kbest', k: null, kAuto: true,
    testRatio: 0.2, conf: 95, maxRows: 20000, ...extra,
  });
  const mlrSummary = (R, full) => {
    const out = {
      n: R.prep.n, nTrain: R.prep.yTrain.length, nTest: R.prep.yTest.length, p: R.prep.features.length,
      k: R.k, bestK: R.bestK, selected: R.feats.map(f => f.name), excluded: [...R.st.excluded],
      train: R.trainM, cv: R.cvSel, test: R.testM, coverage: R.coverage,
      info: { droppedTarget: R.prep.info.droppedTarget, imputed: R.prep.info.imputed, constant: R.prep.info.constant,
              dummies: R.prep.info.dummies, categorical: R.prep.info.categorical.map(c => ({ name: c.name, levels: c.levels.length, base: c.base })) },
    };
    if (full) {
      const ci = R.model.confInt(R.alpha);
      Object.assign(out, {
        rankOrder: R.rank.order.map(j => R.prep.features[j].name),
        rankScore: R.rank.order.map(j => R.rank.score[j]), rankLabel: R.rank.label,
        curve: R.curve,
        coef: R.model.names.map((nm, i) => ({ name: nm, coef: R.model.coef[i], se: R.model.se[i], t: R.model.t[i], p: R.model.pval[i], lo: ci[i][0], hi: ci[i][1] })),
        stdCoef: R.feats.map((f, i) => ({ name: f.name, beta: R.model.stdCoef[i] })),
        ols: { r2: R.model.r2, adjR2: R.model.adjR2, F: R.model.F, Fp: R.model.Fp, aic: R.model.aic, bic: R.model.bic, n: R.model.n, df: R.model.df, sigma: R.model.sigma },
        compare: R.compare.map(c => ({ name: c.name, desc: c.desc, k: c.k, cv: c.cv, test: c.test })),
        tests: R.tests.rows, vifMax: R.tests.vif ? Math.max(...R.tests.vif) : null,
        corr: R.prep.features.map((f, j) => ({ name: f.name, r: R.r[j] })),
        featureStats: R.prep.features.map(f => ({ name: f.name, mean: f.mean, sd: f.sd, min: f.min, max: f.max })),
      });
    }
    return out;
  };
  const out = {};

  const flood = await load('data/flood_train_sample.csv', 'flood');
  out.flood = mlrSummary(M.runMLR(flood, mlrSettings(flood, 'FloodProbability')), true);
  out.flood.rows = flood.rows.length;
  out.floodMethods = {};
  for (const method of ['kbest', 'rfe', 'forward', 'lasso']) {
    const t0 = performance.now();
    const R = M.runMLR(flood, mlrSettings(flood, 'FloodProbability', { method }));
    out.floodMethods[method] = { label: M.FS_METHODS[method].label, type: M.FS_METHODS[method].type, bestK: R.bestK,
      top5: R.rank.order.slice(0, 5).map(j => R.prep.features[j].name), testR2: R.testM.R2, ms: Math.round(performance.now() - t0),
      curve: R.curve.map(c => c.cvRmse) };
  }
  const k5 = M.runMLR(flood, mlrSettings(flood, 'FloodProbability', { k: 5, kAuto: false }));
  out.floodK5 = { test: k5.testM, coverage: k5.coverage };

  const startups = await load(urls.startups, 'startups');
  out.startups = mlrSummary(M.runMLR(startups, mlrSettings(startups, 'Profit')), true);
  const cars = await load(urls.cars, 'cars');
  out.cars = mlrSummary(M.runMLR(cars, mlrSettings(cars, 'price')), false);
  const scores = await load(urls.scores, 'scores');
  out.scores = mlrSummary(M.runMLR(scores, mlrSettings(scores, 'Scores')), false);

  const tsmc = await load('data/2330_TW.csv', 'tsmc');
  const arBase = { dateCol: 'Date', valueCol: 'Close', lagMode: 'aic', pManual: 5, maxLag: 30, testRatio: 0.2, horizon: 20, conf: 95 };
  out.ar = {};
  for (const scale of ['log', 'level']) {
    const R = A.runAR(tsmc, { ...arBase, scale });
    const ci = R.model.confInt(R.alpha);
    out.ar[scale] = {
      n: R.n, nTrain: R.nTrain, nTest: R.nTest, p: R.p, maxLag: R.maxLag,
      dates: { start: R.series.dates[0], trainEnd: R.series.dates[R.nTrain - 1], testStart: R.series.dates[R.nTrain], end: R.series.dates[R.n - 1] },
      first: R.y[0], last: R.y[R.n - 1], testFirst: R.y[R.nTrain],
      metrics: R.metrics, naive: R.naive, coverage: R.coverage, train: R.trainM,
      bestAic: R.scan.reduce((b, r) => r.aic < b.aic ? r : b).p, bestBic: R.scan.reduce((b, r) => r.bic < b.bic ? r : b).p,
      coef: R.model.names.map((nm, i) => ({ name: nm, coef: R.model.coef[i], se: R.model.se[i], t: R.model.t[i], p: R.model.pval[i], lo: ci[i][0], hi: ci[i][1] })),
      phiSum: arr(R.model.b).reduce((s, v) => s + v, 0), r2: R.model.r2, sigma: R.model.sigma,
      lb: R.lb, forecast: R.futureDates.slice(0, 10).map((d, i) => ({ date: d, mean: R.fc.mean[i], lo: R.fc.lo[i], hi: R.fc.hi[i] })),
    };
  }
  return out;
}
"""


def wait_server():
    for _ in range(50):
        try:
            urllib.request.urlopen(BASE, timeout=1)
            return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("本機伺服器沒有啟動")


def tab(page, name):
    page.evaluate(f"document.querySelector('#tabs button[data-tab=\"{name}\"]').click()")
    page.wait_for_timeout(900)


def shot_card(page, title, name, parent_grid=False):
    card = page.locator("section.card", has=page.locator("h3", has_text=title)).first
    target = card.locator("xpath=..") if parent_grid else card
    target.scroll_into_view_if_needed()
    page.wait_for_timeout(400)
    target.screenshot(path=str(FIG / name))
    print("  圖：", name)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(["node", "scripts/dev-server.js"], cwd=ROOT, env={**__import__("os").environ, "PORT": str(PORT)}, stdout=subprocess.DEVNULL)
    try:
        wait_server()
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="light", locale="zh-TW")
            page = ctx.new_page()
            page.goto(f"{BASE}/#mlr")
            page.wait_for_selector("#kpis .kpi", timeout=60000)

            print("擷取數字…")
            results = page.evaluate(EXTRACT_JS, {"startups": URL_STARTUPS, "cars": URL_CARS, "scores": URL_SCORES})
            (ROOT / "report" / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

            print("截圖（問題 ①：只留總覽與自訂網址）…")
            page.goto(f"{BASE}/#mlr")
            page.wait_for_selector("#kpis .kpi", timeout=60000)
            page.wait_for_timeout(800)
            page.screenshot(path=str(FIG / "mlr01_overview.png"))
            tab(page, "eval")
            shot_card(page, "預測圖：測試集預測值", "mlr02_prediction_interval.png")
            page.evaluate("window.scrollTo(0,0)")
            page.click("#mlr-mode button[data-value='url']")
            page.fill("#mlr-url", URL_STARTUPS)
            page.click("#mlr-url-load")
            page.wait_for_function("document.querySelector('#mlr-status').innerText.includes('50_Startups')", timeout=60000)
            tab(page, "model")
            page.evaluate("window.scrollTo(0,0)")
            page.wait_for_timeout(600)
            page.screenshot(path=str(FIG / "mlr03_custom_url.png"))

            print("截圖（問題 ②：自回歸六個分頁）…")
            page.goto(f"{BASE}/#ar")
            page.reload()
            page.wait_for_selector("#kpis .kpi", timeout=60000)
            page.wait_for_timeout(1000)
            page.screenshot(path=str(FIG / "ar01_page.png"))
            shot_card(page, "Business Understanding", "ar02_business.png")
            tab(page, "data")
            shot_card(page, "價格走勢", "ar03_price.png")
            shot_card(page, "ACF 自我相關函數", "ar04_acf_pacf.png", parent_grid=True)
            shot_card(page, "日報酬率分布", "ar05_returns.png", parent_grid=True)
            tab(page, "prep")
            shot_card(page, "Data Preparation 資料準備步驟", "ar06_prep_steps.png")
            shot_card(page, "落後特徵矩陣", "ar07_lag_matrix.png")
            shot_card(page, "訓練期與測試期", "ar08_split.png")
            tab(page, "model")
            shot_card(page, "特徵選擇：落後期數 p", "ar09_lag_text.png")
            shot_card(page, "AIC／BIC vs 落後期數", "ar10_lag_selection.png", parent_grid=True)
            shot_card(page, "迴歸結果（OLS）", "ar11_ar_table.png")
            tab(page, "eval")
            shot_card(page, "評估指標", "ar12_metrics.png")
            shot_card(page, "預測圖：歷史價格", "ar13_prediction.png")
            shot_card(page, "測試期放大", "ar14_test_interval.png")
            shot_card(page, "殘差時間序列", "ar15_residuals.png", parent_grid=True)
            shot_card(page, "殘差自我相關", "ar16_resid_acf.png")
            tab(page, "deploy")
            shot_card(page, "Deployment 部署", "ar17_forecast.png")
            shot_card(page, "預測明細", "ar18_forecast_table.png")

            print("截圖（原始價格尺度對照）…")
            page.select_option("#ar-scale", "level")
            page.wait_for_timeout(1500)
            tab(page, "eval")
            shot_card(page, "評估指標", "ar19_metrics_level.png")

            print("截圖（換資料：股票代號、CSV 網址）…")
            page.goto(f"{BASE}/#ar")
            page.reload()
            page.wait_for_selector("#kpis .kpi", timeout=60000)
            page.click("#ar-mode button[data-value='ticker']")
            page.fill("#ar-ticker", "0050.TW")
            page.click("#ar-ticker-load")
            page.wait_for_function("document.querySelector('#ar-status').innerText.includes('0050')", timeout=60000)
            tab(page, "eval")
            page.evaluate("window.scrollTo(0,0)")
            page.wait_for_timeout(800)
            page.screenshot(path=str(FIG / "ar20_ticker.png"))
            page.click("#ar-mode button[data-value='url']")
            page.click("#ar-url-load")
            page.wait_for_function("document.querySelector('#ar-status').innerText.includes('finance-charts-apple')", timeout=60000)
            tab(page, "data")
            page.evaluate("window.scrollTo(0,0)")
            page.wait_for_timeout(800)
            page.screenshot(path=str(FIG / "ar21_csv_url.png"))

            print("截圖（手機版）…")
            m = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, color_scheme="light", is_mobile=True)
            mp = m.new_page()
            mp.goto(f"{BASE}/#ar")
            mp.wait_for_selector("#kpis .kpi", timeout=60000)
            mp.wait_for_timeout(1000)
            mp.screenshot(path=str(FIG / "ar22_mobile.png"))
            browser.close()
    finally:
        server.terminate()
    print("完成：report/results.json、report/figures/")


if __name__ == "__main__":
    sys.exit(main())
