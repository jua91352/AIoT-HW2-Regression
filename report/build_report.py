"""產生報告書（PDF）與資訊圖表（PNG＋PDF），內容以問題 ② 自迴歸為主。

用法（專案根目錄）：
  .venv/Scripts/python report/collect.py        # 先擷取數字與截圖
  .venv/Scripts/python report/build_report.py   # 再產生報告書與資訊圖表
"""
import html
import io
import json
from datetime import date
from pathlib import Path

import qrcode
import qrcode.image.svg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "report"
R = json.loads((REP / "results.json").read_text(encoding="utf-8"))
STUDENT = "游雅婷"
SID = "5115056031"
COURSE = "物聯網應用與資料分析（7716）"
URL_WEB = "https://aiot-hw2-regression.vercel.app"
URL_PAGES = "https://jua91352.github.io/AIoT-HW2-Regression/"
URL_REPO = "https://github.com/jua91352/AIoT-HW2-Regression"
URL_KAGGLE = "https://www.kaggle.com/competitions/playground-series-s4e5"
URL_STARTUPS = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
TODAY = date.today().isoformat()

AR = R["ar"]["log"]
ARL = R["ar"]["level"]
FL = R["flood"]
SU = R["startups"]
NB_STOCK = Path(__file__).with_name("notebooklm_summary_stock.txt").read_text(encoding="utf-8").strip()
NB_FLOOD = Path(__file__).with_name("notebooklm_summary_flood.txt").read_text(encoding="utf-8").strip()


def render_summary(text):
    """NotebookLM 摘要原文 → HTML：保留文字，只轉換標題、清單與 LaTeX 公式的顯示方式。"""
    math = {r"$\hat{Y}_{t+1} = Y_t$": "Ŷ<sub>t+1</sub> = Y<sub>t</sub>", "$R^2$": "R²"}
    out, mode = [], None

    def close():
        nonlocal mode
        if mode:
            out.append(f"</{mode}>")
            mode = None

    def inline(s):
        s = esc(s)
        for k, v in math.items():
            s = s.replace(esc(k), v)
        if "：" in s:
            head, rest = s.split("：", 1)
            s = f"<b>{head}</b>：{rest}"
        return s

    import re

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        num = re.match(r"^\d+\.\s+(.*)", line)
        if line.startswith("* "):
            if mode != "ul":
                close()
                out.append("<ul>")
                mode = "ul"
            out.append(f"<li>{inline(line[2:])}</li>")
        elif num and "：" in line:
            if mode != "ol":
                close()
                out.append("<ol>")
                mode = "ol"
            out.append(f"<li>{inline(num.group(1))}</li>")
        elif num:
            close()
            out.append(f'<p class="nbh">{esc(line)}</p>')
        else:
            close()
            out.append(f"<p>{inline(line) if not line.startswith('以下') else esc(line)}</p>")
    close()
    return "".join(out)

# 1014 HW2（Python 全資料 1,117,957 筆）的模型比較結果，作為主流解法對照
PY_MODELS = [
    ("MLR（OLS，20 個特徵）", "線性", 0.84488),
    ("MLR + 列統計特徵工程", "線性", 0.86579),
    ("HistGradientBoosting（原始特徵）", "樹模型", 0.83552),
    ("HistGradientBoosting + 特徵工程", "樹模型", 0.86885),
]


def fmt(v, d=None):
    if v is None or v != v:
        return "—"
    a = abs(v)
    if d is not None:
        return f"{v:,.{d}f}"
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 1:
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    if a >= 1e-3:
        return f"{v:.4g}"
    return f"{v:.2e}"


def pct(v, d=1):
    return f"{v * 100:.{d}f}%"


def fp(p):
    if p is None or p != p:
        return "—"
    if p < 1e-16:
        return "&lt; 1e-16"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


def esc(s):
    return html.escape(str(s))


def table(head, rows, num=()):
    th = "".join(f'<th class="{"num" if i in num else ""}">{h}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


class Figs:
    def __init__(self):
        self.n = 0

    def __call__(self, name, caption, width="100%"):
        self.n += 1
        return f'<figure><img src="figures/{name}" style="width:{width}"><figcaption>圖 {self.n}　{caption}</figcaption></figure>'


def qr_svg(url):
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=1)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode("utf-8").split("?>", 1)[-1]


# ---------------------------------------------------------------- 報告書
REPORT_CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft JhengHei", "Noto Sans TC", sans-serif; font-size: 10.5pt; line-height: 1.7; color: #1d1d1b; margin: 0; }
h1 { font-size: 24pt; line-height: 1.3; margin: 0 0 8px; color: #0b0b0b; }
h2 { font-size: 15pt; color: #1c5cab; border-bottom: 2px solid #2a78d6; padding-bottom: 4px; margin: 26px 0 10px; break-after: avoid; }
h3 { font-size: 12pt; color: #0b0b0b; margin: 18px 0 6px; break-after: avoid; }
p { margin: 6px 0; }
ul, ol { margin: 6px 0; padding-left: 22px; } li { margin: 2px 0; }
a { color: #1c5cab; word-break: break-all; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f0efeb; padding: 0 4px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9.5pt; break-inside: avoid; }
th, td { border: 1px solid #e1e0d9; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f0efeb; font-weight: bold; }
td.num, th.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
figure { margin: 10px 0 14px; text-align: center; break-inside: avoid; }
figure img { border: 1px solid #e1e0d9; border-radius: 6px; max-width: 100%; }
figcaption { font-size: 9pt; color: #52514e; margin-top: 4px; }
.cover { height: 250mm; display: flex; flex-direction: column; break-after: page; }
.cover .tag { color: #2a78d6; font-weight: bold; letter-spacing: .08em; font-size: 11pt; }
.cover .sub { font-size: 13pt; color: #52514e; margin: 0 0 18px; }
.cover .meta { margin-top: auto; font-size: 11pt; line-height: 2; border-top: 1px solid #e1e0d9; padding-top: 12px; }
.cover img { width: 100%; border: 1px solid #e1e0d9; border-radius: 8px; margin: 10px 0; }
.box { background: #f4f8fd; border: 1px solid #cde2fb; border-radius: 8px; padding: 10px 14px; margin: 10px 0; break-inside: avoid; }
.warn { background: #fff8e8; border-color: #fbe3a6; }
.label { display: inline-block; background: #2a78d6; color: #fff; font-size: 9pt; font-weight: bold; padding: 1px 10px; border-radius: 999px; margin-right: 6px; vertical-align: middle; }
.label.nb { background: #1baf7a; }
.kpis { display: flex; gap: 8px; margin: 10px 0; }
.kpi { flex: 1; background: #f0efeb; border-radius: 8px; padding: 8px 10px; }
.kpi b { display: block; font-size: 15pt; color: #0b0b0b; }
.kpi span { font-size: 8.5pt; color: #52514e; }
.toc li { margin: 3px 0; }
.pb { break-before: page; }
.keep { break-inside: avoid; }
.ok { color: #006300; font-weight: bold; } .ng { color: #9a6500; font-weight: bold; }
blockquote { margin: 8px 0; padding: 8px 14px; border-left: 4px solid #1baf7a; background: #f3faf6; }
blockquote .nbh { font-weight: bold; color: #0b6b4a; margin-top: 10px; }
blockquote ul, blockquote ol { margin: 4px 0 6px; }
"""


def build_report_html():
    f = Figs()
    a, al = AR, ARL
    m, nv, ml = a["metrics"], a["naive"], al["metrics"]
    fc = a["forecast"]
    phi = a["coef"][1]
    const = a["coef"][0]
    req_rows = [
        ["加分 +1：下拉視窗呈現兩種 problem", "頁首「選擇問題 Problem」：① 多元線性回歸、② 自迴歸 Auto Regression", "第二章、圖 1"],
        ["加分 +2：可任意輸入 Dataset", "範例資料／股票代號／任意 CSV 網址／上傳檔案；50_Startups 網址實測", "第四章"],
        ["線性回歸：單純、多元或 Auto Regression", "主體為 AR(p)（以落後值為特徵的多元線性回歸）；問題 ① 為 MLR", "第三章、第五章"],
        ["Kaggle 資料集、特徵 > 13、標示來源", "問題 ①：Kaggle Predict the Flood，20 個特徵，附連結", "第五章"],
        ["特徵選擇 Feature Selection", "AR：以 AIC／BIC 選擇落後期數 p；MLR：4 種方法＋CV", "3.4 節"],
        ["模型評估 Model Evaluation", "MAE、MSE、RMSE、MAPE、R²、方向準確率、Ljung-Box；與 Naive 比較", "3.5 節"],
        ["預測圖＋信賴區間或預測區間", "測試期一步預測＋95% CI／PI；未來 20 日預測＋95% PI", "圖 13、14、18"],
        ["CRISP-DM 六步驟", "網頁六個分頁；本報告第三章逐步說明", "第三章"],
        ["GitHub + web", f'<a href="{URL_REPO}">{URL_REPO}</a><br><a href="{URL_WEB}">{URL_WEB}</a>', "3.6 節"],
        ["NotebookLM 摘要、GPT 輔助內容", "股價預測摘要（第七章）、洪水預測摘要（5.1 節）；GPT 輔助內容（第八章）；AI 對話紀錄 PDF 另行上傳", "第七、八章、5.1 節"],
    ]
    metric_rows = [
        ["MAE", fmt(m["MAE"]), fmt(nv["MAE"]), fmt(ml["MAE"])],
        ["MSE", fmt(m["MSE"]), fmt(nv["MSE"]), fmt(ml["MSE"])],
        ["RMSE", fmt(m["RMSE"]), fmt(nv["RMSE"]), fmt(ml["RMSE"])],
        ["MAPE", pct(m["MAPE"], 2), pct(nv["MAPE"], 2), pct(ml["MAPE"], 2)],
        ["R²", fmt(m["R2"]), fmt(nv["R2"]), fmt(ml["R2"])],
        ["漲跌方向準確率", pct(m["dirAcc"]), "—", pct(ml["dirAcc"])],
        ["95% 預測區間涵蓋率", f'<span class="ok">{pct(a["coverage"])}</span>', "—", f'<span class="ng">{pct(al["coverage"])}</span>'],
    ]
    coef_rows = [[esc(c["name"]), fmt(c["coef"], 6), fmt(c["se"], 6), fmt(c["t"], 2), fp(c["p"]), fmt(c["lo"], 6), fmt(c["hi"], 6)] for c in a["coef"]]
    fc_rows = [[x["date"], fmt(x["mean"], 1), fmt(x["lo"], 1), fmt(x["hi"], 1)] for x in fc]
    fl_methods = [[v["label"], v["type"], str(v["bestK"]), "、".join(v["top5"][:3]) + "…", f'{v["ms"]} ms'] for v in R["floodMethods"].values()]
    py_rows = [[n, t, fmt(r2, 4)] for n, t, r2 in PY_MODELS]
    py_rows.insert(0, ["本平台：MLR（OLS，瀏覽器，抽樣 20,000 筆）", "線性", fmt(FL["test"]["R2"], 4)])
    py_rows.append(["Kaggle 前幾名（CatBoost／XGBoost／LightGBM 集成）", "集成", "≈ 0.869"])
    su_coef = "、".join(f'{esc(c["name"])} {fmt(c["coef"])}' for c in SU["coef"][1:])

    body = f"""
<section class="cover">
  <div class="tag">HW2：MULTIPLE LINEAR REGRESSION</div>
  <h1>線性回歸預測平台 報告書<br>自迴歸（Auto Regression）股價預測</h1>
  <p class="sub">依 CRISP-DM 流程｜互動網頁：下拉選單切換兩種問題、使用者可自行更換資料來源</p>
  <img src="figures/ar01_page.png">
  <div class="meta">
    課程：{COURSE}<br>
    學生：{STUDENT}　學號：{SID}<br>
    網頁：<a href="{URL_WEB}">{URL_WEB}</a>（GitHub Pages：<a href="{URL_PAGES}">{URL_PAGES}</a>）<br>
    原始碼：<a href="{URL_REPO}">{URL_REPO}</a><br>
    日期：{TODAY}
  </div>
</section>

<h2>摘要</h2>
<div class="box">
<p>本作業製作一個可直接部署的互動式線性回歸預測平台：頁首下拉選單可切換「① 多元線性回歸」與「② 自迴歸 Auto Regression」兩種問題，使用者可用範例資料、股票代號、任意 CSV 網址或上傳檔案更換資料來源，所有統計計算都在瀏覽器中完成，並部署於 Vercel 與 GitHub Pages。</p>
<p>本報告以問題 ② 為主：以台積電（2330.TW）{a["n"]:,} 個交易日的收盤價，依 CRISP-DM 建立 AR(p) 模型。以 AIC／BIC 選擇落後期數（特徵選擇）得到 AR({a["p"]})，估計係數 φ₁ = {fmt(phi["coef"], 4)}，顯示股價接近隨機漫步。測試期（{a["dates"]["testStart"]} ～ {a["dates"]["end"]}，{a["nTest"]} 天）一步預測 RMSE = {fmt(m["RMSE"])}，略優於 Naive 基準的 {fmt(nv["RMSE"])}；改用對數價格建模後，95% 預測區間的實際涵蓋率從 {pct(al["coverage"])} 提升到 {pct(a["coverage"])}。</p>
</div>

<h3>目錄</h3>
<ol class="toc">
  <li>作業需求對照</li><li>平台概觀</li><li>CRISP-DM：自迴歸股價預測（主體）</li><li>可任意輸入資料（加分項）</li>
  <li>問題 ①：多元線性回歸（簡述）</li><li>主流解法比較</li><li>NotebookLM 摘要</li><li>GPT 輔助內容</li>
  <li>數值正確性驗證</li><li>結論與限制</li>
</ol>

<h2 class="pb">一、作業需求對照</h2>
{table(["作業要求", "本作業的做法", "報告位置"], req_rows)}

<h2 class="pb">二、平台概觀</h2>
<p>網頁分成三個區塊：頁首右側是<b>問題下拉選單</b>；左側是資料來源、欄位設定、模型設定；主畫面依 CRISP-DM 分成六個分頁，最上方的指標卡顯示測試期 RMSE、MAPE、漲跌方向準確率與下一交易日預測。</p>
{f("ar01_page.png", "問題 ② 自迴歸的首頁：頁首下拉選單切換兩種問題、左側可更換資料來源")}
<p>多元線性回歸（問題 ①）已於 DIC2 繳交過一個版本，本報告只在第五章簡述，主體為自迴歸。</p>

<h2 class="pb">三、CRISP-DM：自迴歸股價預測</h2>

<h3>3.1 Business Understanding 商業理解</h3>
<ul>
  <li><b>問題：</b>用過去 p 個交易日的收盤價，預測下一個交易日的收盤價，並往後預測 20 個交易日、附上預測區間。</li>
  <li><b>模型：</b>自迴歸 AR(p)：y<sub>t</sub> = c + φ₁y<sub>t−1</sub> + … + φ<sub>p</sub>y<sub>t−p</sub> + ε<sub>t</sub>。本質上是把「前幾天的價格」當成特徵的多元線性回歸，因此可以沿用 OLS、信賴區間與預測區間。</li>
  <li><b>使用者與決策：</b>學生與投資人了解價格的慣性與不確定性；預測區間用來評估合理價格範圍與風險，而不是買賣訊號。</li>
  <li><b>成功標準：</b>測試期 RMSE 低於 Naive 基準（明天價格 = 今天價格）；95% 預測區間的實際涵蓋率接近 95%。</li>
</ul>
{f("ar02_business.png", "網頁「商業理解」分頁")}

<h3>3.2 Data Understanding 資料理解</h3>
<ul>
  <li><b>資料來源：</b>Yahoo Finance 台積電 2330.TW 日線資料（<a href="https://finance.yahoo.com/quote/2330.TW/history">finance.yahoo.com/quote/2330.TW</a>），由本平台的 <code>/api/stock</code> 下載並內建於網站。</li>
  <li><b>期間：</b>{a["dates"]["start"]} ～ {a["dates"]["end"]}，共 {a["n"]:,} 個交易日；欄位為 Date、Open、High、Low、Close、Adj Close、Volume，預測目標為 Close。</li>
  <li><b>走勢：</b>收盤價從 {fmt(a["first"])} 元漲到 {fmt(a["last"])} 元；測試期第一天為 {fmt(a["testFirst"])} 元，測試期內上漲約 {pct(a["last"] / a["testFirst"] - 1, 0)}。</li>
  <li><b>自我相關：</b>ACF 緩慢遞減、PACF 在第 1 期之後落入信賴界線內 → 價格具強烈慣性且非定態，適合從 AR(1) 開始建模。</li>
</ul>
{f("ar03_price.png", "台積電收盤價 K 線圖（紅漲綠跌，可拖曳縮放）")}
{f("ar04_acf_pacf.png", "ACF 與 PACF：ACF 緩慢遞減、PACF 在 lag 1 截斷")}
{f("ar05_returns.png", "日報酬率分布與最近 10 筆資料")}

<h3>3.3 Data Preparation 資料準備</h3>
<ol>
  <li>解析日期並由舊到新排序；移除缺失與重複日期（本資料皆為 0 筆）。</li>
  <li><b>對數轉換：</b>以 log(Close) 建模。價格越高、每日波動的金額越大；取對數後誤差變成百分比尺度，變異數較穩定，預測完再以 exp() 轉回價格。</li>
  <li><b>建立落後特徵：</b>第 t 天的特徵為前 1～p 天的值。</li>
  <li><b>依時間切分、不打亂順序：</b>訓練期 {a["nTrain"]} 天（{a["dates"]["start"]} ～ {a["dates"]["trainEnd"]}），測試期 {a["nTest"]} 天（{a["dates"]["testStart"]} ～ {a["dates"]["end"]}），避免用未來資料預測過去。</li>
</ol>
{f("ar06_prep_steps.png", "網頁「資料準備」分頁的步驟說明")}
{f("ar07_lag_matrix.png", "落後特徵矩陣（前 6 筆）")}
{f("ar08_split.png", "訓練期（藍）與測試期（橘）")}

<h3>3.4 Modeling 建模（含特徵選擇）</h3>
<p><b>特徵選擇＝選擇落後期數 p。</b>在訓練期對 p = 1～{a["maxLag"]} 分別配適 AR(p)，使用相同的樣本區間比較 AIC 與 BIC。對數尺度下 AIC 與 BIC 都在 <b>p = {a["bestAic"]}</b> 最小；原始價格尺度下 AIC 選 p = {al["bestAic"]}、BIC 選 p = {al["bestBic"]}。BIC 懲罰較重、偏好簡單模型；兩種尺度的 BIC 都支持 AR(1)。</p>
{f("ar09_lag_text.png", "網頁「建模」分頁：落後期數選擇說明")}
{f("ar10_lag_selection.png", "AIC／BIC 與測試期 RMSE 隨落後期數的變化")}
<div class="keep"><p><b>AR({a["p"]}) 估計結果（訓練期，對數價格）：</b></p>
{table(["變數", "係數", "標準誤", "t 值", "p 值", "95% CI 下限", "95% CI 上限"], coef_rows, num=(1, 2, 3, 4, 5, 6))}</div>
<p>φ₁ = {fmt(phi["coef"], 4)}（95% CI {fmt(phi["lo"], 4)} ～ {fmt(phi["hi"], 4)}，涵蓋 1），截距不顯著（p = {fp(const["p"])}）。這表示 log 價格幾乎是「昨天的值 + 隨機誤差」，也就是隨機漫步；殘差標準差 σ = {fmt(a["sigma"], 4)}，相當於每日約 {pct(a["sigma"], 2)} 的波動。訓練期 R² = {fmt(a["r2"], 4)} 很高，是因為價格水準本身的慣性，不代表能預測漲跌。</p>
{f("ar11_ar_table.png", "網頁上的 AR 迴歸係數表")}

<h3>3.5 Evaluation 評估</h3>
<p>測試期採「一步預測」：每天都用真實的前 p 天價格預測隔天，與 Naive 基準比較，並列出原始價格尺度的 AR({al["p"]}) 作為對照。</p>
{table(["指標", f"AR({a['p']})，對數價格（採用）", "Naive（明天 = 今天）", f"AR({al['p']})，原始價格"], metric_rows, num=(1, 2, 3))}
<ul>
  <li><b>準確度：</b>AR({a["p"]}) 的 RMSE 比 Naive 低 {pct(1 - m["RMSE"] / nv["RMSE"], 2)}，差距很小，符合「股價接近隨機漫步、難以贏過 Naive」的理論。原始價格的 AR({al["p"]}) RMSE 略低（{fmt(ml["RMSE"])}），方向準確率 {pct(ml["dirAcc"])}。</li>
  <li><b>不確定性：</b>原始價格模型的 95% 預測區間只涵蓋 {pct(al["coverage"])} 的測試期天數——測試期股價從 {fmt(a["testFirst"])} 漲到 {fmt(a["last"])} 元，每日波動的金額變大，用早期資料估計的誤差太小。對數價格模型的涵蓋率為 {pct(a["coverage"])}，接近 95%，因此網頁預設採用對數尺度。</li>
  <li><b>殘差檢定：</b>Ljung-Box（lag {a["lb"]["lags"]}）Q = {fmt(a["lb"]["q"], 2)}，p = {fmt(a["lb"]["p"], 3)} &gt; 0.05，殘差近似白噪音，AR({a["p"]}) 已捕捉到可用的自我相關。</li>
</ul>
{f("ar12_metrics.png", "網頁「評估」分頁：AR 與 Naive 的指標比較（對數價格）")}
{f("ar13_prediction.png", "預測圖：歷史價格、測試期一步預測（橘）與未來 20 日預測（綠），色帶為 95% 預測區間")}
{f("ar14_test_interval.png", "測試期放大：深色帶 = 95% 信賴區間（平均值）、淺色帶 = 95% 預測區間（單日價格）")}
{f("ar19_metrics_level.png", "對照：改用原始價格建模時，95% 預測區間涵蓋率只有約 64%")}
{f("ar15_residuals.png", "訓練期殘差的時間序列與分布（log 尺度）")}
{f("ar16_resid_acf.png", "殘差自我相關與 Ljung-Box 檢定")}

<h3>3.6 Deployment 部署</h3>
<p>以全部資料重新估計 AR({a["p"]}) 後遞迴預測未來 20 個交易日。第 2 天以後要用預測值當作落後值，誤差會累積，因此預測區間越來越寬。內建資料到 {a["dates"]["end"]} 為止，所以預測從下一個交易日開始；下表列出前 10 天，完整 20 天可在網頁下載 CSV。若在網頁用股票代號重新下載，會以最新資料預測。</p>
{table(["日期", "預測收盤價", "95% PI 下限", "95% PI 上限"], fc_rows, num=(1, 2, 3))}
{f("ar17_forecast.png", "網頁「部署」分頁：未來 20 個交易日預測與 95% 預測區間")}
<ul>
  <li><b>網站：</b>Vercel（<a href="{URL_WEB}">{URL_WEB}</a>）與 GitHub Pages（<a href="{URL_PAGES}">{URL_PAGES}</a>），推送到 GitHub 後自動重新部署。</li>
  <li><b>架構：</b>純前端 HTML + JavaScript（統計計算在瀏覽器完成），Vercel Serverless Function <code>api/stock.js</code> 代理 Yahoo Finance 股價；詳見 <code>design.md</code>。</li>
  <li><b>使用：</b>預測結果可下載成 CSV；網頁支援手機瀏覽與深色模式。</li>
</ul>
{f("ar22_mobile.png", "手機版畫面", width="28%")}
<div class="box warn">股價接近隨機漫步，本模型僅供課堂學習線性回歸與時間序列，不構成任何投資建議。</div>

<h2 class="pb">四、可任意輸入資料（加分項）</h2>
<p>使用者不需修改程式，就能在側邊欄更換資料來源：</p>
{table(["方式", "說明", "示範"], [
    ["範例資料", "台積電 2330.TW（內建）、Apple AAPL（plotly 公開資料）", "本報告第三章"],
    ["股票代號", "輸入任何 Yahoo Finance 代號，台股加 .TW／.TWO，可選 1～10 年", "元大台灣50 0050.TW（圖 20）"],
    ["CSV 網址", "任何含日期與數值欄位的公開 CSV；GitHub 頁面網址自動轉 raw", "plotly finance-charts-apple.csv（圖 21）"],
    ["上傳檔案", "CSV 只在瀏覽器處理，不會上傳伺服器", "—"],
])}
{f("ar20_ticker.png", "輸入股票代號 0050.TW 後自動下載並重新建模")}
{f("ar21_csv_url.png", "貼上 Apple 股價 CSV 網址，自動辨識日期與收盤價欄位")}
<p>問題 ① 也可輸入任意 CSV 網址。以老師指定的 50_Startups 為例（圖 22），系統自動把 State 做 one-hot、以交叉驗證選出 {SU["k"]} 個特徵（{"、".join(SU["selected"])}），測試集 R² = {fmt(SU["test"]["R2"], 4)}；係數為 {su_coef}。</p>
{f("mlr03_custom_url.png", "問題 ①：貼上 50_Startups.csv 網址即可完成特徵選擇與建模")}

<h2 class="pb">五、問題 ①：多元線性回歸（簡述，詳見 DIC2）</h2>
<p>資料集為 Kaggle Playground Series S4E5「Predict the Flood」（<a href="{URL_KAGGLE}">{URL_KAGGLE}</a>），20 個洪水風險因子預測洪水機率，符合「特徵 &gt; 13」的要求；網頁使用訓練集隨機抽樣 {FL["rows"]:,} 筆，訓練／測試 = {FL["nTrain"]:,}／{FL["nTest"]:,}。</p>
{table(["特徵選擇方法", "類型", "CV 最佳 k", "前 3 名特徵", "計算時間"], fl_methods, num=(2, 4))}
<p>四種方法都選出全部 20 個特徵：每個特徵與目標的相關係數都只有 {fmt(min(c["r"] for c in FL["corr"]), 3)}～{fmt(max(c["r"] for c in FL["corr"]), 3)}、標準化係數幾乎相同，缺一不可（只用 5 個特徵時測試 R² 僅 {fmt(R["floodK5"]["test"]["R2"], 3)}）。OLS 測試集 R² = {fmt(FL["test"]["R2"], 4)}、RMSE = {fmt(FL["test"]["RMSE"], 5)}，95% 預測區間涵蓋率 {pct(FL["coverage"], 2)}。</p>
{f("mlr02_prediction_interval.png", "問題 ① 測試集預測圖＋95% 信賴區間／預測區間")}
<div class="keep">
<h3>5.1 <span class="label nb">NotebookLM 摘要</span>Kaggle 洪水預測的同主題解法</h3>
<p>以 NotebookLM 研究網路上 Kaggle Playground Series S4E5（Predict the Flood）的同主題解法，摘要如下：</p>
<blockquote>{"".join(f"<p>{esc(p)}</p>" for p in NB_FLOOD.split(chr(10)) if p.strip())}</blockquote>
</div>

<h2>六、主流解法比較</h2>
<h3>6.1 股價預測（問題 ②）</h3>
{table(["方法", "做法", "與本作業的關係"], [
    ["Naive／隨機漫步", "明天價格 = 今天價格", "股價預測的標準基準；本作業 AR(1) 只略優於它"],
    ["AR／ARIMA（Box-Jenkins）", "以落後值（及差分、移動平均項）建模", "本作業採用 AR(p)，以 AIC／BIC 選階數"],
    ["GARCH 類模型", "讓誤差變異數隨時間變化", "可改善波動度變大時預測區間偏窄的問題（3.5 節）"],
    ["機器學習（LSTM、XGBoost 等）", "以大量技術指標或序列特徵訓練非線性模型", "網路上常見，但樣本外常難以穩定贏過 Naive；超出「線性回歸」範圍"],
])}
<p>本作業的結果（φ₁ ≈ 1、RMSE 與 Naive 幾乎相同）與效率市場假說的看法一致：只用過去價格很難預測明天的漲跌；線性模型的價值在於提供可解釋的基準與合理的預測區間。</p>
<h3>6.2 Kaggle 洪水預測（問題 ①）</h3>
{table(["模型", "類型", "R²"], py_rows, num=(2,))}
<p>Python 版數據取自 1014 版分析（Kaggle 全部 1,117,957 筆，20% holdout）。線性模型已達 0.845，加入列統計特徵或樹模型集成可再提升到約 0.869。</p>

<h2 class="pb">七、<span class="label nb">NotebookLM 摘要</span>股價預測的同主題解法</h2>
<p>使用 NotebookLM 研究網路上以時間序列預測股價的同主題解法，請它比較主流方法、說明 AR 為何難以贏過 Naive，並給出建議。摘要如下（保留原文，只把公式改為一般寫法）：</p>
<blockquote class="nb">{render_summary(NB_STOCK)}</blockquote>
<h3>7.2 與本作業結果的對照</h3>
{table(["NotebookLM 摘要的論點", "本作業的結果"], [
    ["對非平穩股價建立 AR 模型，參數會趨近 1、退化為 Naive",
     f'AR({a["p"]}) 的 φ₁ = {fmt(phi["coef"], 4)}（95% CI {fmt(phi["lo"], 4)} ～ {fmt(phi["hi"], 4)}，涵蓋 1）；測試期 RMSE {fmt(m["RMSE"])}，與 Naive 的 {fmt(nv["RMSE"])} 幾乎相同，完全吻合。對數價格的一階差分就是報酬率，φ₁ ≈ 1 代表報酬率接近白噪音（Ljung-Box p = {fmt(a["lb"]["p"], 3)}）'],
    ["直接預測原始股價會出現偽迴歸與虛高的 R²",
     f'訓練期 R² = {fmt(a["r2"], 4)}、測試期 R² = {fmt(m["R2"], 4)} 都很高，但 Naive 的 R² 也有 {fmt(nv["R2"], 4)}；因此本報告以 Naive 為主要比較基準，不以 R² 判斷模型好壞'],
    ["GARCH 能捕捉「波動聚集」、評估風險",
     f'原始價格模型的 95% 預測區間涵蓋率只有 {pct(al["coverage"])}；改用對數價格後為 {pct(a["coverage"])}，但換成 0050.TW 仍只有約 88%，顯示波動度會隨時間改變，GARCH 類模型是下一步'],
    ["務必以 Naive 為對照組，並採滾動式樣本外驗證",
     "本作業已以 Naive 為對照組，並依時間切分、每天用真實歷史值做一步預測；尚未做多個起點的滾動驗證，列為未來改進"],
    ["改為預測報酬率或漲跌方向",
     f'網頁已計算漲跌方向準確率（{pct(m["dirAcc"])}，接近擲硬幣）；以報酬率為目標並加入技術指標的模型列為未來改進'],
])}

<h2>八、<span class="label">GPT 輔助內容</span>AI 協助說明</h2>
<p>本作業使用 AI 助理 <b>Claude Code（Anthropic Claude）</b>協助完成。完整對話過程已匯出為 PDF（<code>{SID}_hw2_AI對話紀錄.pdf</code>），另行上傳至 iLearning。</p>
{table(["項目", "由誰決定／完成"], [
    ["作業需求解讀（讀取 iLearning 作業頁與 Notion 補充）", "AI 讀取整理，學生確認"],
    ["不使用 Streamlit，改為可部署到 GitHub／Vercel 的純前端網頁", "學生決定"],
    ["兩種 problem 的設計（多元線性回歸＋課堂教的股價自迴歸）", "AI 提案，學生採用"],
    ["統計核心、網頁介面、圖表、股價 API 的程式碼", "AI 撰寫"],
    ["數值正確性驗證（與 Python statsmodels 對照）", "AI 執行"],
    ["部署方式（GitHub + Vercel）與 repo 名稱", "學生決定"],
    ["依最新作業需求複查、決定報告以自迴歸為主", "學生決定"],
    ["本報告書、design.md、資訊圖表", "AI 依網頁實際執行結果撰寫，學生審閱"],
])}

<div class="keep"><h2>九、數值正確性驗證</h2>
<p>統計方法全部以 JavaScript 自行實作，因此逐項與 Python statsmodels 對照：</p>
{table(["項目", "測試資料", "結果"], [
    ["AR(3) 係數與 5 步預測區間", "台積電 2330.TW 收盤價", "與 <code>AutoReg(...).get_prediction()</code> 一致"],
    ["OLS 係數、標準誤、p 值、R²、F、AIC、BIC", "50_Startups 全資料（含 State one-hot）", "完全一致"],
    ["新資料點的信賴區間與預測區間", "同上", "一致（小數點後 3 位內）"],
    ["Durbin-Watson、Jarque-Bera、Breusch-Pagan、RESET", "同上", "一致"],
])}</div>

<h2>十、結論與限制</h2>
<ul>
  <li>以 AIC／BIC 選出的 AR({a["p"]}) 模型 φ₁ ≈ {fmt(phi["coef"], 4)}，台積電股價接近隨機漫步；測試期 RMSE {fmt(m["RMSE"])} 僅略優於 Naive 的 {fmt(nv["RMSE"])}。</li>
  <li>建模尺度對不確定性影響很大：對數價格讓 95% 預測區間涵蓋率從 {pct(al["coverage"])} 提升到 {pct(a["coverage"])}。</li>
  <li>平台滿足兩個加分項：下拉選單切換兩種問題、可任意更換資料來源（股票代號、CSV 網址、上傳）。</li>
  <li><b>限制：</b>AR 假設誤差變異數固定，市場波動度改變時預測區間仍會失準（例如 0050.TW 的涵蓋率為 88%）；未來日期只排除週末、未扣除國定假日；外部 CSV 需允許跨網域讀取。</li>
  <li><b>未來改進：</b>依 NotebookLM 摘要的建議，改以報酬率或漲跌方向為預測目標、採滾動式樣本外驗證（Rolling Walk-Forward），並加入 GARCH 波動度模型與成交量等外生變數（ARX）。</li>
</ul>

<h2>參考資料</h2>
<ol>
  <li>Yahoo Finance，2330.TW 歷史股價：https://finance.yahoo.com/quote/2330.TW/history</li>
  <li>Kaggle Playground Series S4E5：{URL_KAGGLE}</li>
  <li>50_Startups 資料集：{URL_STARTUPS}</li>
  <li>statsmodels：AutoReg、OLS 文件 https://www.statsmodels.org/</li>
  <li>網頁使用的函式庫：Plotly.js、PapaParse、jStat</li>
</ol>
"""
    return f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>HW2 報告書 {SID}</title><style>{REPORT_CSS}</style></head><body>{body}</body></html>'


# ---------------------------------------------------------------- 資訊圖表
INFO_CSS = """
* { box-sizing: border-box; }
body { margin: 0; width: 1200px; font-family: "Microsoft JhengHei", "Noto Sans TC", sans-serif; color: #0b0b0b; background: #f6f6f3; }
.wrap { padding: 0 0 36px; }
header { background: #0d366b; color: #fff; padding: 44px 56px 38px; }
header .tag { color: #9ec5f4; font-weight: bold; letter-spacing: .12em; font-size: 15px; }
header h1 { font-size: 46px; margin: 8px 0 6px; line-height: 1.2; }
header p { margin: 0; font-size: 19px; color: #cde2fb; }
header .who { margin-top: 18px; font-size: 15px; color: #b7d3f6; }
section { margin: 26px 40px 0; background: #fcfcfb; border: 1px solid rgba(11,11,11,.08); border-radius: 18px; padding: 26px 30px; }
h2 { margin: 0 0 14px; font-size: 25px; display: flex; align-items: center; gap: 12px; }
h2 .n { display: inline-grid; place-items: center; width: 34px; height: 34px; border-radius: 50%; background: #2a78d6; color: #fff; font-size: 17px; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.stat { background: #f0efeb; border-radius: 14px; padding: 16px 18px; }
.stat b { display: block; font-size: 36px; line-height: 1.15; color: #1c5cab; }
.stat span { font-size: 15px; color: #52514e; }
.stat small { display: block; font-size: 13px; color: #898781; margin-top: 4px; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.prob { border: 2px solid #e1e0d9; border-radius: 14px; padding: 16px 18px; }
.prob.on { border-color: #2a78d6; background: #f4f8fd; }
.prob h3 { margin: 0 0 6px; font-size: 20px; }
.prob p { margin: 0; font-size: 15px; color: #52514e; line-height: 1.6; }
.pill { display: inline-block; background: #2a78d6; color: #fff; border-radius: 999px; font-size: 13px; padding: 2px 10px; margin-left: 6px; vertical-align: middle; }
.flow { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }
.step { background: #f4f8fd; border-top: 5px solid #2a78d6; border-radius: 10px; padding: 12px 12px 14px; }
.step b { display: block; font-size: 13px; color: #1c5cab; letter-spacing: .03em; }
.step h4 { margin: 2px 0 6px; font-size: 18px; }
.step p { margin: 0; font-size: 13.5px; color: #3a3936; line-height: 1.55; }
img.shot { width: 100%; border-radius: 10px; border: 1px solid #e1e0d9; display: block; }
.cap { font-size: 14px; color: #52514e; margin-top: 8px; }
.insight { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.card2 { background: #f0efeb; border-radius: 14px; padding: 18px 20px; }
.card2 h3 { margin: 0 0 8px; font-size: 19px; }
.card2 p { margin: 6px 0 0; font-size: 15px; color: #3a3936; line-height: 1.6; }
.big { font-size: 40px; font-weight: bold; color: #1c5cab; }
.src { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.src div { background: #f4f8fd; border-radius: 12px; padding: 14px 16px; font-size: 14px; color: #3a3936; line-height: 1.5; }
.src b { display: block; font-size: 18px; color: #0b0b0b; margin-bottom: 4px; }
footer { margin: 26px 40px 0; display: grid; grid-template-columns: 1fr 150px; gap: 20px; align-items: center; background: #0d366b; color: #cde2fb; border-radius: 18px; padding: 22px 28px; font-size: 15px; line-height: 1.8; }
footer b { color: #fff; }
footer .qr { background: #fff; border-radius: 10px; padding: 8px; }
footer .qr svg { width: 100%; height: auto; display: block; }
.note { font-size: 12.5px; color: #898781; margin: 14px 40px 0; }
"""


def coverage_svg(level, log):
    w, h, x0, bw = 520, 210, 150, 330
    rows = [("原始價格 AR(10)", level, "#eb6834"), ("對數價格 AR(1)", log, "#2a78d6")]
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" font-family="Microsoft JhengHei" xmlns="http://www.w3.org/2000/svg">']
    for i, (lab, v, col) in enumerate(rows):
        y = 34 + i * 72
        parts.append(f'<text x="0" y="{y + 26}" font-size="16" fill="#3a3936">{lab}</text>')
        parts.append(f'<rect x="{x0}" y="{y}" width="{bw}" height="40" rx="6" fill="#e1e0d9"/>')
        parts.append(f'<rect x="{x0}" y="{y}" width="{bw * v:.1f}" height="40" rx="6" fill="{col}"/>')
        parts.append(f'<text x="{x0 + bw * v - 10:.1f}" y="{y + 27}" font-size="18" font-weight="bold" fill="#fff" text-anchor="end">{v * 100:.1f}%</text>')
    tx = x0 + bw * 0.95
    parts.append(f'<line x1="{tx:.1f}" y1="22" x2="{tx:.1f}" y2="186" stroke="#0b0b0b" stroke-dasharray="5 4" stroke-width="2"/>')
    parts.append(f'<text x="{tx:.1f}" y="204" font-size="14" fill="#0b0b0b" text-anchor="middle">目標 95%</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_infographic_html():
    a, al = AR, ARL
    m, nv = a["metrics"], a["naive"]
    phi = a["coef"][1]["coef"]
    fc0 = a["forecast"][0]
    body = f"""
<div class="wrap">
<header>
  <div class="tag">HW2 · MULTIPLE LINEAR REGRESSION · CRISP-DM</div>
  <h1>用自迴歸預測台積電股價</h1>
  <p>線性回歸預測平台：一個網頁、兩種問題、資料來源隨你換</p>
  <div class="who">{COURSE}｜{STUDENT}　{SID}</div>
</header>

<section>
  <h2><span class="n">★</span>重點數字（台積電 2330.TW，AR({a["p"]})，對數價格）</h2>
  <div class="stats">
    <div class="stat"><span>測試期 RMSE</span><b>{fmt(m["RMSE"], 2)}</b><small>Naive 基準 {fmt(nv["RMSE"], 2)}</small></div>
    <div class="stat"><span>測試期 MAPE</span><b>{pct(m["MAPE"], 2)}</b><small>{a["nTest"]} 個交易日一步預測</small></div>
    <div class="stat"><span>95% 預測區間涵蓋率</span><b>{pct(a["coverage"])}</b><small>原始價格建模只有 {pct(al["coverage"])}</small></div>
    <div class="stat"><span>下一交易日預測（{fc0["date"]}）</span><b>{fmt(fc0["mean"])}</b><small>95% PI {fmt(fc0["lo"])} ～ {fmt(fc0["hi"])}</small></div>
  </div>
</section>

<section>
  <h2><span class="n">1</span>一個網頁、兩種問題（下拉選單切換）</h2>
  <div class="two">
    <div class="prob"><h3>① 多元線性回歸</h3><p>表格資料：Kaggle Predict the Flood（20 個特徵）、50_Startups 等。4 種特徵選擇＋交叉驗證，測試 R² {fmt(FL["test"]["R2"], 3)}。</p></div>
    <div class="prob on"><h3>② 自迴歸 Auto Regression<span class="pill">本圖重點</span></h3><p>時間序列：用前 p 天的收盤價預測明天，AIC／BIC 選 p，附信賴區間與預測區間，並往後預測 20 天。</p></div>
  </div>
</section>

<section>
  <h2><span class="n">2</span>CRISP-DM 六步驟</h2>
  <div class="flow">
    <div class="step"><b>1 BUSINESS</b><h4>商業理解</h4><p>預測下一交易日收盤價；成功標準：贏過 Naive、PI 涵蓋率約 95%</p></div>
    <div class="step"><b>2 DATA</b><h4>資料理解</h4><p>{a["n"]:,} 個交易日；{fmt(a["first"])} → {fmt(a["last"])} 元；PACF 在 lag 1 截斷</p></div>
    <div class="step"><b>3 PREPARATION</b><h4>資料準備</h4><p>日期排序、log 轉換、落後特徵、依時間切分 {a["nTrain"]}／{a["nTest"]} 天</p></div>
    <div class="step"><b>4 MODELING</b><h4>建模</h4><p>特徵選擇＝選 p：AIC、BIC 都選 p = {a["bestAic"]}；φ₁ = {fmt(phi, 4)}</p></div>
    <div class="step"><b>5 EVALUATION</b><h4>評估</h4><p>RMSE {fmt(m["RMSE"], 2)} vs {fmt(nv["RMSE"], 2)}；Ljung-Box p = {fmt(a["lb"]["p"], 2)}（殘差為白噪音）</p></div>
    <div class="step"><b>6 DEPLOYMENT</b><h4>部署</h4><p>Vercel＋GitHub Pages；未來 20 日預測＋95% PI，可下載 CSV</p></div>
  </div>
</section>

<section>
  <h2><span class="n">3</span>預測圖：測試期一步預測＋未來 20 天</h2>
  <img class="shot" src="figures/ar13_prediction.png">
  <div class="cap">藍線：實際收盤價；橘線與色帶：測試期一步預測與 95% 預測區間；綠色：未來 20 個交易日預測與 95% 預測區間。</div>
</section>

<section>
  <h2><span class="n">4</span>兩個發現</h2>
  <div class="insight">
    <div class="card2">
      <h3>股價接近隨機漫步</h3>
      <div class="big">φ₁ = {fmt(phi, 4)}</div>
      <p>log 價格 ≈ 昨天的值 + 隨機誤差。AR 模型的 RMSE 只比「明天 = 今天」低 {pct(1 - m["RMSE"] / nv["RMSE"], 2)}，與效率市場假說一致。</p>
    </div>
    <div class="card2">
      <h3>對數尺度讓預測區間更可靠</h3>
      {coverage_svg(al["coverage"], a["coverage"])}
      <p>測試期股價從 {fmt(a["testFirst"])} 漲到 {fmt(a["last"])} 元，波動金額變大；改用 log 後誤差以百分比計，涵蓋率接近 95%。</p>
    </div>
  </div>
</section>

<section>
  <h2><span class="n">5</span>資料來源隨你換（加分項）</h2>
  <div class="src">
    <div><b>範例資料</b>台積電 2330.TW（內建 5 年）、Apple AAPL</div>
    <div><b>股票代號</b>輸入 0050.TW、2317.TW、NVDA…自動下載 Yahoo Finance 日線</div>
    <div><b>CSV 網址</b>貼上任何公開 CSV，例如老師的 50_Startups.csv</div>
    <div><b>上傳檔案</b>電腦裡的 CSV 只在瀏覽器處理，不會上傳</div>
  </div>
</section>

<footer>
  <div>
    <b>線上網頁</b>　{URL_WEB}<br>
    <b>GitHub</b>　{URL_REPO}<br>
    <b>技術</b>　純前端 JavaScript 計算 OLS／AR／CI／PI，Plotly.js 互動圖表，Vercel Serverless 股價 API<br>
    <b>驗證</b>　AR 係數與預測區間、OLS 結果皆與 Python statsmodels 一致
  </div>
  <div class="qr">{qr_svg(URL_WEB)}</div>
</footer>
<div class="note">資料期間 {a["dates"]["start"]} ～ {a["dates"]["end"]}（Yahoo Finance）。本圖僅供課堂學習，不構成投資建議。本資訊圖表由 AI（Claude Code）依網頁實際執行結果協助製作。</div>
</div>
"""
    return f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>HW2 資訊圖表 {SID}</title><style>{INFO_CSS}</style></head><body>{body}</body></html>'


def main():
    (REP / "report.html").write_text(build_report_html(), encoding="utf-8")
    (REP / "infographic.html").write_text(build_infographic_html(), encoding="utf-8")
    footer = '<div style="font-size:8px;width:100%;text-align:center;color:#898781;font-family:Microsoft JhengHei">HW2 報告書｜' + SID + '｜<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page()
        page.goto((REP / "report.html").as_uri())
        page.wait_for_load_state("networkidle")
        page.pdf(path=str(REP / f"{SID}_hw2_report.pdf"), format="A4", print_background=True,
                 display_header_footer=True, header_template="<div></div>", footer_template=footer,
                 margin={"top": "16mm", "bottom": "18mm", "left": "15mm", "right": "15mm"})
        page = browser.new_page(viewport={"width": 1200, "height": 900}, device_scale_factor=2)
        page.goto((REP / "infographic.html").as_uri())
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(REP / f"{SID}_hw2_infographic.png"), full_page=True)
        h = page.evaluate("document.documentElement.scrollHeight")
        page.pdf(path=str(REP / f"{SID}_hw2_infographic.pdf"), width="1200px", height=f"{h + 2}px", print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        browser.close()
    print("完成：", f"{SID}_hw2_report.pdf", f"{SID}_hw2_infographic.png", f"{SID}_hw2_infographic.pdf")


if __name__ == "__main__":
    main()
