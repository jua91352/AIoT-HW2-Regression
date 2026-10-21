# HW2 線性回歸預測平台（Multiple Linear Regression ＋ Auto Regression）

物聯網應用與資料分析（7716）｜游雅婷｜5115056031

純前端互動網頁：所有統計計算（OLS、交叉驗證、特徵選擇、Lasso／Ridge、AR）都在瀏覽器中以 JavaScript 完成，可以直接部署到 **GitHub Pages** 或 **Vercel**，不需要 Python 伺服器。

- 線上版（Vercel）：https://aiot-hw2-regression.vercel.app
- 線上版（GitHub Pages）：https://jua91352.github.io/AIoT-HW2-Regression/
- 直接開啟問題 ②：在網址後面加上 `#ar`
- GPT 輔助內容與 NotebookLM 摘要：見本文最後兩節（AI 對話紀錄 PDF 另行上傳至 iLearning）

## 文件

| 文件 | 內容 |
|---|---|
| [報告書 PDF](report/5115056031_hw2_report.pdf) | 依 CRISP-DM 說明自迴歸股價預測（主體）、加分項、多元線性回歸簡述、主流解法比較、NotebookLM 摘要、GPT 輔助內容 |
| [資訊圖表 PNG](report/5115056031_hw2_infographic.png)／[PDF](report/5115056031_hw2_infographic.pdf) | 一頁看懂：重點數字、CRISP-DM 六步驟、預測圖、兩個發現、資料來源 |
| [design.md](design.md) | 設計文件：架構、模組、資料流程、演算法、介面、錯誤處理、部署與驗證 |

報告書與資訊圖表的數字和截圖都由程式從網頁實際執行結果產生：`report/collect.py` 擷取數字與截圖，`report/build_report.py` 產生 PDF 與 PNG。

兩個版本功能相同；股票代號下載由 Vercel 的 `/api/stock` 提供（GitHub Pages 版會呼叫 Vercel 的 API）。

## 加分項對應

| 加分要求 | 實作 |
|---|---|
| 有下拉視窗呈現兩種 problem（+1） | 頁面頂部「選擇問題 Problem」下拉選單：① 多元線性回歸（表格資料）② 自迴歸 Auto Regression（股價時間序列預測） |
| 可任意輸入 Dataset（+2） | 側邊欄「資料來源」：內建範例／**貼上任意 CSV 網址**（例如 `https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`）／上傳 CSV 檔；問題 ② 另可輸入任意**股票代號**（2330.TW、AAPL…） |

載入任何 CSV 後會自動判斷欄位型態：數值欄位當特徵、文字欄位自動 one-hot、ID 與高基數文字欄位預設排除，使用者可自行選擇預測目標與排除欄位。

## 功能與 CRISP-DM 對應

每個問題都有 6 個分頁，對應 CRISP-DM 六個步驟：

| 步驟 | ① 多元線性回歸 | ② Auto Regression |
|---|---|---|
| 1 Business Understanding | 問題定義、資料來源、成功標準 | 預測目標、AR(p) 模型說明、與 Naive 比較的成功標準 |
| 2 Data Understanding | 欄位摘要、缺失值、目標分布、相關係數與熱圖 | K 線／走勢圖、日報酬分布、ACF／PACF |
| 3 Data Preparation | 排除 ID、one-hot、訓練集中位數補值、常數欄位移除、訓練／測試切分 | 日期排序、去重、落後特徵矩陣、**依時間切分**、對數轉換 |
| 4 Modeling | **4 種特徵選擇**（SelectKBest、RFE、Forward Selection、Lasso）＋ 5-fold CV 自動選 k；OLS 係數表（標準誤、t、p、CI）；與 Ridge／Lasso／Baseline 比較 | **AIC／BIC 選擇落後期數 p**（特徵選擇）；AR 係數表 |
| 5 Evaluation | MAE、MSE、RMSE、R²、Adj. R²；**預測圖＋信賴區間／預測區間**；殘差圖、Q-Q 圖；RESET、Durbin-Watson、Breusch-Pagan、Jarque-Bera、VIF | 與 Naive 比較 MAE／RMSE／MAPE／方向準確率；**預測圖＋信賴區間／預測區間**；Ljung-Box 殘差檢定 |
| 6 Deployment | 輸入新資料即時預測（含 CI／PI）、下載預測結果 CSV 與模型參數 JSON | 未來 N 個交易日預測（含 PI）、下載 CSV |

## 內建範例資料

| 問題 | 資料集 | 來源 |
|---|---|---|
| ① | Kaggle Predict the Flood（20 個特徵，抽樣 20,000 筆） | https://www.kaggle.com/competitions/playground-series-s4e5 |
| ① | 50_Startups（老師範例） | https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv |
| ① | Student Scores（簡單線性回歸） | https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/studentscores.csv |
| ① 網址範例 | Kaggle Car Price（24 個特徵） | https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv |
| ② | 台積電 2330.TW 5 年日資料 | Yahoo Finance |
| ② | Apple AAPL 2015–2017 | https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv |

## 主要結果（預設設定）

| 資料 | 模型 | 測試集結果 |
|---|---|---|
| Kaggle Flood | MLR，20 個特徵（CV 選出） | R² = 0.843、RMSE = 0.0200、95% PI 涵蓋率 94.8% |
| 50_Startups | MLR，2 個特徵（R&D Spend、Marketing Spend） | R² = 0.946 |
| 台積電 2330.TW | AR(1)，對數價格，AIC 選出 | RMSE 41.76（Naive 41.91）、MAPE 1.53%、95% PI 涵蓋率 94.2% |

台積電若改用原始價格建模，95% 預測區間涵蓋率只有約 64%：測試期股價從約 1,400 漲到 2,500，每日波動的金額跟著變大，用早期資料估計的誤差變異數太小。改用對數價格後誤差以百分比計，涵蓋率回到 94% 左右。

## 數值正確性驗證

JS 統計核心（`js/stats.js`）與 Python `statsmodels` 對照（50_Startups 全資料 OLS、台積電 AR(3)）：

- 係數、標準誤、p 值、R²、Adj. R²、F、AIC、BIC：完全一致
- 新資料點的平均值信賴區間與預測區間：一致（小數點後 3 位內）
- Durbin-Watson、Jarque-Bera、Breusch-Pagan、RESET：一致
- AR 係數與多步預測區間：與 `AutoReg(...).get_prediction()` 一致

## 本機執行

需要 Node.js（只用來提供靜態檔案與模擬 `/api/stock`，不需安裝任何套件）：

```bash
node scripts/dev-server.js
```

開啟 http://localhost:5173 。

## 部署

- **Vercel**：匯入此 GitHub 專案即可（Framework Preset 選 Other，不需建置指令）。`api/stock.js` 會自動成為 Serverless Function，提供股票代號下載。
- **GitHub Pages**：Settings → Pages → Deploy from a branch → `main` / root。GitHub Pages 沒有後端，股票代號功能透過 `js/config.js` 的 `REMOTE_STOCK_API` 呼叫 Vercel 上的 API。

## 檔案結構

```
index.html            頁面骨架（頂部問題下拉選單、側邊欄、CRISP-DM 分頁）
css/style.css         樣式（自動支援深色模式、手機版）
js/app.js             問題切換、資料來源、設定與分頁
js/stats.js           OLS、CI/PI、K-fold CV、前向選擇、RFE、Lasso、Ridge、回歸假設檢定、ACF/PACF、AR 預測
js/data.js            CSV 讀取（網址／上傳、Big5 編碼）、欄位判斷、one-hot、補值、切分、時間序列整理
js/mlr.js             問題 ①：多元線性回歸的計算與畫面
js/ar.js              問題 ②：Auto Regression 的計算與畫面
js/ui.js              表格、卡片、格式化、Plotly 繪圖工具
js/config.js          部署設定（GitHub Pages 使用遠端股價 API 的網址）
api/stock.js          Vercel Serverless Function：代理 Yahoo Finance 日線資料
scripts/dev-server.js 本機開發伺服器
data/                 內建範例資料（Flood 抽樣、台積電股價）
design.md             設計文件
report/               報告書、資訊圖表、截圖（figures/）與產生程式
```

## GPT 輔助內容

本專案的程式碼、README 與部署流程由 AI 助理 **Claude Code（Anthropic Claude）** 協助完成，作為作業要求的「GPT 輔助內容」。完整對話過程已匯出為 PDF（`5115056031_hw2_AI對話紀錄.pdf`），另行上傳至 iLearning。

紀錄範圍為 2026-10-15 的網頁開發與部署過程，依時間列出學生的提問、AI 的回覆、AI 提出的選項與學生的選擇；AI 執行的動作（讀網頁、寫程式、測試、部署）以一行摘要呈現，不含 AI 內部思考與工具原始輸出。

AI 協助的範圍與學生的決定：

| 項目 | 由誰決定／完成 |
|---|---|
| 作業需求解讀（讀取 iLearning 作業頁與 Notion 補充） | AI 讀取整理，學生確認 |
| 不使用 Streamlit、改為可部署到 GitHub／Vercel 的純前端網頁 | 學生決定 |
| 兩種 problem 的設計（多元線性回歸＋課堂教的股價 Auto Regression） | AI 提案，學生採用 |
| 統計核心、網頁介面、圖表、股價 API 的程式碼 | AI 撰寫 |
| 數值正確性驗證（與 Python statsmodels 對照） | AI 執行，結果記錄於上方「數值正確性驗證」 |
| 部署方式（GitHub + Vercel）與 repo 名稱 | 學生決定 |

## NotebookLM 摘要

### 股價預測（問題 ② 自迴歸）

使用 NotebookLM 研究網路上以時間序列預測股價的同主題解法，摘要如下（保留原文）：

> **1. 主流預測方法比較**
>
> - **Naive／隨機漫步基準**：做法為假設明日股價等於今日股價（$\hat{Y}_{t+1} = Y_t$）。優點是極簡且零參數，完全無過擬合風險；缺點是無法預測趨勢轉折或波幅。
> - **AR／ARIMA**：利用歷史價格（自迴歸 AR）與殘差（移動平均 MA）捕獲線性規律。優點是理論完備、適合平穩序列；缺點是無法處理非線性動態，面對非平穩股價易退化。
> - **GARCH**：針對條件變異數建模，專注於捕捉金融市場的「波動聚集（Volatility Clustering）」現象。優點是精準評估風險與波動度，缺點是其預測標的為波動率而非價格方向。
> - **LSTM (機器學習)**：透過門控機制學習非線性長短期依賴。優點是擬合能力強；缺點是需要大量數據、容易過擬合高噪音股價、模型為黑盒子且常見預測滯後（Lagging）。
>
> **2. 為什麼 AR 很難贏過 Naive 基準？**
>
> 根據效率市場假說（EMH），股價價格序列近似隨機漫步，其報酬率自相關性極低且高噪音。若直接對非平穩的原始股價建立 AR 模型，參數會趨近於 1，實際退化為 Naive 模型；若對差分後的平穩報酬率建模，由於信躁比極低，估計出的 AR 參數多為樣本雜訊（過擬合），使得樣本外（Out-of-Sample）預測誤差往往大於零參數的 Naive 基準。
>
> **3. 給用「線性回歸」預測股價學生的建議**
>
> 1. **切勿直接預測原始股價**：原始股價是非平穩序列，直接做 OLS 迴歸會產生「偽迴歸（Spurious Regression）」及虛高 $R^2$ 的假象。
> 2. **改為預測「報酬率」或「漲跌方向」**：將數據轉換為平穩序列後，再結合技術指標或總體經濟特徵進行預測。
> 3. **務必以 Naive Model 為對照組**：採用滾動式樣本外驗證（Rolling Walk-Forward），並比較其 RMSE/MAE 是否真的擊敗簡單的 Naive Baseline。

與本專案的對照：台積電 AR(1) 的 φ₁ = 1.0006（95% CI 涵蓋 1），測試期 RMSE 41.76 與 Naive 的 41.91 幾乎相同，正是摘要所說「參數趨近 1、退化為 Naive」；網頁因此以 Naive 為主要比較基準、不以 R² 判斷好壞。改以報酬率為目標、滾動式驗證與 GARCH 波動度模型列為未來改進。

### Kaggle 洪水預測（問題 ① 多元線性回歸）

以 NotebookLM 研究網路上 Kaggle Playground Series S4E5（Predict the Flood）的同主題解法，摘要如下：

> 在 Kaggle S4E5 洪水預測競賽中，單純多元線性回歸展現出驚人的基準表現，因目標值主要由特徵總和決定，僅靠線性擬合即可達到 0.845 的 R² 分數，優於未經特徵工程的原始樹狀模型；但其限制在於無法捕捉細微的非線性殘差與交互作用，導致分數到達瓶頸。社群隨後發現此資料結構，開發出關鍵的列級特徵工程，包含計算特徵總和（fsum）、按數值大小對列特徵進行排序，以及計算列標準差、極值與高於閾值的特徵數量等統計量。
>
> 前幾名優勝解法進一步結合 CatBoost、XGBoost、LightGBM 與 AutoGluon 等多元模型，並利用 Ridge 或 Lasso 回歸對多組折外預測（OOF）進行加權集成，成功將 R² 分數提升至 0.869 以上。這給予使用多元線性回歸的學生重要啟示：線性模型是快速驗證全局規律與建立強大 Baseline 的優良起點，但若要達到極致預測精度，仍需深入進行 EDA，結合列統計特徵開發與多模型集成，才能補足單一線性模型的侷限。

與本專案的對照：網頁中 Flood 資料的多元線性回歸測試集 R² 為 0.843，與摘要提到的線性模型基準（約 0.845）一致；摘要中的列統計特徵與多模型集成屬於非線性方法，超出本作業「線性回歸」的範圍，因此未納入網頁。
