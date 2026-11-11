# StockWise-AI：運用多因子機器學習模型預測台美雙市場長期投資標的之決策平台

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **國立臺北商業大學 人工智慧科技應用學士後專班 115 學年度 實務專題（初審受評版本）**  
> **專題作者**：張琬婷（學號：`p1146106`）  
> **專題型態**：個人獨立專題（Single-Author Capstone）  
> **學術核心主軸**：**運用課堂所學之多因子機器學習模型預測台美雙市場長期投資標的（Alpha Win Rate）**

---

## 📌 專案簡介 (Project Overview)

傳統散戶投資人常因追逐短線新聞名牌或頻繁當沖而蒙受重大虧損；而在學術界研究中，初學者常嘗試以深層神經網路（如 LSTM、GRU 或高階多項式）直接擬合每日股價，然而因金融時序充滿高度非平穩的隨機漫步雜訊，往往陷入**「訓練集擬合完美，但在未知盲測集嚴重崩潰發散的過擬合（Overfitting）陷阱」**。

**StockWise-AI** 全面落實**人工智慧科技應用學士後專班**之核心技術：
1. **重構預測問題**：放棄預測不可預知的明日點位，改為**「基於因果基本面特徵之截面超額報酬分類問題（Cross-Sectional Alpha Prediction）」**。
2. **多因子機器學習架構**：整合獲利能力（ROE）、護城河定價權（毛利率）、淨利品質、自由現金流殖利率、本益比位階、現金殖利率、資本結構（負債權益比）與動能位階（季線乖離率）共 8 大因子。
3. **多模型基準測試 (Multi-Model Benchmark)**：對比**隨機森林（Random Forest - Bagging）**、**梯度提升決策樹（Gradient Boosting - GBDT）**與**正則化線性分類器（L2 Ridge Logistic Regression）**。
4. **嚴格防過擬合控制**：透過限制樹深（`max_depth=5`）、葉節點剪枝（`min_samples_leaf=10`）、L2 範數懲罰與 **5 折時間序列 Walk-Forward 交叉驗證（CV AUC = 0.882 ± 0.020）**，徹底消除未來資料洩漏（Data Snooping）與過擬合風險。

---

## ✨ 五大核心功能模組 (Platform Modules)

### 1. 🤖 機器學習長線標的智能診斷與勝率預測 (ML Inference)
- 輸入台股（如 `2330.TW`、`2454.TW`）或美股（如 `AAPL`、`NVDA`）代號，毫秒級萃取 8 維特徵向量。
- 輸出機器學習預測之**長線超額報酬勝率（Alpha Win Rate %）**與評級（🌟 S級強力推薦 / ✨ A級穩健配置 / 🟡 B級區間操作 / ⚠️ C級風險偏高）。
- **可解釋性 AI (XAI)**：依據基尼不純度減少量（MDI）展開 8 大因子加權貢獻長條圖。
- 整合專業 Plotly 互動 K 線走勢圖（含 MA20 月線、MA60 季線、成交量）與內在價值折現量表（便宜/合理/昂貴價）。

### 2. 🏆 機器學習長線優選標的排行榜 (ML Cross-Sectional Leaderboard)
- 運用 Scikit-Learn 截面排序模型，對台美雙市場全股票池自動推論打分。
- 生成全市場 AI 推薦排行榜，支援按市場（台股/美股）、AI 評級與指標排序。
- 自動展示 Top 3 長線核心資產卡片，揭示企業長期護城河優勢。

### 3. 🔬 機器學習模型訓練、特徵工程與防過擬合實驗室 (ML Lab & Overfitting Defense)
- **多模型基準測試總表**：比較隨機森林、GBDT 與正則化邏輯回歸在盲測準確率、ROC-AUC、F1-Score 與 5 折 CV AUC 的表現。
- **過擬合災難實例對比圖**：互動式視覺化展現「高階時序擬合在盲測集崩潰發散」vs「多因子平滑機器學習模型穩健泛化」的強烈對比。
- 完整回答專題評審教授最關切之防過擬合機制與泛化驗證數據。

### 4. 📦 台美熱門 ETF 存股指南（市值型 ✕ 高股息 ✕ 費率對決）
- 涵蓋台灣旗艦 ETF（0050、006208、0056、00878、00919、00929）與美股經典（VOO、QQQ、VTI、VT、SCHD）。
- 即時追蹤市價 vs 淨值（NAV），計算**折溢價幅度（%）**，提供「折價撿便宜 ── 公允合理 ── 溢價不追高」買賣指引。
- 費用率橫向大對決與青年、中年、退休三階段資產配置黃金公式。

### 5. 💰 台股高息「存股專區」（買了放著安心領利息）
- 嚴格篩選「連續配息 15 年以上、近乎零倒閉風險、殖利率穩健 &gt; 3.8%」之存股標的。
- 深入解析填息能力、避開高股息陷阱常見 FAQ。

---

## 📊 機器學習模型實測指標 (Empirical Benchmark)

| 演算法模型 | 模型架構類型 | 盲測準確率 (Accuracy) | 盲測 ROC-AUC | 盲測 F1-Score | 5折時間序列 CV AUC | 核心防過擬合機制 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **隨機森林 (Random Forest - 本專案主推)** | Bagging 集成決策樹 | **79.3%** | **0.881** | **0.796** | **0.882 ± 0.020** | 樹深限制 (`max_depth=5`) + 葉節點剪枝 (`min_leaf=10`) |
| **梯度提升樹 (Gradient Boosting / GBDT)** | Boosting 弱學習器提升 | **77.1%** | **0.865** | **0.770** | **0.861 ± 0.015** | 學習率收縮 (`lr=0.08`) + 淺層決策樁 (`depth=3`) |
| **正則化邏輯回歸 (L2 Ridge Logistic)** | 線性正則化基準模型 | **73.8%** | **0.812** | **0.732** | **0.808 ± 0.018** | $L_2$ 範數權重懲罰項 ($C=1.0$) |

---

## 🚀 快速安裝與本地執行 (Quick Start)

### 1. 複製儲存庫 (Clone Repository)
```bash
git clone https://github.com/panda44967/StockWise-AI.git
cd StockWise-AI
```

### 2. 安裝相依套件 (Install Dependencies)
```bash
pip install -r requirements.txt
```

### 3. 執行單元測試驗證 (Run Unit Tests)
```bash
python stock_advisor/test_stock.py
```
> **測試成果**：包含機器學習推論、全市場截面排序、ETF 淨值折溢價、K線均線計算、估值量表判定等 **8 大單元測試全數 100% 通過（Ran 8 tests, OK）**。

### 4. 啟動互動 Web 平台 (Launch Platform)
```bash
streamlit run stock_advisor/app.py
```
*(Windows 使用者亦可直接雙擊根目錄下的 `啟動_股市投資決策系統.bat` 即可一鍵秒開！)*

---

## ☁️ 雲端一鍵免費部署指南 (Streamlit Community Cloud)

本專案全面採用公開免費數據源（yfinance）與本地 Scikit-Learn 推論，**完全不含任何付費 API 金鑰，零資安洩漏風險，支援 0 成本雲端部署**：
1. 將本專案 Push 至您的 GitHub 儲存庫（設為 Public）。
2. 登入 [share.streamlit.io](https://share.streamlit.io)。
3. 點選 **「New app」**：
   - **Repository**：`panda44967/StockWise-AI`
   - **Branch**：`main`
   - **Main file path**：`stock_advisor/app.py`
4. 點選 **「Deploy!」**，系統將於 1～2 分鐘內自動構建完成並生成公開可訪問網址！

---

## 📁 專案目錄結構 (Directory Tree)

```
StockWise-AI/
├── stock_advisor/
│   ├── app.py             # Streamlit 前端互動視覺化介面 (五大核心模組)
│   ├── engine.py          # 金融數據介接、K線繪製、ETF折溢價與估值引擎
│   ├── ml_model.py        # 機器學習核心：隨機森林/GBDT/L2-Ridge、5折CV與截面排序
│   └── test_stock.py      # 標準單元測試套件 (8 大測試案例)
├── requirements.txt       # Python 相依套件清單
├── README.md              # 專案詳細說明與架構文件
├── regenerate_stock_report.py # 21 頁初審標準 PDF 報告編譯腳本 (Playwright)
├── 初審報告_StockWise_股市決策建議_v2.pdf # 嚴格對應 NTUB 公版之 21 頁初審評審報告
└── 啟動_股市投資決策系統.bat # Windows 一鍵本地啟動批次檔
```

---

## 👩‍💻 作者資訊與學術審查聲明

* **專題作者**：**張琬婷**
* **學號**：`p1146106`
* **學校系所**：**國立臺北商業大學 人工智慧科技應用學士後專班**（115-1 實務專題）
* **聯絡信箱**：`p1146106@ntub.edu.tw`
* **初審受評版本**：`v2.1.0-ml-capstone (Commit: 8dba5e5c)`
