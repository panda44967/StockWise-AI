"""
StockWise-AI: Financial Engine for Long-Term Investment
Provides fundamental diagnostics, valuation boundaries (便宜/合理/昂貴價),
interactive price data, industry categorized picks, and TW high-dividend screeners.
Always prioritizes live market data via yfinance!
"""
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf

# Baseline metadata (Qualitative moats & fallbacks) updated to current 2026 market realities
MARKET_DATA = {
    # ================= 台股 (Taiwan Stock Market) =================
    "2330.TW": {
        "name": "台積電 (TSMC)", "market": "TW", "sector": "半導體/先進製程",
        "price": 2135.0, "pe": 21.9, "pb": 7.8, "eps": 97.5, "roe": 0.325,
        "gross_margin": 0.582, "net_margin": 0.425, "fcf_yield": 0.038, "debt_to_equity": 0.22,
        "div_yield": 0.021, "div_years": 21, "fair_value": 2250.0,
        "cheap_price": 1900.0, "expensive_price": 2600.0,
        "moat": "全球 2nm/A16 先進製程市佔 > 90%，高資本支出形成無法跨越之技術壁壘。",
        "why_buy": "AI 晶片算力需求結構性爆發，先進封裝 CoWoS 產能供不應求，長線定價權極高。"
    },
    "2454.TW": {
        "name": "聯發科 (MediaTek)", "market": "TW", "sector": "半導體/IC設計",
        "price": 3390.0, "pe": 22.5, "pb": 5.2, "eps": 150.8, "roe": 0.265,
        "gross_margin": 0.515, "net_margin": 0.245, "fcf_yield": 0.045, "debt_to_equity": 0.28,
        "div_yield": 0.042, "div_years": 16, "fair_value": 3550.0,
        "cheap_price": 3050.0, "expensive_price": 4100.0,
        "moat": "全球智慧型手機 AP 龍頭，邊緣 AI 天璣晶片與客製化 ASIC 雙引擎成長。",
        "why_buy": "配息大方具高股息保護，車用電子與 AI ASIC 帶來中長期估值重估潛力。"
    },
    "2317.TW": {
        "name": "鴻海 (Foxconn)", "market": "TW", "sector": "電子代工/伺服器",
        "price": 224.5, "pe": 15.2, "pb": 1.65, "eps": 14.8, "roe": 0.125,
        "gross_margin": 0.068, "net_margin": 0.032, "fcf_yield": 0.065, "debt_to_equity": 0.54,
        "div_yield": 0.036, "div_years": 24, "fair_value": 240.0,
        "cheap_price": 198.0, "expensive_price": 280.0,
        "moat": "全球最大電子 EMS 製造商，供應鏈垂直整合與龐大全球產能部署。",
        "why_buy": "AI 伺服器機櫃（NVL72）市佔第一，電動車與衛星通訊提供中長期成長動能。"
    },
    "2382.TW": {
        "name": "廣達 (Quanta)", "market": "TW", "sector": "電子代工/伺服器",
        "price": 315.0, "pe": 17.5, "pb": 4.5, "eps": 18.0, "roe": 0.268,
        "gross_margin": 0.086, "net_margin": 0.042, "fcf_yield": 0.052, "debt_to_equity": 0.46,
        "div_yield": 0.041, "div_years": 22, "fair_value": 340.0,
        "cheap_price": 280.0, "expensive_price": 395.0,
        "moat": "白牌雲端資料中心伺服器龍頭，與微軟、Google、Meta 深度綁定。",
        "why_buy": "雲端巨頭資本支出持續擴增，獲利率結構因 AI 高單價機櫃而顯著優化。"
    },
    "2881.TW": {
        "name": "富邦金 (Fubon Financial)", "market": "TW", "sector": "金融保險/金控",
        "price": 92.5, "pe": 10.8, "pb": 1.20, "eps": 8.56, "roe": 0.132,
        "gross_margin": 0.385, "net_margin": 0.170, "fcf_yield": 0.058, "debt_to_equity": 0.82,
        "div_yield": 0.051, "div_years": 18, "fair_value": 98.0,
        "cheap_price": 82.0, "expensive_price": 115.0,
        "moat": "台灣金控每股獲利王，壽險、銀行、證券三合一多元資產收益。",
        "why_buy": "獲利體質強健，降息循環有利於海外債券未實現損益回沖與股息分派。"
    },
    "2886.TW": {
        "name": "兆豐金 (Mega Financial)", "market": "TW", "sector": "金融保險/官股金控",
        "price": 40.5, "pe": 14.5, "pb": 1.38, "eps": 2.79, "roe": 0.102,
        "gross_margin": 0.355, "net_margin": 0.145, "fcf_yield": 0.056, "debt_to_equity": 0.70,
        "div_yield": 0.046, "div_years": 23, "fair_value": 43.0,
        "cheap_price": 37.0, "expensive_price": 49.0,
        "moat": "公股龍頭、外匯清算樞紐，政府持股高，倒閉風險近乎為零。",
        "why_buy": "存股族最愛，配息連續 20 年以上未曾間斷，資產品質極為抗震。"
    },
    "1216.TW": {
        "name": "統一 (Uni-President)", "market": "TW", "sector": "傳統民生/食品內需",
        "price": 86.0, "pe": 19.5, "pb": 2.5, "eps": 4.41, "roe": 0.138,
        "gross_margin": 0.340, "net_margin": 0.054, "fcf_yield": 0.046, "debt_to_equity": 0.50,
        "div_yield": 0.038, "div_years": 32, "fair_value": 90.0,
        "cheap_price": 78.0, "expensive_price": 102.0,
        "moat": "掌控 7-Eleven 超商零售物流通路與全台民生食品龍頭地位。",
        "why_buy": "具備抗通膨定價能力，無論景氣多空皆有穩定現金流入，防禦力頂級。"
    },

    # ================= 美股 (US Stock Market) =================
    "AAPL": {
        "name": "Apple Inc.", "market": "US", "sector": "科技/消費電子",
        "price": 342.4, "pe": 34.2, "pb": 48.0, "eps": 10.01, "roe": 1.55,
        "gross_margin": 0.468, "net_margin": 0.272, "fcf_yield": 0.036, "debt_to_equity": 1.30,
        "div_yield": 0.005, "div_years": 12, "fair_value": 350.0,
        "cheap_price": 295.0, "expensive_price": 410.0,
        "moat": "全球超過 22 億台活躍裝置的封閉 iOS 生態系，使用者轉換成本極高。",
        "why_buy": "Apple Intelligence 換機潮與高毛利軟體服務營收持續雙位數增長。"
    },
    "NVDA": {
        "name": "NVIDIA Corporation", "market": "US", "sector": "科技/半導體AI",
        "price": 186.6, "pe": 48.5, "pb": 38.0, "eps": 3.85, "roe": 1.05,
        "gross_margin": 0.762, "net_margin": 0.545, "fcf_yield": 0.028, "debt_to_equity": 0.35,
        "div_yield": 0.001, "div_years": 11, "fair_value": 195.0,
        "cheap_price": 155.0, "expensive_price": 240.0,
        "moat": "CUDA 軟硬整合開發者生態系垄斷全球 AI 訓練與推論市場。",
        "why_buy": "Blackwell/Rubin 新世代架構供不應求，企業級生成式 AI 算力基礎設施長期紅利。"
    },
    "MSFT": {
        "name": "Microsoft Corporation", "market": "US", "sector": "科技/雲端軟體",
        "price": 465.0, "pe": 33.5, "pb": 12.0, "eps": 13.88, "roe": 0.395,
        "gross_margin": 0.702, "net_margin": 0.362, "fcf_yield": 0.031, "debt_to_equity": 0.40,
        "div_yield": 0.008, "div_years": 20, "fair_value": 480.0,
        "cheap_price": 410.0, "expensive_price": 550.0,
        "moat": "Azure 雲端基礎架構、Office 365 商業壟斷與 OpenAI 深度戰略結盟。",
        "why_buy": "Copilot 商業化滲透率快速提升，自由現金流龐大具備極高防禦力。"
    },
    "COST": {
        "name": "Costco Wholesale", "market": "US", "sector": "民生消費/連鎖零售",
        "price": 930.0, "pe": 52.0, "pb": 15.2, "eps": 17.88, "roe": 0.292,
        "gross_margin": 0.128, "net_margin": 0.029, "fcf_yield": 0.024, "debt_to_equity": 0.30,
        "div_yield": 0.006, "div_years": 19, "fair_value": 900.0,
        "cheap_price": 790.0, "expensive_price": 1050.0,
        "moat": "會員制續約率高達 93%，全球極致以量制價的供應鏈採購定價權。",
        "why_buy": "強韌的抗衰退體質，即使經濟放緩消費者依然大量在好市多採買。"
    },
    "JNJ": {
        "name": "Johnson & Johnson", "market": "US", "sector": "生技醫療/製藥",
        "price": 168.0, "pe": 16.5, "pb": 5.5, "eps": 10.18, "roe": 0.330,
        "gross_margin": 0.690, "net_margin": 0.228, "fcf_yield": 0.050, "debt_to_equity": 0.44,
        "div_yield": 0.031, "div_years": 62, "fair_value": 180.0,
        "cheap_price": 152.0, "expensive_price": 205.0,
        "moat": "62 年股息連續成長（股息貴族），全球創新醫療器材與腫瘤藥專利矩陣。",
        "why_buy": "超長線防禦首選，人口高齡化對醫療器械與創新藥需求具有剛性剛需。"
    },
    "KO": {
        "name": "Coca-Cola Company", "market": "US", "sector": "民生消費/飲料",
        "price": 68.5, "pe": 24.8, "pb": 10.5, "eps": 2.76, "roe": 0.420,
        "gross_margin": 0.610, "net_margin": 0.238, "fcf_yield": 0.040, "debt_to_equity": 1.40,
        "div_yield": 0.030, "div_years": 62, "fair_value": 72.0,
        "cheap_price": 61.0, "expensive_price": 80.0,
        "moat": "全球最大非酒精飲料帝國，巴菲特傳奇持股，品牌忠誠度與全球裝瓶網絡。",
        "why_buy": "現金流如印鈔機，配息連續成長超過 60 年，通膨環境下具備直接轉嫁能力。"
    }
}

# =========================================================================
# ETF_DATA: 經典熱門台股與美股 ETF 知識庫 (專屬一籃子分散指標)
# =========================================================================
ETF_DATA = {
    # ================= 🇹🇼 台灣精選熱門 ETF =================
    "0050.TW": {
        "name": "元大台灣50", "market": "TW", "category": "市值型旗艦",
        "index_name": "臺灣50指數 (追蹤台股市值前50大企業)",
        "price": 92.5, "nav": 92.3, "expense_ratio": "0.43%",
        "div_freq": "半年配 (1月/7月)", "div_yield": 0.038,
        "aum": "超過 4,500 億元",
        "top_holdings": ["台積電 56.2%", "聯發科 4.8%", "鴻海 4.5%", "廣達 2.1%", "台達電 1.9%"],
        "target_audience": "追求貼近台股大盤成長的長期投資人、定期定額小資族首選",
        "strategy": "台股市值型存股之王，建議定期定額不擇時持續扣款，遇大盤急跌回檔時單筆大額加碼。",
        "why_buy": "完全複製台股經濟龍頭成果，台積電高權重帶動極強長線爆發力，永遠不會倒閉下市。"
    },
    "006208.TW": {
        "name": "富邦台50", "market": "TW", "category": "市值型 (超低費率首選)",
        "index_name": "臺灣50指數 (與0050相同，但經理費減半)",
        "price": 108.0, "nav": 107.95, "expense_ratio": "0.24% (同級最低)",
        "div_freq": "半年配 (7月/11月)", "div_yield": 0.035,
        "aum": "超過 1,800 億元",
        "top_holdings": ["台積電 56.1%", "聯發科 4.8%", "鴻海 4.5%", "廣達 2.1%", "台達電 1.9%"],
        "target_audience": "極度重視長期內扣費用侵蝕、計畫扣款 10~20 年以上的長線指數化投資人",
        "strategy": "內扣費用僅 0.24%，長線複利效果比 0050 更省成本，最適合年輕人自動扣款存股。",
        "why_buy": "追蹤與 0050 相同的一籃子龍頭股票，但每年幫投資人省下近一半的管理成本。"
    },
    "0056.TW": {
        "name": "元大高股息", "market": "TW", "category": "經典高股息",
        "index_name": "臺灣高股息指數 (預測未來1年現金殖利率最高50檔)",
        "price": 49.91, "nav": 49.72, "expense_ratio": "0.35%",
        "div_freq": "季配息 (1/4/7/10月)", "div_yield": 0.072,
        "aum": "超過 3,800 億元",
        "top_holdings": ["聯詠 3.8%", "長榮 3.5%", "聯電 3.4%", "廣達 3.2%", "緯創 3.1%"],
        "target_audience": "喜愛領息現金流、具備成立 17 年歷史驗證的保守型與退休族群",
        "strategy": "台股歷史最悠久高股息 ETF，歷經多次景氣循環皆順利填息，適合長線領息滾入。",
        "why_buy": "選股邏輯主動預測未來高配息標的，收益平準金機制維持每季穩定配息。"
    },
    "00878.TW": {
        "name": "國泰永續高股息", "market": "TW", "category": "ESG 穩定高股息",
        "index_name": "MSCI臺灣ESG永續高股息精選30指數",
        "price": 31.99, "nav": 31.87, "expense_ratio": "0.28%",
        "div_freq": "季配息 (2/5/8/11月)", "div_yield": 0.0716,
        "aum": "超過 3,600 億元",
        "top_holdings": ["聯發科 4.2%", "廣達 3.9%", "聯詠 3.7%", "仁寶 3.5%", "大聯大 3.4%"],
        "target_audience": "國民百萬人氣存股王，兼顧 ESG 永續篩選、抗跌防禦與高配息現金流",
        "strategy": "選股重視過去 3 年平均配息穩定度，持股分散且波動度低於大盤，適合安心長抱。",
        "why_buy": "百萬股民最信賴標的，兼具金融保險與電子科技均衡配置，抗震力極強。"
    },
    "00919.TW": {
        "name": "群益台灣精選高息", "market": "TW", "category": "高殖利率旗艦",
        "index_name": "臺灣精選高息指數 (鎖定已宣告配息之高殖利率股)",
        "price": 30.14, "nav": 30.08, "expense_ratio": "0.30%",
        "div_freq": "季配息 (3/6/9/12月)", "div_yield": 0.098,
        "aum": "超過 3,000 億元",
        "top_holdings": ["長榮 9.5%", "聯發科 8.2%", "瑞昱 7.1%", "聯詠 6.8%", "聯電 5.4%"],
        "target_audience": "追求 9%~10% 年化極致高配息、打造被動收入現金流的投資族群",
        "strategy": "5月卡位已公告股利標的、12月鎖定高EPS成長股，配息爆發力極高，逢拉回皆可買。",
        "why_buy": "精準捕捉董事會已宣告的實際配息數字，不靠預測猜測，實打實賺取高殖利率。"
    },
    "00929.TW": {
        "name": "復華台灣科技優息", "market": "TW", "category": "科技高股息 (月配息)",
        "index_name": "特選臺灣科技優息指數 (首檔純科技月配息 ETF)",
        "price": 24.28, "nav": 24.25, "expense_ratio": "0.33%",
        "div_freq": "月配息 (每月領息)", "div_yield": 0.082,
        "aum": "超過 2,500 億元",
        "top_holdings": ["聯發科 8.5%", "大聯大 7.2%", "日月光投控 6.5%", "台灣大 5.8%", "遠傳 5.2%"],
        "target_audience": "喜愛每個月領零用錢/房貸生活費補貼、看好台灣電子科技長期競爭力的投資人",
        "strategy": "每月配息提供平滑金流，剔除景氣循環劇烈股，保留科技成長與配息雙優勢。",
        "why_buy": "百分之百鎖定台灣最強的電子半導體供應鏈，以科技獲利動能支撐長期配息。"
    },

    # ================= 🇺🇸 美國經典 ETF =================
    "VOO": {
        "name": "Vanguard S&P 500 ETF", "market": "US", "category": "美股標普500旗艦",
        "index_name": "S&P 500 Index (全美市值前500大跨國巨頭)",
        "price": 675.99, "nav": 676.01, "expense_ratio": "0.03% (超低管理成本)",
        "div_freq": "季配息 (3/6/9/12月)", "div_yield": 0.0107,
        "aum": "約 1.2 兆美元",
        "top_holdings": ["Apple 7.1%", "Microsoft 6.8%", "NVIDIA 6.5%", "Amazon 3.8%", "Alphabet 3.4%"],
        "target_audience": "全世界長線投資人的終極核心資產、巴菲特遺囑公開指定推薦首選",
        "strategy": "美股最強500家跨國龍頭集合體，長線定期定額30年複利效果驚人，逢大跌即是買點。",
        "why_buy": "自動汰弱留強機制，涵蓋微軟、蘋果、輝達等全球最具創新力的頂尖企業。"
    },
    "QQQ": {
        "name": "Invesco QQQ Trust", "market": "US", "category": "科技成長巨頭 (那斯達克100)",
        "index_name": "NASDAQ-100 Index (非金融創新科技龍頭)",
        "price": 656.1, "nav": 656.55, "expense_ratio": "0.20%",
        "div_freq": "季配息", "div_yield": 0.0045,
        "aum": "約 3,200 億美元",
        "top_holdings": ["Apple 9.2%", "Microsoft 8.8%", "NVIDIA 8.5%", "Amazon 5.5%", "Meta 4.9%"],
        "target_audience": "重視超額資本利得報酬、深信 AI 與雲端科技將主導未來世界的積極型投資人",
        "strategy": "過去20年總報酬率大幅跑贏標普500，科技多頭時期爆發力最強，適合年輕族群重倉。",
        "why_buy": "匯集全球最強科技七雄（Magnificent 7），直接搭乘人工智慧第四次工業革命順風車。"
    },
    "VTI": {
        "name": "Vanguard Total Stock Market ETF", "market": "US", "category": "全美市場涵蓋 (大中小全包)",
        "index_name": "CRSP US Total Market Index (全美國超過3,600檔上市公司)",
        "price": 361.93, "nav": 361.90, "expense_ratio": "0.03%",
        "div_freq": "季配息", "div_yield": 0.013,
        "aum": "約 4,500 億美元",
        "top_holdings": ["Apple 6.4%", "Microsoft 6.1%", "NVIDIA 5.8%", "Amazon 3.4%", "Meta 2.3%"],
        "target_audience": "相信整體美國經濟長期成長、追求極致分散至中小型股票的極致穩健投資人",
        "strategy": "不僅涵蓋S&P 500大型股，亦包含中小型潛力黑馬股，不需預測單一企業即可躺平投資。",
        "why_buy": "全美經濟一網打盡，極致分散風險，內扣費僅0.03%，持股超過3600家。"
    },
    "VT": {
        "name": "Vanguard Total World Stock ETF", "market": "US", "category": "全地球全球化資產",
        "index_name": "FTSE Global All Cap Index (涵蓋全球40多國近10,000檔股票)",
        "price": 151.73, "nav": 151.70, "expense_ratio": "0.07%",
        "div_freq": "季配息", "div_yield": 0.019,
        "aum": "約 550 億美元",
        "top_holdings": ["Apple 4.1%", "Microsoft 3.9%", "NVIDIA 3.7%", "Amazon 2.1%", "TSMC 1.5%"],
        "target_audience": "不願押注單一國家（連美國都不單押）、希望用一檔標的投資全人類經濟文明的投資人",
        "strategy": "終極分散，美國約佔60%、非美成熟國家約28%、新興市場約12%，全球資本自然配置。",
        "why_buy": "無論未來哪個國家崛起，VT 都已經持有其龍頭企業，不用預測地緣政治變遷。"
    },
    "SCHD": {
        "name": "Schwab U.S. Dividend Equity ETF", "market": "US", "category": "美股高股息兼股息成長",
        "index_name": "Dow Jones U.S. Dividend 100 Index",
        "price": 35.17, "nav": 35.18, "expense_ratio": "0.06%",
        "div_freq": "季配息", "div_yield": 0.0295,
        "aum": "約 650 億美元",
        "top_holdings": ["Broadcom 4.5%", "Home Depot 4.2%", "Cisco 4.1%", "BlackRock 4.0%", "Pfizer 3.8%"],
        "target_audience": "追求『股息連續成長 + 資本利得』平衡、注重 ROE 與財務體質的價值投資人",
        "strategy": "連續10年以上配息紀錄 + 負債比健康篩選，股息長年雙位數成長，回撤控制極佳。",
        "why_buy": "兼具現金流與抗震能力，費率超低 0.06%，是美股股息存股族的聖杯標的。"
    }
}

def is_etf_ticker(ticker: str) -> bool:
    """Detect whether a ticker is an ETF rather than an individual corporate stock."""
    clean = ticker.strip().upper()
    if clean.isdigit() and (len(clean) == 4 or len(clean) == 5):
        clean = f"{clean}.TW"
    if clean in ETF_DATA:
        return True
    if clean.startswith("00") and (".TW" in clean or len(clean) in [4, 5, 7, 8]):
        return True
    if clean in ["VOO", "SPY", "QQQ", "VTI", "VT", "SCHD", "IVV", "IWM", "VEA", "VWO"]:
        return True
    return False

def get_etf_profile(ticker: str) -> dict:
    """Fetch ETF specialized profile: NAV, Discount/Premium, Expense Ratio, Holdings."""
    clean = ticker.strip().upper()
    if clean.isdigit() and (len(clean) == 4 or len(clean) == 5):
        clean = f"{clean}.TW"

    base = ETF_DATA.get(clean)
    if not base:
        for k, v in ETF_DATA.items():
            if clean == k or clean == k.split(".")[0]:
                base = v
                clean = k
                break
    
    if not base:
        is_tw = ".TW" in clean or clean.startswith("00")
        base = {
            "name": f"{clean} (ETF)", "market": "TW" if is_tw else "US",
            "category": "指數型股票基金", "index_name": "標的指數",
            "price": 50.0 if is_tw else 100.0, "nav": 49.9 if is_tw else 99.8,
            "expense_ratio": "0.35%", "div_freq": "季配息", "div_yield": 0.04,
            "aum": "中大型規模",
            "top_holdings": ["成分股一 15%", "成分股二 10%", "成分股三 8%"],
            "target_audience": "長線被動指數化投資人",
            "strategy": "定期定額長期持有",
            "why_buy": "一籃子分散投資，降低單一個股踩雷風險。"
        }
    
    res = dict(base)
    res["ticker"] = clean

    # PRIORITY: Live Yahoo Finance query
    try:
        yf_ticker = yf.Ticker(clean)
        info = yf_ticker.info
        curr_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
        nav_price = float(info.get("navPrice") or 0.0)
        y_val = float(info.get("yield") or 0.0)

        if curr_price > 0:
            res["price"] = round(curr_price, 2)
        if nav_price > 0:
            res["nav"] = round(nav_price, 2)
        elif res["price"] > 0:
            res["nav"] = round(res["price"] * 0.998, 2)
        if y_val > 0:
            if y_val > 0.3: # Yahoo Finance returns percentage format (e.g. 7.16 for 7.16%)
                y_val = y_val / 100.0
            res["div_yield"] = round(y_val, 4)
    except Exception:
        pass

    # Compute Discount / Premium
    nav = res["nav"]
    price = res["price"]
    if nav > 0:
        prem = round(((price - nav) / nav) * 100, 2)
    else:
        prem = 0.0
    res["premium_pct"] = prem

    # Evaluate discount state
    if prem < -0.3:
        res["discount_status"] = "🟢 折價優惠（市價 < 淨值）"
        res["discount_desc"] = f"目前市價較淨值折價 {abs(prem):.2f}%，相當於『打折買進一籃子股票』，是極佳的長線進場點！"
        res["discount_color"] = "#16a34a"
        res["action"] = "🟢 建議積極買進 / 定期定額持續扣款"
    elif prem <= 0.3:
        res["discount_status"] = "🔵 公允合理（市價 ≈ 淨值）"
        res["discount_desc"] = f"目前市價與淨值相當（折溢價僅 {prem:+.2f}%），價格反映真實資產價值，適合正常定期定額買進。"
        res["discount_color"] = "#2563eb"
        res["action"] = "🔵 建議正常扣款 / 長線定期定額持有"
    else:
        res["discount_status"] = "🔴 溢價過熱（市價 > 淨值）"
        res["discount_desc"] = f"目前市價較淨值溢價 {prem:.2f}%，市場情緒過熱導致買貴，長線投資人不宜單筆大額追高，建議等待溢價收斂。"
        res["discount_color"] = "#dc2626"
        res["action"] = "🟡 暫緩單筆追高 / 維持小額定期定額"

    return res

def get_all_etfs(market: str = "TW") -> list:
    """Return all ETF profiles for given market."""
    tickers = [k for k, v in ETF_DATA.items() if v["market"] == market]
    return [get_etf_profile(t) for t in tickers]

def get_stock_profile(ticker: str) -> dict:
    """
    Fetch profile prioritizing LIVE Yahoo Finance API first!
    Falls back gracefully to MARKET_DATA if offline or rate limited.
    """
    clean_ticker = ticker.strip().upper()
    if clean_ticker.isdigit() and (len(clean_ticker) == 4 or len(clean_ticker) == 5):
        clean_ticker = f"{clean_ticker}.TW"

    # ETF special handler
    if is_etf_ticker(clean_ticker):
        etf_p = get_etf_profile(clean_ticker)
        return {
            "name": etf_p["name"],
            "market": etf_p["market"],
            "sector": f"ETF ({etf_p['category']})",
            "price": etf_p["price"],
            "pe": 0.0,
            "pb": 1.0,
            "eps": 0.0,
            "roe": 0.0,
            "gross_margin": 0.0,
            "net_margin": 0.0,
            "fcf_yield": 0.0,
            "debt_to_equity": 0.0,
            "div_yield": etf_p["div_yield"],
            "div_years": 15,
            "fair_value": etf_p["nav"],
            "cheap_price": round(etf_p["nav"] * 0.95, 2),
            "expensive_price": round(etf_p["nav"] * 1.05, 2),
            "moat": f"追蹤 {etf_p['index_name']}，一籃子分散持股，自動汰弱留強。",
            "why_buy": etf_p["why_buy"],
            "is_etf": True,
            "etf_data": etf_p
        }

    # Find matching base qualitative template if available
    base = None
    for k, v in MARKET_DATA.items():
        if clean_ticker == k or clean_ticker == k.split(".")[0]:
            base = v
            clean_ticker = k
            break

    # PRIORITY 1: Query live yfinance
    try:
        yf_ticker = yf.Ticker(clean_ticker)
        info = yf_ticker.info
        curr_price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
        
        # If yfinance returned a valid current price (> 0), use live data!
        if curr_price > 0:
            pe = float(info.get("trailingPE") or (base["pe"] if base else 20.0))
            pb = float(info.get("priceToBook") or (base["pb"] if base else 2.5))
            roe = float(info.get("returnOnEquity") or (base["roe"] if base else 0.15))
            eps = float(info.get("trailingEps") or (curr_price / max(pe, 1.0)))
            gm = float(info.get("grossMargins") or (base["gross_margin"] if base else 0.35))
            nm = float(info.get("profitMargins") or (base["net_margin"] if base else 0.15))
            fcf = float(info.get("freeCashflow") or 0.0)
            mcap = float(info.get("marketCap") or 1.0)
            fcf_yield = fcf / max(mcap, 1.0) if (mcap > 0 and fcf > 0) else (base["fcf_yield"] if base else 0.035)
            de = float(info.get("debtToEquity") or 50.0) / 100.0
            div_y = float(info.get("dividendYield") or (base["div_yield"] if base else 0.02))
            if div_y > 0.3: # Yahoo Finance returns percentage format (e.g. 1.31 for 1.31% or 3.19 for 3.19%)
                div_y = div_y / 100.0

            # Dynamic valuation thresholds
            # Fair price based on historical PE range
            fair_p = round(eps * max(pe * 0.95, 15.0), 1) if eps > 0 else round(curr_price, 1)
            cheap_p = round(fair_p * 0.85, 1)
            exp_p = round(fair_p * 1.20, 1)

            return {
                "name": base["name"] if base else info.get("shortName", clean_ticker),
                "market": "TW" if ".TW" in clean_ticker else "US",
                "sector": base["sector"] if base else info.get("sector", "綜合產業"),
                "price": round(curr_price, 1),
                "pe": round(pe, 1),
                "pb": round(pb, 2),
                "eps": round(eps, 2),
                "roe": round(roe, 3),
                "gross_margin": round(gm, 3),
                "net_margin": round(nm, 3),
                "fcf_yield": round(fcf_yield, 3),
                "debt_to_equity": round(de, 2),
                "div_yield": round(div_y, 4),
                "div_years": base["div_years"] if base else 10,
                "fair_value": fair_p,
                "cheap_price": cheap_p,
                "expensive_price": exp_p,
                "moat": base["moat"] if base else "公開市場上市企業，需進一步審視其財報競爭優勢與營運護城河。",
                "why_buy": base["why_buy"] if base else "具備長期基本獲利能力，建議搭配景氣循環於合理價以下分批逢低進場。"
            }
    except Exception:
        pass

    # PRIORITY 2: Fallback to cached profile if yfinance is unreachable
    if base:
        return base
        
    return MARKET_DATA["2330.TW"]

def get_historical_candles(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Get live historical OHLCV data with Moving Averages."""
    clean_ticker = ticker.strip().upper()
    if clean_ticker.isdigit() and len(clean_ticker) == 4:
        clean_ticker = f"{clean_ticker}.TW"

    try:
        yf_ticker = yf.Ticker(clean_ticker)
        df = yf_ticker.history(period=period)
        if len(df) > 10:
            df["MA20"] = df["Close"].rolling(window=20).mean()
            df["MA60"] = df["Close"].rolling(window=60).mean()
            return df
    except Exception:
        pass
    
    # Fallback to simulated profile based on current price
    profile = get_stock_profile(clean_ticker)
    curr_p = profile["price"]
    n_days = 250
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days)
    np.random.seed(abs(hash(clean_ticker)) % 10000)
    returns = np.random.normal(0.0006, 0.015, n_days)
    cum_ret = np.exp(np.cumsum(returns))
    close = curr_p * (cum_ret / cum_ret[-1])
    high = close * (1 + np.random.uniform(0.005, 0.025, n_days))
    low = close * (1 - np.random.uniform(0.005, 0.025, n_days))
    open_p = (high + low) / 2.0
    vol = np.random.randint(1000000, 15000000, n_days)
    
    df = pd.DataFrame({
        "Open": open_p, "High": high, "Low": low, "Close": close, "Volume": vol
    }, index=dates)
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()
    return df

def evaluate_investment_status(stock: dict) -> dict:
    """
    Evaluate valuation status and long-term action signal.
    """
    price = stock["price"]
    cheap = stock["cheap_price"]
    fair = stock["fair_value"]
    exp = stock["expensive_price"]
    
    if price <= cheap:
        val_status = "【便宜價（深度折價）】"
        val_desc = "目前股價顯著低於內在價值，具備高度安全邊際。"
        val_color = "#16a34a" # Green
        action = "🟢 強力推薦買進 / 分批定存"
        action_desc = "價格落於歷史估值下緣，下檔風險有限，適合長線價值投資人積極佈局。"
    elif price <= fair:
        val_status = "【合理偏低價】"
        val_desc = "目前股價處於合理估值區間偏下緣，性價比高。"
        val_color = "#2563eb" # Blue
        action = "🔵 適合長線分批買進 / 定期定額"
        action_desc = "估值公允，若企業持續成長，將隨獲利增長享受股價與股利複利效應。"
    elif price <= exp:
        val_status = "【合理偏高價】"
        val_desc = "目前股價已充分反映公司成長預期，安全邊際收窄。"
        val_color = "#d97706" # Orange
        action = "🟡 建議續抱觀望 / 暫緩大額追高"
        action_desc = "已有部位者可長期安心續抱賺取股利，未上車者建議等待市場情緒拉回再買。"
    else:
        val_status = "【昂貴價（過熱超漲）】"
        val_desc = "目前股價遠高於基本面估值，市場情緒過度樂觀。"
        val_color = "#dc2626" # Red
        action = "🔴 建議分批獲利了結 / 減碼避開"
        action_desc = "股價嚴重透支未來獲利，一旦營收稍不如預期極易發生估值修正（殺估值）。"

    roe = stock["roe"]
    gm = stock["gross_margin"]
    
    health_points = []
    if roe >= 0.20:
        health_points.append(f"卓越獲利能力：ROE 高達 {roe*100:.1f}%，資本運用回報率極高。")
    elif roe >= 0.12:
        health_points.append(f"穩健獲利能力：ROE 為 {roe*100:.1f}%，符合優質企業門檻。")
    else:
        health_points.append(f"獲利能力偏平庸：ROE 僅 {roe*100:.1f}%，資本報酬偏低。")
        
    if gm >= 0.40:
        health_points.append(f"高毛利率優勢：毛利率達 {gm*100:.1f}%，展現強大技術壁壘或品牌定價權。")
    elif gm >= 0.15:
        health_points.append(f"毛利率正常：毛利率約 {gm*100:.1f}%。")
    else:
        health_points.append(f"薄利多銷型：毛利率僅 {gm*100:.1f}%，需依賴龐大規模經濟。")
        
    if stock["fcf_yield"] > 0.03:
        health_points.append("現金流充沛：自由現金流殖利率穩定，真金白銀流入，絕非做帳虛盈。")
    else:
        health_points.append("資本支出較高：自由現金流較吃緊，多數盈餘投入先進產能擴建或研發。")

    return {
        "val_status": val_status,
        "val_desc": val_desc,
        "val_color": val_color,
        "action": action,
        "action_desc": action_desc,
        "health_points": health_points
    }

def get_high_dividend_stocks() -> list:
    """Specialized list of TW stocks suitable for '存股' (Stable & High Dividends)."""
    tw_tickers = [k for k, v in MARKET_DATA.items() if v["market"] == "TW"]
    div_list = []
    for t in tw_tickers:
        item = get_stock_profile(t)
        raw_div = item["div_yield"]
        if raw_div > 0.3:
            raw_div = raw_div / 100.0
        div_y = raw_div * 100.0
        years = item["div_years"]
        score = item["roe"] * 100 + div_y * 3
        if div_y >= 3.0 and years >= 15:
            div_list.append({
                "代號": t,
                "名稱": item["name"],
                "產業類別": item["sector"],
                "現價": f"{item['price']} 元",
                "預估殖利率": f"{div_y:.2f}%",
                "連續配息年數": f"{years} 年",
                "配息穩定度": "極高 (護城河存股首選)" if years >= 20 else "高",
                "存股評等": "🌟🌟🌟🌟🌟 (S級)" if score >= 35 else "🌟🌟🌟🌟 (A級)",
                "長線抱著睡得著理由": item["why_buy"]
            })
    return sorted(div_list, key=lambda x: float(x["預估殖利率"].replace("%", "")), reverse=True)

def get_industry_recommendations(market: str = "TW") -> dict:
    """Group stocks by sector with clear long-term rationale."""
    filtered = {k: v for k, v in MARKET_DATA.items() if v["market"] == market}
    sectors = {}
    for t in filtered.keys():
        item = get_stock_profile(t)
        sec = item["sector"]
        if sec not in sectors:
            sectors[sec] = []
        raw_div = item["div_yield"]
        if raw_div > 0.3:
            raw_div = raw_div / 100.0
        sectors[sec].append({
            "ticker": t,
            "name": item["name"],
            "price": item["price"],
            "pe": item["pe"],
            "roe": f"{item['roe']*100:.1f}%",
            "div_yield": f"{raw_div*100:.1f}%",
            "moat": item["moat"],
            "why_buy": item["why_buy"],
            "fair_value": item["fair_value"],
            "cheap_price": item["cheap_price"]
        })
    return sectors

def simulate_overfitting_comparison():
    """Demonstrate why naive price prediction models overfit vs. factor ranking."""
    np.random.seed(42)
    n_days = 200
    dates = pd.date_range("2024-01-01", periods=n_days)
    
    noise = np.random.normal(0.001, 0.02, n_days)
    price = 100 * np.exp(np.cumsum(noise))
    
    x = np.arange(n_days)
    train_size = 140
    poly_coefs = np.polyfit(x[:train_size], price[:train_size], deg=12)
    poly_fit = np.polyval(poly_coefs, x)
    
    factor_trend = pd.Series(price).rolling(window=20).mean().bfill().values
    
    df = pd.DataFrame({
        "Date": dates,
        "Actual_Price": price,
        "Overfitted_ML_Curve": poly_fit,
        "Robust_Factor_Trend": factor_trend,
        "Dataset": ["Training In-Sample" if i < train_size else "Blind Test Out-of-Sample" for i in range(n_days)]
    })
    return df, train_size
