"""
StockWise-AI: 運用多因子機器學習模型預測台美雙市場長期投資標的之決策決策平台
國立臺北商業大學 人工智慧科技應用學士後專班 115 學年度 實務專題
專題作者：張琬婷 (學號: p1146106)
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import importlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import engine
    importlib.reload(engine)
except ImportError:
    from stock_advisor import engine
    importlib.reload(engine)

try:
    import ml_model
    importlib.reload(ml_model)
except ImportError:
    from stock_advisor import ml_model
    importlib.reload(ml_model)

from engine import (
    get_stock_profile, get_historical_candles, evaluate_investment_status,
    get_high_dividend_stocks, get_industry_recommendations,
    simulate_overfitting_comparison, is_etf_ticker, get_etf_profile,
    get_all_etfs, MARKET_DATA, ETF_DATA
)
from ml_model import predict_stock_ai, get_ai_model_diagnostics, rank_universe_stocks

st.set_page_config(
    page_title="StockWise-AI 機器學習長線投資決策平台",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 4px;
    }
    .status-box {
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🤖 **StockWise-AI**")
st.sidebar.caption("多因子機器學習預測 ✕ 長期價值投資決策平台")

nav_choice = st.sidebar.radio(
    "功能導覽",
    [
        "🤖 機器學習長線標的智能診斷與勝率預測 (ML Inference)",
        "🏆 機器學習長線優選標的排行榜 (ML Cross-Sectional Leaderboard)",
        "🔬 機器學習模型訓練、特徵工程與防過擬合實驗室 (ML Lab & Overfitting Defense)",
        "📦 台美熱門 ETF 存股指南（市值型 ✕ 高股息 ✕ 費率對決）",
        "💰 台股高息「存股專區」（買了放著安心領利息）"
    ]
)

st.sidebar.divider()
st.sidebar.info("""
💡 **學術核心理念**：
短期股價受隨機漫步與市場情緒干擾，傳統高頻時序模型極易**過擬合**。
本專案採用**截面多因子機器學習模型（隨機森林與 GBDT）**，以企業真實賺錢體質為核心，預測 1 年期超越基準指數之勝率（Alpha Outperformance）。
""")

st.sidebar.markdown("""
<div style="font-size: 12px; color: #475569; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; line-height: 1.6;">
    🎓 <b>國立臺北商業大學 人工智慧科技應用學士後專班</b><br>
    👩‍💻 專題作者：<b>張琬婷</b>（<code>p1146106</code>）<br>
    🤖 115 學年度 實務專題初審評審系統<br>
    📚 主軸：<b>機器學習預測台美雙市場長期投資標的</b>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# PAGE 1: 機器學習長線標的智能診斷與勝率預測 (ML Inference & XAI)
# =========================================================================
if nav_choice == "🤖 機器學習長線標的智能診斷與勝率預測 (ML Inference)":
    st.markdown('<div class="main-header">🤖 機器學習長線標的智能診斷與勝率預測</div>', unsafe_allow_html=True)
    st.caption("輸入台股或美股代號，Scikit-Learn 機器學習模型即刻推論長線超額報酬勝率，並深度解析 8 大核心財務因子貢獻。")

    col_m, col_t, col_btn = st.columns([1.3, 2.4, 1])
    with col_m:
        market_mode = st.radio("選擇市場與標的類型", ["🇹🇼 台股個股", "📦 台灣熱門 ETF", "🇺🇸 美股個股", "📦 美國經典 ETF"], horizontal=False)
    
    with col_t:
        if market_mode == "🇹🇼 台股個股":
            quick_options = [
                "2330.TW - 台積電 (晶圓代工龍頭)",
                "2454.TW - 聯發科 (IC設計晶片王)",
                "2317.TW - 鴻海 (AI伺服器代工龍頭)",
                "2382.TW - 廣達 (雲端資料中心龍頭)",
                "2881.TW - 富邦金 (台灣金控獲利王)",
                "2886.TW - 兆豐金 (存股公股金控首選)",
                "1216.TW - 統一 (抗通膨民生食品王)"
            ]
            selected_label = st.selectbox("熱門台股個股快速選", quick_options)
            default_t = selected_label.split(" - ")[0]
            ticker_input = st.text_input("或自訂台股代號（例：2308.TW, 2412.TW）", value=default_t)
        elif market_mode == "📦 台灣熱門 ETF":
            quick_options = [
                "0050.TW - 元大台灣50 (市值型旗艦/台積電大盤成長)",
                "006208.TW - 富邦台50 (同臺灣50/超低0.24%費率首選)",
                "0056.TW - 元大高股息 (17年老牌高股息/季配息)",
                "00878.TW - 國泰永續高股息 (百萬人氣ESG抗跌存股/季配息)",
                "00919.TW - 群益台灣精選高息 (精準宣告高配息10%/季配息)",
                "00929.TW - 復華台灣科技優息 (純科技半導體/月配息)"
            ]
            selected_label = st.selectbox("台灣熱門 ETF 快速選", quick_options)
            default_t = selected_label.split(" - ")[0]
            ticker_input = st.text_input("或自訂台灣 ETF 代號（例：00713.TW, 00692.TW）", value=default_t)
        elif market_mode == "🇺🇸 美股個股":
            quick_options = [
                "AAPL - Apple (消費電子與iOS生態系)",
                "NVDA - NVIDIA (全球AI算力霸主)",
                "MSFT - Microsoft (企業雲端與Copilot)",
                "COST - Costco (好市多抗衰退零售)",
                "JNJ - Johnson & Johnson (62年股息貴族)",
                "KO - Coca-Cola (可口可樂抗通膨印鈔機)"
            ]
            selected_label = st.selectbox("熱門美股個股快速選", quick_options)
            default_t = selected_label.split(" - ")[0]
            ticker_input = st.text_input("或自訂美股代號（例：GOOGL, AMZN, TSLA）", value=default_t)
        else:
            quick_options = [
                "VOO - Vanguard S&P 500 (巴菲特指定標普500旗艦/費率0.03%)",
                "QQQ - Invesco QQQ Trust (那斯達克科技七雄/資本利得王者)",
                "VTI - Vanguard Total Stock (全美市場大中小全包/極致分散)",
                "VT - Vanguard Total World (全球股票型一網打盡/地球文明)",
                "SCHD - Schwab U.S. Dividend (美股高股息兼股息成長聖杯)"
            ]
            selected_label = st.selectbox("美國熱門 ETF 快速選", quick_options)
            default_t = selected_label.split(" - ")[0]
            ticker_input = st.text_input("或自訂美股 ETF 代號（例：DIA, SPY, SMH）", value=default_t)

    with col_btn:
        st.write("")
        st.write("")
        analyze_btn = st.button("🚀 啟動 AI 智能診斷", type="primary", use_container_width=True)

    ticker = ticker_input.strip().upper() if ticker_input else "2330.TW"
    is_etf = is_etf_ticker(ticker)

    with st.spinner(f"正在擷取 {ticker} 最新財報與行情資料，執行機器學習推論..."):
        if is_etf:
            etf = get_etf_profile(ticker)
            stock = None
        else:
            stock = get_stock_profile(ticker)
            etf = None
            status = evaluate_investment_status(stock)
            ai_pred = predict_stock_ai(stock)

    # -------------------------------------------------------------
    # HERO 1: AI 機器學習長線勝率推論結果 (僅個股顯示)
    # -------------------------------------------------------------
    if not is_etf and stock:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%); border: 2px solid #38bdf8; border-radius: 10px; padding: 16px 20px; margin-top: 10px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 15px; font-weight: 700; color: #0369a1;">🤖 Scikit-Learn 多因子機器學習模型推論報告</span>
                <span style="font-size: 13px; background: {ai_pred['color']}; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold;">{ai_pred['badge']}</span>
            </div>
            <div style="display: flex; flex-wrap: wrap; align-items: baseline; gap: 16px; margin-bottom: 6px;">
                <div style="font-size: 26px; font-weight: 800; color: #0f172a;">
                    長線超額報酬勝率：<span style="color: {ai_pred['color']};">{ai_pred['win_rate']}%</span>
                </div>
                <div style="font-size: 14px; font-weight: 600; color: #475569;">
                    AI 建議操作：<span style="color: {ai_pred['color']}; font-weight: 700;">{ai_pred['action_tag']}</span>
                </div>
            </div>
            <div style="font-size: 14px; color: #334155; line-height: 1.6; background: rgba(255,255,255,0.7); padding: 8px 12px; border-radius: 6px;">
                <b>🎯 AI 綜合判定解析：</b>{ai_pred['rating_desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Top Information Row
    # -------------------------------------------------------------
    if is_etf:
        c1, c2, c3, c4 = st.columns(4)
        currency = "NT$" if etf["market"] == "TW" else "US$"
        c1.metric("標的名稱", f"{etf['name']}")
        c2.metric("即時市價", f"{currency} {etf['price']}", delta=f"淨值 {currency} {etf['nav']}")
        prem = etf['premium_pct']
        c3.metric("目前折溢價率", f"{prem:+.2f}%", delta="折價便宜" if prem < 0 else "溢價注意", delta_color="normal" if prem < 0 else "inverse")
        c4.metric("預估年化殖利率", f"{etf['div_yield']*100:.2f}%", f"配息：{etf['div_freq']}")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        currency = "NT$" if stock["market"] == "TW" else "US$"
        c1.metric("標的名稱", f"{stock['name']}")
        c2.metric("最新股價", f"{currency} {stock['price']}", f"本益比: {stock['pe']:.1f}x")
        c3.metric("每股盈餘 (EPS)", f"{currency} {stock['eps']}", f"ROE: {stock['roe']*100:.1f}%")
        div_y_display = stock['div_yield'] if stock['div_yield'] <= 0.3 else stock['div_yield'] / 100.0
        c4.metric("現金殖利率", f"{div_y_display*100:.2f}%", f"連續配息: {stock.get('div_years', 10)}年")
        c5.metric("內在估值位階", status["val_status"], delta=status["action"], delta_color="normal" if "買進" in status["action"] else ("off" if "續抱" in status["action"] else "inverse"))

    st.write("")

    # -------------------------------------------------------------
    # Main Tabs / Two-Column Layout
    # -------------------------------------------------------------
    col_chart, col_val = st.columns([1.6, 1.2])

    with col_chart:
        st.subheader("📈 互動式 K 線走勢圖 (含成交量與均線支撐)")
        hist_df = get_historical_candles(ticker, period="1y")
        
        if hist_df.empty:
            st.warning("⚠️ 暫時無法取得該標的歷史 K 線數據，請檢查代號是否正確。")
        else:
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.04, row_heights=[0.75, 0.25]
            )
            fig.add_trace(go.Candlestick(
                x=hist_df.index,
                open=hist_df['Open'], high=hist_df['High'],
                low=hist_df['Low'], close=hist_df['Close'],
                name='K線走勢',
                increasing_line_color='#ef4444',
                decreasing_line_color='#22c55e'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df['MA20'],
                mode='lines', line=dict(color='#3b82f6', width=1.5),
                name='月線 (MA20)'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df['MA60'],
                mode='lines', line=dict(color='#f59e0b', width=1.5),
                name='季線生命線 (MA60)'
            ), row=1, col=1)

            colors = ['#ef4444' if c >= o else '#22c55e' for c, o in zip(hist_df['Close'], hist_df['Open'])]
            fig.add_trace(go.Bar(
                x=hist_df.index, y=hist_df['Volume'],
                marker_color=colors, name='成交量'
            ), row=2, col=1)

            fig.update_layout(
                height=450, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_val:
        if is_etf:
            st.subheader("🎯 ETF 淨值追蹤與折溢價量表")
            prem_pct = etf["premium_pct"]
            if prem_pct < -0.3:
                etf_status = "🟢 折價中（市價低於淨值 ➔ 撿便宜買進時機）"
                etf_desc = f"目前市價比淨值便宜 <b>{abs(prem_pct):.2f}%</b>！等於以打折價購入一籃子資產，建議分批加碼。"
                etf_color = "#16a34a"
            elif prem_pct <= 0.3:
                etf_status = "🔵 公允價（市價與淨值相當 ➔ 正常定期定額）"
                etf_desc = f"目前折溢價幅度僅 <b>{prem_pct:+.2f}%</b>，處於正常合理範圍，適合照既定計畫定期定額扣款。"
                etf_color = "#2563eb"
            else:
                etf_status = "🔴 溢價過高（市價高於淨值 ➔ 不宜追高）"
                etf_desc = f"目前溢價達 <b>{prem_pct:.2f}%</b>！買在溢價等於買貴資產，建議等待折溢價收斂後再進場。"
                etf_color = "#dc2626"

            st.markdown(f"""
            <div class="status-box" style="background-color: {etf_color}15; border-left: 5px solid {etf_color};">
                <h4 style="color: {etf_color}; margin: 0 0 6px 0;">{etf_status}</h4>
                <div style="font-size: 13px; color: #334155;">{etf_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            clamped_prem = max(min(prem_pct, 1.5), -1.5)
            etf_pct = ((clamped_prem + 1.5) / 3.0) * 100
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; font-weight: 600; margin-bottom: 6px;">
                    <span>🟢 折價區 (-1.5%)</span>
                    <span>🔵 合理區 (0%)</span>
                    <span>🔴 溢價區 (+1.5%)</span>
                </div>
                <div style="position: relative; height: 12px; background: linear-gradient(to right, #10b981 0%, #10b981 45%, #3b82f6 45%, #3b82f6 55%, #ef4444 55%, #ef4444 100%); border-radius: 6px;">
                    <div style="position: absolute; left: calc({etf_pct}% - 6px); top: -4px; width: 12px; height: 20px; background: #0f172a; border: 2px solid #ffffff; border-radius: 4px;"></div>
                </div>
                <div style="text-align: right; font-size: 12px; color: #475569; margin-top: 5px;">
                    市價 <b>{currency} {etf['price']}</b> vs 淨值 <b>{currency} {etf['nav']}</b>（折溢價 <b>{etf['premium_pct']:+.2f}%</b>）
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### 🎚️ 拖動拉桿：試算不同買進市價之折溢價")
            slider_min = float(round(etf["nav"] * 0.90, 2))
            slider_max = float(round(etf["nav"] * 1.10, 2))
            step_val = 0.05 if etf["price"] < 100 else 0.5
            sim_price = st.slider(
                "目標買進市價試算：",
                min_value=slider_min, max_value=slider_max,
                value=float(etf["price"]), step=step_val
            )
            sim_prem = ((sim_price - etf["nav"]) / etf["nav"]) * 100
            st.info(f"若以 **{currency} {sim_price:,.2f}** 買進：折溢價率為 **{sim_prem:+.2f}%**")

        else:
            st.subheader("🎯 估價分析與長線投資建議")
            st.markdown(f"""
            <div class="status-box" style="background-color: {status['val_color']}15; border-left: 5px solid {status['val_color']};">
                <h3 style="color: {status['val_color']}; margin: 0 0 6px 0;">{status['action']}</h3>
                <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">目前位階：{status['val_status']}</div>
                <div style="font-size: 13px; color: #334155;">{status['action_desc']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📏 股價反映實際價值量表")
            st.caption("依據長期盈餘 EPS、歷史本益比與安全邊際折現模型計算：")

            st.markdown(f"""
            <div style="display: flex; gap: 8px; justify-content: space-between; margin-bottom: 12px;">
                <div style="flex: 1; background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 10px 6px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 600; color: #047857; margin-bottom: 4px;">🟢 便宜價 (下檔保護)</div>
                    <div style="font-size: 16px; font-weight: 700; color: #065f46;">{stock['cheap_price']} {currency}</div>
                </div>
                <div style="flex: 1; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 6px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 600; color: #1d4ed8; margin-bottom: 4px;">🔵 合理價 (內在價值)</div>
                    <div style="font-size: 16px; font-weight: 700; color: #1e40af;">{stock['fair_value']} {currency}</div>
                </div>
                <div style="flex: 1; background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 10px 6px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 600; color: #b91c1c; margin-bottom: 4px;">🔴 昂貴價 (超額獲利)</div>
                    <div style="font-size: 16px; font-weight: 700; color: #991b1b;">{stock['expensive_price']} {currency}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            cheap_p = stock["cheap_price"]
            exp_p = stock["expensive_price"]
            curr_p = stock["price"]
            pct = 0.0 if exp_p == cheap_p else ((curr_p - cheap_p) / (exp_p - cheap_p)) * 100
            pct = max(min(pct, 100), 0)

            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; font-weight: 600; margin-bottom: 6px;">
                    <span>🟢 便宜價區間</span>
                    <span>🔵 合理價區間</span>
                    <span>🔴 昂貴價區間</span>
                </div>
                <div style="position: relative; height: 12px; background: linear-gradient(to right, #10b981 0%, #10b981 33%, #3b82f6 33%, #3b82f6 66%, #ef4444 66%, #ef4444 100%); border-radius: 6px;">
                    <div style="position: absolute; left: calc({pct}% - 6px); top: -4px; width: 12px; height: 20px; background: #0f172a; border: 2px solid #ffffff; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Interactive Slider
            st.markdown("##### 🎚️ 拖動拉桿：試算不同買進價位之估值變化")
            slider_min = float(round(stock["cheap_price"] * 0.75, 1))
            slider_max = float(round(stock["expensive_price"] * 1.25, 1))
            step_val = 0.5 if stock["price"] < 100 else 5.0
            sim_price = st.slider(
                "目標買進價位試算：",
                min_value=slider_min, max_value=slider_max,
                value=float(stock["price"]), step=step_val
            )

            sim_profile = stock.copy()
            sim_profile["price"] = sim_price
            sim_status = evaluate_investment_status(sim_profile)
            sim_yield = (stock["price"] * div_y_display) / sim_price if sim_price > 0 else 0

            st.markdown(f"""
            <div style="background-color: {sim_status['val_color']}15; border-left: 4px solid {sim_status['val_color']}; border-radius: 6px; padding: 8px 12px; font-size: 13px;">
                若以 <b>{currency} {sim_price:,.1f}</b> 買進：位階為 <b style="color:{sim_status['val_color']}">{sim_status['val_status']}</b> ➔ <b>{sim_status['action']}</b>（殖利率約 <b>{sim_yield*100:.2f}%</b>）
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Section 2: 企業獲利體質 & 機器學習可解釋性 (XAI)
    # -------------------------------------------------------------
    if not is_etf and stock:
        st.write("")
        st.subheader("🏢 這家公司有沒有賺錢？（基本面體質深度檢視 ✕ XAI 特徵因子貢獻）")
        c_p1, c_p2 = st.columns(2)

        with c_p1:
            st.markdown("##### 💵 核心賺錢能力指標")
            p_table = [
                {"指標項目": "營業毛利率 (Gross Margin)", "數值": f"{stock['gross_margin']*100:.1f}%", "解讀說明": "代表產品成本競爭力與定價權高低"},
                {"指標項目": "稅後純益率 (Net Margin)", "數值": f"{stock['net_margin']*100:.1f}%", "解讀說明": "扣除營業費用後，公司真正放進口袋的淨利比率"},
                {"指標項目": "股東權益報酬率 (ROE)", "數值": f"{stock['roe']*100:.1f}%", "解讀說明": "巴菲特最看重指標：每一百元股東本金能賺回幾元"},
                {"指標項目": "自由現金流殖利率 (FCF Yield)", "數值": f"{stock['fcf_yield']*100:.1f}%", "解讀說明": "營運扣除資本支出後的真現金流量"}
            ]
            st.dataframe(pd.DataFrame(p_table), use_container_width=True, hide_index=True)

            st.markdown("##### 📌 體質評核摘要清單：")
            for pt in status["health_points"]:
                st.write(f"• {pt}")

        with c_p2:
            st.markdown("##### 🛡️ 護城河優勢與長線投資理由")
            st.info(f"🏰 **企業護城河（為什麼不怕對手競爭）**：\n\n{stock['moat']}")
            st.success(f"🚀 **為什麼適合長期投資？**\n\n{stock['why_buy']}")

            st.markdown("##### 🤖 XAI 機器學習因子權重解釋")
            st.caption("隨機森林依據 8 大因子對該標的進行之綜合特徵加權分佈：")
            df_xai = pd.DataFrame(ai_pred["feature_contributions"])
            st.dataframe(df_xai, use_container_width=True, hide_index=True)


# =========================================================================
# PAGE 2: 機器學習長線優選標的排行榜 (ML Cross-Sectional Leaderboard)
# =========================================================================
elif nav_choice == "🏆 機器學習長線優選標的排行榜 (ML Cross-Sectional Leaderboard)":
    st.markdown('<div class="main-header">🏆 機器學習長線優選標的排行榜</div>', unsafe_allow_html=True)
    st.caption("運用 Scikit-Learn 多因子隨機森林截面排序模型，對台股與美股全市場標的進行長線超額報酬勝率評估。")

    col_flt1, col_flt2, col_flt3 = st.columns([1.2, 1.2, 1.6])
    with col_flt1:
        market_filter = st.selectbox("篩選市場", ["全部市場 (台美雙向)", "🇹🇼 台股市場", "🇺🇸 美股市場"])
    with col_flt2:
        tier_filter = st.selectbox("AI 評級等級", ["全部等級", "S級強力推薦 (勝率 ≥ 80%)", "A級穩健配置 (勝率 ≥ 65%)", "B級區間操作 (勝率 ≥ 45%)"])
    with col_flt3:
        sort_by = st.selectbox("排序依據", ["AI 長線勝率 (高到低)", "股東權益報酬率 ROE (高到低)", "預估殖利率 (高到低)"])

    m_code = "ALL"
    if "台股" in market_filter:
        m_code = "TW"
    elif "美股" in market_filter:
        m_code = "US"

    df_ranked = rank_universe_stocks(MARKET_DATA, market_filter=m_code)

    if "S級" in tier_filter:
        df_ranked = df_ranked[df_ranked["AI 評級"].str.contains("S級")]
    elif "A級" in tier_filter:
        df_ranked = df_ranked[df_ranked["AI 評級"].str.contains("A級")]
    elif "B級" in tier_filter:
        df_ranked = df_ranked[df_ranked["AI 評級"].str.contains("B級")]

    if sort_by == "股東權益報酬率 ROE (高到低)":
        df_ranked["temp_roe"] = df_ranked["股東權益報酬率 (ROE)"].str.replace("%", "").astype(float)
        df_ranked = df_ranked.sort_values("temp_roe", ascending=False).drop(columns=["temp_roe"])
    elif sort_by == "預估殖利率 (高到低)":
        df_ranked["temp_yield"] = df_ranked["預估殖利率"].str.replace("%", "").astype(float)
        df_ranked = df_ranked.sort_values("temp_yield", ascending=False).drop(columns=["temp_yield"])

    df_ranked["AI 推薦排行"] = [f"第 {i+1} 名" for i in range(len(df_ranked))]

    # Metric Row
    s_count = len(df_ranked[df_ranked["AI 評級"].str.contains("S級")])
    a_count = len(df_ranked[df_ranked["AI 評級"].str.contains("A級")])
    avg_win = df_ranked["AI 長線勝率"].mean() if not df_ranked.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("評估標的總數", f"{len(df_ranked)} 檔")
    k2.metric("🌟 S級強力推薦", f"{s_count} 檔", "勝率 ≥ 80%")
    k3.metric("✨ A級穩健配置", f"{a_count} 檔", "勝率 65~80%")
    k4.metric("平均 AI 預估勝率", f"{avg_win:.1f}%")

    st.write("")
    st.subheader("🥇 AI 模型首選 Top 3 長線核心資產")
    top3 = df_ranked.head(3).to_dict(orient="records")
    top_cols = st.columns(3)
    for idx, item in enumerate(top3):
        with top_cols[idx]:
            st.markdown(f"""
            <div style="background: white; border: 2px solid #2563eb; border-radius: 8px; padding: 14px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 12px; background: #dbeafe; color: #1e40af; font-weight: bold; padding: 2px 8px; border-radius: 4px;">{item['AI 推薦排行']} ｜ {item['市場']}</span>
                    <span style="font-size: 12px; background: #dcfce7; color: #166534; font-weight: bold; padding: 2px 8px; border-radius: 4px;">{item['AI 評級']}</span>
                </div>
                <h3 style="margin: 4px 0 6px 0; color: #0f172a;">{item['企業名稱']} <span style="font-size: 14px; color: #64748b;">({item['標的代號']})</span></h3>
                <div style="font-size: 13px; color: #334155; margin-bottom: 6px;">
                    產業：<b>{item['產業領域']}</b> ｜ 現價：<b>{item['現價']}</b>
                </div>
                <div style="background: #eff6ff; border-radius: 6px; padding: 8px; margin-bottom: 8px; font-size: 13px;">
                    AI 勝率：<b style="color: #2563eb; font-size: 16px;">{item['AI 長線勝率']}%</b> ｜ ROE：<b>{item['股東權益報酬率 (ROE)']}</b>
                </div>
                <div style="font-size: 12px; color: #475569; line-height: 1.5;">
                    <b>🏰 護城河：</b>{item['長線護城河核心']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.subheader("📋 全市場 AI 截面排序總表")
    st.dataframe(df_ranked, use_container_width=True, hide_index=True)


# =========================================================================
# PAGE 3: 機器學習模型訓練、特徵工程與防過擬合實驗室 (ML Lab)
# =========================================================================
elif nav_choice == "🔬 機器學習模型訓練、特徵工程與防過擬合實驗室 (ML Lab & Overfitting Defense)":
    st.markdown('<div class="main-header">🔬 機器學習模型訓練、特徵工程與防過擬合實驗室</div>', unsafe_allow_html=True)
    st.caption("專題初審答辯關鍵核心：多演算法基準測試 (Multi-Model Benchmark)、5 折時間序列 Walk-Forward 交叉驗證、以及過擬合防禦架構。")

    diag = get_ai_model_diagnostics()
    metrics = diag["metrics"]
    importances = diag["feature_importances"]
    benchmarks = diag["benchmark_comparison"]

    st.subheader("1. 多演算法模型盲測基準對比 (Multi-Model Empirical Benchmark)")
    st.write("本研究對比了三種機器學習演算法在未見盲測集（Out-of-Sample Test Set）與 5 折時間序列交叉驗證下的表現：")
    st.dataframe(pd.DataFrame(benchmarks), use_container_width=True, hide_index=True)

    st.write("")
    st.subheader("2. 防過擬合核心工程指標 (Walk-Forward Cross Validation)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("🎯 5折時間序列交叉驗證 AUC", f"{metrics['cv_auc_mean']}")
    m_col2.metric("🧪 盲測集 (Out-of-Sample) 準確率", f"{metrics['test_accuracy']}%")
    m_col3.metric("📊 盲測 ROC-AUC 分數", f"{metrics['test_auc']}")
    m_col4.metric("⚖️ 盲測 F1-Score", f"{metrics['test_f1']}")

    st.write("")
    st.subheader("3. 過擬合災難實例對比 (In-Sample vs. Out-of-Sample)")
    df_sim, train_idx = simulate_overfitting_comparison()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sim["Date"], y=df_sim["Actual_Price"], mode='lines', name='真實股價 (Ground Truth)', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=df_sim["Date"][:train_idx], y=df_sim["Overfitted_ML_Curve"][:train_idx], mode='lines', name='過擬合模型 (訓練集擬合完美)', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=df_sim["Date"][train_idx:], y=df_sim["Overfitted_ML_Curve"][train_idx:], mode='lines', name='過擬合模型 (盲測集崩潰發散)', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=df_sim["Date"], y=df_sim["Robust_Factor_Trend"], mode='lines', name='多因子平滑長期趨勢 (本專案採用)', line=dict(color='blue', width=2)))
    
    fig.add_vline(x=df_sim["Date"].iloc[train_idx], line_dash="dash", line_color="gray", annotation_text="訓練 / 盲測 分界點")
    fig.update_layout(title="時序價格預測模型 vs. 長期基本面因子模型 泛化能力對比", height=420)
    st.plotly_chart(fig, use_container_width=True)

    c_exp1, c_exp2 = st.columns(2)
    with c_exp1:
        st.error("""
        🚨 **傳統時序預測的致命過擬合**：
        高階時序模型（如高階多項式或深度 LSTM）試圖擬合每日股價，在訓練集（綠線）誤差極低；但因金融價格包含大量隨機漫步雜訊，一進入未知盲測集（紅線）直接嚴重發散！
        """)
    with c_exp2:
        st.success("""
        ✅ **本專案採用的三大防過擬合策略**：
        1. **問題重構**：放棄預測每日點位，改為預測「截面超額報酬機率」。<br>
        2. **剪枝與正則化**：隨機森林限定樹深 (`max_depth=5`)，葉節點最小樣本數 (`min_samples_leaf=10`)，遏制孤立雜訊記憶。<br>
        3. **Walk-Forward 交叉驗證**：嚴格依時間順序驗證，徹底防範未來資訊洩漏 (Data Snooping)。
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("4. 8 大財務動能因子重要性排序 (Feature Importance)")
    st.caption("隨機森林分類器透過基尼不純度減少量（MDI）自動學習出之因子權重分佈：")

    df_imp = pd.DataFrame(list(importances.items()), columns=["特徵因子", "重要性佔比 (%)"]).sort_values("重要性佔比 (%)", ascending=True)
    fig_imp = go.Figure(go.Bar(
        x=df_imp["重要性佔比 (%)"],
        y=df_imp["特徵因子"],
        orientation='h',
        marker_color='#2563eb'
    ))
    fig_imp.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)


# =========================================================================
# PAGE 4: 台美熱門 ETF 存股指南（市值型 ✕ 高股息 ✕ 費率對決）
# =========================================================================
elif nav_choice == "📦 台美熱門 ETF 存股指南（市值型 ✕ 高股息 ✕ 費率對決）":
    st.markdown('<div class="main-header">📦 台美熱門 ETF 存股指南（市值型 ✕ 高股息 ✕ 費率大對決）</div>', unsafe_allow_html=True)
    st.write("投資個股需要持續追蹤單一公司的財報與護城河；**ETF（指數股票型基金）**則是一次買進一籃子優質企業，具備**『自動汰弱留強、永遠不會倒閉、省時省力』**的長線存股優勢！")

    etf_market_tab = st.radio("選擇欲查看之市場", ["🇹🇼 台灣精選熱門 ETF (0050 / 006208 / 0056 / 00878 / 00919 / 00929)", "🇺🇸 美國經典熱門 ETF (VOO / QQQ / VTI / VT / SCHD)"], horizontal=True)
    m_key = "TW" if "台灣" in etf_market_tab else "US"
    etf_list = get_all_etfs(m_key)
    curr_s = "NT$" if m_key == "TW" else "US$"

    cats = ["全部標的", "📈 市值型", "💰 高股息型", "🚀 科技成長型"]
    cat_choice = st.segmented_control("篩選 ETF 類型", cats, default="全部標的")

    st.subheader(f"📋 {'台灣' if m_key == 'TW' else '美國'}精選 ETF 核心清單")
    
    for item in etf_list:
        if cat_choice != "全部標的" and cat_choice.split(" ")[1] not in item["category"]:
            continue
        
        prem_val = item["premium_pct"]
        prem_color = "#16a34a" if prem_val < -0.3 else ("#2563eb" if prem_val <= 0.3 else "#dc2626")
        prem_text = f"折價 {abs(prem_val):.2f}% (撿便宜)" if prem_val < 0 else (f"溢價 {prem_val:.2f}% (公允)" if prem_val <= 0.3 else f"溢價 {prem_val:.2f}% (注意)")

        with st.expander(f"{item['category']} ｜ {item['name']} ({item['ticker']}) ｜ 現價 {curr_s} {item['price']} ｜ 殖利率 {item['div_yield']*100:.2f}% ({item['div_freq']})", expanded=True):
            ec1, ec2, ec3 = st.columns([1.2, 1.8, 1.2])
            with ec1:
                st.markdown(f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px;">
                    <div style="font-size:12px; color:#64748b;">市價 / 淨值 (NAV)</div>
                    <div style="font-size:20px; font-weight:700; color:#0f172a;">{curr_s} {item['price']} <span style="font-size:14px; color:#475569;">/ {item['nav']}</span></div>
                    <div style="margin-top:6px; font-size:13px; font-weight:600; color:{prem_color};">折溢價：{item['premium_pct']:+.2f}% ({prem_text})</div>
                    <hr style="margin:8px 0; border:none; border-top:1px solid #cbd5e1;">
                    <div style="font-size:12px;"><b>總內扣費：</b><span style="color:#0f766e; font-weight:bold;">{item['expense_ratio']}</span></div>
                    <div style="font-size:12px;"><b>配息頻率：</b>{item['div_freq']}</div>
                    <div style="font-size:12px;"><b>基金規模：</b>{item['aum']}</div>
                </div>
                """, unsafe_allow_html=True)
            with ec2:
                st.markdown(f"**🎯 追蹤指數：** {item['index_name']}")
                st.markdown(f"**🍰 前三大成分股：** {', '.join(item['top_holdings'][:3])}")
                st.info(f"💡 **核心存股理由：** {item['why_buy']}")
            with ec3:
                st.success(f"👥 **適合對象：**\n\n{item['target_audience']}")
                st.warning(f"🚀 **長線扣款策略：**\n\n{item['strategy']}")

    st.write("")
    st.subheader("📊 台美熱門 ETF 規格橫向大對決")
    table_rows = []
    for it in etf_list:
        table_rows.append({
            "代號": it["ticker"],
            "名稱": it["name"],
            "分類": it["category"],
            "市價": f"{curr_s} {it['price']}",
            "淨值": f"{curr_s} {it['nav']}",
            "折溢價": f"{it['premium_pct']:+.2f}%",
            "費用率": it["expense_ratio"],
            "配息頻率": it["div_freq"],
            "預估殖利率": f"{it['div_yield']*100:.2f}%",
            "適合族群": it["target_audience"].split("、")[0] if "、" in it["target_audience"] else it["target_audience"][:18] + "..."
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# =========================================================================
# PAGE 5: 台股高息「存股專區」（買了放著安心領利息）
# =========================================================================
elif nav_choice == "💰 台股高息「存股專區」（買了放著安心領利息）":
    st.markdown('<div class="main-header">💰 台股穩定高息「存股專區」（買了放著安心領利息）</div>', unsafe_allow_html=True)
    st.write("台灣投資人最熱愛的**「存股領息」**哲學！挑選存股標的不能只看殖利率高低，必須符合三大標準：")
    
    col_k1, col_k2, col_k3 = st.columns(3)
    col_k1.success("✅ **連續配息 15 年以上**\n\n經歷過金融海嘯、疫情依然年年發放現金股利。")
    col_k2.info("✅ **倒閉風險近乎為零**\n\n具備官股背景、大到不能倒金控或民生必需品龍頭。")
    col_k3.warning("✅ **殖利率穩定 > 3.8%**\n\n每年穩健領取被動現金流，適合當退休養老金。")

    st.subheader("📋 適合「買了放著抱得住」的高息存股清單")
    div_stocks = get_high_dividend_stocks()
    for row in div_stocks:
        val_str = str(row.get("預估殖利率", "")).replace("%", "").strip()
        try:
            val_num = float(val_str)
            if val_num > 30.0:
                val_num = val_num / 100.0
                row["預估殖利率"] = f"{val_num:.2f}%"
        except ValueError:
            pass
    df_div = pd.DataFrame(div_stocks)
    st.dataframe(df_div, use_container_width=True, hide_index=True)

    st.write("")
    st.subheader("💡 存股族常見痛點與解答：")
    with st.expander("Q1：高股息會不會『賺了利息、賠了差價』？"):
        st.write("答：如果挑選的是沒有競爭力、獲利衰退的夕陽產業，確實會發生無法填息的情況。但本專區嚴選的兆豐金、富邦金、統一與聯發科，長期 ROE 均維持在 10%~25% 以上，公司賺得比配得多，長期還原權息的股價趨勢持續向上，填息機率超過 90%。")
        
    with st.expander("Q2：存股什麼時候買最好？一定要等便宜價嗎？"):
        st.write("答：存股族最佳策略是『**定期定額＋拉回大跌時加碼**』。當股價落在合理偏低或便宜價時，殖利率會被放大（例如兆豐金跌至 36 元以下時殖利率破 5%），此時是擴大本金持股部位的最佳時機。")
