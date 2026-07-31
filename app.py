import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import requests
import numpy as np
import joblib
import os

# --- 頁面基本配置 ---
st.set_page_config(
    page_title="專屬智慧交易儀表板 V4.1",
    page_icon="📈",
    layout="wide"
)

st.title("📈 專屬智慧交易與選股儀表板 V4.1 (AI 機器學習彈性控盤版)")

# --- 載入預先訓練好的 AI 模型 ---
@st.cache_resource
def load_ml_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    return None

ml_model = load_ml_model()

if ml_model:
    st.sidebar.success("🤖 AI 機器學習預測引擎：已連線 (model.pkl)")
else:
    st.sidebar.warning("⚠️ 尚未偵測到 model.pkl，請先執行 train_model.py 訓練模型。")

# --- 自動抓取台股清單 ---
@st.cache_data(ttl=86400)
def get_all_taiwan_stocks():
    stock_dict = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
        resp = requests.get(url_twse, headers=headers, timeout=10)
        if resp.status_code == 200:
            for item in resp.json():
                code = str(item.get('Code', '')).strip()
                name = str(item.get('Name', '')).strip()
                if len(code) == 4 and code.isdigit():
                    stock_dict[f"{code}.TW"] = name
    except Exception:
        pass
        
    try:
        url_tpex = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
        resp = requests.get(url_tpex, headers=headers, timeout=10)
        if resp.status_code == 200:
            for item in resp.json():
                code = str(item.get('SecuritiesCompanyCode', '')).strip()
                name = str(item.get('CompanyName', '')).strip()
                if len(code) == 4 and code.isdigit():
                    stock_dict[f"{code}.TWO"] = name
    except Exception:
        pass

    return stock_dict

TOP100_CODES = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "2881.TW", "2882.TW",
    "3260.TW", "8271.TWO", "3362.TW", "2451.TW", "4967.TW", "3441.TWO", "3504.TW", "4976.TW", "3312.TW"
]

# --- 側邊欄選單 ---
st.sidebar.header("📌 功能選單")
menu = st.sidebar.radio("", ["1. AI 機器學習預測當沖篩選", "2. 美股即時行情觀測", "3. 玉山證券帳務與交易"])

# ==========================================
# 功能一：AI 機器學習選股模組
# ==========================================
if menu == "1. AI 機器學習預測當沖篩選":
    st.header("🎯 機器學習勝率預測與彈性風控模組")
    
    all_stocks = get_all_taiwan_stocks()
    st.success(f"🌐 已成功載入全台股資料庫，共收錄 **{len(all_stocks)}** 檔普通股。")
    
    st.subheader("⚙️ 多維度篩選與風控彈性設定")
    
    # 第一排參數控制
    r1_1, r1_2, r1_3 = st.columns(3)
    with r1_1:
        scan_scope = st.radio("掃描範圍：", ["🚀 全台股大掃描 (1800+ 檔)", "🔥 精選熱門當沖庫"], horizontal=True)
    with r1_2:
        min_win_prob = st.slider("🤖 AI 預測勝率門檻 (%)", 40, 90, 50, step=5, help="若搜尋數量過少，可適度拉低勝率門檻（如 50%~55%）")
    with r1_3:
        sl_pct = st.number_input("風控停損比例 (%)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

    # 第二排參數控制
    r2_1, r2_2, r2_3 = st.columns(3)
    with r2_1:
        gap_min, gap_max = st.slider("📈 開盤溢價率區間 (%)", 0.0, 10.0, (1.0, 7.0), step=0.5)
    with r2_2:
        min_vol = st.number_input("📊 昨日成交量門檻（張）", min_value=0, value=500, step=100)
    with r2_3:
        min_current_pct = st.number_input("⚡ 最低當前漲幅 (%)", min_value=-5.0, max_value=5.0, value=0.0, step=0.5)

    # 第三排開關控制
    r3_1, r3_2 = st.columns(2)
    with r3_1:
        exclude_ky = st.checkbox("🛡️ 自動排除 -KY 股", value=True)
    with r3_2:
        enable_orb = st.checkbox("🔥 ORB 續強過濾 (現價 ≥ 開盤價)", value=True, help="取消勾選可釋放更多潛在開高拉回的標的")

    if st.button("🚀 開始 AI 大數據即時勝率掃描"):
        target_stocks = all_stocks if "全台股" in scan_scope else {k: all_stocks.get(k, k) for k in TOP100_CODES if k in all_stocks}
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tickers_list = list(target_stocks.keys())
        total_count = len(tickers_list)
        
        chunk_size = 100
        chunks = [tickers_list[i:i + chunk_size] for i in range(0, total_count, chunk_size)]
        
        matched_count = 0
        processed_count = 0
        
        for chunk_idx, chunk_tickers in enumerate(chunks):
            status_text.text(f"AI 引擎正在計算全市場特徵矩陣 (第 {chunk_idx + 1} / {len(chunks)} 批次)...")
            try:
                data = yf.download(chunk_tickers, period="30d", interval="1d", group_by='ticker', threads=True, progress=False)
                
                for ticker in chunk_tickers:
                    processed_count += 1
                    try:
                        df_stock = data[ticker] if len(chunk_tickers) > 1 else data
                        df_clean = df_stock.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
                        
                        if len(df_clean) >= 20:
                            open_p = float(df_clean['Open'].iloc[-1])
                            prev_close = float(df_clean['Close'].iloc[-2])
                            current_price = float(df_clean['Close'].iloc[-1])
                            prev_vol_lots = float(df_clean['Volume'].iloc[-2]) / 1000.0
                            stock_name = target_stocks.get(ticker, "未知")
                            
                            # 1. KY 股過濾
                            if exclude_ky and ("KY" in stock_name or "KY" in ticker):
                                continue
                                
                            # 2. 成交量門檻過濾
                            if prev_vol_lots < min_vol:
                                continue
                                
                            gap_rate = ((open_p - prev_close) / prev_close) * 100
                            current_pct = ((current_price - prev_close) / prev_close) * 100
                            daytrade_ret = ((current_price - open_p) / open_p) * 100
                            
                            # 3. 溢價率與漲幅過濾
                            if gap_min <= gap_rate <= gap_max and current_pct >= min_current_pct:
                                # 4. ORB 過濾
                                orb_pass = (current_price >= open_p) if enable_orb else True
                                
                                if orb_pass:
                                    # --- 提取即時特徵向量 ---
                                    vol_ma5 = float(df_clean['Volume'].iloc[-7:-2].mean())
                                    vol_ratio = (prev_vol_lots * 1000) / (vol_ma5 + 1e-5)
                                    ma5 = float(df_clean['Close'].iloc[-6:-1].mean())
                                    ma20 = float(df_clean['Close'].iloc[-21:-1].mean())
                                    dist_ma5 = ((prev_close - ma5) / (ma5 + 1e-5)) * 100
                                    dist_ma20 = ((prev_close - ma20) / (ma20 + 1e-5)) * 100
                                    max_20d = float(df_clean['High'].iloc[-21:-1].max())
                                    is_breakout = 1 if prev_close >= max_20d else 0
                                    
                                    highs = df_clean['High'].iloc[-15:-1].values
                                    lows = df_clean['Low'].iloc[-15:-1].values
                                    closes = df_clean['Close'].iloc[-16:-2].values
                                    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - closes), np.abs(lows - closes)))
                                    atr14 = (np.mean(tr) / (prev_close + 1e-5)) * 100
                                    
                                    X_input = pd.DataFrame([{
                                        'gap_rate': gap_rate,
                                        'vol_ratio': vol_ratio,
                                        'dist_ma5': dist_ma5,
                                        'dist_ma20': dist_ma20,
                                        'is_breakout': is_breakout,
                                        'atr14': atr14
                                    }])
                                    
                                    # --- AI 預測勝率 ---
                                    if ml_model:
                                        win_prob = float(ml_model.predict_proba(X_input)[0][1]) * 100
                                    else:
                                        win_prob = 50.0
                                        
                                    if win_prob >= min_win_prob:
                                        matched_count += 1
                                        stop_loss = round(open_p * (1 - sl_pct / 100), 2)
                                        take_profit = round(open_p * 1.03, 2)
                                        code_clean = ticker.replace(".TW", "").replace(".TWO", "")
                                        
                                        results.append({
                                            "AI 預估勝率": f"{win_prob:.1f}%",
                                            "股票代號": code_clean,
                                            "股票名稱": stock_name,
                                            "昨日成交量(張)": int(prev_vol_lots),
                                            "開盤溢價率 (%)": f"{gap_rate:.2f}%",
                                            "今日漲跌幅 (%)": f"{current_pct:+.2f}%",
                                            "目前最新價": round(current_price, 2),
                                            "當沖潛在報酬 (%)": f"{daytrade_ret:+.2f}%",
                                            "建議停損價 (SL)": stop_loss,
                                            "目標停利 (TP)": take_profit,
                                            "raw_prob": win_prob
                                        })
                    except Exception:
                        continue
            except Exception:
                pass
            
            progress_bar.progress(processed_count / total_count)
            
        status_text.success(f"🎉 掃描完成！達到 AI 預測勝率 {min_win_prob}% 以上標的共 {len(results)} 檔！")
        
        if results:
            res_df = pd.DataFrame(results).sort_values(by="raw_prob", ascending=False).drop(columns=["raw_prob"])
            st.dataframe(res_df, use_container_width=True)
        else:
            st.warning("今日暫無達標之高勝率 AI 推薦標的（建議調整勝率門檻或溢價率區間重試）。")

# ==========================================
# 功能二：美股即時行情觀測
# ==========================================
elif menu == "2. 美股即時行情觀測":
    st.header("🇺🇸 美股重點標的與技術圖表")
    DEFAULT_US_STOCKS = ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMD"]
    us_symbol = st.selectbox("選擇美股標的", DEFAULT_US_STOCKS)
    df_us = yf.Ticker(us_symbol).history(period="3mo")
    
    if not df_us.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df_us.index, open=df_us['Open'], high=df_us['High'], low=df_us['Low'], close=df_us['Close'], name="K線"
        )])
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 功能三：玉山證券帳務與交易
# ==========================================
elif menu == "3. 玉山證券帳務與交易":
    st.header("🏦 玉山證券富果 API 連線測試")
    st.info("可在這裡整合玉山富果 API 進行自動下單連動。")