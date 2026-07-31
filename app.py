import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import os
import requests
import time
from fugle_marketdata import RestClient

st.set_page_config(page_title="AI 2.0 量化當沖系統", page_icon="⚡", layout="wide")
st.title("⚡ AI 2.0 專業量化當沖儀表板 (AI x VWAP 雙劍合璧)")

# --- 側邊欄：富果 API 設定 ---
st.sidebar.header("🔑 機構級資料源設定")
fugle_token = st.sidebar.text_input("NmY3MDM1NjYtNzNlNC00NWJiLWFiNjgtZTc1NWI0MDgwY2FjIGI2YjQ5N2QyLTBhMTctNDM3OC1hNGJiLWJkOTZmNGM4NTg5Nw==", type="password", help="請至 developer.fugle.tw 獲取")
if fugle_token:
    st.sidebar.success("✅ Token 已輸入，VWAP 引擎待命中！")
else:
    st.sidebar.warning("⚠️ 請輸入 Token 以啟用 VWAP 與 Tick 級別過濾。")

# --- 1. 載入 AI 模型 ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl") if os.path.exists("model.pkl") else None

ml_model = load_model()
if ml_model:
    st.sidebar.success("🤖 AI 2.0 預測引擎：已連線")
else:
    st.sidebar.error("⚠️ 找不到 model.pkl，請先執行 train_model.py")

# --- 2. 抓取全台股清單 ---
@st.cache_data(ttl=86400)
def get_all_taiwan_stocks():
    stock_dict = {}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL", headers=headers, timeout=10)
        if resp.status_code == 200:
            for item in resp.json():
                code = str(item.get('Code', '')).strip()
                if len(code) == 4 and code.isdigit(): stock_dict[f"{code}.TW"] = str(item.get('Name', '')).strip()
        resp2 = requests.get("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", headers=headers, timeout=10)
        if resp2.status_code == 200:
            for item in resp2.json():
                code = str(item.get('SecuritiesCompanyCode', '')).strip()
                if len(code) == 4 and code.isdigit(): stock_dict[f"{code}.TWO"] = str(item.get('CompanyName', '')).strip()
    except: pass
    return stock_dict

all_stocks = get_all_taiwan_stocks()

# --- 3. UI 控制面板 ---
st.subheader("⚙️ 第一階段：AI 宏觀濾網 (Macro Filters)")
r1_1, r1_2, r1_3 = st.columns(3)
with r1_1: scan_scope = st.radio("掃描範圍：", ["🔥 精選熱門當沖庫 (100檔)", "🚀 全台股大掃描 (1700+ 檔)"], horizontal=True)
with r1_2: min_win_prob = st.slider("🎯 AI 預測勝率門檻 (%)", 40, 90, 55, step=5)
with r1_3: min_rvol = st.number_input("🔥 RVOL 爆發動能 (>1為爆量)", min_value=0.5, value=1.2, step=0.1)

st.subheader("🎯 第二階段：VWAP 狙擊濾網 (Micro Filters)")
r2_1, r2_2 = st.columns(2)
with r2_1: require_vwap = st.checkbox("☑️ 嚴格要求現價站上 VWAP (主力成本線)", value=True)
with r2_2: require_ema = st.checkbox("📈 嚴格要求 EMA 多頭排列", value=False)

TOP100_CODES = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "2881.TW", "3260.TWO", "3231.TW", "2603.TW", "3711.TW", "3443.TW", "3035.TW"]

# --- 4. 執行雙劍合璧掃描 ---
if st.button("🚀 啟動 AI x VWAP 雙階段掃描"):
    if not ml_model:
        st.error("系統缺少 AI 模型 (model.pkl)，無法執行。")
        st.stop()
        
    target_stocks = all_stocks if "全台股" in scan_scope else {k: all_stocks.get(k, k) for k in TOP100_CODES if k in all_stocks}
    tickers_list = list(target_stocks.keys())
    
    # 存放第一階段通過的標的
    stage1_passed = []
    
    st.markdown("### ⏳ 第一階段：全市場大數據 AI 過濾中...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    chunk_size = 100
    chunks = [tickers_list[i:i + chunk_size] for i in range(0, len(tickers_list), chunk_size)]
    processed_count = 0
    
    for chunk_idx, chunk_tickers in enumerate(chunks):
        status_text.text(f"計算日K特徵矩陣 (第 {chunk_idx + 1} / {len(chunks)} 批次)...")
        try:
            data = yf.download(chunk_tickers, period="60mo", interval="1d", group_by='ticker', threads=True, progress=False)
            for ticker in chunk_tickers:
                processed_count += 1
                try:
                    df_stock = data[ticker] if len(chunk_tickers) > 1 else data
                    df = df_stock.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
                    if len(df) < 50: continue
                    
                    stock_name = target_stocks.get(ticker, "未知")
                    if "KY" in stock_name or "KY" in ticker: continue
                    
                    vol_today_lots = float(df['Volume'].iloc[-1]) / 1000.0
                    if vol_today_lots < 1000: continue # 基礎量能過濾
                    
                    open_p = float(df['Open'].iloc[-1])
                    prev_close = float(df['Close'].iloc[-2])
                    current_p = float(df['Close'].iloc[-1])
                    vol_today = float(df['Volume'].iloc[-1])
                    
                    # AI 特徵計算
                    gap_pct = ((open_p - prev_close) / prev_close) * 100
                    vol_ma20 = float(df['Volume'].iloc[-21:-1].mean())
                    rvol = vol_today / (vol_ma20 + 1e-5)
                    
                    if rvol < min_rvol: continue # RVOL 過濾
                    
                    ema5 = float(df['Close'].ewm(span=5).mean().iloc[-2])
                    ema10 = float(df['Close'].ewm(span=10).mean().iloc[-2])
                    ema20 = float(df['Close'].ewm(span=20).mean().iloc[-2])
                    ema60 = float(df['Close'].ewm(span=60).mean().iloc[-2])
                    ema_bull = int(ema5 > ema10 and ema10 > ema20 and ema20 > ema60)
                    
                    if require_ema and not ema_bull: continue
                    
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-2]
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-2]
                    rsi = 100 - (100 / (1 + (gain / (loss + 1e-5))))
                    
                    macd_line = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                    macd_sig = macd_line.ewm(span=9).mean()
                    macd_hist = float((macd_line - macd_sig).iloc[-2])
                    
                    sma20 = float(df['Close'].rolling(20).mean().iloc[-2])
                    std20 = float(df['Close'].rolling(20).std().iloc[-2])
                    
                    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
                    atr_14_val = float(tr.rolling(14).mean().iloc[-2])
                    
                    feature_dict = {
                        'Gap_0_2': int(0 <= gap_pct < 2), 'Gap_2_4': int(2 <= gap_pct < 4),
                        'Gap_4_6': int(4 <= gap_pct < 6), 'Gap_6_9': int(6 <= gap_pct < 9), 'Gap_Over_9': int(gap_pct >= 9),
                        'RVOL': rvol, 'EMA_Bullish': ema_bull, 'RSI_14': rsi, 'RSI_GoldenZone': int(55 < rsi < 75),
                        'MACD_Hist_Pos': int(macd_hist > 0), 'Close_Above_BB': int(prev_close > sma20 + 2 * std20),
                        'High_20D': int(prev_close > float(df['High'].rolling(20).max().iloc[-3])),
                        'High_55D': int(prev_close > float(df['High'].rolling(55).max().iloc[-3])),
                        'High_120D': int(prev_close > float(df['High'].rolling(120).max().iloc[-3])),
                        'is_InsideBar': int((df['High'].iloc[-2] < df['High'].iloc[-3]) and (df['Low'].iloc[-2] > df['Low'].iloc[-3])),
                        'is_Marubozu': int((abs(prev_close - float(df['Open'].iloc[-2])) / ((df['High'].iloc[-2] - df['Low'].iloc[-2]) + 1e-5)) > 0.8),
                        'ATR_Ratio': (atr_14_val / prev_close) * 100
                    }
                    
                    X_input = pd.DataFrame([feature_dict])
                    prob = ml_model.predict_proba(X_input)[0][1] * 100
                    
                    if prob >= min_win_prob:
                        stage1_passed.append({
                            "symbol": ticker.replace(".TW", "").replace(".TWO", ""),
                            "name": stock_name,
                            "prob": prob,
                            "rvol": rvol,
                            "current_p": current_p,
                            "sl": round(open_p - (1.2 * atr_14_val), 2),
                            "tp": round(open_p + (2.5 * atr_14_val), 2)
                        })
                except Exception: pass
        except Exception: pass
        progress_bar.progress(processed_count / len(tickers_list))
    
    st.success(f"✔️ 第一階段完成！過濾出 {len(stage1_passed)} 檔具備爆發動能與高勝率的標的。")
    
    # --- 第 2 階段：Fugle VWAP 狙擊 ---
    if len(stage1_passed) > 0:
        if not fugle_token:
            st.warning("⚠️ 由於您未輸入 Fugle Token，將跳過第二階段 VWAP 審核，直接顯示第一階段名單。")
            final_results = stage1_passed
        else:
            st.markdown("### 🎯 第二階段：富果 API 盤中主力成本 (VWAP) 精算中...")
            client = RestClient(api_key=fugle_token)
            final_results = []
            
            vwap_progress = st.progress(0)
            for idx, stock in enumerate(stage1_passed):
                try:
                    trades = client.stock.intraday.trades(symbol=stock["symbol"])
                    ticks = trades.get('data', [])
                    
                    if len(ticks) > 0:
                        total_value = sum([t['price'] * t['volume'] for t in ticks])
                        total_volume = sum([t['volume'] for t in ticks])
                        vwap_price = (total_value / total_volume) if total_volume > 0 else stock["current_p"]
                        
                        is_above_vwap = stock["current_p"] >= vwap_price
                        
                        # 嚴格濾網判斷
                        if require_vwap and not is_above_vwap:
                            pass # 淘汰跌破 VWAP 的弱勢股
                        else:
                            stock["VWAP 成本"] = round(vwap_price, 2)
                            stock["現價 VS 成本"] = "🚀 多頭強勢" if is_above_vwap else "📉 空頭弱勢"
                            final_results.append(stock)
                except Exception:
                    pass
                time.sleep(0.3) # 保護機制
                vwap_progress.progress((idx + 1) / len(stage1_passed))
            
            st.success(f"🎉 狙擊完成！最終共有 {len(final_results)} 檔完美符合 AI 與 VWAP 雙重認證！")
        
        # 整理最終表格並呈現
        if final_results:
            display_df = pd.DataFrame(final_results)
            # 重新命名與排序欄位
            display_df = display_df.rename(columns={
                "symbol": "代號", "name": "股票名稱", "prob": "AI 勝率 (%)", 
                "rvol": "RVOL (倍)", "current_p": "最新價", "sl": "停損(1.2ATR)", "tp": "停利(2.5ATR)"
            })
            
            cols_order = ["代號", "股票名稱", "AI 勝率 (%)", "RVOL (倍)", "最新價"]
            if "VWAP 成本" in display_df.columns:
                cols_order.extend(["VWAP 成本", "現價 VS 成本"])
            cols_order.extend(["停損(1.2ATR)", "停利(2.5ATR)"])
            
            display_df = display_df[cols_order].sort_values(by="AI 勝率 (%)", ascending=False)
            
            # 美化數值顯示
            display_df["AI 勝率 (%)"] = display_df["AI 勝率 (%)"].apply(lambda x: f"{x:.1f}%")
            display_df["RVOL (倍)"] = display_df["RVOL (倍)"].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("今日無任何標的能同時撐過 AI 大數據與 VWAP 嚴格考驗。建議保持空手或微調參數。")