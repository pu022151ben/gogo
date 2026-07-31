import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import os
import requests

st.set_page_config(page_title="AI 2.0 專業量化交易儀表板", page_icon="⚡", layout="wide")
st.title("⚡ AI 2.0 專業量化交易與選股儀表板 (ATR 風控版)")

# --- 1. 載入 AI 模型 ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl") if os.path.exists("model.pkl") else None

ml_model = load_model()
if ml_model:
    st.sidebar.success("🤖 AI 2.0 預測引擎：已連線")
else:
    st.sidebar.warning("⚠️ 尚未偵測到 model.pkl，請先執行 train_model.py")

# --- 2. 抓取全台股清單 ---
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

all_stocks = get_all_taiwan_stocks()
st.success(f"🌐 已成功載入全台股資料庫，共收錄 **{len(all_stocks)}** 檔普通股。")

# --- 3. UI 控制面板 ---
st.subheader("⚙️ 專業量化濾網設定 (Quant Filters)")
r1_1, r1_2, r1_3 = st.columns(3)
with r1_1:
    scan_scope = st.radio("掃描範圍：", ["🚀 全台股大掃描 (1700+ 檔)", "🔥 精選熱門當沖庫 (100檔)"], horizontal=True)
with r1_2:
    min_win_prob = st.slider("🎯 AI 預測勝率門檻 (%)", 40, 90, 55, step=5, help="盤勢較悶時可調降至 50~55%")
with r1_3:
    min_rvol = st.number_input("🔥 RVOL 爆發動能 (>1為爆量)", min_value=0.5, value=1.2, step=0.1, help="當日預估量除以20日均量。設定1.2代表量增20%。")

r2_1, r2_2, r2_3 = st.columns(3)
with r2_1:
    require_ema = st.checkbox("📈 嚴格要求 EMA 多頭排列", value=False, help="勾選後只會選出短中長期趨勢向上的股票")
with r2_2:
    min_vol = st.number_input("📊 最低成交量門檻（張）", min_value=100, value=1000, step=100)
with r2_3:
    exclude_ky = st.checkbox("🛡️ 自動排除 -KY 股", value=True)

TOP100_CODES = ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "2881.TW", "2882.TW", "3260.TWO", "3231.TW", "2603.TW", "3711.TW"]

# --- 4. 執行大數據掃描 ---
if st.button("🚀 啟動全市場 AI 掃描"):
    target_stocks = all_stocks if "全台股" in scan_scope else {k: all_stocks.get(k, k) for k in TOP100_CODES if k in all_stocks}
    tickers_list = list(target_stocks.keys())
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    chunk_size = 100
    chunks = [tickers_list[i:i + chunk_size] for i in range(0, len(tickers_list), chunk_size)]
    processed_count = 0
    
    for chunk_idx, chunk_tickers in enumerate(chunks):
        status_text.text(f"AI 引擎正在計算全市場特徵矩陣 (第 {chunk_idx + 1} / {len(chunks)} 批次)...")
        try:
            # 批次下載以增加速度
            data = yf.download(chunk_tickers, period="60d", interval="1d", group_by='ticker', threads=True, progress=False)
            
            for ticker in chunk_tickers:
                processed_count += 1
                try:
                    df_stock = data[ticker] if len(chunk_tickers) > 1 else data
                    df = df_stock.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
                    if len(df) < 50: continue
                    
                    stock_name = target_stocks.get(ticker, "未知")
                    if exclude_ky and ("KY" in stock_name or "KY" in ticker): continue
                    
                    vol_today_lots = float(df['Volume'].iloc[-1]) / 1000.0
                    if vol_today_lots < min_vol: continue
                    
                    open_p = float(df['Open'].iloc[-1])
                    prev_close = float(df['Close'].iloc[-2])
                    current_p = float(df['Close'].iloc[-1])
                    vol_today = float(df['Volume'].iloc[-1])
                    
                    # --- AI 特徵計算 (對齊 train_model.py) ---
                    gap_pct = ((open_p - prev_close) / prev_close) * 100
                    gap_bins = {
                        'Gap_0_2': int(0 <= gap_pct < 2),
                        'Gap_2_4': int(2 <= gap_pct < 4),
                        'Gap_4_6': int(4 <= gap_pct < 6),
                        'Gap_6_9': int(6 <= gap_pct < 9),
                        'Gap_Over_9': int(gap_pct >= 9)
                    }
                    
                    vol_ma20 = float(df['Volume'].iloc[-21:-1].mean())
                    rvol = vol_today / (vol_ma20 + 1e-5)
                    if rvol < min_rvol: continue # 受到控制面板 RVOL 過濾
                    
                    ema5 = float(df['Close'].ewm(span=5, adjust=False).mean().iloc[-2])
                    ema10 = float(df['Close'].ewm(span=10, adjust=False).mean().iloc[-2])
                    ema20 = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-2])
                    ema60 = float(df['Close'].ewm(span=60, adjust=False).mean().iloc[-2])
                    ema_bull = int(ema5 > ema10 and ema10 > ema20 and ema20 > ema60)
                    
                    if require_ema and not ema_bull: continue # 受到控制面板 EMA 過濾
                    
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-2]
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-2]
                    rsi = 100 - (100 / (1 + (gain / (loss + 1e-5))))
                    
                    macd_line = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                    macd_sig = macd_line.ewm(span=9).mean()
                    macd_hist = float((macd_line - macd_sig).iloc[-2])
                    
                    sma20 = float(df['Close'].rolling(20).mean().iloc[-2])
                    std20 = float(df['Close'].rolling(20).std().iloc[-2])
                    
                    tr1 = df['High'] - df['Low']
                    tr2 = (df['High'] - df['Close'].shift(1)).abs()
                    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr_14_val = float(tr.rolling(14).mean().iloc[-2])
                    
                    feature_dict = {
                        **gap_bins,
                        'RVOL': rvol,
                        'EMA_Bullish': ema_bull,
                        'RSI_14': rsi,
                        'RSI_GoldenZone': int(55 < rsi < 75),
                        'MACD_Hist_Pos': int(macd_hist > 0),
                        'Close_Above_BB': int(prev_close > sma20 + 2 * std20),
                        'High_20D': int(prev_close > float(df['High'].rolling(20).max().iloc[-3])),
                        'High_55D': int(prev_close > float(df['High'].rolling(55).max().iloc[-3])),
                        'High_120D': int(prev_close > float(df['High'].rolling(120).max().iloc[-3])),
                        'is_InsideBar': int((df['High'].iloc[-2] < df['High'].iloc[-3]) and (df['Low'].iloc[-2] > df['Low'].iloc[-3])),
                        'is_Marubozu': int((abs(prev_close - float(df['Open'].iloc[-2])) / ((df['High'].iloc[-2] - df['Low'].iloc[-2]) + 1e-5)) > 0.8),
                        'ATR_Ratio': (atr_14_val / prev_close) * 100
                    }
                    
                    if ml_model:
                        X_input = pd.DataFrame([feature_dict])
                        prob = ml_model.predict_proba(X_input)[0][1] * 100
                        
                        if prob >= min_win_prob:
                            sl_price = round(open_p - (1.2 * atr_14_val), 2)
                            tp_price = round(open_p + (2.5 * atr_14_val), 2)
                            
                            results.append({
                                "AI 勝率": f"{prob:.1f}%",
                                "代號": ticker.replace(".TW", "").replace(".TWO", ""),
                                "股票名稱": stock_name,
                                "成交量(張)": int(vol_today_lots),
                                "RVOL 動能": f"{rvol:.2f} 倍",
                                "EMA 狀態": "多頭" if ema_bull else "未達標",
                                "最新價": round(current_p, 2),
                                "停損 (1.2ATR)": sl_price,
                                "停利 (2.5ATR)": tp_price,
                                "raw_prob": prob
                            })
                except Exception:
                    pass
        except Exception:
            pass
        progress_bar.progress(processed_count / len(tickers_list))
        
    status_text.success(f"🎉 掃描完成！達到高標準的潛力標的共 {len(results)} 檔！")
    
    if results:
        res_df = pd.DataFrame(results).sort_values(by="raw_prob", ascending=False).drop(columns=["raw_prob"])
        st.dataframe(res_df, use_container_width=True)
    else:
        st.warning("今日市場動能較弱，無股票能同時通過 AI 預測與 RVOL 嚴格標準。建議可微調面板中的參數。")