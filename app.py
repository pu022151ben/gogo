import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import os
import requests

st.set_page_config(page_title="AI 2.0 專業量化交易儀表板", page_icon="⚡", layout="wide")
st.title("⚡ AI 2.0 專業量化交易與選股儀表板 (ATR 風控版)")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl") if os.path.exists("model.pkl") else None

ml_model = load_model()
if ml_model:
    st.sidebar.success("🤖 AI 2.0 預測引擎：已連線")
else:
    st.sidebar.warning("⚠️ 請先執行 train_model.py")

st.subheader("⚙️ 專業量化濾網設定 (Quant Filters)")
r1_1, r1_2, r1_3 = st.columns(3)
with r1_1:
    min_win_prob = st.slider("🎯 AI 預測勝率門檻 (%)", 50, 90, 60, step=5)
with r1_2:
    min_rvol = st.number_input("🔥 RVOL 爆發動能 (>2)", min_value=0.5, value=1.5, step=0.1)
with r1_3:
    require_ema = st.checkbox("📈 嚴格要求 EMA 多頭排列", value=False)

if st.button("🚀 啟動全市場 AI 掃描"):
    # 測試用清單 (實務可替換為上市櫃全檔)
    watchlist = ["2330.TW", "2317.TW", "2454.TW", "2382.TW", "2603.TW", "3231.TW", "3711.TW", "3035.TW", "3443.TW", "3260.TWO"]
    results = []
    
    progress = st.progress(0)
    for idx, ticker in enumerate(watchlist):
        try:
            df = yf.download(ticker, period="60d", interval="1d", progress=False).dropna()
            if len(df) < 50: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            open_p = float(df['Open'].iloc[-1])
            prev_close = float(df['Close'].iloc[-2])
            current_p = float(df['Close'].iloc[-1])
            vol_today = float(df['Volume'].iloc[-1])
            
            # --- 核心特徵計算 (與訓練模組 100% 映射) ---
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
            if rvol < min_rvol: continue # RVOL 濾網
            
            ema5 = float(df['Close'].ewm(span=5, adjust=False).mean().iloc[-2])
            ema10 = float(df['Close'].ewm(span=10, adjust=False).mean().iloc[-2])
            ema20 = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-2])
            ema60 = float(df['Close'].ewm(span=60, adjust=False).mean().iloc[-2])
            ema_bull = int(ema5 > ema10 and ema10 > ema20 and ema20 > ema60)
            
            if require_ema and not ema_bull: continue # EMA 濾網
            
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
            
            # 建立輸入矩陣
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
                    # 動態 ATR 風控價格
                    sl_price = round(open_p - (1.2 * atr_14_val), 2)
                    tp_price = round(open_p + (2.5 * atr_14_val), 2)
                    
                    results.append({
                        "代號": ticker.replace(".TW", "").replace(".TWO", ""),
                        "AI 勝率": f"{prob:.1f}%",
                        "RVOL": f"{rvol:.2f}",
                        "EMA 狀態": "多頭" if ema_bull else "盤整/空頭",
                        "現價": round(current_p, 2),
                        "停損 (SL 1.2ATR)": sl_price,
                        "停利 (TP 2.5ATR)": tp_price
                    })
        except Exception:
            pass
        progress.progress((idx + 1) / len(watchlist))
        
    if results:
        st.success(f"掃描完成！共 {len(results)} 檔達標。")
        st.dataframe(pd.DataFrame(results))
    else:
        st.warning("今日無達標標的。")