import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import os
import requests
import time
import gc
from datetime import datetime
from fugle_marketdata import RestClient

st.set_page_config(page_title="AI 2.0 量化當沖系統", page_icon="⚡", layout="wide")
st.title("⚡ AI 2.0 專業量化當沖儀表板")
st.markdown("### 🏆 華爾街級戰略：T-1 日大數據選股 ➔ 09:10 微觀狙擊")

# --- 側邊欄：富果 API 設定 ---
st.sidebar.header("🔑 機構級資料源設定")
fugle_token = st.sidebar.text_input("請輸入富果 (Fugle) API Token", type="password")
if fugle_token: 
    st.sidebar.success("✅ 富果 Token 已就緒，狙擊引擎待命中！")

# 初始化 Session State
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# --- 1. 載入 AI 模型 ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl") if os.path.exists("model.pkl") else None

ml_model = load_model()
if ml_model: 
    st.sidebar.success("🤖 AI 2.0 預測引擎：已連線")
else: 
    st.sidebar.error("⚠️ 找不到 model.pkl")

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

# 精選熱門當沖庫代碼 (預防全台股 API Rate Limit 時的備用高效庫)
TOP_DAYTRADE_CODES = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "3231.TW", "2603.TW",
    "3711.TW", "3443.TW", "3035.TW", "2379.TW", "3661.TW", "6669.TW", "1519.TW", "1504.TW",
    "1513.TW", "1514.TW", "2368.TW", "3017.TW", "3324.TWO", "3583.TW", "6274.TWO", "8096.TWO",
    "3260.TWO", "6187.TWO", "8054.TWO", "3037.TW", "2356.TW", "2408.TW", "2609.TW", "2618.TW"
]

# ==========================================
# 介面分頁
# ==========================================
tab1, tab2 = st.tabs(["🌙 步驟一：盤前 AI 選股 (無 API 限制)", "☀️ 步驟二：09:10 當沖狙擊 (Fugle 微觀)"])

# ------------------------------------------
# Tab 1: 盤前選股 (Yahoo 歷史資料)
# ------------------------------------------
with tab1:
    st.info("💡 操作提示：請在「前一天晚上」或「開盤前 08:30」執行此步驟。AI 將使用收盤後的正確資料進行大範圍掃描。")
    
    scan_scope = st.radio("🔍 選擇掃描目標範圍：", ["🔥 精選熱門高流動性當沖庫 (~200檔，穩定防爆)", "🚀 全台股大掃描 (1700+檔，耗時較長)"], horizontal=True)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: min_win_prob = st.slider("🎯 AI 勝率門檻 (%)", 0, 90, 40, step=5)
    with c2: min_atr_ratio = st.number_input("⚡ 最低日震幅 ATR (%)", min_value=1.0, value=2.0, step=0.5, help="過濾低波動金融/牛皮股")
    with c3: min_vol = st.number_input("📉 最低成交量(張)", value=1000, step=500)
    with c4: min_rvol = st.number_input("🔥 昨日 RVOL", min_value=0.5, value=1.0, step=0.1)
    with c5: min_rs = st.number_input("💪 最低 RS (%)", value=-5.0, step=1.0)

    if st.button("🚀 啟動盤前 AI 大掃描"):
        if not ml_model: st.stop()

        # 根據選擇設定目標標的
        if "精選" in scan_scope:
            target_stocks = {k: all_stocks.get(k, k) for k in TOP_DAYTRADE_CODES if k in all_stocks}
            if not target_stocks:
                target_stocks = {k: k for k in TOP_DAYTRADE_CODES}
        else:
            target_stocks = all_stocks

        tickers_list = list(target_stocks.keys())
        
        death_toll = {
            "1. yfinance 無資料或下載失敗": 0,
            "2. 有效交易日少於 125 天": 0,
            "3. KY 股排除": 0,
            f"4. 日震幅 ATR% 低於 {min_atr_ratio}%": 0,
            f"5. 成交量少於 {min_vol} 張": 0,
            "6. RVOL 未達標": 0,
            "7. 特徵計算或 AI 預測異常": 0,
            "8. AI 勝率未達標": 0
        }
        
        max_prob_found = 0.0
        all_scored = []
        
        st.markdown("### ⏳ AI 宏觀過濾中，請稍候...")
        progress_bar = st.progress(0)
        
        chunk_size = 40
        chunks = [tickers_list[i:i + chunk_size] for i in range(0, len(tickers_list), chunk_size)]
        processed = 0
        
        # 🚨 頂層防爆機制：防止整支 App 崩潰
        try:
            for chunk in chunks:
                try:
                    data = yf.download(chunk, period="1y", interval="1d", group_by='ticker', threads=True, progress=False)
                except Exception:
                    death_toll["1. yfinance 無資料或下載失敗"] += len(chunk)
                    continue
                    
                for ticker in chunk:
                    processed += 1
                    try:
                        df_stock = data[ticker] if len(chunk) > 1 else data
                        df = df_stock.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
                        df = df[df['Volume'] > 0]
                        
                        if len(df) < 125:
                            death_toll["2. 有效交易日少於 125 天"] += 1
                            continue
                        
                        stock_name = target_stocks.get(ticker, "未知")
                        if "KY" in stock_name or "KY" in ticker:
                            death_toll["3. KY 股排除"] += 1
                            continue
                        
                        open_p = float(df['Open'].iloc[-1])
                        prev_close = float(df['Close'].iloc[-2])
                        current_p = float(df['Close'].iloc[-1])
                        vol_today = float(df['Volume'].iloc[-1])
                        
                        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
                        atr_14_val = float(tr.rolling(14).mean().iloc[-2])
                        atr_ratio = (atr_14_val / prev_close) * 100
                        
                        if atr_ratio < min_atr_ratio:
                            death_toll[f"4. 日震幅 ATR% 低於 {min_atr_ratio}%"] += 1
                            continue
                        
                        if (vol_today / 1000.0) < min_vol:
                            death_toll[f"5. 成交量少於 {min_vol} 張"] += 1
                            continue
                        
                        gap_pct = ((open_p - prev_close) / prev_close) * 100
                        vol_ma20 = float(df['Volume'].iloc[-21:-1].mean())
                        rvol = vol_today / (vol_ma20 + 1e-5)
                        
                        if rvol < min_rvol:
                            death_toll["6. RVOL 未達標"] += 1
                            continue
                        
                        ema5 = float(df['Close'].ewm(span=5).mean().iloc[-2])
                        ema10 = float(df['Close'].ewm(span=10).mean().iloc[-2])
                        ema20 = float(df['Close'].ewm(span=20).mean().iloc[-2])
                        ema60 = float(df['Close'].ewm(span=60).mean().iloc[-2])
                        ema_bull = int(ema5 > ema10 and ema10 > ema20 and ema20 > ema60)
                        
                        delta = df['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(14).mean().iloc[-2]
                        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-2]
                        rsi = 100 - (100 / (1 + (gain / (loss + 1e-5))))
                        
                        macd_line = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                        macd_sig = macd_line.ewm(span=9).mean()
                        macd_hist = float((macd_line - macd_sig).iloc[-2])
                        
                        sma20 = float(df['Close'].rolling(20).mean().iloc[-2])
                        std20 = float(df['Close'].rolling(20).std().iloc[-2])
                        
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
                            'ATR_Ratio': atr_ratio
                        }
                        
                        X_input = pd.DataFrame([feature_dict]).fillna(0)
                        if hasattr(ml_model, "feature_names_in_"):
                            X_input = X_input.reindex(columns=ml_model.feature_names_in_, fill_value=0)
                        
                        prob = float(ml_model.predict_proba(X_input)[0][1] * 100)
                        if prob > max_prob_found:
                            max_prob_found = prob

                        clean_symbol = ticker.replace(".TW", "").replace(".TWO", "")
                        scored_entry = {
                            "symbol": clean_symbol,
                            "name": stock_name,
                            "prob": round(prob, 1),
                            "rvol": round(rvol, 2),
                            "atr": round(atr_ratio, 2),
                            "sl": round(current_p - (1.2 * atr_14_val), 2),
                            "tp": round(current_p + (2.5 * atr_14_val), 2)
                        }
                        all_scored.append(scored_entry)
                            
                    except Exception:
                        death_toll["7. 特徵計算或 AI 預測異常"] += 1
                        
                progress_bar.progress(processed / len(tickers_list))
                gc.collect() # 手動觸發記憶體回收，預防 OOM 崩潰

            st.markdown("### 📊 【除錯報告】篩選統計")
            st.json(death_toll)

            if all_scored:
                sorted_candidates = sorted(all_scored, key=lambda x: x["prob"], reverse=True)[:30]
                st.markdown("### 🎯 請選擇欲同步至「09:10 微觀狙擊」的觀察名單：")
                
                options_dict = {f"{item['symbol']} {item['name']} (AI勝率: {item['prob']}%, ATR: {item['atr']}%)": item for item in sorted_candidates}
                
                selected_keys = st.multiselect(
                    "勾選標的（可隨時新增/刪除）：",
                    options=list(options_dict.keys()),
                    default=list(options_dict.keys())
                )
                
                st.session_state.watchlist = [options_dict[k] for k in selected_keys]
                st.success(f"✔️ 已成功同步 `{len(st.session_state.watchlist)}` 檔選取標的至「步驟二：09:10 當沖狙擊池」！")
                
                if st.session_state.watchlist:
                    st.dataframe(pd.DataFrame(st.session_state.watchlist).rename(columns={
                        "symbol": "代號", "name": "名稱", "prob": "AI勝率(%)", "rvol": "昨日RVOL", "atr": "日震幅ATR(%)", "sl": "建議停損", "tp": "建議停利"
                    }), use_container_width=True)

        except Exception as global_err:
            st.error(f"⚠️ 掃描中途遇到網路連線限制或 API 阻斷：{str(global_err)}")
            st.info("💡 建議將目標範圍切換為『🔥 精選熱門高流動性當沖庫』重新嘗試，可 100% 避免連線拒絕問題。")

# ------------------------------------------
# Tab 2: 09:10 當沖狙擊 (Fugle 盤中資料)
# ------------------------------------------
with tab2:
    st.info("💡 操作提示：請在早上 09:10 分準時執行！系統將針對步驟一勾選的股票進行微觀籌碼與 VWAP 精確打擊。")
    
    if len(st.session_state.watchlist) == 0:
        st.warning("⚠️ 目前觀察名單為空，請先至「步驟一」進行掃描並選擇標的。")
    else:
        st.markdown(f"**目前鎖定目標：** `{len(st.session_state.watchlist)}` 檔標的")
        
        c2_1, c2_2 = st.columns(2)
        with c2_1: min_gap = st.number_input("🚀 今日跳空開高要求 (%)", min_value=0.0, value=1.0, step=0.5)
        with c2_2: require_vwap = st.checkbox("☑️ 現價必須大於 09:10 VWAP (極嚴格)", value=True)

        st.markdown("##### 💰 資金與風控設定（依 ATR 停損反推建議張數）")
        c2_3, c2_4 = st.columns(2)
        with c2_3: account_size = st.number_input("帳戶總資金 (TWD)", min_value=10000, value=500000, step=10000)
        with c2_4: risk_pct = st.slider("單筆交易最大可承受虧損 (% of 總資金)", 0.5, 5.0, 1.0, step=0.5)
        
        if st.button("🎯 執行 09:10 終極狙擊 (連線 Fugle API)"):
            if not fugle_token:
                st.error("🚨 請先在左側欄輸入您的 Fugle API Token！")
                st.stop()
                
            client = RestClient(api_key=fugle_token)
            final_targets = []
            
            st.markdown("### 🔍 讀取富果微觀 Tick K 線中...")
            sniper_progress = st.progress(0)
            
            for idx, stock in enumerate(st.session_state.watchlist):
                symbol = stock["symbol"]
                try:
                    quote = client.stock.intraday.quote(symbol=symbol)
                    current_p = quote.get('closePrice', 0)
                    open_p = quote.get('openPrice', 0)
                    prev_close = quote.get('previousClose', 1)
                    
                    if open_p == 0 or current_p == 0: continue
                    today_gap = ((open_p - prev_close) / prev_close) * 100
                    
                    trades = client.stock.intraday.trades(symbol=symbol)
                    ticks = trades.get('data', [])
                    
                    if len(ticks) > 0:
                        total_value = sum([t['price'] * t['volume'] for t in ticks])
                        total_volume = sum([t['volume'] for t in ticks])
                        vwap_price = (total_value / total_volume) if total_volume > 0 else current_p
                        
                        cond_gap = today_gap >= min_gap
                        cond_vwap = current_p >= vwap_price if require_vwap else True
                        cond_momentum = current_p >= open_p
                        
                        if cond_gap and cond_vwap and cond_momentum:
                            stock["今日跳空(%)"] = round(today_gap, 2)
                            stock["09:10 現價"] = current_p
                            stock["盤中 VWAP"] = round(vwap_price, 2)
                            stock["動能狀態"] = "🔥 主力點火中"

                            risk_budget = account_size * (risk_pct / 100.0)
                            risk_per_share = current_p - stock["sl"]
                            if risk_per_share > 0:
                                max_shares = int(risk_budget / risk_per_share)
                                suggested_lots = max_shares // 1000
                                actual_risk = suggested_lots * 1000 * risk_per_share
                            else:
                                suggested_lots, actual_risk = 0, 0
                            stock["建議張數"] = suggested_lots
                            stock["預估風險金額"] = round(actual_risk, 0)

                            final_targets.append(stock)
                            
                except Exception: pass
                
                time.sleep(0.3)
                sniper_progress.progress((idx + 1) / len(st.session_state.watchlist))
                
            if final_targets:
                st.success(f"🎉 狙擊完成！萬中選一，這 {len(final_targets)} 檔是今日勝率最高、符合您 09:30 停利目標的飆股！")
                df_final = pd.DataFrame(final_targets)
                cols_order = ["symbol", "name", "prob", "今日跳空(%)", "09:10 現價", "盤中 VWAP", "動能狀態", "sl", "tp", "建議張數", "預估風險金額"]
                df_final = df_final[cols_order].rename(columns={
                    "symbol": "代號", "name": "名稱", "prob": "AI 勝率", "sl": "破此價停損", "tp": "建議停利"
                })
                st.dataframe(df_final, use_container_width=True)
            else:
                st.warning("⚠️ 抱歉，今天您的觀察名單中，沒有任何一檔股票撐過『跳空』與『VWAP』的嚴格審查。最好的交易就是今天不交易！")