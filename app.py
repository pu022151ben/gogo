import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import os
import requests
import time
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

# 初始化 Session State 來記憶「盤前選股名單」
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# --- 1. 載入 AI 模型 ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl") if os.path.exists("model.pkl") else None

ml_model = load_model()
if ml_model: st.sidebar.success("🤖 AI 2.0 預測引擎：已連線")
else: st.sidebar.error("⚠️ 找不到 model.pkl")

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

# ==========================================
# 介面分頁：完美切割「盤前」與「盤中」
# ==========================================
tab1, tab2 = st.tabs(["🌙 步驟一：盤前 AI 選股 (無 API 限制)", "☀️ 步驟二：09:10 當沖狙擊 (Fugle 微觀)"])

# ------------------------------------------
# Tab 1: 盤前選股 (Yahoo 歷史資料)
# ------------------------------------------
with tab1:
    st.info("💡 操作提示：請在「前一天晚上」或「開盤前 08:30」執行此步驟。AI 將使用收盤後的正確資料進行大範圍掃描。")
    c1, c2, c3, c4 = st.columns(4)
    with c1: min_win_prob = st.slider("🎯 AI 預測勝率門檻 (%)", 40, 90, 50, step=5)
    with c2: min_rvol = st.number_input("🔥 昨日 RVOL 爆發動能", min_value=0.5, value=1.0, step=0.1)
    with c3: min_rs = st.number_input("💪 最低相對強度 RS (vs 大盤, %)", value=0.0, step=1.0, help="個股近20日報酬 減去 大盤(加權指數)近20日報酬。>0 代表跑贏大盤，屬於強勢股，符合『強者恆強』選股邏輯。")
    with c4: use_market_filter = st.checkbox("🛡️ 大盤多頭過濾", value=True, help="加權指數需站上5日均線，才視為大盤偏多；大盤偏空時當沖做多勝率通常較低。")

    if st.button("🚀 啟動盤前 AI 大掃描 (1700+ 檔)"):
        if not ml_model: st.stop()

        # --- 大盤(加權指數)基準資料：用於相對強度 RS 與多空過濾 ---
        market_return_20d = None
        market_bullish = True
        try:
            twii = yf.download("^TWII", period="6mo", interval="1d", progress=False)
            twii_close = twii['Close'].dropna()
            if len(twii_close) >= 21:
                market_return_20d = float((twii_close.iloc[-2] / twii_close.iloc[-21] - 1) * 100)
                ma5 = float(twii_close.iloc[-6:-1].mean())
                market_bullish = float(twii_close.iloc[-2]) > ma5
        except Exception:
            pass

        if use_market_filter and not market_bullish:
            st.warning("⚠️ 大盤目前偏空（加權指數跌破5日均線），已啟用『大盤多頭過濾』。此模式下當沖做多勝率通常較低，建議謹慎操作或暫停進場。")
            st.stop()
        elif market_return_20d is not None:
            st.caption(f"📊 大盤近20日報酬：{round(market_return_20d, 2)}%（{'偏多 🟢' if market_bullish else '偏空 🔴'}），將作為個股相對強度 RS 的基準。")

        tickers_list = list(all_stocks.keys())
        temp_watchlist = []
        
        st.markdown("### ⏳ AI 宏觀過濾中，請稍候...")
        progress_bar = st.progress(0)
        
        chunk_size = 100
        chunks = [tickers_list[i:i + chunk_size] for i in range(0, len(tickers_list), chunk_size)]
        processed = 0
        
        for chunk in chunks:
            # 抓取 1 年歷史資料
            data = yf.download(chunk, period="1y", interval="1d", group_by='ticker', threads=True, progress=False)
            for ticker in chunk:
                processed += 1
                try:
                    df_stock = data[ticker] if len(chunk) > 1 else data
                    df = df_stock.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
                    if len(df) < 125: continue
                    
                    stock_name = all_stocks.get(ticker, "未知")
                    if "KY" in stock_name or "KY" in ticker: continue
                    
                    # 使用最後一筆完整的 K 線 (昨日)
                    open_p = float(df['Open'].iloc[-1])
                    prev_close = float(df['Close'].iloc[-2])
                    current_p = float(df['Close'].iloc[-1])
                    vol_today = float(df['Volume'].iloc[-1])
                    
                    if (vol_today / 1000.0) < 500: continue # 剔除死魚股
                    
                    gap_pct = ((open_p - prev_close) / prev_close) * 100
                    vol_ma20 = float(df['Volume'].iloc[-21:-1].mean())
                    rvol = vol_today / (vol_ma20 + 1e-5)
                    
                    if rvol < min_rvol: continue
                    
                    # --- 相對強度 RS (個股近20日報酬 - 大盤近20日報酬) ---
                    # 注意：RS 僅作為「額外篩選/排序指標」，不餵入 ML 模型特徵，
                    # 避免與 model.pkl 訓練時的特徵欄位不一致而導致預測錯誤。
                    if len(df) >= 21 and market_return_20d is not None:
                        stock_return_20d = float((df['Close'].iloc[-2] / df['Close'].iloc[-21] - 1) * 100)
                        rs_score = stock_return_20d - market_return_20d
                    else:
                        rs_score = 0.0
                    if rs_score < min_rs: continue
                    
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
                        clean_symbol = ticker.replace(".TW", "").replace(".TWO", "")
                        temp_watchlist.append({
                            "symbol": clean_symbol,
                            "name": stock_name,
                            "prob": round(prob, 1),
                            "rvol": round(rvol, 2),
                            "rs": round(rs_score, 2),
                            "sl": round(current_p - (1.2 * atr_14_val), 2),
                            "tp": round(current_p + (2.5 * atr_14_val), 2)
                        })
                except Exception: pass
            progress_bar.progress(processed / len(tickers_list))
            
        # 依相對強度 RS 排序（強者恆強：跑贏大盤越多的排越前面）
        temp_watchlist = sorted(temp_watchlist, key=lambda x: x["rs"], reverse=True)

        # 將結果存入記憶體供 Tab 2 使用
        st.session_state.watchlist = temp_watchlist
        st.success(f"✔️ 盤前選股完成！成功抓出 {len(temp_watchlist)} 檔高潛力觀察股。已自動同步至「步驟二」。")
        if temp_watchlist:
            st.dataframe(pd.DataFrame(temp_watchlist).rename(columns={"symbol": "代號", "name": "名稱", "prob": "AI勝率(%)", "rvol": "昨日RVOL", "rs": "相對強度RS(%)", "sl": "建議停損", "tp": "建議停利"}), use_container_width=True)


# ------------------------------------------
# Tab 2: 09:10 當沖狙擊 (Fugle 盤中資料)
# ------------------------------------------
with tab2:
    st.info("💡 操作提示：請在早上 09:10 分準時執行！系統將針對「步驟一」篩選出的股票進行微觀籌碼與 VWAP 精確打擊。")
    
    if len(st.session_state.watchlist) == 0:
        st.warning("⚠️ 目前觀察名單為空，請先至「步驟一」執行盤前大掃描。")
    else:
        st.markdown(f"**目前鎖定目標：** `{len(st.session_state.watchlist)}` 檔潛力股")
        
        c2_1, c2_2 = st.columns(2)
        with c2_1: min_gap = st.number_input("🚀 今日跳空開高要求 (%)", min_value=0.0, value=1.0, step=0.5, help="主力盤前點火的證據")
        with c2_2: require_vwap = st.checkbox("☑️ 現價必須大於 09:10 VWAP (極嚴格)", value=True, help="這確保主力開盤沒有倒貨")

        st.markdown("##### 💰 資金與風控設定（依 ATR 停損反推建議張數）")
        c2_3, c2_4 = st.columns(2)
        with c2_3: account_size = st.number_input("帳戶總資金 (TWD)", min_value=10000, value=500000, step=10000)
        with c2_4: risk_pct = st.slider("單筆交易最大可承受虧損 (% of 總資金)", 0.5, 5.0, 1.0, step=0.5, help="職業當沖常見紀律：單筆虧損不超過總資金的1~2%，以此反推該買幾張。")
        
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
                    # 1. 抓取今日即時報價
                    quote = client.stock.intraday.quote(symbol=symbol)
                    current_p = quote.get('closePrice', 0)
                    open_p = quote.get('openPrice', 0)
                    prev_close = quote.get('previousClose', 1)
                    
                    if open_p == 0 or current_p == 0: continue
                    
                    # 計算今日跳空
                    today_gap = ((open_p - prev_close) / prev_close) * 100
                    
                    # 2. 抓取 Tick 算今日開盤至今的 VWAP
                    trades = client.stock.intraday.trades(symbol=symbol)
                    ticks = trades.get('data', [])
                    
                    if len(ticks) > 0:
                        total_value = sum([t['price'] * t['volume'] for t in ticks])
                        total_volume = sum([t['volume'] for t in ticks])
                        vwap_price = (total_value / total_volume) if total_volume > 0 else current_p
                        
                        # 3. 戰略條件判斷
                        cond_gap = today_gap >= min_gap
                        cond_vwap = current_p >= vwap_price if require_vwap else True
                        cond_momentum = current_p >= open_p # 開高走高，不買開高走低
                        
                        if cond_gap and cond_vwap and cond_momentum:
                            stock["今日跳空(%)"] = round(today_gap, 2)
                            stock["09:10 現價"] = current_p
                            stock["盤中 VWAP"] = round(vwap_price, 2)
                            stock["動能狀態"] = "🔥 主力點火中"

                            # --- ATR 部位風控：單筆最大虧損不超過帳戶資金的 risk_pct% ---
                            risk_budget = account_size * (risk_pct / 100.0)
                            risk_per_share = current_p - stock["sl"]
                            if risk_per_share > 0:
                                max_shares = int(risk_budget / risk_per_share)
                                suggested_lots = max_shares // 1000  # 台股一張=1000股，僅供整張參考
                                actual_risk = suggested_lots * 1000 * risk_per_share
                            else:
                                suggested_lots, actual_risk = 0, 0
                            stock["建議張數"] = suggested_lots
                            stock["預估風險金額"] = round(actual_risk, 0)

                            final_targets.append(stock)
                            
                except Exception as e: pass
                
                # 保護 API 呼叫頻率
                time.sleep(0.3)
                sniper_progress.progress((idx + 1) / len(st.session_state.watchlist))
                
            if final_targets:
                st.success(f"🎉 狙擊完成！萬中選一，這 {len(final_targets)} 檔是今日勝率最高、符合您 09:30 停利目標的飆股！")
                
                df_final = pd.DataFrame(final_targets)
                cols_order = ["symbol", "name", "prob", "rs", "今日跳空(%)", "09:10 現價", "盤中 VWAP", "動能狀態", "sl", "tp", "建議張數", "預估風險金額"]
                df_final = df_final[cols_order].rename(columns={
                    "symbol": "代號", "name": "名稱", "prob": "AI 勝率", "rs": "相對強度RS(%)", "sl": "破此價停損", "tp": "建議停利"
                })
                st.dataframe(df_final, use_container_width=True)

                total_risk = df_final["預估風險金額"].sum()
                st.caption(f"📌 若上列標的『全部』進場，依目前風控設定，估計最大總風險金額約為 {total_risk:,.0f} 元（佔總資金 {round(total_risk/account_size*100, 2)}%）。實務上建議依信心程度篩選少數幾檔，而非全數進場。")
                st.info("💡 **實戰下單建議**：請在 09:10~09:15 之間進場上列表格中的標的。若跌破『盤中 VWAP』請果斷停損，若快速衝高達標請嚴格停利 (09:30 沖銷)！『建議張數』係依 ATR 停損反推的風控參考值，非買賣建議，請自行評估流動性與委買賣掛單量。")
            else:
                st.warning("⚠️ 抱歉，今天您的觀察名單中，沒有任何一檔股票撐過『跳空』與『VWAP』的嚴格審查。最好的交易就是今天不交易！")