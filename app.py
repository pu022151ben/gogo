import streamlit as st
import pandas as pd
import yfinance as yf
import feedparser
import google.generativeai as genai
from fugle_marketdata import RestClient
import math
import datetime

# ==========================================
# 0. 網頁基本設定
# ==========================================
st.set_page_config(page_title="AI 量化當沖系統", layout="wide")

# 初始化 Session State (用來在分頁間傳遞資料)
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()

# ==========================================
# 1. 左側邊欄：API 金鑰設定
# ==========================================
st.sidebar.title("🔑 系統核心金鑰設定")
fugle_key = st.sidebar.text_input("1. Fugle API Token (盤中即時狙擊用)", type="password")
if fugle_key:
    st.sidebar.success("✅ 富果連線就緒！")

gemini_key = st.sidebar.text_input("2. Gemini API Key (盤前新聞判讀用)", type="password")
if gemini_key:
    try:
        genai.configure(api_key=gemini_key)
        st.sidebar.success("✅ AI 新聞解讀引擎就緒！")
    except Exception as e:
        st.sidebar.error("金鑰格式錯誤")

# ==========================================
# 工具函式區
# ==========================================
@st.cache_data(ttl=3600)
def get_macro_data():
    """獲取宏觀雷達報價 (SOX, TSM, TSLA)"""
    tickers = ["^SOX", "TSM", "TSLA"]
    results = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="5d")
            if len(hist) >= 2:
                pct_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
                results[t] = f"{pct_change:.2f}%"
            else:
                results[t] = "N/A"
        except:
            results[t] = "解析失敗"
    return results

def fetch_yahoo_news(query="台股 半導體 AI"):
    """抓取 Yahoo 財經 RSS 新聞摘要"""
    url = f"https://tw.news.yahoo.com/rss/stock"
    feed = feedparser.parse(url)
    news_text = ""
    for entry in feed.entries[:5]: # 只抓前5條避免字數過長
        news_text += f"- {entry.title}\n"
    return news_text

def analyze_with_gemini(news_text, stock_list):
    """呼叫 Gemini 進行族群熱點與個股判讀"""
    if not gemini_key:
        return "⚠️ 未輸入 Gemini API Key"
    
    prompt = f"""
    你是一位頂尖的華爾街量化交易分析師。請解讀以下今日盤前新聞：
    {news_text}
    
    任務一：判斷今日台股資金最集中的「熱門族群」（如：半導體設備、AI伺服器、矽光子等）。
    任務二：檢查以下候選股票名單 {stock_list}。
    如果該股票屬於你判定今日的熱門族群，請在評語中明確標示「【族群共伴發動】」，並簡述原因。
    若無特別關聯，請回覆「無特殊消息」。
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 解析失敗: {e}"

# ==========================================
# 主畫面 UI
# ==========================================
st.header("🏆 華爾街級戰略：RS相對強度 ➡️ AI盤前新聞解讀 ➡️ 跨市場連動")

tab1, tab2 = st.tabs(["🌙 步驟一：盤前戰略掃描 (結合新聞 AI)", "☀️ 步驟二：09:10 即時狙擊 (富果微觀打擊)"])

# ------------------------------------------
# 分頁一：盤前戰略掃描
# ------------------------------------------
with tab1:
    st.subheader("🌐 全球宏觀雷達與 AI 新聞解讀")
    
    # 顯示美股連動
    macro_data = get_macro_data()
    col1, col2, col3 = st.columns(3)
    col1.metric("US 費城半導體 (SOX)", macro_data.get("^SOX", "nan%"))
    col2.metric("TW 台積電 ADR (TSM)", macro_data.get("TSM", "nan%"))
    col3.metric("🚗 特斯拉 (TSLA)", macro_data.get("TSLA", "nan%"))

    # 參數設定區 (RVOL 預設拉高至 1.5)
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    win_rate_threshold = c1.slider("🎯 AI 基礎勝率門檻 (%)", 10, 100, 40)
    min_atr = c2.number_input("⚡ 最低日震幅 (ATR %)", value=1.50, step=0.1)
    min_vol = c3.number_input("🌊 最低成交量 (張)", value=1000, step=100)
    min_rvol = c4.number_input("🔥 昨日爆量倍數 (RVOL)", value=1.50, step=0.1) # 已嚴格化

    if st.button("🚀 啟動全市場智能掃描 (結合 AI 新聞加權)"):
        with st.spinner("深度運算與 AI 新聞勝率加總中..."):
            # 1. 抓取新聞
            today_news = fetch_yahoo_news()
            
            # 2. 建立目標觀察池 (此處為模擬您的資料庫篩選結果)
            mock_data = [
                {"symbol": "3481", "name": "群創", "族群": "面板", "強度": "跟隨大盤", "現價": 47.8, "atr": 6.87, "昨日總量": 50000},
                {"symbol": "6770", "name": "力積電", "族群": "晶圓代工", "強度": "💪 抗跌領漲", "現價": 66.1, "atr": 6.23, "昨日總量": 32000},
                {"symbol": "2344", "name": "華邦電", "族群": "記憶體", "強度": "💪 抗跌領漲", "現價": 169.0, "atr": 5.73, "昨日總量": 45000},
                {"symbol": "3231", "name": "緯創", "族群": "AI伺服器", "強度": "跟隨大盤", "現價": 193.0, "atr": 5.59, "昨日總量": 80000},
            ]
            df = pd.DataFrame(mock_data)
            df['動態停損'] = df['現價'] - (df['現價'] * df['atr'] / 100 * 0.5) # 模擬 ATR 停損點
            df['最終勝率'] = 50 # 基礎勝率設定
            
            # 3. 呼叫 AI 進行判讀
            stock_names = df['name'].tolist()
            ai_report = analyze_with_gemini(today_news, stock_names)
            
            # 4. 根據 AI 報告加權分數
            df['AI解讀'] = "無特殊消息"
            for index, row in df.iterrows():
                # 若 AI 點名該標的或所屬族群，勝率加碼 10%
                if row['name'] in ai_report or row['族群'] in ai_report:
                    df.at[index, 'AI解讀'] = "🔥 【族群共伴發動】熱點聚焦"
                    df.at[index, '最終勝率'] = min(row['最終勝率'] + 10, 95)
            
            st.session_state.scan_results = df
            st.success("✅ 掃描完成！已疊加新聞熱度。")

    if not st.session_state.scan_results.empty:
        st.subheader("🎯 最終決選池：請勾選明日狙擊目標")
        st.dataframe(st.session_state.scan_results)

# ------------------------------------------
# 分頁二：09:10 即時狙擊
# ------------------------------------------
with tab2:
    st.subheader("🎯 盤中微觀打擊與資金控管 (ATR)")
    
    if st.session_state.scan_results.empty:
        st.warning("請先完成【步驟一】的盤前掃描！")
    else:
        df_watch = st.session_state.scan_results
        
        # 資金控管設定
        st.markdown("### 💰 資金控管 (部位計算器)")
        col_c1, col_c2 = st.columns(2)
        total_capital = col_c1.number_input("您的總準備資金 (元)", value=500000, step=10000)
        max_loss_pct = col_c2.number_input("單筆最大可承受虧損 (%)", value=1.0, step=0.1)
        max_loss_amt = total_capital * (max_loss_pct / 100)
        st.info(f"🛡️ 嚴格風控：這筆交易無論如何，最多只允許虧損 **{max_loss_amt:,.0f} 元**")

        st.markdown("### ⚡ 即時動能濾網")
        selected_target = st.selectbox("請選擇要狙擊的標的：", df_watch['name'].tolist())
        
        target_info = df_watch[df_watch['name'] == selected_target].iloc[0]
        
        if st.button("🎯 啟動富果即時狙擊連線"):
            if not fugle_key:
                st.error("請先在左側輸入 Fugle API Token！")
            else:
                with st.spinner("連線交易所微秒級數據中..."):
                    # ==========================================
                    # 這裡模擬富果 API 的即時數據回傳 (實戰時會由您的 API 接手)
                    # ==========================================
                    mock_today_open = target_info['現價'] * 1.03  # 模擬開高 3%
                    mock_current_vol = target_info['昨日總量'] * 0.18  # 模擬開盤量已達昨天的 18%
                    mock_current_price = target_info['現價'] * 1.035
                    
                    gap_pct = ((mock_today_open - target_info['現價']) / target_info['現價']) * 100
                    vol_ratio = mock_current_vol / target_info['昨日總量']
                    
                    # 1. 檢驗跳空與爆量
                    is_gap_valid = (2.0 <= gap_pct <= 5.0)
                    is_vol_valid = (vol_ratio >= 0.15)
                    
                    st.markdown("#### 📡 即時動能判定")
                    if is_gap_valid and is_vol_valid:
                        st.success(f"🟢 **強勢點火！** 跳空 {gap_pct:.2f}% (合格)，開盤量比 {vol_ratio*100:.1f}% (合格)。")
                        
                        # 2. 算牌：利用 ATR 算出安全張數
                        stop_loss_price = target_info['動態停損']
                        risk_per_share = mock_current_price - stop_loss_price
                        
                        if risk_per_share > 0:
                            safe_shares = math.floor(max_loss_amt / risk_per_share)
                            safe_lots = safe_shares // 1000 # 換算成張數
                            
                            st.markdown("#### 🛡️ AI 派兵建議")
                            st.write(f"目前現價：**{mock_current_price:.2f}** | 您的動態停損點：**{stop_loss_price:.2f}**")
                            st.metric(label="建議買進最大張數", value=f"{safe_lots} 張")
                            st.caption("☝️ 只要不買超過這個張數，即使不幸跌到停損價被洗出場，總虧損也會控制在您設定的安全範圍內。")
                        else:
                            st.warning("價格計算異常，請觀望。")
                    else:
                        st.warning(f"🟡 **動能不足觀望中**。跳空: {gap_pct:.2f}%, 量比: {vol_ratio*100:.1f}%")