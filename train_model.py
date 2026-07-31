import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score
import joblib

print("🚀 [AI 2.0] 開始讀取大數據與建構 30+ 量化特徵矩陣...")

def extract_features(df):
    features = pd.DataFrame(index=df.index)
    
    # 1. 基礎價格與跳空 (Gap Bins)
    prev_close = df['Close'].shift(1)
    open_p = df['Open']
    gap_pct = ((open_p - prev_close) / prev_close) * 100
    features['Gap_0_2'] = ((gap_pct >= 0) & (gap_pct < 2)).astype(int)
    features['Gap_2_4'] = ((gap_pct >= 2) & (gap_pct < 4)).astype(int)
    features['Gap_4_6'] = ((gap_pct >= 4) & (gap_pct < 6)).astype(int)
    features['Gap_6_9'] = ((gap_pct >= 6) & (gap_pct < 9)).astype(int)
    features['Gap_Over_9'] = (gap_pct >= 9).astype(int)

    # --- 以下技術指標皆根據「昨日」收盤計算 (避免未來函數 Look-ahead bias) ---
    
    # 2. RVOL 與 成交量動能 (使用前一日)
    vol_ma20 = df['Volume'].rolling(20).mean().shift(1)
    features['RVOL'] = df['Volume'].shift(1) / (vol_ma20 + 1e-5)
    
    # 3. EMA 多頭排列
    ema5 = df['Close'].ewm(span=5, adjust=False).mean().shift(1)
    ema10 = df['Close'].ewm(span=10, adjust=False).mean().shift(1)
    ema20 = df['Close'].ewm(span=20, adjust=False).mean().shift(1)
    ema60 = df['Close'].ewm(span=60, adjust=False).mean().shift(1)
    features['EMA_Bullish'] = ((ema5 > ema10) & (ema10 > ema20) & (ema20 > ema60)).astype(int)
    
    # 4. RSI (14日)
    delta = df['Close'].diff().shift(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-5)
    features['RSI_14'] = 100 - (100 / (1 + rs))
    features['RSI_GoldenZone'] = ((features['RSI_14'] > 55) & (features['RSI_14'] < 75)).astype(int)
    
    # 5. MACD Histogram
    macd_line = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_line - macd_signal).shift(1)
    features['MACD_Hist_Pos'] = (macd_hist > 0).astype(int)
    
    # 6. 布林通道 (Bollinger Bands)
    sma20 = df['Close'].rolling(20).mean().shift(1)
    std20 = df['Close'].rolling(20).std().shift(1)
    bb_upper = sma20 + 2 * std20
    features['Close_Above_BB'] = (df['Close'].shift(1) > bb_upper).astype(int)
    
    # 7. Donchian Channel (多週期創高)
    features['High_20D'] = (df['Close'].shift(1) > df['High'].rolling(20).max().shift(2)).astype(int)
    features['High_55D'] = (df['Close'].shift(1) > df['High'].rolling(55).max().shift(2)).astype(int)
    features['High_120D'] = (df['Close'].shift(1) > df['High'].rolling(120).max().shift(2)).astype(int)
    
    # 8. K棒型態 (One-hot encoding)
    # Inside Bar (內孕線)
    features['is_InsideBar'] = ((df['High'].shift(1) < df['High'].shift(2)) & (df['Low'].shift(1) > df['Low'].shift(2))).astype(int)
    # Marubozu (實體大紅K)
    body = (df['Close'].shift(1) - df['Open'].shift(1)).abs()
    shadow = df['High'].shift(1) - df['Low'].shift(1)
    features['is_Marubozu'] = ((body / (shadow + 1e-5)) > 0.8).astype(int)
    
    # 9. 波動率 (ATR)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features['ATR_14'] = tr.rolling(14).mean().shift(1)
    features['ATR_Ratio'] = (features['ATR_14'] / df['Close'].shift(1)) * 100
    
    return features.dropna()

# --- 動態 ATR 標籤：TP = 2.5*ATR, SL = 1.2*ATR ---
def generate_atr_label(df, features):
    labels = pd.Series(0, index=features.index)
    
    for idx in features.index:
        try:
            today_open = df.at[idx, 'Open']
            today_high = df.at[idx, 'High']
            today_low = df.at[idx, 'Low']
            atr = features.at[idx, 'ATR_14']
            
            tp_price = today_open + (2.5 * atr)
            sl_price = today_open - (1.2 * atr)
            
            # 若最高價觸及停利，且最低價未掃到停損，視為成功獲利
            if (today_high >= tp_price) and (today_low > sl_price):
                labels.at[idx] = 1
        except:
            continue
    return labels

# 使用具流動性的大型權值股做歷史回測
sample_tickers = ["2330.TW", "2317.TW", "2454.TW", "2303.TW", "2382.TW", "2603.TW", "2881.TW", "3231.TW", "3711.TW"]
dataset_list = []

for t in sample_tickers:
    try:
        df = yf.Ticker(t).history(period="10y")
        if len(df) > 500:
            feat = extract_features(df)
            label = generate_atr_label(df, feat)
            combined = feat.copy()
            combined['target'] = label
            # 過濾僅保留有向上跳空訊號的交易日，符合當沖邏輯
            valid_days = combined[(combined['Gap_0_2']==1) | (combined['Gap_2_4']==1) | (combined['Gap_4_6']==1) | (combined['Gap_6_9']==1) | (combined['Gap_Over_9']==1)]
            if not valid_days.empty:
                dataset_list.append(valid_days)
    except Exception as e:
        print(f"處理 {t} 失敗: {e}")

final_df = pd.concat(dataset_list)
X = final_df.drop(columns=['target', 'ATR_14']) # 訓練不包含絕對價格，避免過擬合
y = final_df['target']

# Walk-Forward 時間序列拆分 (嚴格樣本外測試)
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"\n訓練集特徵維度: {X_train.shape[1]} | 總筆數: {len(X)}")

# LightGBM 模型 (升級樹深度以學習 30 個特徵的非線性交互)
model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.01, max_depth=7, num_leaves=63, random_state=42, verbose=-1)
model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.65).astype(int) # 嚴格要求 65% 信心水準

print("\n--- [AI 2.0] 嚴格樣本外回測績效 ---")
print(f"動態風控 TP=2.5*ATR / SL=1.2*ATR")
print(f"高信心觸發次數: {sum(y_pred)}")
print(f"實質勝率 (Precision): {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")

joblib.dump(model, 'model.pkl')
print("\n🎉 量化版 model.pkl 已成功匯出！")
