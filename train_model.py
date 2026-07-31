import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score
import joblib

print("🚀 開始讀取與處理歷史大數據...")

# 1. 定義標籤生成公式 (Target Label)
def generate_label(df, target_pct=0.03, sl_pct=0.015):
    open_p = df['Open']
    high_p = df['High']
    low_p = df['Low']
    
    max_return = (high_p - open_p) / open_p
    min_return = (low_p - open_p) / open_p
    
    y = np.where((max_return >= target_pct) & (min_return > -sl_pct), 1, 0)
    return pd.Series(y, index=df.index)

# 2. 定義特徵工程公式 (Feature Engineering)
def extract_features(df):
    features = pd.DataFrame(index=df.index)
    prev_close = df['Close'].shift(1)
    open_p = df['Open']
    
    features['gap_rate'] = ((open_p - prev_close) / prev_close) * 100
    vol_ma5 = df['Volume'].rolling(5).mean().shift(1)
    features['vol_ratio'] = df['Volume'].shift(1) / (vol_ma5 + 1e-5)
    
    ma5 = df['Close'].rolling(5).mean().shift(1)
    ma20 = df['Close'].rolling(20).mean().shift(1)
    features['dist_ma5'] = ((prev_close - ma5) / (ma5 + 1e-5)) * 100
    features['dist_ma20'] = ((prev_close - ma20) / (ma20 + 1e-5)) * 100
    
    max_20d = df['High'].rolling(20).max().shift(1)
    features['is_breakout'] = np.where(prev_close >= max_20d, 1, 0)
    
    tr = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)))
    features['atr14'] = tr.rolling(14).mean().shift(1) / (prev_close + 1e-5) * 100
    
    return features

# 使用穩定且歷史悠久的台股核心權值股作為 AI 訓練樣本
sample_tickers = ["2330.TW", "2317.TW", "2454.TW", "2303.TW", "2382.TW", "2881.TW", "2882.TW", "2891.TW", "3711.TW", "2308.TW"]

dataset_list = []

for t in sample_tickers:
    print(f"正在分析標的近 10 年歷史數據: {t}")
    try:
        df = yf.Ticker(t).history(period="10y")
        if len(df) > 500:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            feat = extract_features(df)
            label = generate_label(df)
            
            # 在同一個表格內合併，避免索引對齊報錯
            combined = feat.copy()
            combined['target'] = label
            
            # 過濾開盤溢價率介於 2%~6% 的歷史樣本
            mask = (combined['gap_rate'] >= 2.0) & (combined['gap_rate'] <= 6.0)
            filtered_df = combined[mask].dropna()
            
            if not filtered_df.empty:
                dataset_list.append(filtered_df)
    except Exception as e:
        print(f"處理 {t} 時發生錯誤: {e}")

if not dataset_list:
    raise ValueError("沒有成功收集到任何訓練資料！")

final_df = pd.concat(dataset_list)

X = final_df.drop(columns=['target'])
y = final_df['target']

# 時間序列拆分訓練 (前 80% 訓練，後 20% 測試)
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"\n訓練集筆數: {len(X_train)} | 測試集筆數: {len(X_test)}")

# 訓練 LightGBM 模型
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# 模型成效驗證
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.6).astype(int)

print("\n--- 模型測試集實測績效 ---")
print(f"準確率 (Accuracy): {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"高信心樣本實質勝率 (Precision): {precision_score(y_test, y_pred, zero_division=0)*100:.2f}%")

# 匯出模型
joblib.dump(model, 'model.pkl')
print("\n🎉 AI 模型已成功匯出為 'model.pkl'！")