import time
import requests
from fugle_marketdata import RestClient

# --- 設定區 ---
API_TOKEN = "NmY3MDM1NjYtNzNlNC00NWJiLWFiNjgtZTc1NWI0MDgwY2FjIGI2YjQ5N2QyLTBhMTctNDM3OC1hNGJiLWJkOTZmNGM4NTg5Nw=="
client = RestClient(api_key=API_TOKEN)

# 這裡放入您每天盤前用 APP (AI 第一階段) 跑出來的「高勝率觀察名單」
# 或是直接放入昨日強勢股代號
watchlist = ["2330", "2317", "2603", "3231", "3443", "3035", "2382", "3711"] 

print("🎯 [09:10 早盤狙擊器] 啟動！尋找開盤爆發股...\n")

for symbol in watchlist:
    try:
        # 1. 抓取今日即時報價與昨日收盤價
        quote = client.stock.intraday.quote(symbol=symbol)
        name = quote.get('name', symbol)
        current_price = quote.get('closePrice', 0)
        open_price = quote.get('openPrice', 0)
        prev_close = quote.get('previousClose', 1) # 昨日收盤價
        vol_today = quote.get('total', {}).get('tradeVolume', 0)
        
        # 2. 計算跳空幅度 (Gap)
        gap_pct = ((open_price - prev_close) / prev_close) * 100
        
        # 3. 取得 09:00 ~ 09:10 的 Tick 明細計算 VWAP
        trades = client.stock.intraday.trades(symbol=symbol)
        ticks = trades.get('data', [])
        
        if len(ticks) > 0 and open_price > 0:
            total_value = sum([t['price'] * t['volume'] for t in ticks])
            total_volume = sum([t['volume'] for t in ticks])
            vwap = (total_value / total_volume) if total_volume > 0 else current_price
            
            # --- 核心邏輯判斷 ---
            cond1_gap = gap_pct >= 1.5  # 跳空大於 1.5%
            cond2_vwap = current_price >= vwap # 現價站穩開盤 VWAP
            cond3_momentum = current_price >= open_price # 09:10 現價不可跌破開盤價 (拒絕開高走低)
            
            if cond1_gap and cond2_vwap and cond3_momentum:
                print(f"🔥 [強烈買進訊號] {symbol} {name}")
                print(f"   ➔ 跳空: {gap_pct:.2f}% | 現價: {current_price} | 開盤 VWAP: {vwap:.2f}")
                print(f"   ➔ 戰術: 09:10 現價買進，跌破 {vwap:.2f} 停損，09:30 前拉高停利！\n")
            else:
                print(f"⏳ [觀望] {symbol} {name} (未達早盤爆發標準)")

        time.sleep(0.5) # 避免呼叫頻率過高
        
    except Exception as e:
        pass

print("\n✅ 掃描完畢，請專注操作上述 [強烈買進訊號] 的標的！")