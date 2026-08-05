from fugle_marketdata import RestClient
import time

# 把這行換成您的 Token
API_TOKEN = "NmY3MDM1NjYtNzNlNC00NWJiLWFiNjgtZTc1NWI0MDgwY2FjIGI2YjQ5N2QyLTBhMTctNDM3OC1hNGJiLWJkOTZmNGM4NTg5Nw=="
client = RestClient(api_key=API_TOKEN)

# 我們建立一個熱門當沖觀察名單 (實務上可以擴充到上百檔)
watchlist = ["2330", "2317", "2603", "2454", "3231", "3711", "3443", "3035", "2382", "3260"]

print("🚀 啟動熱門股 VWAP 深度掃描...\n")

for symbol in watchlist:
    try:
        # 1. 抓取即時報價 (這個您剛剛測試過，權限是完全沒問題的！)
        quote = client.stock.intraday.quote(symbol=symbol)
        price = quote.get('closePrice', 0)
        vol = quote.get('total', {}).get('tradeVolume', 0)
        name = quote.get('name', symbol)
        
        # 2. 抓取 Tick 明細算 VWAP
        trades = client.stock.intraday.trades(symbol=symbol)
        ticks = trades.get('data', [])
        
        if len(ticks) > 0:
            # VWAP 公式：總成交金額 / 總成交量
            total_value = sum([t['price'] * t['volume'] for t in ticks])
            total_volume = sum([t['volume'] for t in ticks])
            vwap = (total_value / total_volume) if total_volume > 0 else price
            
            # 判斷趨勢：當沖客最看重的強弱分界線
            if price > vwap:
                status = "🚀 站上 VWAP (強勢多頭)"
            elif price < vwap:
                status = "📉 跌破 VWAP (弱勢空頭)"
            else:
                status = "➖ 貼齊 VWAP (盤整)"
                
            print(f"[{symbol} {name}] 成交量: {vol}張 | 最新價: {price} | VWAP: {vwap:.2f} | 狀態: {status}")
        
        # ⚠️ 關鍵保護機制：暫停 0.5 秒，避免連續呼叫被伺服器封鎖 (Rate Limit)
        time.sleep(0.5)
        
    except Exception as e:
        print(f"[{symbol}] 抓取失敗: {e}")

print("\n✅ 掃描完成！")