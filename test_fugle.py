from fugle_marketdata import RestClient

# 把這行換成您在開發者中心複製的 Token
API_TOKEN = "NmY3MDM1NjYtNzNlNC00NWJiLWFiNjgtZTc1NWI0MDgwY2FjIGI2YjQ5N2QyLTBhMTctNDM3OC1hNGJiLWJkOTZmNGM4NTg5Nw=="

# 初始化客戶端
client = RestClient(api_key=API_TOKEN)

# 測試：抓取台積電 (2330) 今天的最新即時報價與 VWAP 需要的成交明細
stock_id = "2330"
try:
    # 抓取即時行情 (包含開盤價、總量、最新價等)
    quote = client.stock.intraday.quote(symbol=stock_id)
    print(f"📊 {stock_id} 即時行情抓取成功！")
    print(f"最新成交價: {quote['closePrice']}")
    print(f"今日累積成交量: {quote['total']['tradeVolume']} 張")
    
    # 抓取盤中 Tick 明細 (這就是未來計算 VWAP 和 Opening Drive 的關鍵數據)
    trades = client.stock.intraday.trades(symbol=stock_id)
    print(f"\n🔍 成功抓取 Tick 明細，共 {len(trades['data'])} 筆成交紀錄。")
    
except Exception as e:
    print(f"連線失敗，請檢查 Token 是否正確：{e}")