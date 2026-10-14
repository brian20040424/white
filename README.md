# white：twbacktest 台股回測系統

用 Python 回測台股**大盤（加權指數）**與**個股／ETF**的交易策略，並依台股實際規定扣除手續費與證交稅。

- **資料**：Yahoo Finance（還原權息）→ 證交所官網 → 本機快取；最新資料抓不到時，自動退回抓近 5 年、近 3 年；也能匯入自己的 CSV
- **策略**：均線交叉、EMA 交叉、RSI、MACD、布林通道、KD、通道突破、買進持有
- **交易模擬**：收盤產生訊號、次日開盤成交；可選整張或零股；手續費折扣與最低手續費；證交稅（股票 0.3%／ETF 0.1%）；滑價；停損停利
- **績效**：CAGR、夏普、索提諾、最大回撤、卡瑪、勝率、獲利因子，以及相對大盤的超額報酬、Alpha、Beta
- **報告**：HTML 報告、圖表、交易明細 CSV
- **參數最佳化**：網格搜尋，可切分樣本內／樣本外驗證

## 快速開始

安裝：

```
git clone https://github.com/brian20040424/white.git
cd white
pip install -r requirements.txt
```

下載台積電與大盤資料，再用 20/60 日均線交叉回測台積電（`--lot 1` 允許零股，避免高價股買不起一張）：

```
python -m twbacktest fetch 2330 TAIEX
python -m twbacktest run 2330 --lot 1
```

多檔比較、參數最佳化：

```
python -m twbacktest run 2330 0050 -s macd --lot 1
python -m twbacktest optimize 2330 -g fast=5:30:5 -g slow=40:120:20 --split 0.7 --lot 1
```

完整說明請看 **[使用手冊](docs/使用手冊.md)**。

## 測試

```
pip install pytest
python -m pytest
```

> 回測結果只反映歷史資料在特定假設下的表現，不代表未來績效，也不構成投資建議。
