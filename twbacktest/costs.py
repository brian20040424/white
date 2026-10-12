"""台股交易成本模型：券商手續費、證券交易稅與滑價。

預設值依台灣證券市場現行規定：
- 手續費：成交金額 × 0.1425%，買賣各收一次，未滿 20 元以 20 元計，可設定券商折扣。
- 證交稅：只在賣出時課徵，一般股票 0.3%，ETF 0.1%。
- 手續費與證交稅計算後無條件捨去至整數元。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

STOCK_TAX_RATE = 0.003
ETF_TAX_RATE = 0.001


def default_tax_rate(symbol: str) -> float:
    """台股 ETF 代號多以 00 開頭（如 0050、00878），證交稅為 0.1%。"""
    code = symbol.split(".")[0]
    return ETF_TAX_RATE if code.startswith("00") else STOCK_TAX_RATE


@dataclass
class CostModel:
    commission_rate: float = 0.001425
    commission_discount: float = 1.0   # 例：六折填 0.6
    min_commission: float = 20.0
    tax_rate: float = STOCK_TAX_RATE
    slippage: float = 0.0              # 單邊滑價比例，例：0.001 = 0.1%

    @property
    def effective_rate(self) -> float:
        return self.commission_rate * self.commission_discount

    def commission(self, value: float) -> float:
        if value <= 0:
            return 0.0
        return float(max(math.floor(value * self.effective_rate), self.min_commission))

    def tax(self, value: float) -> float:
        if value <= 0:
            return 0.0
        return float(math.floor(value * self.tax_rate))

    def buy_price(self, price: float) -> float:
        return price * (1 + self.slippage)

    def sell_price(self, price: float) -> float:
        return price * (1 - self.slippage)
