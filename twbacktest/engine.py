"""逐日回測引擎（只做多）。

執行規則：
1. 第 t 日收盤後依策略訊號決定目標部位。
2. 第 t+1 日開盤價成交（含滑價、手續費、證交稅）。
3. 若設定停損/停利，盤中觸價即以觸發價出場；若開盤就跳空越過，則以開盤價出場。
   停損/停利出場後，要等策略訊號先回到 0 再轉為 1 才會重新進場。
4. 每日以收盤價計算資產淨值。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    lot_size: int = 1000              # 1 張 = 1000 股；設為 1 代表允許零股交易
    position_size: float = 1.0        # 每次進場投入現金的比例 (0~1]
    stop_loss: float | None = None    # 例：0.1 = 自進場價下跌 10% 停損
    take_profit: float | None = None  # 例：0.2 = 自進場價上漲 20% 停利
    costs: CostModel = field(default_factory=CostModel)

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital 必須大於 0")
        if self.lot_size < 1:
            raise ValueError("lot_size 至少為 1")
        if not 0 < self.position_size <= 1:
            raise ValueError("position_size 必須介於 0 與 1 之間")
        for name in ("stop_loss", "take_profit"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} 必須大於 0")


@dataclass
class BacktestResult:
    equity: pd.DataFrame        # 每日：cash, shares, close, equity, signal
    trades: pd.DataFrame        # 每筆交易
    config: BacktestConfig
    skipped_entries: int = 0    # 買進訊號因資金不足一個交易單位而未能進場的次數


TRADE_COLUMNS = ["entry_date", "entry_price", "exit_date", "exit_price", "shares",
                 "buy_cost", "sell_cost", "pnl", "return_pct", "holding_days", "exit_reason"]


def run_backtest(data: pd.DataFrame, signal: pd.Series, config: BacktestConfig | None = None) -> BacktestResult:
    config = config or BacktestConfig()
    cm = config.costs
    if data.empty:
        raise ValueError("沒有行情資料可以回測")
    data = data.sort_index()
    signal = signal.reindex(data.index).fillna(0.0)

    opens = data["open"].to_numpy(float)
    highs = data["high"].to_numpy(float)
    lows = data["low"].to_numpy(float)
    closes = data["close"].to_numpy(float)
    sig = signal.to_numpy(float)
    dates = data.index

    cash = float(config.initial_capital)
    shares = 0
    entry = None            # dict：進場資訊
    blocked = False         # 停損/停利後等待訊號重置
    skipped = 0
    skip_counted = False    # 同一段買進訊號只計一次「資金不足」
    trades: list[dict] = []
    rows = []

    def close_position(i: int, raw_price: float, reason: str):
        nonlocal cash, shares, entry
        price = cm.sell_price(raw_price)
        value = shares * price
        fee = cm.commission(value)
        tax = cm.tax(value)
        cash += value - fee - tax
        cost_basis = entry["shares"] * entry["price"] + entry["fee"]
        pnl = value - fee - tax - cost_basis
        trades.append({
            "entry_date": entry["date"], "entry_price": entry["price"],
            "exit_date": dates[i], "exit_price": price, "shares": shares,
            "buy_cost": entry["fee"], "sell_cost": fee + tax,
            "pnl": pnl, "return_pct": pnl / cost_basis,
            "holding_days": (dates[i] - entry["date"]).days, "exit_reason": reason,
        })
        shares = 0
        entry = None

    for i in range(len(dates)):
        prev_sig = sig[i - 1] if i > 0 else 0.0
        if prev_sig < 0.5:
            blocked = False
            skip_counted = False

        # 1) 依昨日收盤訊號，在今日開盤執行
        if i > 0 and not np.isnan(opens[i]):
            if shares > 0 and prev_sig < 0.5:
                close_position(i, opens[i], "signal")
            elif shares == 0 and prev_sig >= 0.5 and not blocked:
                price = cm.buy_price(opens[i])
                budget = cash * config.position_size
                per_share = price * (1 + cm.effective_rate)
                qty = int(budget // per_share) // config.lot_size * config.lot_size
                while qty > 0 and qty * price + cm.commission(qty * price) > cash:
                    qty -= config.lot_size
                if qty > 0:
                    fee = cm.commission(qty * price)
                    cash -= qty * price + fee
                    shares = qty
                    entry = {"date": dates[i], "price": price, "fee": fee, "shares": qty}
                elif not skip_counted:
                    skipped += 1
                    skip_counted = True

        # 2) 盤中停損 / 停利
        if shares > 0:
            ref = entry["price"]
            if config.stop_loss is not None:
                stop = ref * (1 - config.stop_loss)
                if lows[i] <= stop:
                    fill = min(opens[i], stop) if entry["date"] < dates[i] else stop
                    close_position(i, fill, "stop_loss")
                    blocked = True
            if shares > 0 and config.take_profit is not None:
                target = ref * (1 + config.take_profit)
                if highs[i] >= target:
                    fill = max(opens[i], target) if entry["date"] < dates[i] else target
                    close_position(i, fill, "take_profit")
                    blocked = True

        rows.append((cash, shares, closes[i], cash + shares * closes[i], sig[i]))

    equity = pd.DataFrame(rows, index=dates, columns=["cash", "shares", "close", "equity", "signal"])

    # 期末仍持有：以最後收盤價估算（含假設賣出成本），只列入交易明細，不影響淨值曲線
    if shares > 0:
        last = len(dates) - 1
        value = shares * cm.sell_price(closes[last])
        fee, tax = cm.commission(value), cm.tax(value)
        cost_basis = entry["shares"] * entry["price"] + entry["fee"]
        pnl = value - fee - tax - cost_basis
        trades.append({
            "entry_date": entry["date"], "entry_price": entry["price"],
            "exit_date": dates[last], "exit_price": cm.sell_price(closes[last]), "shares": shares,
            "buy_cost": entry["fee"], "sell_cost": fee + tax,
            "pnl": pnl, "return_pct": pnl / cost_basis,
            "holding_days": (dates[last] - entry["date"]).days, "exit_reason": "open",
        })

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return BacktestResult(equity=equity, trades=trades_df, config=config, skipped_entries=skipped)
