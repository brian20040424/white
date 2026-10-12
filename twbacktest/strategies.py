"""內建交易策略。

每個策略接收含 open/high/low/close/volume 的 DataFrame，回傳與日期對齊的目標部位序列：
1 = 持有、0 = 空手。訊號在當日收盤後決定，回測引擎會在「下一個交易日開盤」執行，
因此策略可以放心使用當日收盤資料，不會產生未來函數。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class Strategy:
    name: str
    title: str
    description: str
    func: Callable[..., pd.Series]
    defaults: dict = field(default_factory=dict)

    def resolve_params(self, params: dict | None) -> dict:
        params = dict(params or {})
        unknown = set(params) - set(self.defaults)
        if unknown:
            allowed = "、".join(self.defaults) or "（無參數）"
            raise ValueError(f"策略 {self.name} 不支援參數 {sorted(unknown)}，可用參數：{allowed}")
        merged = {**self.defaults, **params}
        # 依預設值的型別轉換參數
        return {k: type(self.defaults[k])(v) for k, v in merged.items()}

    def signals(self, df: pd.DataFrame, params: dict | None = None) -> pd.Series:
        p = self.resolve_params(params)
        sig = self.func(df, **p)
        return sig.reindex(df.index).fillna(0).clip(0, 1).astype(float)


def _hold_between(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """在 enter 為真時進場、exit_ 為真時出場，其餘日子維持前一狀態。"""
    state = pd.Series(np.nan, index=enter.index)
    state[exit_.fillna(False)] = 0.0
    state[enter.fillna(False)] = 1.0
    return state.ffill().fillna(0.0)


def buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.Series:
    if fast >= slow:
        raise ValueError("fast 必須小於 slow")
    f, s = ind.sma(df["close"], fast), ind.sma(df["close"], slow)
    return (f > s).astype(float).where(s.notna(), 0.0)


def ema_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    if fast >= slow:
        raise ValueError("fast 必須小於 slow")
    f, s = ind.ema(df["close"], fast), ind.ema(df["close"], slow)
    return (f > s).astype(float).where(s.notna(), 0.0)


def rsi_reversion(df: pd.DataFrame, period: int = 14, lower: float = 30.0, upper: float = 70.0) -> pd.Series:
    r = ind.rsi(df["close"], period)
    return _hold_between(r < lower, r > upper)


def macd_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m = ind.macd(df["close"], fast, slow, signal)
    return (m["dif"] > m["dea"]).astype(float).where(m["dea"].notna(), 0.0)


def bollinger_reversion(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.Series:
    b = ind.bollinger(df["close"], period, k)
    return _hold_between(df["close"] < b["lower"], df["close"] > b["mid"])


def kd_cross(df: pd.DataFrame, n: int = 9, oversold: float = 20.0, overbought: float = 80.0) -> pd.Series:
    """K 值在超賣區向上穿越 D 值時進場，在超買區向下穿越時出場。"""
    k = ind.kd(df, n)
    cross_up = (k["k"] > k["d"]) & (k["k"].shift(1) <= k["d"].shift(1))
    cross_dn = (k["k"] < k["d"]) & (k["k"].shift(1) >= k["d"].shift(1))
    return _hold_between(cross_up & (k["k"] < oversold), cross_dn & (k["k"] > overbought))


def breakout(df: pd.DataFrame, entry: int = 20, exit: int = 10) -> pd.Series:
    """海龜式通道突破：收盤創 entry 日新高進場，跌破 exit 日新低出場。"""
    up = ind.donchian(df, entry)["upper"]
    dn = ind.donchian(df, exit)["lower"]
    return _hold_between(df["close"] > up, df["close"] < dn)


STRATEGIES: dict[str, Strategy] = {s.name: s for s in [
    Strategy("buy_hold", "買進持有", "第一天買進後一路持有，作為比較基準。", buy_hold, {}),
    Strategy("sma_cross", "均線黃金/死亡交叉",
             "短均線在長均線之上時持有，跌破時出場。", sma_cross, {"fast": 20, "slow": 60}),
    Strategy("ema_cross", "指數均線交叉",
             "與 sma_cross 相同，但使用反應較快的指數移動平均。", ema_cross, {"fast": 12, "slow": 26}),
    Strategy("rsi", "RSI 超買超賣",
             "RSI 低於 lower 時買進，高於 upper 時賣出。", rsi_reversion,
             {"period": 14, "lower": 30.0, "upper": 70.0}),
    Strategy("macd", "MACD 交叉",
             "DIF 在訊號線（DEA）之上時持有。", macd_cross, {"fast": 12, "slow": 26, "signal": 9}),
    Strategy("bollinger", "布林通道均值回歸",
             "收盤跌破下軌時買進，回到中軌之上時賣出。", bollinger_reversion, {"period": 20, "k": 2.0}),
    Strategy("kd", "KD 指標交叉",
             "K 值在超賣區黃金交叉買進，在超買區死亡交叉賣出。", kd_cross,
             {"n": 9, "oversold": 20.0, "overbought": 80.0}),
    Strategy("breakout", "通道突破（海龜）",
             "收盤突破前 entry 日最高價買進，跌破前 exit 日最低價賣出。", breakout,
             {"entry": 20, "exit": 10}),
]}


def get_strategy(name: str) -> Strategy:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(f"找不到策略 {name}，可用策略：{', '.join(STRATEGIES)}") from None
