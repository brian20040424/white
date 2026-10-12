"""常用技術指標。所有函式只使用當日（含）以前的資料，不會偷看未來。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & avg_gain.notna()] = 100.0
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": dif - dea})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + k * std, "lower": mid - k * std})


def kd(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """台灣常用的 KD 隨機指標：K = (1-1/k)·K昨 + (1/k)·RSV，初始值 50。"""
    low_n = df["low"].rolling(n, min_periods=n).min()
    high_n = df["high"].rolling(n, min_periods=n).max()
    rng = (high_n - low_n).replace(0, np.nan)
    rsv = ((df["close"] - low_n) / rng * 100).fillna(50.0)
    rsv[low_n.isna()] = np.nan

    k_vals, d_vals = [], []
    k_prev = d_prev = 50.0
    for v in rsv.to_numpy():
        if np.isnan(v):
            k_vals.append(np.nan)
            d_vals.append(np.nan)
            continue
        k_prev = (1 - 1 / k_smooth) * k_prev + v / k_smooth
        d_prev = (1 - 1 / d_smooth) * d_prev + k_prev / d_smooth
        k_vals.append(k_prev)
        d_vals.append(d_prev)
    return pd.DataFrame({"k": k_vals, "d": d_vals}, index=df.index)


def donchian(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """前 n 日（不含當日）的最高價與最低價通道。"""
    return pd.DataFrame({
        "upper": df["high"].rolling(n, min_periods=n).max().shift(1),
        "lower": df["low"].rolling(n, min_periods=n).min().shift(1),
    })
