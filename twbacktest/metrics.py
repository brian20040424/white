"""績效指標計算。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import BacktestResult

TRADING_DAYS = 252

# 指標名稱 → (中文說明, 顯示格式)
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "start": ("回測起日", "s"),
    "end": ("回測迄日", "s"),
    "initial_capital": ("初始資金", "money"),
    "final_equity": ("期末資產", "money"),
    "total_return": ("總報酬率", "pct"),
    "cagr": ("年化報酬率 (CAGR)", "pct"),
    "annual_volatility": ("年化波動率", "pct"),
    "sharpe": ("夏普比率", "f2"),
    "sortino": ("索提諾比率", "f2"),
    "max_drawdown": ("最大回撤 (MDD)", "pct"),
    "max_drawdown_days": ("最長回撤期間（交易日）", "int"),
    "calmar": ("卡瑪比率", "f2"),
    "exposure": ("持倉時間比例", "pct"),
    "num_trades": ("交易次數（已平倉）", "int"),
    "win_rate": ("勝率", "pct"),
    "profit_factor": ("獲利因子", "f2"),
    "avg_trade_return": ("平均每筆報酬", "pct"),
    "avg_holding_days": ("平均持有天數", "f1"),
    "total_fees": ("手續費＋證交稅合計", "money"),
    "benchmark_total_return": ("大盤同期報酬率", "pct"),
    "benchmark_cagr": ("大盤年化報酬率", "pct"),
    "benchmark_max_drawdown": ("大盤最大回撤", "pct"),
    "excess_return": ("超額報酬（相對大盤）", "pct"),
    "alpha": ("Alpha（年化）", "pct"),
    "beta": ("Beta", "f2"),
}


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    peak = equity.cummax()
    dd = equity / peak - 1
    # 最長回撤期間：連續低於前高的交易日數
    under = (dd < 0).astype(int).to_numpy()
    longest = run = 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)
    return float(dd.min()), int(longest)


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def _cagr(equity: pd.Series) -> float:
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (365.25 / days) - 1)


def compute_metrics(result: BacktestResult, benchmark: pd.Series | None = None,
                    risk_free: float = 0.0) -> dict:
    eq = result.equity["equity"]
    rets = eq.pct_change().dropna()
    ann_vol = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS)) if len(rets) else float("nan")
    ann_mean = float(rets.mean() * TRADING_DAYS) if len(rets) else float("nan")
    downside = rets[rets < 0]
    down_vol = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)) if len(downside) else float("nan")
    mdd, mdd_days = _max_drawdown(eq)
    cagr = _cagr(eq)

    t = result.trades
    closed = t[t["exit_reason"] != "open"] if len(t) else t
    wins = closed[closed["pnl"] > 0]["pnl"].sum() if len(closed) else 0.0
    losses = -closed[closed["pnl"] < 0]["pnl"].sum() if len(closed) else 0.0
    fees = float((closed["buy_cost"] + closed["sell_cost"]).sum()) if len(closed) else 0.0
    if len(t) and (t["exit_reason"] == "open").any():
        fees += float(t.loc[t["exit_reason"] == "open", "buy_cost"].sum())

    m = {
        "start": eq.index[0].strftime("%Y-%m-%d"),
        "end": eq.index[-1].strftime("%Y-%m-%d"),
        "initial_capital": result.config.initial_capital,
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "cagr": cagr,
        "annual_volatility": ann_vol,
        "sharpe": (ann_mean - risk_free) / ann_vol if ann_vol and ann_vol > 0 else float("nan"),
        "sortino": (ann_mean - risk_free) / down_vol if down_vol and down_vol > 0 else float("nan"),
        "max_drawdown": mdd,
        "max_drawdown_days": mdd_days,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "exposure": float((result.equity["shares"] > 0).mean()),
        "num_trades": int(len(closed)),
        "win_rate": float((closed["pnl"] > 0).mean()) if len(closed) else float("nan"),
        "profit_factor": float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else float("nan")),
        "avg_trade_return": float(closed["return_pct"].mean()) if len(closed) else float("nan"),
        "avg_holding_days": float(closed["holding_days"].mean()) if len(closed) else float("nan"),
        "total_fees": fees,
    }

    if benchmark is not None:
        bench = benchmark.reindex(eq.index).ffill().dropna()
        if len(bench) > 1:
            b_mdd, _ = _max_drawdown(bench)
            m["benchmark_total_return"] = float(bench.iloc[-1] / bench.iloc[0] - 1)
            m["benchmark_cagr"] = _cagr(bench)
            m["benchmark_max_drawdown"] = b_mdd
            m["excess_return"] = m["total_return"] - m["benchmark_total_return"]
            joined = pd.concat([rets, bench.pct_change()], axis=1, join="inner").dropna()
            if len(joined) > 2 and joined.iloc[:, 1].var() > 0:
                cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1], ddof=0)
                beta = cov[0, 1] / cov[1, 1]
                alpha = (joined.iloc[:, 0].mean() - beta * joined.iloc[:, 1].mean()) * TRADING_DAYS
                m["beta"] = float(beta)
                m["alpha"] = float(alpha)
    return m


def format_value(key: str, value) -> str:
    fmt = METRIC_LABELS.get(key, (key, "f2"))[1]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, float) and np.isinf(value):
        return "∞"
    if fmt == "pct":
        return f"{value * 100:.2f}%"
    if fmt == "money":
        return f"{value:,.0f}"
    if fmt == "int":
        return f"{int(value)}"
    if fmt == "f1":
        return f"{value:.1f}"
    if fmt == "f2":
        return f"{value:.2f}"
    return str(value)


def format_metrics(m: dict) -> list[tuple[str, str]]:
    return [(METRIC_LABELS[k][0], format_value(k, m[k])) for k in METRIC_LABELS if k in m]
