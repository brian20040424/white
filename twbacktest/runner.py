"""把資料、策略、引擎、指標串起來的高階介面（CLI 與 Python 皆使用）。"""
from __future__ import annotations

import datetime as dt
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data as datamod
from .costs import CostModel, default_tax_rate
from .engine import BacktestConfig, BacktestResult, run_backtest
from .metrics import compute_metrics
from .strategies import get_strategy

log = logging.getLogger(__name__)


@dataclass
class RunOutput:
    symbol: str
    strategy: str
    params: dict
    result: BacktestResult
    metrics: dict
    data_info: str
    benchmark_info: str
    report_path: Path | None = None


def load_prices(symbol: str, start=None, end=None, *, csv: str | None = None, source: str = "auto",
                offline: bool = False) -> tuple[pd.DataFrame, str]:
    if csv:
        df = datamod.read_csv(csv)
        if start:
            df = df.loc[pd.Timestamp(start):]
        if end:
            df = df.loc[:pd.Timestamp(end)]
        if df.empty:
            raise datamod.DataUnavailable(f"{csv} 在指定期間內沒有資料")
        return df, f"{symbol}：{df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}，共 {len(df)} 筆，來源 CSV {csv}"
    df, info = datamod.get_history(symbol, start, end, source=source, offline=offline)
    return df, str(info)


def load_benchmark(prices: pd.DataFrame, benchmark: str | None, *, csv: str | None = None,
                   source: str = "auto", offline: bool = False) -> tuple[pd.Series | None, str]:
    if not benchmark or benchmark.lower() == "none":
        return None, "未使用"
    try:
        df, info = load_prices(benchmark, prices.index[0], prices.index[-1], csv=csv,
                               source=source, offline=offline)
        return df["close"], info
    except Exception as e:  # 大盤抓不到不影響個股回測
        log.warning("無法取得基準 %s：%s", benchmark, e)
        return None, f"無法取得（{e}）"


def build_config(symbol: str, *, capital: float = 1_000_000, lot: int = 1000, position: float = 1.0,
                 discount: float = 1.0, min_fee: float = 20.0, tax: float | None = None,
                 slippage: float = 0.0, stop_loss: float | None = None,
                 take_profit: float | None = None) -> BacktestConfig:
    costs = CostModel(commission_discount=discount, min_commission=min_fee,
                      tax_rate=default_tax_rate(symbol) if tax is None else tax, slippage=slippage)
    return BacktestConfig(initial_capital=capital, lot_size=lot, position_size=position,
                          stop_loss=stop_loss, take_profit=take_profit, costs=costs)


def backtest(symbol: str, strategy: str = "sma_cross", params: dict | None = None, *,
             start=None, end=None, csv: str | None = None, benchmark: str | None = "TAIEX",
             benchmark_csv: str | None = None, source: str = "auto", offline: bool = False,
             risk_free: float = 0.0, report_dir: str | Path | None = None,
             prices: pd.DataFrame | None = None, bench: pd.Series | None = None,
             bench_info: str | None = None, **config_kwargs) -> RunOutput:
    """執行一次回測。

    prices / bench 可直接傳入已載入的資料以避免重複下載；
    同時傳入 bench_info 代表基準已處理過（bench 可為 None 表示不比較）。
    """
    strat = get_strategy(strategy)
    params = strat.resolve_params(params)
    if prices is None:
        prices, data_info = load_prices(symbol, start, end, csv=csv, source=source, offline=offline)
    else:
        data_info = f"{symbol}：{prices.index[0]:%Y-%m-%d} ~ {prices.index[-1]:%Y-%m-%d}，共 {len(prices)} 筆"
    if bench_info is None:
        if bench is None:
            bench, bench_info = load_benchmark(prices, benchmark, csv=benchmark_csv, source=source,
                                               offline=offline)
        else:
            bench_info = str(benchmark)

    config = build_config(symbol, **config_kwargs)
    result = run_backtest(prices, strat.signals(prices, params), config)
    metrics = compute_metrics(result, bench, risk_free=risk_free)
    out = RunOutput(symbol, strategy, params, result, metrics, data_info, bench_info)

    if report_dir:
        from .report import write_report
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = datamod.normalize_symbol(symbol).replace("^", "IDX_")
        folder = Path(report_dir) / f"{name}_{strategy}_{stamp}"
        c = config.costs
        meta = {
            "股票": datamod.normalize_symbol(symbol),
            "策略": f"{strat.title} ({strategy})",
            "策略參數": ", ".join(f"{k}={v}" for k, v in params.items()) or "無",
            "資料": data_info,
            "基準": bench_info,
            "初始資金": f"{config.initial_capital:,.0f}",
            "交易單位": "零股（1 股）" if config.lot_size == 1 else f"{config.lot_size} 股",
            "資金使用比例": f"{config.position_size:.0%}",
            "手續費": f"{c.commission_rate:.4%} × 折扣 {c.commission_discount}，最低 {c.min_commission:.0f} 元",
            "證交稅": f"{c.tax_rate:.2%}（賣出時）",
            "滑價": f"{c.slippage:.2%}",
            "停損": f"{config.stop_loss:.0%}" if config.stop_loss else "未設定",
            "停利": f"{config.take_profit:.0%}" if config.take_profit else "未設定",
            "成交規則": "收盤產生訊號，次一交易日開盤價成交",
        }
        if result.skipped_entries:
            meta["資金不足未進場次數"] = result.skipped_entries
        title = f"{datamod.normalize_symbol(symbol)} · {strat.title}"
        chart_title = f"{datamod.normalize_symbol(symbol)} - {strategy}"
        out.report_path = write_report(result, metrics, folder, title, meta, bench, chart_title)
    return out


def optimize(symbol: str, strategy: str, grid: dict[str, list], *, metric: str = "sharpe",
             split: float | None = None, start=None, end=None, csv: str | None = None,
             source: str = "auto", offline: bool = False, risk_free: float = 0.0,
             **config_kwargs) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """參數網格搜尋。

    split（例如 0.7）會把資料切成前 70% 樣本內（用來挑參數）與後 30% 樣本外（用來驗證），
    以降低過度最佳化的風險。回傳 (樣本內排名表, 樣本外驗證表或 None)。
    """
    strat = get_strategy(strategy)
    prices, _ = load_prices(symbol, start, end, csv=csv, source=source, offline=offline)
    if split:
        if not 0.1 <= split <= 0.95:
            raise ValueError("split 必須介於 0.1 與 0.95 之間")
        cut = int(len(prices) * split)
        ins, oos = prices.iloc[:cut], prices.iloc[cut:]
    else:
        ins, oos = prices, None

    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        try:
            p = strat.resolve_params(params)
            res = run_backtest(ins, strat.signals(ins, p), build_config(symbol, **config_kwargs))
        except ValueError as e:  # 例如 fast >= slow 的無效組合
            log.debug("略過參數 %s：%s", params, e)
            continue
        m = compute_metrics(res, risk_free=risk_free)
        rows.append({**params, **{k: m[k] for k in
                                  ("total_return", "cagr", "sharpe", "max_drawdown", "calmar",
                                   "win_rate", "num_trades")}})
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("沒有任何有效的參數組合")
    if metric not in table:
        raise ValueError(f"無法依 {metric} 排序，可用：total_return, cagr, sharpe, max_drawdown, calmar, win_rate")
    table = table.sort_values(metric, ascending=False, na_position="last").reset_index(drop=True)

    oos_table = None
    if oos is not None and len(oos) > 1:
        best = {k: table.loc[0, k] for k in keys}
        p = strat.resolve_params(best)
        res = run_backtest(oos, strat.signals(prices, p).loc[oos.index], build_config(symbol, **config_kwargs))
        m = compute_metrics(res, risk_free=risk_free)
        oos_table = pd.DataFrame([{**best, "期間": f"{oos.index[0]:%Y-%m-%d} ~ {oos.index[-1]:%Y-%m-%d}",
                                   **{k: m[k] for k in ("total_return", "cagr", "sharpe", "max_drawdown",
                                                        "num_trades")}}])
    return table, oos_table
