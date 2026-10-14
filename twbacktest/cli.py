"""命令列介面：python -m twbacktest <指令> ..."""
from __future__ import annotations

import argparse
import logging
import sys
import unicodedata

import pandas as pd

from . import data as datamod
from .metrics import METRIC_LABELS, format_metrics, format_value
from .runner import backtest, load_benchmark, load_prices, optimize
from .strategies import STRATEGIES


def _ljust(text: str, width: int) -> str:
    """依顯示寬度補空白（中文字佔兩格），讓終端機表格對齊。"""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - w, 0)


def _parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _parse_params(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"參數格式錯誤：{item}（應為 名稱=數值，例如 fast=20）")
        k, v = item.split("=", 1)
        out[k.strip()] = _parse_value(v.strip())
    return out


def _parse_grid(items: list[str] | None) -> dict:
    grid = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"網格格式錯誤：{item}（例：fast=5,10,20 或 fast=5:30:5）")
        k, v = item.split("=", 1)
        v = v.strip()
        if ":" in v:  # 起:迄:間隔（含迄）
            a, b, *s = [_parse_value(x) for x in v.split(":")]
            step = s[0] if s else 1
            vals, x = [], a
            while x <= b + 1e-9:
                vals.append(round(x, 10) if isinstance(x, float) else x)
                x += step
        else:
            vals = [_parse_value(x) for x in v.split(",") if x.strip()]
        grid[k.strip()] = vals
    if not grid:
        raise SystemExit("請至少提供一個 --grid，例如 --grid fast=5,10,20")
    return grid


def _add_common(p: argparse.ArgumentParser):
    g = p.add_argument_group("資料")
    g.add_argument("--start", help="起始日 YYYY-MM-DD（預設：迄日往前 5 年）")
    g.add_argument("--end", help="結束日 YYYY-MM-DD（預設：今天）")
    g.add_argument("--source", default="auto", choices=["auto", "yfinance", "twse"], help="資料來源")
    g.add_argument("--offline", action="store_true", help="不連網，只使用本機快取")
    g.add_argument("--csv", help="改用自備 CSV 行情檔（只適用單一股票）")

    g = p.add_argument_group("交易設定")
    g.add_argument("--capital", type=float, default=1_000_000, help="初始資金（預設 1,000,000）")
    g.add_argument("--lot", type=int, default=1000, help="交易單位股數，1000=整張、1=零股（預設 1000）")
    g.add_argument("--position", type=float, default=1.0, help="每次進場投入現金比例 0~1（預設 1）")
    g.add_argument("--discount", type=float, default=1.0, help="手續費折扣，例如六折填 0.6（預設 1）")
    g.add_argument("--min-fee", type=float, default=20, help="最低手續費（預設 20 元）")
    g.add_argument("--tax", type=float, help="證交稅率（預設：股票 0.003、00 開頭 ETF 0.001）")
    g.add_argument("--slippage", type=float, default=0.0, help="單邊滑價比例，例如 0.001（預設 0）")
    g.add_argument("--stop-loss", type=float, help="停損比例，例如 0.1 = 10%%")
    g.add_argument("--take-profit", type=float, help="停利比例，例如 0.2 = 20%%")
    g.add_argument("--risk-free", type=float, default=0.0, help="無風險利率（年化，用於夏普比率，預設 0）")


def _config_kwargs(a) -> dict:
    return dict(capital=a.capital, lot=a.lot, position=a.position, discount=a.discount, min_fee=a.min_fee,
                tax=a.tax, slippage=a.slippage, stop_loss=a.stop_loss, take_profit=a.take_profit)


def cmd_strategies(a) -> int:
    for s in STRATEGIES.values():
        params = ", ".join(f"{k}={v}" for k, v in s.defaults.items()) or "（無參數）"
        print(f"{s.name:<10} {s.title}\n           {s.description}\n           預設參數：{params}\n")
    return 0


def cmd_fetch(a) -> int:
    ok = True
    for sym in a.symbols:
        try:
            _, info = datamod.get_history(sym, a.start, a.end, source=a.source)
            print(f"✔ {info}")
        except Exception as e:
            ok = False
            print(f"✘ {sym}：{e}", file=sys.stderr)
    print(f"資料已存到 {datamod.cache_dir().resolve()}")
    return 0 if ok else 1


def cmd_run(a) -> int:
    params = _parse_params(a.param)
    if a.csv and len(a.symbols) > 1:
        raise SystemExit("--csv 只能搭配單一股票")
    summary = []
    for sym in a.symbols:
        try:
            prices, info = load_prices(sym, a.start, a.end, csv=a.csv, source=a.source, offline=a.offline)
            bench, binfo = load_benchmark(prices, a.benchmark, csv=a.benchmark_csv, source=a.source,
                                          offline=a.offline)
            out = backtest(sym, a.strategy, params, prices=prices, bench=bench, bench_info=binfo,
                           risk_free=a.risk_free, report_dir=None if a.no_report else a.out,
                           **_config_kwargs(a))
        except Exception as e:
            print(f"✘ {sym}：{e}", file=sys.stderr)
            continue
        print(f"\n=== {datamod.normalize_symbol(sym)} · {STRATEGIES[a.strategy].title} "
              f"({', '.join(f'{k}={v}' for k, v in out.params.items()) or '無參數'}) ===")
        print(f"資料：{info}")
        print(f"基準：{binfo}")
        for label, val in format_metrics(out.metrics):
            print(f"  {_ljust(label, 26)}{val:>16}")
        if out.result.skipped_entries:
            print(f"  ⚠ 有 {out.result.skipped_entries} 次訊號因資金不足一個交易單位而未進場，可改用 --lot 1（零股）")
        if out.report_path:
            print(f"報告：{out.report_path.resolve()}")
        summary.append({"股票": datamod.normalize_symbol(sym), **{
            METRIC_LABELS[k][0]: format_value(k, out.metrics.get(k))
            for k in ("total_return", "cagr", "sharpe", "max_drawdown", "num_trades", "benchmark_total_return")}})
    if len(summary) > 1:
        print("\n=== 彙總比較 ===")
        print(pd.DataFrame(summary).to_string(index=False))
    return 0 if summary else 1


def _warn_skipped(count: int, scope: str) -> None:
    if count:
        print(f"\n⚠ {scope}有 {int(count)} 次買進訊號因資金不足一個交易單位而未進場（skipped 欄），"
              f"結果會被低估。可改用 --lot 1（零股）或提高 --capital 重跑。")


def cmd_optimize(a) -> int:
    grid = _parse_grid(a.grid)
    table, oos = optimize(a.symbol, a.strategy, grid, metric=a.metric, split=a.split, start=a.start,
                          end=a.end, csv=a.csv, source=a.source, offline=a.offline,
                          risk_free=a.risk_free, **_config_kwargs(a))
    show = table.head(a.top).copy()
    for k in ("total_return", "cagr", "max_drawdown", "win_rate"):
        show[k] = show[k].map(lambda v, k=k: format_value(k, v))
    for k in ("sharpe", "calmar"):
        show[k] = show[k].map(lambda v, k=k: format_value(k, v))
    print(f"=== {datamod.normalize_symbol(a.symbol)} · {a.strategy} 參數最佳化"
          f"（依 {a.metric} 排序，共 {len(table)} 組{'，樣本內' if a.split else ''}）===")
    print(show.to_string(index=False))
    if oos is not None:
        o = oos.copy()
        for k in ("total_return", "cagr", "max_drawdown", "sharpe"):
            o[k] = o[k].map(lambda v, k=k: format_value(k, v))
        print("\n=== 最佳參數在樣本外的表現 ===")
        print(o.to_string(index=False))
    _warn_skipped(show["skipped"].sum(), "樣本內前幾名")
    if oos is not None:
        oos_skipped = int(oos["skipped"].iloc[0])
        _warn_skipped(oos_skipped, "樣本外")
        if oos_skipped and oos["num_trades"].iloc[0] == 0:
            print("  樣本外 0 筆交易是因為買不起一個交易單位，不代表這組參數沒有訊號。")
    if a.save:
        table.to_csv(a.save, index=False, encoding="utf-8-sig")
        print(f"\n完整結果已存到 {a.save}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m twbacktest", description="台股回測系統")
    p.add_argument("-v", "--verbose", action="store_true", help="顯示詳細除錯訊息")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("strategies", help="列出內建策略與預設參數")
    s.set_defaults(func=cmd_strategies)

    s = sub.add_parser("fetch", help="下載並快取行情資料")
    s.add_argument("symbols", nargs="+", help="股票代號或 TAIEX（大盤），例如 2330 0050 TAIEX")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--source", default="auto", choices=["auto", "yfinance", "twse"])
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("run", help="執行回測")
    s.add_argument("symbols", nargs="+", help="一或多個股票代號，例如 2330 2317 0050")
    s.add_argument("-s", "--strategy", default="sma_cross", choices=list(STRATEGIES))
    s.add_argument("-p", "--param", action="append", help="策略參數，可重複，例如 -p fast=10 -p slow=60")
    s.add_argument("--benchmark", default="TAIEX", help="比較基準（預設 TAIEX，填 none 不比較）")
    s.add_argument("--benchmark-csv", help="改用自備 CSV 當比較基準")
    s.add_argument("--out", default="reports", help="報告輸出資料夾（預設 reports）")
    s.add_argument("--no-report", action="store_true", help="不產生報告檔，只在終端機顯示結果")
    _add_common(s)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("optimize", help="參數網格最佳化")
    s.add_argument("symbol")
    s.add_argument("-s", "--strategy", default="sma_cross", choices=list(STRATEGIES))
    s.add_argument("-g", "--grid", action="append", help="參數範圍，例如 -g fast=5,10,20 或 -g slow=40:120:20")
    s.add_argument("--metric", default="sharpe", help="排序依據：sharpe、cagr、total_return、calmar…")
    s.add_argument("--split", type=float, help="樣本內比例，例如 0.7（用後 30%% 資料驗證最佳參數）")
    s.add_argument("--top", type=int, default=10, help="顯示前幾名（預設 10）")
    s.add_argument("--save", help="把完整結果存成 CSV")
    _add_common(s)
    s.set_defaults(func=cmd_optimize)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if not args.verbose:
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        return args.func(args)
    except (ValueError, datamod.DataUnavailable) as e:
        print(f"錯誤：{e}", file=sys.stderr)
        return 1
