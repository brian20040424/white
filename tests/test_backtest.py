"""測試使用程式產生的價格序列（非真實行情），只用來驗證計算邏輯。"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from twbacktest import data as datamod
from twbacktest.cli import main
from twbacktest.costs import CostModel, default_tax_rate
from twbacktest.engine import BacktestConfig, run_backtest
from twbacktest.metrics import compute_metrics
from twbacktest.runner import backtest, optimize
from twbacktest.strategies import STRATEGIES, get_strategy


def make_prices(closes, opens=None, start="2022-01-03"):
    idx = pd.bdate_range(start, periods=len(closes))
    closes = np.asarray(closes, float)
    opens = closes if opens is None else np.asarray(opens, float)
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.01,
                         "low": np.minimum(opens, closes) * 0.99, "close": closes,
                         "volume": 1000.0}, index=idx)


def random_walk(n=800, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    opens = closes * (1 + rng.normal(0, 0.003, n))
    return make_prices(closes, opens)


# ---------------------------------------------------------------- 成本
def test_commission_minimum_and_floor():
    cm = CostModel()
    assert cm.commission(1000) == 20            # 1.425 元 → 最低 20 元
    assert cm.commission(1_000_000) == 1425
    assert CostModel(commission_discount=0.6).commission(1_000_000) == 855
    assert cm.tax(1_000_000) == 3000
    assert cm.tax(999) == 2                     # 2.997 → 無條件捨去


def test_etf_tax_rate():
    assert default_tax_rate("0050") == 0.001
    assert default_tax_rate("00878.TW") == 0.001
    assert default_tax_rate("2330") == 0.003


# ---------------------------------------------------------------- 引擎
def test_executes_next_day_open_with_exact_costs():
    closes = [100, 100, 100, 110, 110, 120]
    opens = [100, 100, 101, 110, 111, 120]
    df = make_prices(closes, opens)
    sig = pd.Series([0, 1, 1, 1, 0, 0], index=df.index, dtype=float)
    res = run_backtest(df, sig, BacktestConfig(initial_capital=1_000_000))

    t = res.trades.iloc[0]
    assert t["entry_date"] == df.index[2] and t["entry_price"] == 101  # 次日開盤
    assert t["exit_date"] == df.index[5] and t["exit_price"] == 120
    shares = 9000                                                     # 只能買整張
    assert t["shares"] == shares
    buy_fee = int(shares * 101 * 0.001425)
    sell_fee = int(shares * 120 * 0.001425)
    tax = int(shares * 120 * 0.003)
    expected_cash = 1_000_000 - shares * 101 - buy_fee + shares * 120 - sell_fee - tax
    assert res.equity["equity"].iloc[-1] == pytest.approx(expected_cash)
    assert t["pnl"] == pytest.approx(expected_cash - 1_000_000)


def test_no_lookahead_signal_on_last_day_not_executed():
    df = make_prices([100] * 5)
    sig = pd.Series([0, 0, 0, 0, 1], index=df.index, dtype=float)
    res = run_backtest(df, sig)
    assert res.trades.empty
    assert res.equity["equity"].iloc[-1] == 1_000_000


def test_odd_lot_and_insufficient_cash():
    df = make_prices([500] * 5)
    sig = pd.Series(1.0, index=df.index)
    res = run_backtest(df, sig, BacktestConfig(initial_capital=100_000, lot_size=1000))
    assert res.trades.empty and res.skipped_entries > 0
    res = run_backtest(df, sig, BacktestConfig(initial_capital=100_000, lot_size=1))
    assert res.trades.iloc[0]["shares"] == 199
    assert (res.equity["cash"] >= 0).all()


def test_stop_loss_exits_and_waits_for_reset():
    closes = [100, 100, 100, 85, 85, 85, 85]
    opens = [100, 100, 100, 95, 85, 85, 85]
    df = make_prices(closes, opens)
    df.loc[df.index[3], "low"] = 80
    sig = pd.Series(1.0, index=df.index)
    res = run_backtest(df, sig, BacktestConfig(stop_loss=0.1))
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert t["exit_reason"] == "stop_loss" and t["exit_price"] == pytest.approx(90)
    assert (res.equity["shares"].iloc[4:] == 0).all()  # 訊號未重置前不再進場


def test_stop_loss_gap_down_fills_at_open():
    df = make_prices([100, 100, 100, 85, 85], [100, 100, 100, 85, 85])
    res = run_backtest(df, pd.Series(1.0, index=df.index), BacktestConfig(stop_loss=0.1))
    assert res.trades.iloc[0]["exit_price"] == pytest.approx(85)


def test_open_position_listed_but_not_counted():
    df = random_walk(300)
    res = run_backtest(df, pd.Series(1.0, index=df.index))
    assert res.trades.iloc[-1]["exit_reason"] == "open"
    assert compute_metrics(res)["num_trades"] == 0


# ---------------------------------------------------------------- 策略與指標
@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_strategy_runs(name):
    df = random_walk()
    sig = get_strategy(name).signals(df)
    assert set(sig.unique()) <= {0.0, 1.0}
    res = run_backtest(df, sig)
    m = compute_metrics(res, benchmark=df["close"])
    assert m["max_drawdown"] <= 0
    assert (res.equity["cash"] >= -1e-6).all()


def test_strategy_rejects_unknown_param():
    with pytest.raises(ValueError):
        get_strategy("sma_cross").resolve_params({"fastt": 5})


def test_buy_hold_tracks_benchmark():
    df = random_walk(500)
    res = run_backtest(df, get_strategy("buy_hold").signals(df), BacktestConfig(lot_size=1))
    m = compute_metrics(res, benchmark=df["close"])
    assert m["beta"] == pytest.approx(1.0, abs=0.05)
    assert abs(m["excess_return"]) < 0.05


# ---------------------------------------------------------------- 資料
def test_parse_twse_json_by_field_name():
    payload = {
        "stat": "OK",
        "fields": ["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數"],
        "data": [["113/01/02", "26,059,058", "15,437,000", "590.00", "593.00", "589.00", "593.00", "+0.00", "24,452"],
                 ["113/01/03", "37,106,763", "21,618,000", "584.00", "585.00", "576.00", "578.00", "-15.00", "45,078"],
                 ["113/01/04", "--", "--", "--", "--", "--", "--", "", "0"]],
    }
    df = datamod.parse_twse_json(payload)
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df.loc["2024-01-03", "close"] == 578.0
    assert df.loc["2024-01-02", "volume"] == 26059058
    assert datamod.parse_twse_json({"stat": "很抱歉，沒有符合條件的資料!"}).empty


def test_read_csv_chinese_and_roc_dates(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("日期,開盤價,最高價,最低價,收盤價,成交股數\n113/01/03,584,585,576,\"1,578\",100\n"
                 "113/01/02,590,593,589,593,200\n", encoding="utf-8")
    df = datamod.read_csv(p)
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert df["close"].iloc[-1] == 1578


def test_get_history_falls_back_to_recent_years_then_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TWBT_CACHE", str(tmp_path))
    calls = []
    good = random_walk(800)

    def fake(symbol, start, end):
        calls.append((start, end))
        # 模擬：完整期間抓不到，只有近 5 年可抓
        return good if (end - start).days < 6 * 365 else pd.DataFrame(columns=datamod.COLUMNS)

    monkeypatch.setattr(datamod, "SOURCES", {"yfinance": fake})
    df, info = datamod.get_history("2330", start="2010-01-01")
    assert "近 5 年" in info.note and len(df) == 800
    assert calls[0][0] == dt.date(2010, 1, 1)

    monkeypatch.setattr(datamod, "SOURCES", {"yfinance": lambda *a: pd.DataFrame(columns=datamod.COLUMNS)})
    df2, info2 = datamod.get_history("2330", start="2022-01-01", end="2024-12-31")
    assert info2.source == "cache/yfinance" and len(df2) > 0

    with pytest.raises(datamod.DataUnavailable):
        datamod.get_history("9999")


# ---------------------------------------------------------------- 高階介面與 CLI
def test_backtest_with_csv_writes_report(tmp_path):
    df = random_walk(600)
    stock, bench = tmp_path / "s.csv", tmp_path / "b.csv"
    df.to_csv(stock, index_label="Date")
    (df * 0.5).to_csv(bench, index_label="Date")
    out = backtest("2330", "macd", csv=str(stock), benchmark="TAIEX", benchmark_csv=str(bench),
                   report_dir=tmp_path / "rep", lot=1)
    assert out.report_path.exists()
    for f in ("chart.png", "equity.csv", "trades.csv", "summary.json"):
        assert (out.report_path.parent / f).exists()
    assert "benchmark_total_return" in out.metrics


def test_optimize_with_split(tmp_path):
    df = random_walk(800)
    p = tmp_path / "s.csv"
    df.to_csv(p, index_label="Date")
    table, oos = optimize("2330", "sma_cross", {"fast": [5, 10, 60], "slow": [30, 60]},
                          csv=str(p), split=0.7, lot=1)
    assert len(table) == 4          # fast=60 的兩組（fast >= slow）無效被略過
    assert oos is not None and len(oos) == 1


def test_cli_run_and_optimize(tmp_path, capsys):
    df = random_walk(600)
    p = tmp_path / "s.csv"
    df.to_csv(p, index_label="Date")
    assert main(["run", "2330", "--csv", str(p), "--benchmark", "none", "-s", "rsi", "-p", "lower=25",
                 "--out", str(tmp_path / "r"), "--lot", "1"]) == 0
    captured = capsys.readouterr()
    assert "總報酬率" in captured.out
    assert "無法取得" not in captured.err   # --benchmark none 不應嘗試下載基準
    assert main(["optimize", "2330", "--csv", str(p), "-g", "fast=5:15:5", "-g", "slow=40,60",
                 "--split", "0.7", "--lot", "1"]) == 0
    assert "樣本外" in capsys.readouterr().out
