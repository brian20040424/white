"""輸出回測報告：圖表 (PNG)、淨值與交易明細 (CSV)、指標 (JSON) 與單檔 HTML 報告。"""
from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .engine import BacktestResult  # noqa: E402
from .metrics import drawdown_series, format_metrics  # noqa: E402

EXIT_REASONS = {"signal": "策略訊號", "stop_loss": "停損", "take_profit": "停利", "open": "期末持有中（估算）"}


def plot(result: BacktestResult, title: str, benchmark: pd.Series | None, path: Path) -> Path:
    eq = result.equity
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 2, 1]})
    # 圖表標籤使用英文，避免系統缺少中文字型時顯示成方塊
    ax = axes[0]
    ax.plot(eq.index, eq["close"], color="#555", lw=1, label="Close")
    t = result.trades
    if len(t):
        ax.scatter(t["entry_date"], t["entry_price"], marker="^", color="#d62728", s=50, label="Buy", zorder=3)
        closed = t[t["exit_reason"] != "open"]
        ax.scatter(closed["exit_date"], closed["exit_price"], marker="v", color="#2ca02c", s=50,
                   label="Sell", zorder=3)
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(eq.index, eq["equity"], color="#1f77b4", lw=1.5, label="Strategy")
    if benchmark is not None:
        b = benchmark.reindex(eq.index).ffill().dropna()
        if len(b):
            ax.plot(b.index, b / b.iloc[0] * result.config.initial_capital, color="#ff7f0e",
                    lw=1.2, label="Benchmark")
    ax.set_ylabel("Equity (TWD)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    dd = drawdown_series(eq["equity"]) * 100
    ax.fill_between(dd.index, dd, 0, color="#d62728", alpha=0.35)
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _json_safe(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def write_report(result: BacktestResult, metrics: dict, out_dir: Path, title: str,
                 meta: dict, benchmark: pd.Series | None = None, chart_title: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = plot(result, chart_title or title, benchmark, out_dir / "chart.png")
    result.equity.to_csv(out_dir / "equity.csv")
    trades = result.trades.copy()
    trades["exit_reason"] = trades["exit_reason"].map(lambda r: EXIT_REASONS.get(r, r))
    trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "metrics": {k: _json_safe(v) for k, v in metrics.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    img = base64.b64encode(png.read_bytes()).decode()
    metric_rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>"
                          for k, v in format_metrics(metrics))
    meta_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                        for k, v in meta.items())
    if len(trades):
        show = trades.copy()
        for c in ("entry_date", "exit_date"):
            show[c] = pd.to_datetime(show[c]).dt.strftime("%Y-%m-%d")
        for c in ("entry_price", "exit_price"):
            show[c] = show[c].map(lambda x: f"{x:,.2f}")
        for c in ("buy_cost", "sell_cost", "pnl"):
            show[c] = show[c].map(lambda x: f"{x:,.0f}")
        show["return_pct"] = show["return_pct"].map(lambda x: f"{x * 100:.2f}%")
        show.columns = ["進場日", "進場價", "出場日", "出場價", "股數", "買進手續費", "賣出費用",
                        "損益", "報酬率", "持有天數", "出場原因"]
        trade_html = show.to_html(index=False, border=0, classes="trades")
    else:
        trade_html = "<p>回測期間沒有任何交易。</p>"

    page = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:"Noto Sans TC","Microsoft JhengHei","PingFang TC",sans-serif;margin:0 auto;max-width:1100px;
padding:16px;color:#222;background:#fff}}
h1{{font-size:1.4rem}} h2{{font-size:1.1rem;margin-top:2rem}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th,td{{padding:6px 10px;border-bottom:1px solid #eee;text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}}
img{{max-width:100%;height:auto}} .wrap{{overflow-x:auto}}
.note{{color:#666;font-size:.85rem}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<img src="data:image/png;base64,{img}" alt="回測圖表">
<div class="grid">
<div><h2>績效指標</h2><table>{metric_rows}</table></div>
<div><h2>回測設定</h2><table>{meta_rows}</table></div>
</div>
<h2>交易明細</h2><div class="wrap">{trade_html}</div>
<p class="note">回測結果僅反映歷史資料在指定假設下的表現，不代表未來績效，也不構成投資建議。</p>
</body></html>"""
    path = out_dir / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
