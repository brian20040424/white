"""行情資料取得。

取得順序：
1. Yahoo Finance（yfinance，還原權息後的價格，上市 .TW 與上櫃 .TWO 皆可）
2. 臺灣證券交易所官方網站（只有上市股票與加權指數，未還原權息）
3. 本機快取（data_cache/）

若指定期間（預設到今天）抓不到資料，會自動退回「近 5 年」、再退回「近 3 年」重試；
全部失敗時才使用本機快取中既有的資料。
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume"]
MIN_ROWS = 60  # 少於這個筆數視為抓取失敗

INDEX_ALIASES = {"TAIEX": "^TWII", "^TWII": "^TWII", "TWII": "^TWII", "大盤": "^TWII", "加權指數": "^TWII"}

TWSE_STOCK_URLS = [
    "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY",
    "https://www.twse.com.tw/exchangeReport/STOCK_DAY",
]
TWSE_INDEX_URLS = [
    "https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST",
    "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST",
]

# 證交所欄位名稱 → 標準欄位
TWSE_FIELD_MAP = {
    "日期": "date",
    "開盤價": "open", "開盤指數": "open",
    "最高價": "high", "最高指數": "high",
    "最低價": "low", "最低指數": "low",
    "收盤價": "close", "收盤指數": "close",
    "成交股數": "volume",
}

# 使用者自備 CSV 可接受的欄位名稱
CSV_FIELD_MAP = {
    **TWSE_FIELD_MAP,
    "date": "date", "datetime": "date", "time": "date",
    "open": "open", "high": "high", "low": "low", "close": "close",
    "adj close": "adj_close", "adj_close": "adj_close", "volume": "volume",
    "開盤": "open", "最高": "high", "最低": "low", "收盤": "close", "成交量": "volume",
}


class DataUnavailable(RuntimeError):
    pass


@dataclass
class FetchInfo:
    symbol: str
    source: str
    start: str
    end: str
    rows: int
    note: str = ""

    def __str__(self) -> str:
        msg = f"{self.symbol}：{self.start} ~ {self.end}，共 {self.rows} 筆，來源 {self.source}"
        return msg + (f"（{self.note}）" if self.note else "")


def cache_dir() -> Path:
    return Path(os.environ.get("TWBT_CACHE", "data_cache"))


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip()
    if s.upper() in INDEX_ALIASES or s in INDEX_ALIASES:
        return INDEX_ALIASES.get(s.upper(), INDEX_ALIASES.get(s))
    return s.upper()


def is_index(symbol: str) -> bool:
    return normalize_symbol(symbol).startswith("^")


def stock_code(symbol: str) -> str:
    return normalize_symbol(symbol).split(".")[0]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    if "volume" not in df:
        df["volume"] = 0.0
    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["close"])
    # 缺開高低時以收盤價補（例：只提供收盤價的資料）
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df["close"])
    df = df[df["close"] > 0]
    return df[~df.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------- Yahoo Finance
def fetch_yfinance(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf

    sym = normalize_symbol(symbol)
    candidates = [sym] if (sym.startswith("^") or "." in sym) else [f"{sym}.TW", f"{sym}.TWO"]
    last_err = None
    for c in candidates:
        try:
            raw = yf.download(c, start=start.isoformat(), end=(end + dt.timedelta(days=1)).isoformat(),
                              auto_adjust=True, progress=False, threads=False)
        except Exception as e:  # yfinance 在網路錯誤時可能拋出各種例外
            last_err = e
            continue
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=lambda x: str(x).lower())
        df = _clean(raw)
        if len(df):
            return df
    if last_err:
        log.debug("yfinance 失敗：%s", last_err)
    return pd.DataFrame(columns=COLUMNS)


# ---------------------------------------------------------------- 證交所
def roc_to_date(s: str) -> dt.date:
    """'113/01/02' → 2024-01-02（民國年 + 1911）。"""
    y, m, d = str(s).strip().split("/")
    return dt.date(int(y) + 1911, int(m), int(d))


def _to_float(v) -> float:
    s = str(v).replace(",", "").strip()
    if s in ("", "--", "---", "X0.00"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_twse_json(payload: dict) -> pd.DataFrame:
    """解析證交所 JSON（以欄位名稱對應，不依賴欄位順序）。"""
    if not payload or payload.get("stat") not in ("OK", "ok") or not payload.get("data"):
        return pd.DataFrame(columns=COLUMNS)
    fields = [TWSE_FIELD_MAP.get(str(f).strip()) for f in payload["fields"]]
    rows = []
    for r in payload["data"]:
        rec = {}
        for name, val in zip(fields, r):
            if name == "date":
                rec["date"] = roc_to_date(val)
            elif name:
                rec[name] = _to_float(val)
        if "date" in rec:
            rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return _clean(pd.DataFrame(rows).set_index("date"))


def _month_starts(start: dt.date, end: dt.date):
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        yield cur
        cur = dt.date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)


def fetch_twse(symbol: str, start: dt.date, end: dt.date, pause: float | None = None,
               session=None) -> pd.DataFrame:
    """逐月向證交所抓取。證交所有流量限制，預設每次請求間隔 3 秒。"""
    import requests

    sym = normalize_symbol(symbol)
    if sym.endswith(".TWO"):
        return pd.DataFrame(columns=COLUMNS)  # 上櫃股票不在證交所
    pause = float(os.environ.get("TWBT_TWSE_PAUSE", 3.0)) if pause is None else pause
    index_mode = sym == "^TWII"
    if sym.startswith("^") and not index_mode:
        return pd.DataFrame(columns=COLUMNS)
    urls = TWSE_INDEX_URLS if index_mode else TWSE_STOCK_URLS
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", "Mozilla/5.0 twbacktest")

    frames, failures = [], 0
    for i, month in enumerate(_month_starts(start, end)):
        params = {"date": month.strftime("%Y%m%d"), "response": "json"}
        if not index_mode:
            params["stockNo"] = stock_code(sym)
        payload = None
        for url in urls:
            try:
                resp = sess.get(url, params=params, timeout=15)
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as e:
                log.debug("證交所請求失敗 %s %s：%s", url, params, e)
        if payload is None:
            failures += 1
            if failures >= 3 and not frames:
                break  # 連續失敗，多半是網路不通，不再浪費時間
        else:
            frames.append(parse_twse_json(payload))
        if pause:
            time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames)
    df = _clean(df) if len(df) else df
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)]


SOURCES = {"yfinance": fetch_yfinance, "twse": fetch_twse}


# ---------------------------------------------------------------- 快取
def _cache_path(symbol: str, source: str) -> Path:
    name = normalize_symbol(symbol).replace("^", "IDX_")
    return cache_dir() / source / f"{name}.csv"


def save_cache(df: pd.DataFrame, symbol: str, source: str) -> Path:
    path = _cache_path(symbol, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = read_csv(path)
        df = pd.concat([old[~old.index.isin(df.index)], df]).sort_index()
    df.to_csv(path, index_label="date")
    return path


def load_cache(symbol: str, source: str | None = None) -> tuple[pd.DataFrame, str] | None:
    for src in ([source] if source else list(SOURCES)):
        path = _cache_path(symbol, src)
        if path.exists():
            df = read_csv(path)
            if len(df):
                return df, src
    return None


def read_csv(path: str | Path) -> pd.DataFrame:
    """讀取 CSV，支援英文（Date/Open/High/Low/Close/Volume）或中文（日期/開盤價…）欄位。"""
    raw = pd.read_csv(path)
    mapping = {}
    for c in raw.columns:
        key = str(c).strip()
        std = CSV_FIELD_MAP.get(key.lower(), CSV_FIELD_MAP.get(key))
        if std and std not in mapping.values():
            mapping[c] = std
    raw = raw.rename(columns=mapping)
    if "date" not in raw:
        raise ValueError(f"{path} 找不到日期欄位（需要 Date 或 日期）")
    if "close" not in raw:
        raise ValueError(f"{path} 找不到收盤價欄位（需要 Close 或 收盤價）")
    dates = raw["date"].astype(str)
    if dates.str.match(r"^\d{2,3}/\d{1,2}/\d{1,2}$").all():
        raw["date"] = [roc_to_date(d) for d in dates]
    raw["date"] = pd.to_datetime(raw["date"])
    for c in COLUMNS:
        if c in raw and not pd.api.types.is_numeric_dtype(raw[c]):
            raw[c] = raw[c].map(_to_float)
    return _clean(raw.set_index("date"))


# ---------------------------------------------------------------- 主要入口
def _years_ago(d: dt.date, years: int) -> dt.date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:  # 2/29
        return d.replace(year=d.year - years, day=28)


def _to_date(v) -> dt.date | None:
    if v is None or v == "":
        return None
    return pd.Timestamp(v).date()


def get_history(symbol: str, start=None, end=None, *, source: str = "auto",
                offline: bool = False, use_cache: bool = True) -> tuple[pd.DataFrame, FetchInfo]:
    """取得日線資料。回傳 (DataFrame, FetchInfo)。

    start 預設為 end 往前 5 年；end 預設為今天。
    """
    sym = normalize_symbol(symbol)
    end_d = _to_date(end) or dt.date.today()
    start_d = _to_date(start) or _years_ago(end_d, 5)
    if start_d > end_d:
        raise ValueError("start 不可晚於 end")
    sources = list(SOURCES) if source == "auto" else [source]
    for s in sources:
        if s not in SOURCES:
            raise ValueError(f"未知資料來源 {s}，可用：auto、{'、'.join(SOURCES)}")

    if not offline:
        today = dt.date.today()
        windows = [(start_d, end_d, "")]
        for yrs in (5, 3):
            w = (_years_ago(today, yrs), today, f"指定期間抓不到，改抓近 {yrs} 年資料")
            if w[:2] != (start_d, end_d):
                windows.append(w)
        for w_start, w_end, note in windows:
            for src in sources:
                try:
                    df = SOURCES[src](sym, w_start, w_end)
                except Exception as e:
                    log.debug("%s 取得 %s 失敗：%s", src, sym, e)
                    continue
                if len(df) >= MIN_ROWS:
                    if use_cache:
                        save_cache(df, sym, src)
                    return df, FetchInfo(sym, src, df.index[0].strftime("%Y-%m-%d"),
                                         df.index[-1].strftime("%Y-%m-%d"), len(df), note)

    cached = load_cache(sym, None if source == "auto" else source) if use_cache else None
    if cached:
        df, src = cached
        sliced = df.loc[pd.Timestamp(start_d):pd.Timestamp(end_d)]
        note = "離線模式，使用本機快取" if offline else "網路抓取失敗，使用本機快取"
        if len(sliced) < MIN_ROWS:
            sliced, note = df, note + "（指定期間資料不足，改用快取內全部資料）"
        return sliced, FetchInfo(sym, f"cache/{src}", sliced.index[0].strftime("%Y-%m-%d"),
                                 sliced.index[-1].strftime("%Y-%m-%d"), len(sliced), note)

    raise DataUnavailable(
        f"無法取得 {sym} 的行情資料。請確認網路可連到 query1.finance.yahoo.com 或 www.twse.com.tw，"
        f"或改用 --csv 匯入自己下載的資料。"
    )
