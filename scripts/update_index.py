"""
Nightly incremental update of the War-Peace Index.

Reads data/war_peace_index.csv, re-pulls GDELT for the last OVERLAP_DAYS before the
newest row through yesterday (UTC), splices the fresh counts in, recomputes every
derived column on a complete daily calendar, and writes the CSV back.

Pulling a short window keeps each request small, so GDELT throttling is rare. If a
night fails, the next night's window starts from the last good row and backfills
the gap automatically.

Methodology matches gdelt_news_daily.py (compact queries, label-based decoding,
low-volume rule, blank daily change after a missing day).
"""
from __future__ import annotations

import random
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "war_peace_index.csv"

API = "https://api.gdeltproject.org/api/v2/doc/doc"
OVERLAP_DAYS = 7              # always re-pull the last week so late GDELT revisions are picked up
LOOKBACK_DAYS = 21            # how far back to look for missing or incomplete days to backfill
INCOMPLETE_SHARE = 0.5        # a day with < 50% of normal total volume is treated as not finished
MIN_RELEVANT_ARTICLES = 50
MAX_TRIES = 3
MAX_WAIT = 120
REQUEST_TIMEOUT = 180       # long-window searches can take a couple of minutes to answer

CONTEXT = "(Ukraine OR Kyiv) (Russia OR Russian)"
PEACE = '(ceasefire OR "peace talks" OR "peace deal" OR truce OR armistice OR negotiations)'
WAR = "(invasion OR offensive OR shelling OR airstrike OR bombardment OR mobilization OR escalation)"
QUERIES = {"peace": f"{CONTEXT} {PEACE}", "war": f"{CONTEXT} {WAR}"}


class GdeltUnavailable(Exception):
    """GDELT is down, throttling, or returned nothing usable. Try again next run."""


def pull(query: str, start: str, end: str) -> pd.DataFrame:
    params = {"query": query, "mode": "timelinevolraw", "format": "csv",
              "startdatetime": start, "enddatetime": end, "timelinesmooth": "0"}
    wait = 30
    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = requests.get(API, params=params, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as exc:
            print(f"  [{attempt}] connection error {exc.__class__.__name__}; waiting {wait}s")
            time.sleep(wait); wait = min(wait * 2, MAX_WAIT); continue
        if r.status_code == 429:
            ra = r.headers.get("Retry-After", "")
            delay = min((int(ra) if ra.isdigit() else wait) + random.uniform(0, 10), MAX_WAIT)
            print(f"  [{attempt}] throttled; waiting {delay:.0f}s")
            time.sleep(delay); wait = min(wait * 2, MAX_WAIT); continue
        if r.status_code >= 500:
            print(f"  [{attempt}] GDELT server error {r.status_code}; waiting {wait}s")
            time.sleep(wait); wait = min(wait * 2, MAX_WAIT); continue
        if r.status_code != 200:
            raise GdeltUnavailable(f"HTTP {r.status_code}: {r.text[:200]!r}")
        text = r.text.lstrip("\ufeff\n\r \t")
        if not text:
            raise GdeltUnavailable("GDELT returned an empty response")
        if text.lower().startswith(("<html", "<!doctype")):
            raise GdeltUnavailable(f"GDELT returned a web page instead of data: {text[:200]!r}")
        df = pd.read_csv(StringIO(text))
        if df.shape[1] < 3:
            raise GdeltUnavailable(f"unexpected response: columns {list(df.columns)}, start {text[:200]!r}")
        return df
    raise GdeltUnavailable("GDELT did not answer after several tries (timeouts or throttling)")


def decode(df: pd.DataFrame, name: str) -> pd.DataFrame:
    d = df.iloc[:, :3].copy()
    d.columns = ["date", "series", "value"]
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d["series"] = d["series"].astype(str).str.strip()
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["date", "value"])
    labels = [x for x in d["series"].unique() if x and x.lower() != "nan"]
    norm = [l for l in labels if any(k in l.lower() for k in ("total", "all article", "all news"))]
    match = [l for l in labels if l not in norm]
    if len(norm) != 1 or len(match) != 1:
        raise RuntimeError(f"could not identify series for {name}: {labels}")
    # GDELT returns 15-minute or hourly bins for short windows; summing within each
    # UTC day gives the same daily totals regardless of the resolution it picks.
    p = d.pivot_table(index="date", columns="series", values="value", aggfunc="sum")
    return p[[match[0], norm[0]]].rename(columns={match[0]: name, norm[0]: f"norm_{name}"})


def main() -> int:
    old = pd.read_csv(CSV, parse_dates=["date"]).set_index("date")
    have = old["total_news"].dropna()
    last_good = have.index.max()
    yesterday = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)

    # Re-pull from the earliest recent day that is missing or looks incomplete, so a
    # GDELT outage of any length (up to LOOKBACK_DAYS) is backfilled once it recovers.
    recent = old.loc[old.index >= yesterday - pd.Timedelta(days=LOOKBACK_DAYS), "total_news"]
    recent = recent.reindex(pd.date_range(recent.index.min(), yesterday, freq="D"))
    med = have.iloc[-60:].median()
    weak = recent[recent.isna() | (recent < INCOMPLETE_SHARE * med)]
    win_start = last_good - pd.Timedelta(days=OVERLAP_DAYS)
    if len(weak):
        win_start = min(win_start, weak.index.min())

    start, end = win_start.strftime("%Y%m%d000000"), yesterday.strftime("%Y%m%d235959")
    print(f"Pulling {win_start.date()} to {yesterday.date()}")
    got = {}
    try:
        for i, (name, q) in enumerate(QUERIES.items()):
            if i: time.sleep(30)
            got[name] = decode(pull(q, start, end), name)
    except GdeltUnavailable as exc:
        # Not a bug in this script: GDELT is down or throttling. Leave the data as it
        # is and finish cleanly; the next scheduled run picks up where this left off.
        print(f"GDELT unavailable, data left unchanged: {exc}")
        return 0

    new = got["peace"].join(got["war"], how="outer")
    new = new[(new.index >= win_start) & (new.index <= yesterday)]
    new["total_news"] = new[["norm_peace", "norm_war"]].mean(axis=1)
    new = new[["peace", "war", "total_news"]].dropna()
    if new.empty:
        print("GDELT returned no rows for the window (likely a GDELT outage); leaving data unchanged.")
        return 0

    # Keep every day we already have, and overwrite only the days GDELT just returned.
    # A day GDELT leaves out of this response keeps its old value instead of going blank.
    counts = old[["peace", "war", "total_news"]].dropna()
    counts = pd.concat([counts[~counts.index.isin(new.index)], new]).sort_index()

    # Drop trailing days GDELT has not finished processing (total volume far below
    # normal). They are re-pulled on the next run instead of being published as real.
    ref = counts["total_news"].dropna().iloc[-90:-7].median()
    while len(counts) and not (counts["total_news"].iloc[-1] >= INCOMPLETE_SHARE * ref):
        print(f"  holding back {counts.index[-1].date()} (incomplete: {counts['total_news'].iloc[-1]})")
        counts = counts.iloc[:-1]

    cal = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    c = counts.reindex(cal)
    rel = c["peace"] + c["war"]
    ok = c["total_news"] > 0
    out = pd.DataFrame(index=cal)
    out["peace"] = c["peace"].round().astype("Int64")
    out["war"] = c["war"].round().astype("Int64")
    out["total_news"] = c["total_news"].round().astype("Int64")
    out["salience"] = (rel / c["total_news"]).where(ok)
    out["direction"] = ((c["peace"] - c["war"]) / rel.replace(0, np.nan)).where(ok)
    out["signed_intensity"] = ((c["peace"] - c["war"]) / c["total_news"]).where(ok)
    out["d_signed_intensity"] = out["signed_intensity"].diff()   # NaN across any missing day
    out["low_volume"] = rel.lt(MIN_RELEVANT_ARTICLES).astype("Int64").where(rel.notna())
    out.index.name = "date"
    out.to_csv(CSV, date_format="%Y-%m-%d", float_format="%.8g")
    print(f"Wrote {len(out)} rows through {out.index.max().date()} (previously {last_good.date()}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
