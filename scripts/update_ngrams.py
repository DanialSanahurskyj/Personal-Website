"""
War-Peace Index updater using GDELT Web NGrams 3.0 (replaces the stalled DOC search API).

Source files (about 192 per UTC day, ~27 MB each):
    https://data.gdeltproject.org/gdeltv3/webngrams/YYYYMMDDHHMMSS.webngrams.json.gz
Each line is one word occurrence: {"date", "ngram", "lang", "type", "pos", "pre", "post", "url"}.

Method (same keyword rules as the original index, applied per article URL):
    context : article contains ("ukraine" or "kyiv") AND ("russia" or "russian")
    peace   : ceasefire, truce, armistice, negotiations, "peace talks", "peace deal"
    war     : invasion, offensive, shelling, airstrike, bombardment, mobilization, escalation
    total   : number of distinct articles that day (normalizer)

Consistency with the original series:
    * Rows through API_END are the original DOC-API data and are never changed.
    * Raw NGrams counts are stored per day in data/ngrams_daily.csv.
    * Calibration: over an overlap window ending at API_END, NGrams peace and war shares are
      rescaled so their averages match the original series (k_peace, k_war). Results and the
      correlation between the two series are saved in data/ngrams_calibration.json.
    * Days after API_END are written to the main CSV only after calibration exists, with
      source = "ngrams".

Only complete days are published: a day counts as complete once GDELT has started
publishing the next day and the day has a normal number of files and articles.
Each run processes at most MAX_DAYS_PER_RUN days, so a backlog clears over several runs.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
MAIN_CSV = ROOT / "data" / "war_peace_index.csv"
RAW_CSV = ROOT / "data" / "ngrams_daily.csv"
CAL_FILE = ROOT / "data" / "ngrams_calibration.json"

BASE = "https://data.gdeltproject.org/gdeltv3/webngrams/{s}.webngrams.json.gz"
API_END = pd.Timestamp("2026-10-01")      # last day of the original DOC-API series
CAL_DAYS = 35                              # overlap window for calibration (ending API_END)
MIN_CAL_DAYS = 20                          # matched days needed (the original series has outage gaps)
MAX_DAYS_PER_RUN = 6                       # ~20 min per day of data; keeps a run near 2 hours
WORKERS = 4
MIN_RELEVANT_ARTICLES = 50
MIN_FILES_SHARE = 0.8                      # need >= 80% of a normal day's file count
MIN_DOCS_SHARE = 0.5                       # and >= 50% of a normal day's article count
TYPICAL_FILES = 192

UKRAINE = {b"ukraine", b"kyiv"}
RUSSIA = {b"russia", b"russian"}
PEACE_WORDS = {b"ceasefire", b"truce", b"armistice", b"negotiations"}
WAR_WORDS = {b"invasion", b"offensive", b"shelling", b"airstrike", b"bombardment",
             b"mobilization", b"escalation"}
U, R, P, W = 1, 2, 4, 8


def _field(line: bytes, name: bytes, last: bool = False) -> bytes:
    """Fast extraction of a string field from one JSON line without a full parse.
    Works with or without spaces around the colon ("url":"x" and "url": "x").
    `last=True` takes the final occurrence (the url field ends each record)."""
    key = b'"' + name + b'"'
    i = line.rfind(key) if last else line.find(key)
    if i < 0:
        return b""
    i = line.find(b":", i + len(key))
    if i < 0:
        return b""
    i += 1
    while i < len(line) and line[i:i + 1] in b" \t":
        i += 1
    if line[i:i + 1] != b'"':
        return b""
    j = i + 1
    while True:
        j = line.find(b'"', j)
        if j < 0:
            return b""
        if line[j - 1:j] != b"\\":
            return line[i + 1:j]
        j += 1


def process_file(stamp: str) -> tuple[str, int, dict, list]:
    """Return (stamp, status, {url: flags} for keyword hits, [all urls])."""
    url = BASE.format(s=stamp)
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=180, stream=True)
            break
        except requests.RequestException:
            time.sleep(10)
    else:
        return stamp, -1, {}, []
    if r.status_code != 200:
        return stamp, r.status_code, {}, []
    r.raw.decode_content = False
    hits: dict[bytes, int] = {}
    urls: set[bytes] = set()
    nlines = 0
    try:
        with gzip.GzipFile(fileobj=r.raw) as gz:
            for line in gz:
                nlines += 1
                u = _field(line, b"url", last=True)
                w = _field(line, b"ngram")
                post = None
                if not u or not w:
                    try:                       # fallback: full JSON parse
                        rec = json.loads(line)
                        u = str(rec.get("url", "")).encode()
                        w = str(rec.get("ngram", "")).encode()
                        post = str(rec.get("post", "")).encode()
                    except (ValueError, AttributeError):
                        continue
                if not u:
                    continue
                urls.add(u)
                w = w.lower().strip(b".,;:!?'()\\")
                if w in UKRAINE: f = U
                elif w in RUSSIA: f = R
                elif w in PEACE_WORDS: f = P
                elif w in WAR_WORDS: f = W
                elif w == b"peace":
                    post = (post if post is not None else _field(line, b"post")).lower().lstrip()
                    f = P if post.startswith((b"talks", b"deal")) else 0
                else:
                    f = 0
                if f:
                    hits[u] = hits.get(u, 0) | f
    except (OSError, EOFError):
        return stamp, -2, {}, []
    if nlines and not urls:
        print(f"  {stamp}: {nlines} lines but no articles could be read (format changed?)")
        return stamp, -3, {}, []
    return stamp, 200, {k.decode("utf-8", "replace"): v for k, v in hits.items()}, \
        [x.decode("utf-8", "replace") for x in urls]


def file_exists(stamp: str) -> bool:
    try:
        return requests.head(BASE.format(s=stamp), timeout=30).status_code == 200
    except requests.RequestException:
        return False


def next_day_started(day: pd.Timestamp) -> bool:
    nxt = day + pd.Timedelta(days=1)
    return any(file_exists((nxt + pd.Timedelta(minutes=m)).strftime("%Y%m%d%H%M%S"))
               for m in range(0, 60))


def count_day(day: pd.Timestamp, pool: ProcessPoolExecutor) -> dict:
    stamps = [(day + pd.Timedelta(minutes=m)).strftime("%Y%m%d%H%M%S") for m in range(1440)]
    t0 = time.time()
    flags: dict[str, int] = {}
    all_urls: set[str] = set()
    files = errors = 0
    for stamp, status, hits, urls in pool.map(process_file, stamps, chunksize=8):
        if status == 200:
            files += 1
            all_urls.update(urls)
            for u, f in hits.items():
                flags[u] = flags.get(u, 0) | f
        elif status != 404:
            errors += 1
    ctx = [f for f in flags.values() if f & U and f & R]
    row = {
        "date": day, "files": files, "errors": errors, "total_news": len(all_urls),
        "context": len(ctx), "peace": sum(1 for f in ctx if f & P),
        "war": sum(1 for f in ctx if f & W),
    }
    print(f"  {day.date()}: {files} files, {errors} errors, {row['total_news']:,} articles, "
          f"Ukraine+Russia {row['context']:,}, peace {row['peace']:,}, war {row['war']:,} "
          f"({time.time()-t0:,.0f}s)")
    return row


def load_raw() -> pd.DataFrame:
    if RAW_CSV.exists():
        raw = pd.read_csv(RAW_CSV, parse_dates=["date"]).set_index("date")
        # A day with no articles was a failed read, never a real result: redo it.
        if "complete" in raw.columns:
            raw.loc[raw["total_news"].fillna(0) <= 0, "complete"] = 0
        return raw
    return pd.DataFrame(columns=["files", "errors", "total_news", "context", "peace", "war"])


def is_complete(row: dict, raw: pd.DataFrame) -> bool:
    good = raw[(raw["errors"] == 0) & (raw["total_news"] > 0)] if len(raw) else raw
    typ_files = good["files"].median() if len(good) >= 3 else TYPICAL_FILES
    typ_docs = good["total_news"].median() if len(good) >= 3 else 0
    return (row["errors"] == 0 and row["total_news"] > 0 and row["context"] > 0
            and row["files"] >= MIN_FILES_SHARE * typ_files
            and row["total_news"] >= MIN_DOCS_SHARE * typ_docs)


def calibrate(raw: pd.DataFrame, main: pd.DataFrame) -> dict | None:
    start = API_END - pd.Timedelta(days=CAL_DAYS - 1)
    ng = raw.loc[start:API_END]
    api = main.loc[start:API_END, ["peace", "war", "total_news"]].dropna()
    j = api.join(ng[["peace", "war", "total_news"]], rsuffix="_ng", how="inner")
    j = j[(j["total_news"] > 0) & (j["total_news_ng"] > 0)]
    if len(j) < MIN_CAL_DAYS:
        print(f"Calibration waiting: {len(j)} of {MIN_CAL_DAYS} matched overlap days processed.")
        return None
    ps_api, ws_api = j["peace"] / j["total_news"], j["war"] / j["total_news"]
    ps_ng, ws_ng = j["peace_ng"] / j["total_news_ng"], j["war_ng"] / j["total_news_ng"]
    k_p, k_w = float(ps_api.mean() / ps_ng.mean()), float(ws_api.mean() / ws_ng.mean())
    si_api, si_ng = ps_api - ws_api, k_p * ps_ng - k_w * ws_ng
    cal = {
        "overlap_start": str(j.index.min().date()), "overlap_end": str(j.index.max().date()),
        "overlap_days": int(len(j)), "k_peace": k_p, "k_war": k_w,
        "corr_signed_intensity_daily": float(si_api.corr(si_ng)),
        "corr_signed_intensity_7d": float(si_api.rolling(7, min_periods=5).mean()
                                          .corr(si_ng.rolling(7, min_periods=5).mean())),
        "corr_direction_daily": float(((j["peace"] - j["war"]) / (j["peace"] + j["war"]))
                                      .corr((j["peace_ng"] - j["war_ng"]) / (j["peace_ng"] + j["war_ng"]))),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    CAL_FILE.write_text(json.dumps(cal, indent=2))
    print("Calibration:", json.dumps(cal, indent=2))
    return cal


def write_main(main: pd.DataFrame, raw: pd.DataFrame, cal: dict) -> None:
    api_rows = main.loc[:API_END, ["peace", "war", "total_news"]].copy()
    api_rows["source"] = np.where(api_rows["total_news"].notna(), "api", "")
    ng = raw.loc[raw.index > API_END]
    ng = ng[ng["complete"] == 1]
    new = pd.DataFrame({
        "peace": (ng["peace"] * cal["k_peace"]).round(),
        "war": (ng["war"] * cal["k_war"]).round(),
        "total_news": ng["total_news"].astype(float),
        "source": "ngrams",
    }, index=ng.index)
    counts = pd.concat([api_rows, new]).sort_index()
    counts = counts[~counts.index.duplicated(keep="first")]
    idx = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    c = counts.reindex(idx)
    rel = c["peace"] + c["war"]
    ok = c["total_news"] > 0
    out = pd.DataFrame(index=idx)
    out["peace"] = c["peace"].round().astype("Int64")
    out["war"] = c["war"].round().astype("Int64")
    out["total_news"] = c["total_news"].round().astype("Int64")
    out["salience"] = (rel / c["total_news"]).where(ok)
    out["direction"] = ((c["peace"] - c["war"]) / rel.replace(0, np.nan)).where(ok)
    out["signed_intensity"] = ((c["peace"] - c["war"]) / c["total_news"]).where(ok)
    out["d_signed_intensity"] = out["signed_intensity"].diff()
    out["low_volume"] = rel.lt(MIN_RELEVANT_ARTICLES).astype("Int64").where(rel.notna())
    out["source"] = c["source"].fillna("")
    out.index.name = "date"
    out.to_csv(MAIN_CSV, date_format="%Y-%m-%d", float_format="%.8g")
    print(f"Main CSV written through {out.index.max().date()} ({len(new)} NGrams days).")


def days_to_do(raw: pd.DataFrame, yesterday: pd.Timestamp) -> list[pd.Timestamp]:
    cal_days = list(pd.date_range(API_END - pd.Timedelta(days=CAL_DAYS - 1), API_END))
    new_days = list(pd.date_range(API_END + pd.Timedelta(days=1), yesterday))
    done = set(raw.index[raw.get("complete", pd.Series(dtype=int)) == 1]) if len(raw) else set()
    # Newest days first so the site catches up quickly, then calibration days.
    todo = [d for d in reversed(new_days) if d not in done] + [d for d in cal_days if d not in done]
    return todo


def save_raw(raw: pd.DataFrame) -> None:
    raw = raw.sort_index()
    raw.index.name = "date"
    raw.to_csv(RAW_CSV, date_format="%Y-%m-%d")


def finish(raw: pd.DataFrame, main_df: pd.DataFrame, yesterday: pd.Timestamp) -> int:
    if CAL_FILE.exists():
        cal = json.loads(CAL_FILE.read_text())
    elif any(d <= API_END for d in days_to_do(raw, yesterday)):
        left = sum(d <= API_END for d in days_to_do(raw, yesterday))
        print(f"Calibration waiting: {left} overlap days still to process.")
        return 0
    else:
        cal = calibrate(raw, main_df)
    if cal is None:
        print("Main CSV unchanged until calibration is ready.")
        return 0
    write_main(main_df, raw, cal)
    return 0


def cli() -> int:
    """Backfill helpers used by the parallel workflow:
       --list-days            print the days still to process as a JSON list
       --day YYYY-MM-DD DIR   process one day and write DIR/part-YYYY-MM-DD.csv
       --merge DIR            merge all part files into ngrams_daily.csv, then calibrate/publish
    """
    args = sys.argv[1:]
    yesterday = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    raw = load_raw()
    if "complete" not in raw.columns:
        raw["complete"] = pd.Series(dtype=int)
    if args[0] == "--list-days":
        todo = [d for d in days_to_do(raw, yesterday) if d < yesterday or next_day_started(d)]
        print(json.dumps([str(d.date()) for d in todo]))
        return 0
    if args[0] == "--day":
        day, out_dir = pd.Timestamp(args[1]), Path(args[2])
        out_dir.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(WORKERS) as pool:
            row = count_day(day, pool)
        pd.DataFrame([row]).to_csv(out_dir / f"part-{day.date()}.csv", index=False, date_format="%Y-%m-%d")
        return 0
    if args[0] == "--merge":
        parts = sorted(Path(args[1]).rglob("part-*.csv"))
        print(f"Merging {len(parts)} day files.")
        for f in parts:
            row = pd.read_csv(f, parse_dates=["date"]).iloc[0].to_dict()
            row["complete"] = int(is_complete(row, raw))
            if not row["complete"]:
                print(f"  {row['date'].date()}: incomplete, the nightly run will retry it.")
            keys = [k for k in row if k != "date"]
            raw.loc[row["date"], keys] = [row[k] for k in keys]
        save_raw(raw)
        main_df = pd.read_csv(MAIN_CSV, parse_dates=["date"]).set_index("date")
        return finish(raw, main_df, yesterday)
    print(cli.__doc__); return 2


def main() -> int:
    if len(sys.argv) > 1:
        return cli()
    main_df = pd.read_csv(MAIN_CSV, parse_dates=["date"]).set_index("date")
    raw = load_raw()
    if "complete" not in raw.columns:
        raw["complete"] = pd.Series(dtype=int)
    yesterday = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    todo = days_to_do(raw, yesterday)
    print(f"{len(todo)} days waiting; processing up to {MAX_DAYS_PER_RUN} this run.")

    with ProcessPoolExecutor(WORKERS) as pool:
        for day in todo[:MAX_DAYS_PER_RUN]:
            if day >= yesterday and not next_day_started(day):
                print(f"  {day.date()}: GDELT has not finished this day yet; skipping.")
                continue
            row = count_day(day, pool)
            row["complete"] = int(is_complete(row, raw))
            if not row["complete"]:
                print(f"  {day.date()}: incomplete, will retry next run.")
            keys = [k for k in row if k != "date"]
            raw.loc[day, keys] = [row[k] for k in keys]
            save_raw(raw)   # save progress after every day
            raw = load_raw()

    return finish(raw, main_df, yesterday)


if __name__ == "__main__":
    sys.exit(main())
