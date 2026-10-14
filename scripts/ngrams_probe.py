"""
One-time test of GDELT's Web NGrams files as a source for the War-Peace Index.

It samples minute files from one recent UTC day, applies the index's exact keyword
rules at the article level, and reports file sizes, speed, article counts and matches.
It does NOT change data/war_peace_index.csv.

Files (published every minute, per GDELT):
  https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/YYYYMMDDHHMM00.ngrams.txt.gz
      tab-separated: DOCID <tab> QUADGRAM <tab> COUNT   (DOCID restarts in each file)
  https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/YYYYMMDDHHMM00.toc.json.gz
      one JSON object per article: ID, date, lang, title, url, ...
"""
from __future__ import annotations

import gzip
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import pandas as pd
import requests

BASE = "https://data.gdeltproject.org/gdeltv5/weblegacy/ngrams/"
SAMPLE_EVERY_MINUTES = 15          # sample 96 of the day's 1,440 minutes
WORKERS = 8

# Same rules as the original index (case-insensitive, whole words).
UKRAINE = {"ukraine", "kyiv"}
RUSSIA = {"russia", "russian"}
PEACE_WORDS = {"ceasefire", "truce", "armistice", "negotiations"}
PEACE_PHRASES = ["peace talks", "peace deal"]
WAR_WORDS = {"invasion", "offensive", "shelling", "airstrike", "bombardment", "mobilization", "escalation"}


def fetch(url: str) -> tuple[bytes | None, int]:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(5); continue
        if r.status_code == 200:
            return r.content, 200
        if r.status_code == 404:
            return None, 404
        time.sleep(5)
    return None, -1


def process_minute(stamp: str) -> dict:
    out = {"stamp": stamp, "ok": False, "bytes": 0, "docs": 0, "ctx": 0, "peace": 0, "war": 0,
           "langs": Counter(), "lines": 0, "status": None}
    raw, status = fetch(f"{BASE}{stamp}.ngrams.txt.gz")
    out["status"] = status
    if raw is None:
        return out
    out["bytes"] += len(raw)
    flags: dict[str, set] = {}
    with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            out["lines"] += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            doc, gram = parts[0], " " + parts[1].lower() + " "
            fl = flags.setdefault(doc, set())
            words = set(gram.split())
            if words & UKRAINE: fl.add("u")
            if words & RUSSIA: fl.add("r")
            if words & PEACE_WORDS or any(f" {p} " in gram for p in PEACE_PHRASES): fl.add("p")
            if words & WAR_WORDS: fl.add("w")
    out["docs"] = len(flags)
    for fl in flags.values():
        if "u" in fl and "r" in fl:
            out["ctx"] += 1
            out["peace"] += "p" in fl
            out["war"] += "w" in fl
    toc, _ = fetch(f"{BASE}{stamp}.toc.json.gz")
    if toc:
        out["bytes"] += len(toc)
        for ln in gzip.decompress(toc).decode("utf-8", "replace").splitlines():
            try:
                out["langs"][json.loads(ln).get("lang", "?")] += 1
            except json.JSONDecodeError:
                pass
    out["ok"] = True
    return out


def main() -> int:
    day = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    if len(sys.argv) > 1:
        day = pd.Timestamp(sys.argv[1])
    stamps = [(day + pd.Timedelta(minutes=m)).strftime("%Y%m%d%H%M00")
              for m in range(0, 1440, SAMPLE_EVERY_MINUTES)]
    print(f"Probing {len(stamps)} minute files from {day.date()} (every {SAMPLE_EVERY_MINUTES} min)")
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(process_minute, stamps))
    secs = time.time() - t0

    got = [r for r in res if r["ok"]]
    status = Counter(r["status"] for r in res)
    print(f"HTTP results: {dict(status)}  (200 = file found, 404 = no file for that minute)")
    if not got:
        print("No files retrieved. Check the URL pattern or the date."); return 1
    tot = {k: sum(r[k] for r in got) for k in ("bytes", "docs", "ctx", "peace", "war", "lines")}
    langs = Counter()
    for r in got: langs.update(r["langs"])
    print(f"Files found: {len(got)} of {len(stamps)}   time: {secs:.0f}s")
    print(f"Sample: {tot['bytes']/1e6:,.1f} MB downloaded, {tot['lines']:,} ngram lines, {tot['docs']:,} articles")
    print(f"Sample matches: Ukraine+Russia {tot['ctx']:,} | peace {tot['peace']:,} | war {tot['war']:,}")
    full_mb = tot["bytes"] / 1e6 * SAMPLE_EVERY_MINUTES * len(stamps) / max(len(got), 1)
    print(f"Estimated full day: ~{full_mb:,.0f} MB, ~{secs*SAMPLE_EVERY_MINUTES/60:,.0f} min to process here")
    print(f"Estimated full-day articles: ~{tot['docs']*SAMPLE_EVERY_MINUTES*len(stamps)/max(len(got),1):,.0f}")
    print("Languages in sample:", ", ".join(f"{k}:{v}" for k, v in langs.most_common(12)))
    if tot["ctx"]:
        print(f"Sample signed share (peace - war) / Ukraine+Russia articles: "
              f"{(tot['peace']-tot['war'])/tot['ctx']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
