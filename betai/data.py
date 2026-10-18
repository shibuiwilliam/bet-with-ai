"""Download and cache the public datasets used by the real-data studies.

- football-data.co.uk: results, shots and 1X2 odds of European leagues.
- Kenneth R. French Data Library: daily US market factor and 49 industry portfolios.

Files are cached under data/ and recorded with their SHA-256 in data/MANIFEST.json.
The sources keep growing (the current football season, the monthly French update),
so every loader cuts the data at the end date of the published run. A fresh download
then reproduces the published results as long as past rows are not revised; when a
downloaded file differs from the recorded hash, a note is printed.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
import zipfile
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from .common import DATA

FOOTBALL_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"

# Divisions with odds from many bookmakers. Seasons 1718-1819 are only used to warm up ratings.
DIVISIONS = [
    "E0",
    "E1",
    "E2",
    "E3",
    "SC0",
    "D1",
    "D2",
    "I1",
    "I2",
    "SP1",
    "SP2",
    "F1",
    "F2",
    "N1",
    "B1",
    "P1",
    "T1",
    "G1",
]
WARMUP_SEASONS = ["1718", "1819"]
ODDS_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]

# Last dates in the data used for the published outputs.
FOOTBALL_END = pd.Timestamp("2026-10-14")
FRENCH_END = pd.Timestamp("2026-08-31")

MANIFEST = DATA / "MANIFEST.json"


def _fetch(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(
        url, headers={"User-Agent": "bet-with-ai/0.1 (research; +https://github.com/shibuiwilliam/bet-with-ai)"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            dest.write_bytes(body)
            _record(url, dest, body)
            time.sleep(0.3)  # be polite to the hosts
            return
        except Exception as e:  # noqa: BLE001 - retry any network error
            if attempt == 2:
                raise RuntimeError(f"download failed: {url}: {e}") from e
            time.sleep(2 * (attempt + 1))


def _record(url: str, dest, body: bytes) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    key = str(dest.relative_to(DATA))
    sha = hashlib.sha256(body).hexdigest()
    if key in manifest and manifest[key]["sha256"] != sha:
        print(
            f"  note: {key} differs from the version recorded in MANIFEST.json "
            f"(the source was revised); results may differ from outputs/"
        )
    manifest[key] = {
        "url": url,
        "sha256": sha,
        "bytes": len(body),
        "downloaded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Football
# ---------------------------------------------------------------------------

ODDS_COLS = {
    # name: (home, draw, away) columns
    "avg_open": ("AvgH", "AvgD", "AvgA"),
    "max_open": ("MaxH", "MaxD", "MaxA"),
    "b365_open": ("B365H", "B365D", "B365A"),
    "ps_open": ("PSH", "PSD", "PSA"),
    "avg_close": ("AvgCH", "AvgCD", "AvgCA"),
    "max_close": ("MaxCH", "MaxCD", "MaxCA"),
    "b365_close": ("B365CH", "B365CD", "B365CA"),
    "ps_close": ("PSCH", "PSCD", "PSCA"),
}


def download_football() -> None:
    for season in WARMUP_SEASONS + ODDS_SEASONS:
        for div in DIVISIONS:
            _fetch(FOOTBALL_URL.format(season=season, div=div), DATA / "football" / season / f"{div}.csv")


def _parse_dates(s: pd.Series) -> pd.Series:
    """football-data uses dd/mm/yyyy, with dd/mm/yy in some older files."""
    s = s.astype(str).str.strip()
    long = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return long.fillna(short)


def load_football() -> pd.DataFrame:
    """One row per match, sorted by date, with results, shots and odds arrays."""
    cache = DATA / "football.pkl"
    if cache.exists():
        df = pd.read_pickle(cache)
        return df[df["date"] <= FOOTBALL_END].reset_index(drop=True)
    download_football()
    frames = []
    for season in WARMUP_SEASONS + ODDS_SEASONS:
        for div in DIVISIONS:
            path = DATA / "football" / season / f"{div}.csv"
            raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
            df = pd.DataFrame(
                {
                    "season": season,
                    "div": div,
                    "date": _parse_dates(raw["Date"]),
                    "home": raw["HomeTeam"].astype(str).str.strip(),
                    "away": raw["AwayTeam"].astype(str).str.strip(),
                    "hg": pd.to_numeric(raw["FTHG"], errors="coerce"),
                    "ag": pd.to_numeric(raw["FTAG"], errors="coerce"),
                    "ftr": raw["FTR"].astype(str).str.strip(),
                }
            )
            for c_src, c_dst in [("HS", "hs"), ("AS", "as_"), ("HST", "hst"), ("AST", "ast")]:
                df[c_dst] = pd.to_numeric(raw[c_src], errors="coerce") if c_src in raw else np.nan
            for name, cols in ODDS_COLS.items():
                for side, c in zip("hda", cols, strict=True):
                    df[f"{name}_{side}"] = pd.to_numeric(raw[c], errors="coerce") if c in raw else np.nan
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["ftr"].isin(["H", "D", "A"]) & out["date"].notna()].copy()
    out["y"] = out["ftr"].map({"H": 0, "D": 1, "A": 2}).astype(int)
    out = out.sort_values(["date", "div", "home"]).reset_index(drop=True)
    out.to_pickle(cache)
    return out[out["date"] <= FOOTBALL_END].reset_index(drop=True)


def odds(df: pd.DataFrame, name: str) -> np.ndarray:
    """(n, 3) array of decimal odds for home/draw/away (NaN where missing)."""
    return df[[f"{name}_h", f"{name}_d", f"{name}_a"]].to_numpy(dtype=float)


def devig(o: np.ndarray, method: str = "power") -> np.ndarray:
    """Remove the bookmaker margin from (n, 3) decimal odds.

    "multiplicative" rescales the implied probabilities 1/o to sum to 1. "power"
    finds k with sum((1/o)^k) = 1, which takes more margin from long odds and so
    corrects for the favourite-longshot bias. Rows with missing odds give NaN."""
    inv = 1.0 / o
    if method == "multiplicative":
        return inv / inv.sum(axis=1, keepdims=True)
    lo, hi = np.full(len(o), 0.5), np.full(len(o), 3.0)
    for _ in range(60):  # bisection; sum((1/o)^k) decreases in k
        k = (lo + hi) / 2
        over = (inv ** k[:, None]).sum(axis=1) > 1
        lo, hi = np.where(over, k, lo), np.where(over, hi, k)
    p = inv ** ((lo + hi) / 2)[:, None]
    return p / p.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Kenneth R. French Data Library
# ---------------------------------------------------------------------------


def _french_csv(name: str) -> str:
    dest = DATA / "french" / f"{name}_CSV.zip"
    _fetch(FRENCH_URL.format(name=name), dest)
    with zipfile.ZipFile(dest) as z:
        member = z.namelist()[0]
        return z.read(member).decode("latin-1")


def _parse_french_block(text: str, header_startswith: str | None = None) -> pd.DataFrame:
    """Parse the first daily block (YYYYMMDD rows) after the matching header line."""
    lines = text.splitlines()
    start = 0
    if header_startswith:
        start = next(i for i, ln in enumerate(lines) if ln.strip().startswith(header_startswith))
    # The header row is the first line after `start` that begins with a comma.
    h = next(i for i in range(start, len(lines)) if lines[i].startswith(","))
    cols = [c.strip() for c in lines[h].split(",")[1:]]
    rows = []
    for ln in lines[h + 1 :]:
        parts = ln.split(",")
        key = parts[0].strip()
        if not (len(key) == 8 and key.isdigit()):
            break
        rows.append([key] + [float(x) for x in parts[1 : len(cols) + 1]])
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")
    df = df.replace([-99.99, -999.0], np.nan) / 100.0  # percent -> fraction
    return df


def load_ff_daily() -> pd.DataFrame:
    """Daily Mkt-RF, SMB, HML, RF as fractions."""
    return _parse_french_block(_french_csv("F-F_Research_Data_Factors_daily")).loc[:FRENCH_END]


def load_industries_daily() -> pd.DataFrame:
    """Daily value-weighted returns of 49 industry portfolios as fractions."""
    text = _french_csv("49_Industry_Portfolios_daily")
    return _parse_french_block(text, "Average Value Weighted Returns -- Daily").loc[:FRENCH_END]
