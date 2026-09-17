"""
download_gk2a.py -- pull GK2A L2 (+ L1B) products needed for FF-product evaluation
from the KMA API Hub, per fire event, time-aligned to the FF cadence.

Products (all derived from the SAME AMI scan -> co-temporal by construction):
  * FF    -- forest-fire / wildfire flag (+ confidence) ........ the product under test
  * CLD   -- cloud detection / cloud mask ..................... false-alarm screening
  * CTPS  -- cloud-top product (height/pressure) ............. parallax cloud-displacement
  * SW038 -- L1B ~3.8 um (Ch07) radiance/BT .................... fire-sensitive channel
  * IR112 -- L1B ~11.2 um (Ch14) radiance/BT ................... longwave reference for
                                                                   deltaT = T_3.8 - T_11.2

KMA API Hub URL scheme (confirmed from the console sample URLs):
    https://apihub.kma.go.kr/api/typ05/api/GK2A/{LEVEL}/{TOKEN}/{AREA}/data
        ?date={YYYYMMDDHHMM}&authKey={KEY}
  e.g. .../GK2A/LE1B/NR016/FD/data?date=202210272350&authKey=...

IMPORTANT NOTES
---------------
1. TIMEZONE.  GK2A `date` tokens are **UTC** (satellite observation time). KMA/KFS
   reported ignition times are **KST (UTC+9)**. We convert KST->UTC before building
   the timeline. (e.g. 2022-03-05 01:08 KST ignition -> 2022-03-04 16:08 UTC -- the
   calendar date rolls BACK a day; getting this wrong silently fetches the wrong scans.)
2. AREA + CADENCE.  AMI scans Full Disk (FD) every 10 min and the Korea area (KO)
   every 2 min; KMA produces the wildfire product as fast as every 2 min over Korea.
   Pull FF, CLD, CTPS, SW038 and IR112 on the SAME area + SAME timestamps so they all
   co-register exactly. Default here is KO @ 10-min (step-min 10); switch to FD or
   2-min if you confirmed those are what you want -- but keep all products identical.
3. TOKENS.  FF / CLD / CTPS / SW038 / IR112 below are the logical names; CONFIRM the
   exact product tokens against your API-Hub console sample URLs (GK2A meteorological-
   products page) and edit PRODUCTS if they differ (the cloud-top product in particular
   may be CTP / CTH / CTPS depending on the catalogue).
4. AUTH.  Read the key from env `KMA_API_KEY` (never hard-code). A 403 usually means
   the key is not activated for the typ05 GK2A service, an IP allowlist, or a wrong
   token/area -- this script prints the (small) response body, which carries the reason.
5. TIME WINDOW.  By default the timeline runs from a fixed pre-ignition lead
   (T_STEPS * step_min) through the event end, via build_timeline(). Pass
   --before-min / --after-min to override this with an explicit, event-independent
   window instead -- e.g. for an onset-latency or full-event L1B/FRP study where you
   want exactly "ignition - N min" through "event end + M min" regardless of T_STEPS.

Usage:
    export KMA_API_KEY=...                       # do not hard-code
    python download_gk2a.py --event 2022-03-04_Uljin --dry-run     # print timeline + URLs
    python download_gk2a.py --event 2022-03-04_Uljin               # download
    python download_gk2a.py --all --products FF CLD CTPS --area KO --step-min 10

    # L1B thermal channels, explicit -30/+30 min window around ignition/end:
    python download_gk2a.py --event 2023-04-03_Hampyeong --products SW038 IR112 \\
        --before-min 30 --after-min 30 --dry-run
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #
BASE = "https://apihub.kma.go.kr/api/typ05/api/GK2A"
KST = timezone(timedelta(hours=9))
UTC = timezone.utc
# GK2A began operational service on 2019-07-25; no FF/CLD/CTPS L2 exist before this.
GK2A_OP_START = datetime(2019, 7, 25, tzinfo=UTC)

# logical name -> (level, token, default_area, file_ext).  *** verify TOKENs in console ***
PRODUCTS = {
    "FF":    ("LE2",  "FF",    "KO", "nc"),   # forest-fire / wildfire flag
    "CLD":   ("LE2",  "CLD",   "KO", "nc"),   # cloud detection / mask
    "CTPS":  ("LE2",  "CTPS",  "KO", "nc"),   # cloud-top (height/pressure) for parallax

    # L1B thermal channels for GK2A FRP / onset-latency analysis
    "SW038": ("LE1B", "SW038", "KO", "nc"),   # Ch07, ~3.8 um -- fire-sensitive
    "IR112": ("LE1B", "IR112", "KO", "nc"),   # Ch14, ~11.2 um -- longwave reference
}

DATA_ROOT  = Path(os.environ.get(
    "GK2A_DATA_ROOT",
    "/projectnb/modislc/users/mkmoon/khurshedjon/mangrove_seg/gk2a_eval_data"))
EVENTS_CSV = DATA_ROOT / "events.csv"
T_STEPS    = 5          # causal lead frames to fetch before ignition (matches ff_config.T_STEPS)

# --- NASA FIRMS / VIIRS active-fire (SECONDARY reference; needs a free FIRMS MAP_KEY) ---
# Key: https://firms.modaps.eosdis.nasa.gov/api/map_key/  (env FIRMS_MAP_KEY)
# Endpoint: /api/area/csv/[MAP_KEY]/[SOURCE]/[west,south,east,north]/[DAY_RANGE 1..10]/[YYYY-MM-DD]
FIRMS_BASE      = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
KOREA_BBOX      = "124,33,132,39"                       # west,south,east,north (peninsula)
VIIRS_SOURCES   = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"]  # Standard Processing = full archive
FIRMS_MAX_RANGE = 5                                     # FIRMS area API caps a request at 5 days


# --------------------------------------------------------------------------- #
# time handling                                                               #
# --------------------------------------------------------------------------- #
def parse_dt(s, tz):
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=tz)

def kst_to_utc(dt_kst):
    return dt_kst.astimezone(UTC)

def _snap(dt, step_min, up=False):
    """Snap a datetime to the step grid (floor, or ceil if up=True)."""
    m = dt.minute - (dt.minute % step_min)
    base = dt.replace(minute=m, second=0, microsecond=0)
    if up and base < dt:
        base += timedelta(minutes=step_min)
    return base

def build_timeline(ignition_utc, end_utc, step_min=10, lead_steps=T_STEPS):
    """
    UTC scan timestamps from (ignition - lead_steps*step) to end, on the step grid.
    The lead frames give the model pre-ignition FF context (the miss-rich early window
    between KMA-reported ignition and first FF detection lives just after ignition).

    This is the DEFAULT timeline used when --before-min/--after-min are not given.
    """
    start = _snap(ignition_utc, step_min) - timedelta(minutes=lead_steps * step_min)
    stop  = _snap(end_utc, step_min, up=True)
    out, t = [], start
    while t <= stop:
        out.append(t)
        t += timedelta(minutes=step_min)
    return out


# --------------------------------------------------------------------------- #
# events                                                                       #
# --------------------------------------------------------------------------- #
def load_events(path=EVENTS_CSV):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["event_id"]: r for r in rows}


# --------------------------------------------------------------------------- #
# url + fetch                                                                  #
# --------------------------------------------------------------------------- #
def build_url(level, token, area, dt_utc, key):
    date = dt_utc.strftime("%Y%m%d%H%M")
    return f"{BASE}/{level}/{token}/{area}/data?date={date}&authKey={key}"

def _redact(url, key):
    return url.replace(key, "***") if key else url

def make_session(workers):
    """A requests.Session with a connection pool sized to the worker count
    (shared across threads; requests Sessions are safe for concurrent GETs)."""
    import requests
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    ad = HTTPAdapter(pool_connections=max(4, workers), pool_maxsize=max(4, workers))
    s.mount("https://", ad); s.mount("http://", ad)
    return s

def fetch(url, dest, key, timeout=60, retries=2, session=None):
    """Download one file. Returns (ok, status, note). Reuses `session` if given."""
    import requests                                   # lazy import so --dry-run needs no deps
    getter = session.get if session is not None else requests.get
    for attempt in range(retries + 1):
        try:
            r = getter(url, timeout=timeout)
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(2 ** attempt); continue
            return False, None, f"network error: {e}"
        if r.status_code == 200 and len(r.content) > 256:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return True, 200, f"{len(r.content):,} bytes"
        # non-200 OR tiny body (API Hub returns a short text reason on error)
        body = r.content[:200].decode("utf-8", "replace").strip()
        note = f"HTTP {r.status_code}; body[:200]={body!r}"
        if r.status_code in (429, 500, 502, 503) and attempt < retries:
            time.sleep(2 ** attempt); continue
        return False, r.status_code, note


# --------------------------------------------------------------------------- #
# per-event driver                                                            #
# --------------------------------------------------------------------------- #
def event_dirs(event_id):
    root = DATA_ROOT / "raw" / event_id / "gk2a"
    return {name: root / name.lower() for name in PRODUCTS}

def expected_path(event_id, name, area, dt):
    """Canonical on-disk path for one (event, product, area, scan) -- single source of
    truth shared by the downloader and the scan manifest."""
    level, token, default_area, ext = PRODUCTS[name]
    a = area or default_area
    return event_dirs(event_id)[name] / f"gk2a_{token}_{a}_{dt:%Y%m%d%H%M}.{ext}"

def download_event(ev, products, area=None, step_min=10, key=None, dry_run=False,
                   overwrite=False, workers=6, before_min=None, after_min=None):
    """
    before_min / after_min: if EITHER is given, the timeline is built as an explicit
    window  [ignition - before_min, end + after_min]  on the step grid, OVERRIDING the
    default build_timeline() lead-in/no-tail behaviour. Pass 0 for either side to mean
    "start exactly at ignition" / "stop exactly at end". Leave both None to keep the
    original T_STEPS-based default.
    """
    ignition = parse_dt(ev["ignition_utc"], UTC) if ev.get("ignition_utc") else kst_to_utc(parse_dt(ev["ignition_kst"], KST))
    end      = parse_dt(ev["end_utc"], UTC) if ev.get("end_utc") else kst_to_utc(parse_dt(ev["end_kst"], KST))

    # guard: GK2A L2 products do not exist before operational service (2019-07-25)
    if ignition < GK2A_OP_START:
        print(f"\n=== {ev['event_id']}  ({ev['location']}) ===")
        print(f"  SKIP: ignition {ignition:%Y-%m-%d %H:%M} UTC predates GK2A operations "
              f"({GK2A_OP_START:%Y-%m-%d}); no FF/CLD/CTPS available for this date.")
        return 0, 0, 0

    if before_min is not None or after_min is not None:
        # explicit, event-independent window: ignition-before_min .. end+after_min
        before = 0 if before_min is None else before_min
        after  = 0 if after_min is None else after_min
        start = _snap(ignition - timedelta(minutes=before), step_min)
        stop  = _snap(end + timedelta(minutes=after), step_min, up=True)
        stamps = []
        t = start
        while t <= stop:
            stamps.append(t)
            t += timedelta(minutes=step_min)
    else:
        stamps = build_timeline(ignition, end, step_min)

    print(f"\n=== {ev['event_id']}  ({ev['location']}, {ev['area_ha']} ha) ===")
    print(f"  ignition KST {ev['ignition_kst']}  ->  UTC {ignition:%Y-%m-%d %H:%M}")
    print(f"  end      KST {ev['end_kst']}  ->  UTC {end:%Y-%m-%d %H:%M}")
    if before_min is not None or after_min is not None:
        print(f"  window: EXPLICIT  ignition-{before_min or 0}min .. end+{after_min or 0}min")
    print(f"  timeline: {len(stamps)} scans @ {step_min} min  "
          f"({stamps[0]:%Y-%m-%d %H:%M} .. {stamps[-1]:%Y-%m-%d %H:%M} UTC)")

    # dry-run: print a few sample URLs per product, no requests
    if dry_run:
        for name in products:
            level, token, default_area, ext = PRODUCTS[name]
            a = area or default_area
            for dt in (stamps[0], stamps[len(stamps) // 2], stamps[-1]):
                print(f"    [{name}] {_redact(build_url(level, token, a, dt, key or 'MISSING_KEY'), key)}")
        return 0, 0, 0

    # build the worklist (skip files already on disk -> resumable)
    jobs = []
    for name in products:
        for dt in stamps:
            dest = expected_path(ev["event_id"], name, area, dt)
            if dest.exists() and not overwrite:
                continue
            jobs.append((name, dt, dest))
    n_skip = len(products) * len(stamps) - len(jobs)
    n_ok = n_fail = 0
    auth_failed = threading.Event()
    session = make_session(workers)

    def work(job):
        name, dt, dest = job
        if auth_failed.is_set():
            return "abort", name, dt, None
        level, token, default_area, ext = PRODUCTS[name]
        a = area or default_area
        url = build_url(level, token, a, dt, key)
        ok, status, note = fetch(url, dest, key, session=session)
        if status == 403:
            auth_failed.set()                         # signal all workers to stop
        return ("ok" if ok else "fail"), name, dt, note

    print(f"  downloading {len(jobs)} files with {workers} workers ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            res, name, dt, note = fut.result()
            if res == "ok":
                n_ok += 1
            elif res == "fail":
                n_fail += 1
                print(f"    FAIL [{name}] {dt:%Y%m%d%H%M}: {note}")

    if auth_failed.is_set():
        print("    -> 403 detected: stopped. Check key activation for typ05 GK2A, IP "
              "allowlist, and the product TOKEN/AREA against the console sample URL.")
    print(f"  done: {n_ok} downloaded, {n_skip} skipped (exist), {n_fail} failed "
          f"[{workers} workers]")
    return n_ok, n_skip, n_fail


# --------------------------------------------------------------------------- #
# VIIRS active fire (NASA FIRMS) -- SECONDARY reference, per event             #
# --------------------------------------------------------------------------- #
def viirs_dir(event_id):
    return DATA_ROOT / "raw" / event_id / "viirs"

def viirs_chunks(ignition_utc, end_utc, lead_days=1, max_range=FIRMS_MAX_RANGE):
    """Yield (start_date 'YYYY-MM-DD', day_range) covering [ignition-lead .. end] in
    <=10-day spans (the FIRMS area-API cap). One lead day captures the pass before ignition."""
    start = ignition_utc.date() - timedelta(days=lead_days)
    end   = end_utc.date()
    d = start
    while d <= end:
        rng = min(max_range, (end - d).days + 1)
        yield d.strftime("%Y-%m-%d"), rng
        d += timedelta(days=rng)

def fetch_csv(url, dest, session, timeout=120, retries=2):
    """Fetch a FIRMS CSV (point detections). Saves even a header-only 'no detections' file;
    treats FIRMS error text (e.g. 'Invalid MAP_KEY') as a failure."""
    import requests
    for attempt in range(retries + 1):
        try:
            r = (session or requests).get(url, timeout=timeout)
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(2 ** attempt); continue
            return False, None, f"network error: {e}"
        head = r.content[:120].decode("utf-8", "replace").strip()
        if r.status_code == 200 and "latitude" in head.lower():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
            return True, 200, f"{max(r.content.count(chr(10).encode()[0]) - 1, 0)} detections"
        note = f"HTTP {r.status_code}; body[:120]={head!r}"
        if r.status_code in (429, 500, 502, 503) and attempt < retries:
            time.sleep(2 ** attempt); continue
        return False, r.status_code, note

def download_viirs_event(ev, sources=VIIRS_SOURCES, bbox=KOREA_BBOX, map_key=None,
                         dry_run=False, overwrite=False):
    """Download VIIRS active-fire CSVs (one per source per <=10-day span) into
    raw/<event>/viirs/. Detections carry acq_date/acq_time (UTC) + frp; preprocessing
    snaps each to the nearest 10-min FF frame."""
    ignition = parse_dt(ev["ignition_utc"], UTC) if ev.get("ignition_utc") else kst_to_utc(parse_dt(ev["ignition_kst"], KST))
    end      = parse_dt(ev["end_utc"], UTC) if ev.get("end_utc") else kst_to_utc(parse_dt(ev["end_kst"], KST))
    chunks   = list(viirs_chunks(ignition, end))
    out      = viirs_dir(ev["event_id"])

    print(f"\n=== VIIRS {ev['event_id']}  ({ev['location']}) ===")
    print(f"  span {chunks[0][0]} .. {end:%Y-%m-%d} UTC | bbox {bbox} | sources {', '.join(sources)}")
    n_ok = n_skip = n_fail = 0
    session = make_session(4)
    for src in sources:
        for date, rng in chunks:
            dest = out / f"viirs_{src}_{date}_{rng}d.csv"
            if dest.exists() and not overwrite:
                n_skip += 1; continue
            url = f"{FIRMS_BASE}/{map_key or 'MISSING_KEY'}/{src}/{bbox}/{rng}/{date}"
            if dry_run:
                print(f"    [{src}] {_redact(url, map_key)}")
                continue
            ok, status, note = fetch_csv(url, dest, session)
            if ok:
                n_ok += 1; print(f"    ok  [{src}] {date} +{rng}d: {note}")
            else:
                n_fail += 1; print(f"    FAIL [{src}] {date} +{rng}d: {note}")
                if status in (401, 403) or (note and "Invalid MAP_KEY" in note):
                    print("    -> check FIRMS_MAP_KEY (free key from firms.modaps.eosdis.nasa.gov/api/map_key/).")
                    return n_ok, n_skip, n_fail
                # any other error (e.g. transient 5xx): log and keep going
    if not dry_run:
        print(f"  done: {n_ok} downloaded, {n_skip} skipped (exist), {n_fail} failed")
    return n_ok, n_skip, n_fail


# --------------------------------------------------------------------------- #
# cli                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Download GK2A FF/CLD/CTPS/SW038/IR112 per fire event from KMA API Hub.")
    ap.add_argument("--event", help="event_id from events.csv (e.g. 2022-03-04_Uljin)")
    ap.add_argument("--all", action="store_true", help="process every event in events.csv")
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=list(PRODUCTS))
    ap.add_argument("--area", default=None, help="override area token (KO, FD, ...)")
    ap.add_argument("--step-min", type=int, default=10, help="cadence in minutes (10 = FF L2 default)")
    ap.add_argument("--before-min", type=int, default=None,
                    help="minutes before ignition to start the timeline; if given "
                         "(with/without --after-min) OVERRIDES the default T_STEPS "
                         "lead-in with this explicit window")
    ap.add_argument("--after-min", type=int, default=None,
                    help="minutes after event end to stop the timeline; default is "
                         "no post-event extension unless this or --before-min is set")
    ap.add_argument("--workers", type=int, default=6,
                    help="concurrent download threads (I/O-bound; 4-8 is polite, raise if no 429s)")
    ap.add_argument("--events-csv", default=str(EVENTS_CSV))
    ap.add_argument("--dry-run", action="store_true", help="print timeline + sample URLs, no download")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--viirs", action="store_true",
                    help="also download VIIRS active fire (FIRMS) per event into raw/<event>/viirs/")
    ap.add_argument("--viirs-only", action="store_true", help="download ONLY VIIRS (skip GK2A)")
    ap.add_argument("--viirs-sources", nargs="+", default=VIIRS_SOURCES,
                    help="FIRMS sources (e.g. VIIRS_SNPP_SP VIIRS_NOAA20_SP VIIRS_NOAA21_NRT)")
    ap.add_argument("--bbox", default=KOREA_BBOX, help="VIIRS area bbox: west,south,east,north")
    args = ap.parse_args()

    do_gk2a  = not args.viirs_only
    do_viirs = args.viirs or args.viirs_only

    key = os.environ.get("KMA_API_KEY")
    if do_gk2a and not key and not args.dry_run:
        sys.exit("ERROR: set KMA_API_KEY in your environment (do not hard-code it).")
    map_key = os.environ.get("FIRMS_MAP_KEY")
    if do_viirs and not map_key and not args.dry_run:
        sys.exit("ERROR: set FIRMS_MAP_KEY (free: firms.modaps.eosdis.nasa.gov/api/map_key/).")

    events = load_events(args.events_csv)
    if args.all:
        targets = list(events.values())
    elif args.event:
        if args.event not in events:
            sys.exit(f"unknown event_id '{args.event}'. Known: {', '.join(events)}")
        targets = [events[args.event]]
    else:
        sys.exit("specify --event <id> or --all")

    totals = [0, 0, 0]; vtotals = [0, 0, 0]
    for ev in targets:
        if do_gk2a:
            r = download_event(ev, args.products, area=args.area, step_min=args.step_min,
                               key=key, dry_run=args.dry_run, overwrite=args.overwrite,
                               workers=args.workers, before_min=args.before_min,
                               after_min=args.after_min)
            totals = [a + b for a, b in zip(totals, r)]
        if do_viirs:
            v = download_viirs_event(ev, sources=args.viirs_sources, bbox=args.bbox,
                                     map_key=map_key, dry_run=args.dry_run, overwrite=args.overwrite)
            vtotals = [a + b for a, b in zip(vtotals, v)]
    if not args.dry_run:
        if do_gk2a:
            print(f"\nGK2A  TOTAL: {totals[0]} downloaded, {totals[1]} skipped, {totals[2]} failed")
        if do_viirs:
            print(f"VIIRS TOTAL: {vtotals[0]} downloaded, {vtotals[1]} skipped, {vtotals[2]} failed")


if __name__ == "__main__":
    main()