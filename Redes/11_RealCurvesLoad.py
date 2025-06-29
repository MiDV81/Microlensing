"""
Bulk-download OGLE-IV EWS files (phot.dat + params.dat) for every season.

Folder layout created:
<cwd>/Redes/MicrolensingData/2011/
                                     2011/blg-0001_phot.dat
                                     2011/blg-0001_params.dat
                                     ...
"""
import sys
import os
import requests
from pathlib import Path
from time import sleep
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNFunctions import ROOT_DIR

# ---------- basic settings ----------
BASE_ROOT = "https://www.astrouw.edu.pl/ogle/ogle4/ews"
YEARS      = list(range(2011, 2020)) + [2022, 2023, 2024, 2025]   # 2020-21 were cancelled
PREFIXES   = ["blg", "dg", "gd"]          # bulge + (newer) disk fields
MAX_IDX    = 3000                    # upper bound; loop stops earlier on misses
MISS_LIMIT = 50                      # stop after this many consecutive blanks
TIMEOUT    = 30
PAUSE      = 0.05                # seconds between server hits

# ---------- where to save ----------
ROOT_OUTDIR = ROOT_DIR / "MicrolensingData"
ROOT_OUTDIR.mkdir(parents=True, exist_ok=True)
print("Saving to:", ROOT_OUTDIR)
# ---------- main loop ----------
for year in YEARS:
    year_dir = ROOT_OUTDIR / str(year)
    year_dir.mkdir(exist_ok=True)

    for prefix in PREFIXES:
        misses = 0
        for idx in range(1, MAX_IDX + 1):
            code = f"{prefix}-{idx:04d}"
            base_url = f"{BASE_ROOT}/{year}/{code}"

            # download each requested file type
            for fname in ("phot.dat", "params.dat"):
                url = f"{base_url}/{fname}"
                dest = year_dir / f"{code}_{fname}"

                try:
                    r = requests.get(url, timeout=TIMEOUT)
                    if r.status_code == 200 and r.content.strip():
                        dest.write_bytes(r.content)
                        print(f"✔ {year} {code}/{fname}  ({len(r.content):,} B)")
                        if fname == "phot.dat":
                            misses = 0          # reset only when phot.dat exists
                    else:
                        if fname == "phot.dat":
                            misses += 1
                except Exception as exc:
                    print(f"⚠ {year} {code}/{fname}  {exc}")

                sleep(PAUSE)

            # stop once we hit a stretch of non-existent events
            if misses >= MISS_LIMIT:
                print(f"⨯  reached {MISS_LIMIT} consecutive blanks — "
                      f"stopping {prefix} scan for {year}")
                break

print("\nDone:", datetime.now(), "UTC")