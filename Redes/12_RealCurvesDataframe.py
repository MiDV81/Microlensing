import re
import numpy as np
import pandas as pd
from pathlib import Path
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Functions.NNFunctions import parse_params, parse_phot, save_data, ROOT_DIR

ROOT = ROOT_DIR / "MicrolensingData"

# scan all seasons --------------------------------------------------
rows = []

for year_dir in sorted(ROOT.glob("*")):
    if not year_dir.is_dir() or not year_dir.name.isdigit():
        continue
    year = year_dir.name
    
    for phot_path in year_dir.glob("*_phot.dat"):
        code = phot_path.stem.replace("_phot", "")   # e.g. blg-0123
        print(f"Working on {year}: {code}", end="\r")
        params_path = year_dir / f"{code}_params.dat"
        if not params_path.exists():
            # skip events with no params.dat
            continue

        row_key = f"{year}_{code}"

        # merge dictionaries: parameters + arrays
        record = {}
        record.update(parse_params(params_path))
        record.update(parse_phot(phot_path))
        record["year"] = int(year)     # optional extra column
        record["event_code"] = code

        rows.append((row_key, record))
    # break

# build the frame
index, data = zip(*rows)
df = pd.DataFrame(list(data), index=index)

print(f"{len(df)} events loaded")
print(df.head())

file = "OGLE_IV_all_events.pkl.gz"
save_data(df, file, ROOT)
