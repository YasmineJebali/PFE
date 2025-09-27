# make_analog_ev_full.py
import os
import pandas as pd

# --- CONFIG ---
EV_CSV_CANDIDATES = [
    r"C:\Users\yasmi\Downloads\EVDataExplorer2025 (1).csv",
    r"C:\Users\yasmi\Downloads\EVDataExplorer2025.csv",
]
OUTPUT = r"C:\Users\yasmi\Downloads\analog_ev_full.csv"

# Countries you care about (add/remove freely).
# Names are normalized (lowercase, accents ignored) then mapped to a canonical label.
COUNTRY_ALIASES = {
    "turkiye": "Türkiye", "turkey": "Türkiye", "türkiye": "Türkiye",
    "thailand": "Thailand",
    "south africa": "South Africa",
    "jordan": "Jordan",
    "egypt": "Egypt",
    "morocco": "Morocco",
}

# Accept multiple ways the IEA file may label these parameters
PARAM_ALIASES = {
    "ev stock": "EV stock",
    "electric car stock": "EV stock",
    "ev charging points": "EV charging points",
    "publicly available chargers (units)": "EV charging points",
}

USECOLS = ["region_country", "parameter", "year", "value"]
CHUNK = 100_000
# -------------


def pick_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find an EV CSV at the configured paths.")


def norm(s: str) -> str:
    # lowercase, remove surrounding spaces; keep it simple and robust
    return str(s).strip().lower()


def main():
    csv_path = pick_existing_path(EV_CSV_CANDIDATES)
    print(f"[OK] Using file: {csv_path}")

    # sanity: confirm required columns exist
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [c for c in USECOLS if c not in cols]
    if missing:
        raise ValueError(f"Missing expected columns in source CSV: {missing}\nFound: {cols}")

    chunks = []
    total = both = 0

    # read & filter
    for chunk in pd.read_csv(csv_path, usecols=USECOLS, chunksize=CHUNK):
        total += len(chunk)
        # normalize helper columns
        chunk["_country_norm"] = chunk["region_country"].map(norm)
        chunk["_param_norm"] = chunk["parameter"].map(norm)

        # keep only aliases we recognize
        mask_country = chunk["_country_norm"].isin(COUNTRY_ALIASES.keys())
        mask_param = chunk["_param_norm"].isin(PARAM_ALIASES.keys())
        both_mask = mask_country & mask_param
        both += both_mask.sum()

        sub = chunk.loc[both_mask, USECOLS].copy()
        if not sub.empty:
            # canonicalize labels
            sub["region_country"] = sub["region_country"].map(lambda x: COUNTRY_ALIASES.get(norm(x), x))
            sub["parameter"] = sub["parameter"].map(lambda x: PARAM_ALIASES.get(norm(x), x))
            chunks.append(sub)

    print(f"[STATS] rows read: {total:,} | rows kept: {both:,}")
    if not chunks:
        raise RuntimeError("No matching rows after filtering. Adjust COUNTRY_ALIASES/PARAM_ALIASES.")

    df = pd.concat(chunks, ignore_index=True)

    # pivot to one row per country-year
    wide = df.pivot_table(
        index=["region_country", "year"],
        columns="parameter",
        values="value",
        aggfunc="first"
    ).reset_index()

    # rename to app schema
    wide = wide.rename(columns={
        "region_country": "country",
        "EV stock": "ev_stock",
        "EV charging points": "public_chargers",
    })

    # add ISO3
    ISO3 = {
        "Türkiye": "TUR",
        "Thailand": "THA",
        "South Africa": "ZAF",
        "Jordan": "JOR",
        "Egypt": "EGY",
        "Morocco": "MAR",
    }
    wide["iso3"] = wide["country"].map(ISO3).fillna("")

    # final columns (keep what exists)
    keep = [c for c in ["country", "iso3", "year", "ev_stock", "public_chargers"] if c in wide.columns]
    out = wide[keep].sort_values(["country", "year"]).reset_index(drop=True)

    print("\n[PREVIEW]")
    print(out.head(12))
    out.to_csv(OUTPUT, index=False)
    print(f"\n✅ Saved: {OUTPUT} (rows={len(out)})")


if __name__ == "__main__":
    main()
