import os
import glob
import pandas as pd

# === where to look for the WDI CSV ===
SEARCH_DIRS = [
    r"C:\Users\yasmi\Downloads\WDI_CSV",      # if you unzipped the WDI zip here
    r"C:\Users\yasmi\Downloads",              # fallback: your Downloads
]

# === possible file names used by World Bank dumps ===
CANDIDATE_NAMES = [
    "WDICSV.csv",          # common
    "WDIData.csv",         # sometimes this
    "WDIData*/*.csv",      # extracted subfolder
    "WDI*/*.csv",          # any WDI subfolder
]

# === output path used by the app auto-loader ===
OUT = r"C:\Users\yasmi\Downloads\tn_tech_real.csv"

# Indicators to extract (Tunisia)
INDICATORS = {
    "internet": "Individuals using the Internet (% of population)",   # IT.NET.USER.ZS
    "mobile":   "Mobile cellular subscriptions (per 100 people)",     # IT.CEL.SETS.P2
}

def find_wdi_csv() -> str:
    """Search typical locations / filenames and return the first matching CSV path."""
    for base in SEARCH_DIRS:
        if not os.path.exists(base):
            continue
        # direct candidates
        for name in CANDIDATE_NAMES:
            pattern = os.path.join(base, name)
            matches = glob.glob(pattern)
            for p in matches:
                if p.lower().endswith(".csv"):
                    return p
        # last-resort: any big CSV in the folder that contains 'wdi'
        for p in glob.glob(os.path.join(base, "*.csv")):
            if "wdi" in os.path.basename(p).lower():
                return p
    raise FileNotFoundError("Could not locate a WDI CSV. Put the extracted CSV in one of the SEARCH_DIRS.")

def main():
    csv_path = find_wdi_csv()
    print(f"[OK] Using WDI file: {csv_path}")

    df = pd.read_csv(csv_path)
    needed_cols = {"Country Name", "Indicator Name"}
    if not needed_cols.issubset(df.columns):
        raise ValueError(f"File doesn't look like WDI: missing columns {needed_cols}. Found: {df.columns.tolist()[:10]}...")

    tun = df[df["Country Name"] == "Tunisia"].copy()
    tn = tun[tun["Indicator Name"].isin(INDICATORS.values())].copy()

    if tn.empty:
        raise RuntimeError("No Tunisia rows for the target indicators. Check that the WDI file includes Tunisia.")

    # Wide years to long format (year, value)
    tn_long = tn.melt(
        id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"],
        var_name="year",
        value_name="adoption_pct"
    )
    tn_long = tn_long[pd.to_numeric(tn_long["year"], errors="coerce").notnull()]
    tn_long["year"] = tn_long["year"].astype(int)

    # Map indicator → tech label
    name_to_tech = {v: k for k, v in INDICATORS.items()}
    tn_long["tech"] = tn_long["Indicator Name"].map(name_to_tech)

    # Keep clean columns
    out = (
        tn_long[["tech", "year", "adoption_pct"]]
        .dropna()
        .sort_values(["tech", "year"])
        .reset_index(drop=True)
    )

    # Save where the app expects it
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"✅ Saved Tunisia tech adoption data to: {OUT} (rows={len(out)})")
    print(out.head(12))

if __name__ == "__main__":
    main()
