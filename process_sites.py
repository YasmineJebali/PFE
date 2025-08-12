#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Agil station and charger CSVs to compute nearest charger distance (km)
and produce a ranked list of candidate sites.
Usage:
  python process_sites.py
Inputs (in data/):
  - tunisia_agil_stations.csv
  - tunisia_charging_stations.csv
Output (in data/):
  - processed_sites.csv
"""

import math
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree

DATA_DIR = Path("data")
AGIL_CSV = DATA_DIR / "tunisia_agil_stations.csv"
CHARGERS_CSV = DATA_DIR / "tunisia_charging_stations.csv"
OUT_CSV = DATA_DIR / "processed_sites.csv"

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R * c

def load_clean(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))]
    return df

def compute_nearest_distance(agil_df: pd.DataFrame, ch_df: pd.DataFrame) -> pd.DataFrame:
    if ch_df.empty:
        agil_df["nearest_charger_km"] = np.nan
        return agil_df

    # Build BallTree with chargers in radians (supports haversine)
    ch_coords = np.deg2rad(ch_df[["lat", "lon"]].to_numpy())
    tree = BallTree(ch_coords, metric="haversine")

    agil_coords = np.deg2rad(agil_df[["lat", "lon"]].to_numpy())
    dist_rad, _ = tree.query(agil_coords, k=1)

    # Convert radians to km (Earth radius ~ 6371.0088 km)
    agil_df["nearest_charger_km"] = dist_rad.flatten() * 6371.0088
    return agil_df


def score_sites(df: pd.DataFrame) -> pd.DataFrame:
    # Simple score: farther from existing chargers = higher priority.
    # Add bonus if site has a name (proxy for larger stations)
    base = df["nearest_charger_km"].fillna(df["nearest_charger_km"].max())
    name_bonus = (~df["name"].fillna("").eq("")).astype(int) * 0.2
    df["site_score"] = base + name_bonus
    return df

def main():
    agil_df = load_clean(AGIL_CSV)
    ch_df = load_clean(CHARGERS_CSV)
    print(f"Loaded {len(agil_df)} Agil sites and {len(ch_df)} charging points")

    agil_df = compute_nearest_distance(agil_df, ch_df)
    agil_df = score_sites(agil_df)

    agil_df = agil_df.sort_values("site_score", ascending=False)
    agil_df.to_csv(OUT_CSV, index=False)
    print(f"Saved ranked sites to {OUT_CSV}")

if __name__ == "__main__":
    main()
