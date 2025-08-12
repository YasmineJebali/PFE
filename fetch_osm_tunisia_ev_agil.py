#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Tunisia EV charging stations and Agil fuel stations from Overpass API,
then export as CSV and GeoJSON.

Usage:
  python fetch_osm_tunisia_ev_agil.py

Notes:
- Requires: requests, pandas, geopandas (optional but recommended), shapely
- Install deps: pip install -r requirements.txt
- Overpass public API has rate limits. If you hit 429/504, wait and retry.
"""

import json
import time
from typing import Dict, Any, List, Tuple
import requests
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None
    Point = None

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# 1) All public EV charging stations in Tunisia
QUERY_CHARGERS = r"""
[out:json][timeout:180];
area["ISO3166-1"="TN"]->.a;
(
  node["amenity"="charging_station"](area.a);
  way["amenity"="charging_station"](area.a);
  relation["amenity"="charging_station"](area.a);
);
out center tags;
"""

# 2) All Agil fuel stations (brand variations) in Tunisia
QUERY_AGIL = r"""
[out:json][timeout:180];
area["ISO3166-1"="TN"]->.a;
(
  node["amenity"="fuel"]["brand"~"agil", i](area.a);
  node["amenity"="fuel"]["name"~"agil", i](area.a);
  way["amenity"="fuel"]["brand"~"agil", i](area.a);
  way["amenity"="fuel"]["name"~"agil", i](area.a);
  relation["amenity"="fuel"]["brand"~"agil", i](area.a);
  relation["amenity"="fuel"]["name"~"agil", i](area.a);
);
out center tags;
"""

def run_overpass(query: str, max_retries: int = 3, sleep_seconds: int = 10) -> Dict[str, Any]:
    for attempt in range(1, max_retries + 1):
        resp = requests.post(OVERPASS_URL, data={"data": query})
        if resp.status_code == 200:
            return resp.json()
        if attempt < max_retries:
            time.sleep(sleep_seconds * attempt)
    resp.raise_for_status()
    return {}

def extract_lat_lon(el: Dict[str, Any]) -> Tuple[float, float]:
    if 'lat' in el and 'lon' in el:
        return el['lat'], el['lon']
    center = el.get('center', {})
    return center.get('lat', None), center.get('lon', None)

def elements_to_rows(elements: List[Dict[str, Any]], kind: str) -> List[Dict[str, Any]]:
    rows = []
    for el in elements:
        tags = el.get('tags', {})
        lat, lon = extract_lat_lon(el)
        row = {
            "osm_type": el.get("type"),
            "osm_id": el.get("id"),
            "kind": kind,
            "name": tags.get("name"),
            "brand": tags.get("brand"),
            "operator": tags.get("operator"),
            "addr_full": tags.get("addr:full"),
            "addr_street": tags.get("addr:street"),
            "addr_city": tags.get("addr:city"),
            "addr_postcode": tags.get("addr:postcode"),
            "opening_hours": tags.get("opening_hours"),
            "phone": tags.get("phone"),
            "website": tags.get("website"),
            "lat": lat,
            "lon": lon,
        }
        if kind == "charging_station":
            row.update({
                "socket:schuko": tags.get("socket:schuko"),
                "socket:type2": tags.get("socket:type2"),
                "socket:type2:output": tags.get("socket:type2:output"),
                "socket:ccs": tags.get("socket:ccs"),
                "socket:ccs2": tags.get("socket:ccs2"),
                "socket:chademo": tags.get("socket:chademo"),
                "capacity": tags.get("capacity"),
                "capacity:charging_points": tags.get("capacity:charging_points"),
                "power": tags.get("power"),
                "maxcurrent": tags.get("maxcurrent"),
                "voltage": tags.get("voltage"),
                "authentication": tags.get("authentication"),
                "operator:wikidata": tags.get("operator:wikidata"),
                "network": tags.get("network"),
                "fee": tags.get("fee"),
                "payment:credit_cards": tags.get("payment:credit_cards"),
                "authentication:app": tags.get("authentication:app"),
                "authentication:membership_card": tags.get("authentication:membership_card"),
                "charge:output": tags.get("charge:output"),
            })
        rows.append(row)
    return rows

def dataframe_to_geo(df: pd.DataFrame):
    if gpd is None or Point is None:
        return None
    return gpd.GeoDataFrame(
        df, geometry=[Point(xy) if pd.notnull(xy[0]) and pd.notnull(xy[1]) else None for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )

def save_outputs(df_chargers: pd.DataFrame, df_agil: pd.DataFrame):
    df_chargers.to_csv("data/tunisia_charging_stations.csv", index=False)
    df_agil.to_csv("data/tunisia_agil_stations.csv", index=False)
    if gpd is not None:
        gdf_chargers = dataframe_to_geo(df_chargers)
        gdf_agil = dataframe_to_geo(df_agil)
        if gdf_chargers is not None:
            gdf_chargers.to_file("data/tunisia_charging_stations.geojson", driver="GeoJSON")
        if gdf_agil is not None:
            gdf_agil.to_file("data/tunisia_agil_stations.geojson", driver="GeoJSON")

def spatial_dedupe(df: pd.DataFrame, lat_col="lat", lon_col="lon") -> pd.DataFrame:
    dfx = df.copy()
    dfx = dfx[dfx[lat_col].notna() & dfx[lon_col].notna()]
    dfx["_lat_r"] = dfx[lat_col].round(4)
    dfx["_lon_r"] = dfx[lon_col].round(4)
    dfx = dfx.sort_values(["name", "brand", "_lat_r", "_lon_r", "osm_id"]).drop_duplicates(
        subset=["_lat_r", "_lon_r", "name", "brand"], keep="first")
    return dfx.drop(columns=["_lat_r", "_lon_r"])

def main():
    print("Querying Overpass for charging stations in Tunisia...")
    chargers_json = run_overpass(QUERY_CHARGERS)
    print(f"  -> {len(chargers_json.get('elements', []))} elements")

    print("Querying Overpass for Agil fuel stations in Tunisia...")
    agil_json = run_overpass(QUERY_AGIL)
    print(f"  -> {len(agil_json.get('elements', []))} elements")

    chargers_rows = elements_to_rows(chargers_json.get("elements", []), "charging_station")
    agil_rows = elements_to_rows(agil_json.get("elements", []), "fuel_agil")

    df_chargers = pd.DataFrame(chargers_rows)
    df_agil = pd.DataFrame(agil_rows)

    df_chargers = spatial_dedupe(df_chargers)
    df_agil = spatial_dedupe(df_agil)

    save_outputs(df_chargers, df_agil)

    print("Saved to data/:")
    print("  - tunisia_charging_stations.csv")
    print("  - tunisia_agil_stations.csv")
    try:
        import geopandas  # noqa
        print("  - tunisia_charging_stations.geojson")
        print("  - tunisia_agil_stations.geojson")
    except Exception:
        pass

if __name__ == "__main__":
    main()
