from graphviz import Digraph
from pathlib import Path

# Ensure output folder exists
out_dir = Path("figures/data")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "preprocessing_pipeline"

# Create the diagram
g = Digraph("preprocessing_pipeline", format="png")
g.attr(rankdir="LR", fontsize="12", fontname="Arial")

# --- Step 1: Raw data sources ---
g.node("IEA", "IEA EV Explorer\n(Global EV Data)", shape="folder", style="filled", fillcolor="#dbeafe")
g.node("WDI", "World Bank WDI\n(Socioeconomic Data)", shape="folder", style="filled", fillcolor="#dbeafe")
g.node("OSM", "OpenStreetMap API\n(Geospatial Data)", shape="folder", style="filled", fillcolor="#dbeafe")
g.node("INS", "INS Tunisia\n(National Statistics)", shape="folder", style="filled", fillcolor="#dbeafe")

# --- Step 2: Processing scripts ---
g.node("CLEAN", "Python Cleaning Scripts\n(fetch_osm, process_sites, etc.)", shape="box", style="filled", fillcolor="#e0f2fe")

# --- Step 3: Clean datasets ---
g.node("DATA", "Clean Datasets\n(CSV / Parquet)", shape="component", style="filled", fillcolor="#fef9c3")

# --- Step 4: Streamlit App ---
g.node("APP", "Streamlit App\n(Map, ML, Modeling, ROI, Chatbot)", shape="box3d", style="filled", fillcolor="#dcfce7")

# --- Arrows (data flow) ---
for src in ["IEA", "WDI", "OSM", "INS"]:
    g.edge(src, "CLEAN", label="ingest + preprocess")

g.edge("CLEAN", "DATA", label="save CSV outputs")
g.edge("DATA", "APP", label="load into pages")

# Render (save as PNG)
out_file = g.render(out_path, cleanup=True)
print(f"✅ Diagram saved to: {out_file}")
