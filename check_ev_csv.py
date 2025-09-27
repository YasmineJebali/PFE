import pandas as pd

# Path to your big EV CSV file
ev_csv_path = r"C:\Users\yasmi\Downloads\EVDataExplorer2025 (1).csv"

# Only read two columns to see what names are inside
usecols = ["region_country", "parameter"]
df = pd.read_csv(ev_csv_path, usecols=usecols)

print("\n=== Unique Country Names in dataset ===")
print(df["region_country"].unique())

print("\n=== Unique Parameter Names in dataset ===")
print(df["parameter"].unique())
