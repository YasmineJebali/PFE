# config.py
from pathlib import Path
from dataclasses import dataclass
import os

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Paths:
    analog_csv: Path = Path(os.getenv("ANALOG_CSV", r"C:\Users\yasmi\Downloads\analog_ev_full.csv"))
    tn_tech_csv: Path = Path(os.getenv("TN_TECH_CSV", r"C:\Users\yasmi\Downloads\tn_tech_real.csv"))
    tn_forecast_out: Path = Path(os.getenv("TN_FORECAST_OUT", r"C:\Users\yasmi\Downloads\tn_ev_forecast.csv"))

PATHS = Paths()
