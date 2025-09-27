# schemas.py
import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Allow up to 200 to accommodate indicators that exceed 100 per 100 inhabitants
TnTechSchema = DataFrameSchema({
    "tech": Column(str),
    "year": Column(int, checks=Check.ge(1900)),
    "adoption_pct": Column(
        float,
        checks=Check.in_range(0, 200, include_min=True, include_max=True)  # <-- changed
    ),
})

AnalogSchema = DataFrameSchema({
    "country": Column(str),
    "iso3": Column(str, nullable=True),
    "year": Column(int, checks=Check.ge(1900)),
    "ev_stock": Column(float, checks=Check.ge(0)),
    "public_chargers": Column(float, nullable=True, checks=Check.ge(0)),
})
