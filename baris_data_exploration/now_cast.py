#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os


NOWCAST_PATH = "/home4/s5539099/test/data/met_nowcast.parquet"
OUTDIR       = "/home4/s5539099/test/windAI_rug/plots"
os.makedirs(OUTDIR, exist_ok=True)


df = pd.read_parquet(NOWCAST_PATH)


if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
else:
    df.index = pd.to_datetime(df.index)
df = df.sort_index()


vars_of_interest = [
    "wind_speed_10m",
    "air_temperature_2m",
    "air_pressure_at_sea_level",
    "relative_humidity_2m",
    "precipitation_amount"
]
corr = df[vars_of_interest].corr()
print("Correlation matrix:\n", corr)


plt.figure(figsize=(6,5))
plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
plt.colorbar(label="Pearson r")
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr)), corr.index)
plt.title("Now-cast Variable Correlations")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "nowcast_variable_correlation.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved correlation heatmap to {OUTDIR}/nowcast_variable_correlation.png")
