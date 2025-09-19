import pandas as pd
import matplotlib.pyplot as plt

forecast = pd.read_parquet("/home4/s5539099/test/data/met_forecast.parquet")

# 1.1 Shape & Columns
print("Shape:", forecast.shape)
print("Columns:", list(forecast.columns))

# 1.2 dtypes & null counts
print(forecast.dtypes)
print("\nMissing values per column:")
print(forecast.isna().mean().sort_values(ascending=False).head(95))

# 2.1 How often are forecasts issued?
tref = forecast.time_ref.drop_duplicates().sort_values()
hours = tref.diff().dt.total_seconds() / 3600
print("\nIssue-time intervals (in hours):")
print(hours.value_counts())

# 2.2 Unique lead times
lts = sorted(forecast['lt'].unique())
print("\nLead times available:", lts)
# Check we cover 0–71
print("Min lt, Max lt:", min(lts), max(lts))

# Focus on wind-speed members
ws_cols = [c for c in forecast.columns if c.startswith("ws10m_")]

# Compute % missing for each member, then average by the lt suffix
# Group by lead time and calculate missing percentage for each group
miss_pct = forecast.groupby('lt')[ws_cols].apply(lambda x: x.isna().mean().mean()).to_dict()

plt.figure()
plt.plot(list(miss_pct.keys()), list(miss_pct.values()), marker='o')
plt.title("% Missing ws10m Members vs. Lead Time")
plt.xlabel("Lead Time (hours)")
plt.ylabel("% Missing")
plt.grid(True)
plt.savefig("/home4/s5539099/test/windAI_rug/plots/missing_wind_data.png", dpi=300, bbox_inches='tight')
plt.show()

all_ws = forecast[ws_cols].stack().values

plt.figure()
plt.hist(all_ws, bins=80)
plt.title("Histogram of All ws10m Values")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Count")
plt.savefig("/home4/s5539099/test/windAI_rug/plots/speed.png", dpi=300, bbox_inches='tight')
plt.show()

# Pre-compute
forecast['mean_ws'] = forecast[ws_cols].mean(axis=1)
forecast['spread_ws'] = forecast[ws_cols].std(axis=1)

mean_spread = (
    forecast
    .groupby('lt')[['mean_ws','spread_ws']]
    .mean()
    .loc[48:64]   # just for your Day-2 window
)

plt.figure()
plt.plot(mean_spread.index, mean_spread['mean_ws'], label='Mean WS')
plt.plot(mean_spread.index, mean_spread['spread_ws'], label='Spread WS')
plt.title("Ensemble Mean & Spread vs. Lead Time (48–71h)")
plt.xlabel("Lead Time (h)")
plt.ylabel("Wind Speed (m/s)")
plt.legend()
plt.grid(True)
plt.savefig("/home4/s5539099/test/windAI_rug/plots/uncertainty.png", dpi=300, bbox_inches='tight')
plt.show()

