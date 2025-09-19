import pandas as pd
import numpy as np


FORECAST_FP   = "/home4/s5539099/test/data/met_forecast.parquet"
NOWCAST_FP    = "/home4/s5539099/test/data/met_nowcast.parquet"
METADATA_FP   = "data/windparks_bidzone.csv" 
POWER_FP      = "/home4/s5539099/test/data/wind_power_per_bidzone.parquet"
OUTPUT_FP     = "/home4/s5539099/test/windAI_rug/model_dataset.parquet"
LT_MIN, LT_MAX= 48, 64

MET_FORECAST_COLS = (
    ["sid","time_ref","time","lt"]
    + [f"ws10m_{i:02d}" for i in range(15)]
    + [f"wd10m_{i:02d}" for i in range(15)]
    + [f"t2m_{i:02d}"   for i in range(15)]
    + [f"rh2m_{i:02d}"  for i in range(15)]
    + [f"mslp_{i:02d}"  for i in range(15)]
    + [f"g10m_{i:02d}"  for i in range(15)]
)

NOWCAST_COLS = [
    "windpark",
    "wind_speed_10m","wind_direction_10m",
    "air_temperature_2m","air_pressure_at_sea_level",
    "relative_humidity_2m","precipitation_amount",
]

META_COLS = [
    "eic_code","bidding_area","operating_power_max","prod_start_new"
]

POWER_COLS = ["ELSPOT NO1","ELSPOT NO2","ELSPOT NO3","ELSPOT NO4"]


fc = pd.read_parquet(FORECAST_FP, columns=MET_FORECAST_COLS)
fc["time_ref"] = pd.to_datetime(fc["time_ref"], utc=True)
fc["time"]     = pd.to_datetime(fc["time"],     utc=True)
fc["valid_time"] = fc["time_ref"] + pd.to_timedelta(fc["lt"], unit="h")
fc = fc[(fc["lt"] >= LT_MIN) & (fc["lt"] <= LT_MAX)].copy()
ws_cols = [c for c in fc if c.startswith("ws10m_")]
fc["ws_mean"]   = fc[ws_cols].mean(axis=1)
fc["ws_spread"] = fc[ws_cols].std(axis=1)
fc = fc[["sid","valid_time","lt","ws_mean","ws_spread"]].rename(columns={"sid":"windpark"})


nc = pd.read_parquet(NOWCAST_FP)
nc = nc.reset_index()
time_col = nc.columns[0]
nc = nc.rename(columns={time_col:"valid_time"})
nc["valid_time"] = pd.to_datetime(nc["valid_time"], utc=True)
nc = nc[["windpark","valid_time"] + NOWCAST_COLS[1:]]


df = pd.merge(fc, nc, on=["windpark","valid_time"], how="inner")


md = pd.read_csv(METADATA_FP, usecols=META_COLS + ["substation_name"])
md = md.rename(columns={
    "eic_code": "windpark_id",
    "substation_name": "windpark",  
    "bidding_area": "zone",
    "operating_power_max": "capacity"
})
df = df.merge(md, on="windpark", how="left")
df["zone"] = df["zone"].str.replace(r"^ELSPOT\\s+", "", regex=True)


def cap_wt_mean(g, col):
    return (g[col] * g["capacity"]).sum() / g["capacity"].sum()

zone_feats = (
    df.groupby(["zone","valid_time"], group_keys=False)
      .apply(lambda g: pd.Series({
          "ws_mean":   cap_wt_mean(g, "ws_mean"),
          "ws_spread": cap_wt_mean(g, "ws_spread"),
          "ws_now":    cap_wt_mean(g, "wind_speed_10m"),
          "t2m_now":   cap_wt_mean(g, "air_temperature_2m"),
          "p_slp_now": cap_wt_mean(g, "air_pressure_at_sea_level"),
          "rh_now":    cap_wt_mean(g, "relative_humidity_2m"),
          "tp_now":    cap_wt_mean(g, "precipitation_amount"),
      }), include_groups=False)
      .reset_index()
)
zone_feats["zone"] = zone_feats["zone"].str.replace(r"^ELSPOT\\s+", "", regex=True)


pw = pd.read_parquet(POWER_FP)
pw = pw.reset_index()
time_col = pw.columns[0]
pw = pw.rename(columns={time_col:"valid_time"})
pw["valid_time"] = pd.to_datetime(pw["valid_time"], utc=True)
pw = pw[["valid_time"] + POWER_COLS]


pw_long = pw.melt(
    id_vars=["valid_time"],
    value_vars=POWER_COLS,
    var_name="zone",
    value_name="power"
)


model_df = (
    zone_feats
    .merge(pw_long, on=["zone","valid_time"], how="inner")
    .dropna()
)

print("Forecast shape:", fc.shape)
print(fc.head())
print("Nowcast shape:", nc.shape)
print(nc.head())
print("After merge (fc, nc):", df.shape)
print(df.head())
print("Metadata shape:", md.shape)
print(md.head())
print("After merge (df, md):", df.shape)
print(df.head())
print("Zone feats shape:", zone_feats.shape)
print(zone_feats.head())
print("Power shape:", pw.shape)
print(pw.head())
print("Power long shape:", pw_long.shape)
print(pw_long.head())
print("Final model_df shape:", model_df.shape)
print(model_df.head())



model_df.to_parquet(OUTPUT_FP, index=False)
print(f"Saved aligned dataset ({len(model_df)} rows) to {OUTPUT_FP}")