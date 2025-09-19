import os
import numpy as np
import pandas as pd

# ---------- paths ----------
FORECAST_PATH   = "/home2/s5549329/windAI_rug/WindAi/given_datasets/met_forecast.parquet"
NOWCAST_PATH    = "/home2/s5549329/windAI_rug/WindAi/given_datasets/met_nowcast.parquet"
POWER_PATH      = "/home2/s5549329/windAI_rug/WindAi/given_datasets/wind_power_per_bidzone.parquet"
META_PATH       = "/home2/s5549329/windAI_rug/WindAi/given_datasets/windparks_bidzone.csv"

OUT_DIR = "/home2/s5549329/windAI_rug/WindAi/deep_learning/created_datasets"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- load ----------
df_forecast   = pd.read_parquet(FORECAST_PATH)
df_nowcasting = pd.read_parquet(NOWCAST_PATH)
df_power      = pd.read_parquet(POWER_PATH)
df_metadata   = pd.read_csv(META_PATH)

# ---------- power wide->long ----------
df_power_long = (
    df_power.reset_index()
    .melt(id_vars="index", var_name="bidding_area", value_name="power_MW")
    .rename(columns={"index": "time"})
)
df_power_long["bidding_area"] = df_power_long["bidding_area"].str.replace("ELSPOT ", "", regex=False)
df_power_long["time"] = pd.to_datetime(df_power_long["time"], errors="coerce")

# ---------- normalize nowcasting ----------
# ensure 'sid' and 'time' exist
df_nowcasting = df_nowcasting.rename(columns={"windpark": "sid"})
if "time" not in df_nowcasting.columns:
    df_nowcasting = df_nowcasting.reset_index().rename(columns={"index": "time"})
df_nowcasting["time"] = pd.to_datetime(df_nowcasting["time"], errors="coerce")

df_nowcasting = df_nowcasting.rename(columns={
    "air_temperature_2m": "t2m_now",
    "air_pressure_at_sea_level": "mslp_now",
    "relative_humidity_2m": "rh2m_now",
    "wind_speed_10m": "ws10m_now",
    "wind_direction_10m": "wd10m_now",
    "precipitation_amount": "precip_now",
})

# ---------- crop common time range ----------
start = max(
    pd.to_datetime(df_forecast["time_ref"]).min(),
    df_nowcasting["time"].min(),
    df_power_long["time"].min(),
)
end = min(
    pd.to_datetime(df_forecast["time_ref"]).max(),
    df_nowcasting["time"].max(),
    df_power_long["time"].max(),
)

df_forecast   = df_forecast[(df_forecast["time_ref"] >= start) & (df_forecast["time_ref"] <= end)]
df_nowcasting = df_nowcasting[(df_nowcasting["time"]      >= start) & (df_nowcasting["time"]      <= end)]
df_power_long = df_power_long[(df_power_long["time"]      >= start) & (df_power_long["time"]      <= end)]

# ---------- keep only common sids ----------
meta_set     = set(df_metadata["substation_name"])
nowcast_set  = set(df_nowcasting["sid"])
forecast_set = set(df_forecast["sid"])
common = meta_set & nowcast_set & forecast_set
print(f"Common windparks (sids): {len(common)}")

df_metadata   = df_metadata[df_metadata["substation_name"].isin(common)].copy()
df_nowcasting = df_nowcasting[df_nowcasting["sid"].isin(common)].copy()
df_forecast   = df_forecast[df_forecast["sid"].isin(common)].copy()

# ---------- FORECAST: summary stats per row ----------
def _colpick(df, key):
    return [c for c in df.columns if key in c]

ws_cols  = _colpick(df_forecast, "ws10m_")
wd_cols  = _colpick(df_forecast, "wd10m_")
t_cols   = _colpick(df_forecast, "t2m_")
rh_cols  = _colpick(df_forecast, "rh2m_")
p_cols   = _colpick(df_forecast, "mslp_")
g_cols   = _colpick(df_forecast, "g10m_")

# wind speed
df_forecast["ws10m_mean"] = df_forecast[ws_cols].mean(axis=1)
df_forecast["ws10m_std"]  = df_forecast[ws_cols].std(axis=1)

# wind dir (circular mean) + std of raw angles
angles = np.radians(df_forecast[wd_cols])
mean_angle = np.arctan2(np.sin(angles).mean(axis=1), np.cos(angles).mean(axis=1))
df_forecast["wd10m_mean"] = (np.degrees(mean_angle) + 360) % 360
df_forecast["wd10m_std"]  = df_forecast[wd_cols].std(axis=1)

# temperature
df_forecast["t2m_mean"] = df_forecast[t_cols].mean(axis=1)
df_forecast["t2m_std"]  = df_forecast[t_cols].std(axis=1)

# humidity
df_forecast["rh2m_mean"] = df_forecast[rh_cols].mean(axis=1)
df_forecast["rh2m_std"]  = df_forecast[rh_cols].std(axis=1)

# pressure
df_forecast["mslp_mean"] = df_forecast[p_cols].mean(axis=1)
df_forecast["mslp_std"]  = df_forecast[p_cols].std(axis=1)

# gust
df_forecast["g10m_mean"] = df_forecast[g_cols].mean(axis=1)
df_forecast["g10m_std"]  = df_forecast[g_cols].std(axis=1)

forecast_feature_cols = [
    "ws10m_mean", "ws10m_std",
    "wd10m_mean", "wd10m_std",
    "t2m_mean", "t2m_std",
    "rh2m_mean", "rh2m_std",
    "mslp_mean", "mslp_std",
    "g10m_mean", "g10m_std",
]

# ---------- FORECAST: wide future leads by lt ----------
def make_future_leads_by_lt(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int = 61,
    sid_col: str = "sid",
    tref_col: str = "time_ref",
    lt_col: str = "lt",
) -> pd.DataFrame:
    use_cols = [sid_col, tref_col, lt_col] + feature_cols
    base = df.loc[:, use_cols].copy()
    base[lt_col] = base[lt_col].astype(int)

    # average if duplicates at same (sid, time_ref, lt)
    base = base.groupby([sid_col, tref_col, lt_col], as_index=False)[feature_cols].mean()

    wide = base.set_index([sid_col, tref_col, lt_col])[feature_cols].unstack(lt_col)
    full_cols = pd.MultiIndex.from_product([feature_cols, range(horizon)], names=[None, lt_col])
    wide = wide.reindex(columns=full_cols)
    wide.columns = [f"{feat}_{k}" for feat, k in wide.columns]
    return wide.reset_index()

# choose a target horizon (keep NaNs when lt is missing)
H_TARGET = 61
df_forecast_leads = make_future_leads_by_lt(
    df=df_forecast,
    feature_cols=forecast_feature_cols,
    horizon=H_TARGET,
    sid_col="sid",
    tref_col="time_ref",
    lt_col="lt",
)
df_forecast_leads.to_parquet(os.path.join(OUT_DIR, f"forecast_leads_H{H_TARGET}.parquet"), index=False)

# ---------- NOWCAST: collapse hourly -> one row per (sid, time_ref=12:00) ----------
def hourly_to_daily_anchored_at_noon(df_now: pd.DataFrame, strict_24: bool = True) -> pd.DataFrame:
    df = df_now.copy()
    if "time" not in df.columns:
        df = df.reset_index().rename(columns={"index": "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # anchor hours to day with reference at 12:00
    anchor = pd.Timedelta(hours=12)
    df["time_ref"] = (df["time"] - anchor).dt.floor("D") + anchor

    agg = (
        df.groupby(["sid", "time_ref"], as_index=False)
          .agg(
              t2m_now    = ("t2m_now",    "mean"),
              mslp_now   = ("mslp_now",   "mean"),
              rh2m_now   = ("rh2m_now",   "mean"),
              precip_now = ("precip_now", "mean"),
              ws10m_now  = ("ws10m_now",  "mean"),
              wd10m_now  = ("wd10m_now",  "mean"),
              n_hours    = ("time",       "size"),
          )
    )
    if strict_24:
        agg = agg[agg["n_hours"] >= 24].drop(columns="n_hours")
    else:
        agg = agg.drop(columns="n_hours")

    return agg.sort_values(["sid", "time_ref"]).reset_index(drop=True)

daily_now = hourly_to_daily_anchored_at_noon(df_nowcasting, strict_24=True)
daily_now.to_parquet(os.path.join(OUT_DIR, "nowcast_daily.parquet"), index=False)

# ---------- MERGE forecast (daily at 12:00) + nowcast daily ----------
df_weather = pd.merge(df_forecast_leads, daily_now, on=["sid", "time_ref"], how="inner")

# ---------- attach region (bid zone) & power ----------
df_metadata = df_metadata.rename(columns={"substation_name": "sid"})
df_metadata["bidding_area"] = df_metadata["bidding_area"].str.replace("ELSPOT ", "", regex=False)

# time column for final merge with power
df_weather = df_weather.rename(columns={"time_ref": "time"})
df_weather["time"] = pd.to_datetime(df_weather["time"], errors="coerce")

# map bid zone to each sid & merge power
df_power_long_sid = pd.merge(
    df_power_long,
    df_metadata[["bidding_area", "sid"]],
    on="bidding_area",
    how="left",
)

df_final_sid = pd.merge(
    df_weather,
    df_power_long_sid[["time", "sid", "power_MW", "bidding_area"]],
    on=["time", "sid"],
    how="inner",
)

df_final_sid.to_parquet(os.path.join(OUT_DIR, "weather_nowcast_power_by_sid.parquet"), index=False)

# ---------- per-region series (mean across sids at each time) ----------
regions = ["NO1", "NO2", "NO3", "NO4"]
dfs_by_region = {r: df_final_sid[df_final_sid["bidding_area"] == r].copy() for r in regions}

for r, dfR in dfs_by_region.items():
    if dfR.empty:
        print(f"[WARN] Region {r} is empty after merges; skipping.")
        continue
    dfR = dfR.sort_values("time").reset_index(drop=True)
    num_cols = dfR.select_dtypes(include=[np.number]).columns
    df_region_mean = (
        dfR.groupby("time", as_index=False)[num_cols].mean()
    )
    outp = os.path.join(OUT_DIR, f"region_{r}_timeseries_mean.parquet")
    df_region_mean.to_parquet(outp, index=False)
    print(f"[{r}] saved: {outp}")

print("DONE.")
