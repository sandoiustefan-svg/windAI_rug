import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory to save plots
outdir = "/home4/s5539099/test/windAI_rug/plots"

def load_and_clean(path):
    df = pd.read_parquet(path)
    df_clean = df.dropna(how="any")
    
    # Index by time if needed
    if df_clean.index.name != "time":
        if "time" in df_clean.columns:
            df_clean["time"] = pd.to_datetime(df_clean["time"])
            df_clean = df_clean.set_index("time")
        else:
            df_clean.index = pd.to_datetime(df_clean.index)
    df_clean = df_clean.sort_index()
    return df_clean

def save_plot(fig, name):
    fig.savefig(os.path.join(outdir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)

# Load and clean data
df = load_and_clean("/home4/s5539099/test/data/wind_power_per_bidzone.parquet")

# 1. Interval Histogram
diffs = df.index.to_series().diff().dt.total_seconds() / 3600
fig = plt.figure()
plt.hist(diffs.dropna(), bins=range(0,5), align="left", rwidth=0.8)
plt.title("Histogram of Inter-Timestamp Intervals (h)")
plt.xlabel("Interval (hours)")
plt.ylabel("Count")
save_plot(fig, "interval_histogram.png")

# 2. Time Series Snippet (last 30 days)
snippet = df.iloc[-24*30:]
fig = plt.figure(figsize=(12,5))
for col in df.columns:
    plt.plot(snippet.index, snippet[col], label=col)
plt.title("Hourly Power by Zone (last 30 days)")
plt.xlabel("Time")
plt.ylabel("MW")
plt.legend(ncol=2)
save_plot(fig, "time_series_last_30_days.png")

# 3. Monthly & Weekday Means
df2 = df.copy()
df2["month"] = df2.index.month
df2["weekday"] = df2.index.weekday
monthly = df2.groupby("month").mean()
weekday = df2.groupby("weekday").mean()

# Monthly plot
fig = plt.figure()
for col in df.columns:
    plt.plot(monthly.index, monthly[col], marker="o", label=col)
plt.title("Mean Monthly Power by Zone")
plt.xlabel("Month")
plt.ylabel("MW")
plt.legend()
save_plot(fig, "mean_monthly_power.png")

# Weekday plot
fig = plt.figure()
for col in df.columns:
    plt.plot(weekday.index, weekday[col], marker="o", label=col)
plt.title("Mean Weekday Power by Zone")
plt.xlabel("Weekday (0=Mon)")
plt.ylabel("MW")
plt.legend()
save_plot(fig, "mean_weekday_power.png")

# 4. Diurnal Cycle
df2["hour"] = df2.index.hour
hourly = df2.groupby("hour").mean()
fig = plt.figure()
for col in df.columns:
    plt.plot(hourly.index, hourly[col], marker="o", label=col)
plt.title("Mean Hour-of-Day Power by Zone")
plt.xlabel("Hour of Day")
plt.ylabel("MW")
plt.legend()
save_plot(fig, "diurnal_cycle.png")

# 5. Distributions & Boxplot
# Histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=50)
    axes[i].set_title(f"{col} Distribution")
    axes[i].set_xlabel("MW")
    axes[i].set_ylabel("Count")
save_plot(fig, "zone_histograms.png")

# Boxplot
fig = plt.figure(figsize=(6,4))
plt.boxplot([df[col] for col in df.columns], labels=df.columns, showfliers=False)
plt.title("Power Boxplot by Zone (no fliers)")
plt.ylabel("MW")
save_plot(fig, "zone_boxplot.png")

# 6. Autocorrelation & Persistence Error
# ACF
max_lag = 72
fig = plt.figure(figsize=(12,6))
for col in df.columns:
    acfs = [df[col].autocorr(lag=lag) for lag in range(1, max_lag+1)]
    plt.plot(range(1, max_lag+1), acfs, label=col)
plt.title("Autocorrelation vs Lag (1–72h)")
plt.xlabel("Lag (hours)")
plt.ylabel("Autocorrelation")
plt.legend()
save_plot(fig, "autocorrelation_vs_lag.png")

# Persistence error histogram
pers_err = (df - df.shift(48)).abs().stack()
fig = plt.figure()
plt.hist(pers_err.dropna(), bins=50)
plt.title("Histogram of 48h Persistence Errors")
plt.xlabel("Absolute Error (MW)")
plt.ylabel("Count")
save_plot(fig, "persistence_error_histogram.png")

# 7. Cross-Zone Correlation Matrix
corr = df.corr()
fig = plt.figure()
plt.imshow(corr, vmin=-1, vmax=1, cmap="RdBu", interpolation="nearest")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.index)
plt.title("Cross-Zone Correlation Matrix")
save_plot(fig, "cross_zone_correlation_matrix.png")

print(f"Plots saved to: {outdir}")
