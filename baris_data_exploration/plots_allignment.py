import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the aligned dataset
df = pd.read_parquet("/home4/s5539099/test/windAI_rug/model_dataset.parquet")

print(df.head(15))


# Ensure valid_time is datetime and set as index
df["valid_time"] = pd.to_datetime(df["valid_time"])
df = df.set_index("valid_time").sort_index()

# Directory for saving plots
plot_dir = "/home4/s5539099/test/windAI_rug/plots"
os.makedirs(plot_dir, exist_ok=True)

# 1. Correlation matrix among features + target
corr = df.drop(columns=["zone"]).corr()
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, vmin=-1, vmax=1, interpolation="nearest")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)
ax.set_title("Feature and Target Correlation Matrix")
fig.colorbar(im, ax=ax, label="Correlation")
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, "correlation_matrix.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 2. Scatter: power vs ws_mean
fig, ax = plt.subplots()
ax.scatter(df["ws_mean"], df["power"], alpha=0.3)
ax.set_xlabel("Forecast Mean Wind Speed (m/s)")
ax.set_ylabel("Actual Power (MW)")
ax.set_title("Actual Power vs Forecast Mean Wind Speed")
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, "power_vs_ws_mean.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 3. Scatter: power vs ws_now
fig, ax = plt.subplots()
ax.scatter(df["ws_now"], df["power"], alpha=0.3)
ax.set_xlabel("Now-cast Wind Speed (m/s)")
ax.set_ylabel("Actual Power (MW)")
ax.set_title("Actual Power vs Now-cast Wind Speed")
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, "power_vs_ws_now.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 4. Time series snippet for a single zone (e.g., NO3) over last 14 days
zone = df["zone"].unique()[2]  # e.g., third zone type
snippet = df[df["zone"] == zone].iloc[-24*14:]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(snippet.index, snippet["power"])
ax.set_title(f"Actual Power Time Series for {zone} (Last 14 Days)")
ax.set_xlabel("Time")
ax.set_ylabel("Power (MW)")
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, f"time_series_{zone}.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 5. Diurnal cycle of power (average by hour of day) for each zone
df_hour = df.copy()
df_hour["hour"] = df_hour.index.hour
hourly = df_hour.groupby(["zone", "hour"])["power"].mean().unstack("zone")
fig, ax = plt.subplots()
hourly.plot(ax=ax)
ax.set_title("Diurnal Cycle of Power by Zone")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Power (MW)")
plt.legend(title="Zone")
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, "diurnal_cycle_power.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 6. Monthly average power for each zone
df_month = df.copy()
df_month["month"] = df_month.index.month
monthly = df_month.groupby(["zone", "month"])["power"].mean().unstack("zone")
fig, ax = plt.subplots()
monthly.plot(ax=ax, marker="o")
ax.set_title("Monthly Average Power by Zone")
ax.set_xlabel("Month")
ax.set_ylabel("Average Power (MW)")
plt.legend(title="Zone")
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, "monthly_average_power.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# 7. Distribution histograms of key features and power
features = ["ws_mean", "ws_spread", "ws_now", "power"]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for i, col in enumerate(features):
    axes[i].hist(df[col], bins=50)
    axes[i].set_title(f"{col} Distribution")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
plt.tight_layout()
fig.savefig(os.path.join(plot_dir, "feature_power_distributions.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Plots saved to {plot_dir}")
