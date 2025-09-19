import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import os

df = pd.read_parquet("model_dataset.parquet")
df['valid_time'] = pd.to_datetime(df['valid_time'])
df = df.sort_values(['zone', 'valid_time'])

# 2. Persistence forecast
df['power_pred'] = df.groupby('zone')['power'].shift(1)
df_eval = df.dropna(subset=['power_pred']).copy()

# 3. Daily aggregate across ALL ZONES for a clean global view
daily_global = (
    df_eval
      .set_index('valid_time')
      .resample('D')[['power', 'power_pred']]
      .mean()
)

plt.figure(figsize=(15, 5))
# plot persistence first, then actual on top
plt.plot(daily_global.index, daily_global['power_pred'],
         label='Persistence', color='orange', linestyle='--', linewidth=2)
plt.plot(daily_global.index, daily_global['power'],
         label='Actual',      color='blue',   linewidth=2)
plt.xlabel('Date')
plt.ylabel('Power')
plt.title('Daily Mean: Actual vs. Persistence Forecast (All Zones)')
plt.legend()
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/global_daily_actual_vs_persistence.png')
plt.close()


# 4. Small‐multiples per zone with visible styling
#    (using the daily df to keep each panel legible)
df_plot = daily_global.reset_index().merge(df_eval[['zone','valid_time']], how='right').drop_duplicates()

zones = sorted(df_eval['zone'].unique())
ncols = 3
nrows = -(-len(zones) // ncols)  # ceil without math
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(ncols*5, nrows*3),
                         sharey=True, constrained_layout=True)
axes = axes.flatten()

for ax, zone in zip(axes, zones):
    g = df_eval[df_eval['zone']==zone]\
          .set_index('valid_time')\
          .resample('D')[['power','power_pred']].mean()
    ax.plot(g.index, g['power_pred'],
            label='Persistence', color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.plot(g.index, g['power'],
            label='Actual',      color='blue',   linewidth=1.5, alpha=0.8)
    ax.set_title(zone, fontsize='small')
    ax.set_ylabel('Power')
    ax.legend(fontsize='xx-small')

# clean up extras & label bottom row
for ax in axes[len(zones):]:
    fig.delaxes(ax)
for ax in axes[-ncols:]:
    ax.set_xlabel('Date')

plt.savefig('plots/persistence_small_multiples_visible.png')
plt.show()

# Scatter‐plot of predicted vs. actual power for the whole dataset
plt.figure(figsize=(6, 6))
plt.scatter(df_eval['power'], df_eval['power_pred'], s=2, alpha=0.3)
min_val = min(df_eval['power'].min(), df_eval['power_pred'].min())
max_val = max(df_eval['power'].max(), df_eval['power_pred'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
plt.xlabel('Actual Power')
plt.ylabel('Predicted Power')
plt.title('Actual vs. Predicted Scatter (All Zones)')
plt.tight_layout()
plt.savefig('plots/actual_vs_pred_scatter.png')
plt.show()
