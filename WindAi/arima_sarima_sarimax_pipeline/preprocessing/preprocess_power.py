import pandas as pd
import numpy as np
import os


class Preprocess_power_sarima:
    def __init__(self, path):
        self.path = path
        self.df = None
        self.regions = {}

    def load_and_reshape(self):
        df = pd.read_parquet(self.path)

        df = df.reset_index().melt(
            id_vars="index",
            var_name="bidding_area",
            value_name="power_MW"
        ).rename(columns={"index": "time"})

        self.df = df

    def add_cyclical_time_features(self):

        time_df = pd.to_datetime(self.df.pop("time"), format='%d.%m.%Y %H:%M:%S')
        time_df_seconds = time_df.map(pd.Timestamp.timestamp)

        day = 24 * 60 * 60
        year = 365.2425 * day

        self.df['Day sin'] = np.sin(time_df_seconds * (2 * np.pi / day))
        self.df['Day cos'] = np.cos(time_df_seconds * (2 * np.pi / day))
        self.df['Year sin'] = np.sin(time_df_seconds * (2 * np.pi / year))
        self.df['Year cos'] = np.cos(time_df_seconds * (2 * np.pi / year))
        self.df['time'] = time_df

    def aggregate_per_zone(self):
        self.df = self.df.groupby(['bidding_area', 'time'], as_index=False).agg({
            'power_MW': 'sum',
            'Day sin': 'first',
            'Day cos': 'first',
            'Year sin': 'first',
            'Year cos': 'first'
        })

    def split_by_region(self):
        self.regions = {
            region: group.sort_values("time")
            for region, group in self.df.groupby("bidding_area")
        }

    def clean_final_data(self):
        for region in list(self.regions):
            df = self.regions[region]
            df = df.dropna(subset=['power_MW'])  
            df = df.drop(columns=['bidding_area', 'time'], errors='ignore')  
            self.regions[region] = df

    def save_regions(self, output_dir="WindAi/arima_pipeline/created_datasets"):
        os.makedirs(output_dir, exist_ok=True)
        for region, df_region in self.regions.items():
            filename = f"arima_power_{region.lower().replace(' ', '_')}.parquet"
            path = os.path.join(output_dir, filename)
            df_region.to_parquet(path, index=False)
            print(f"Saved {region} to {path}")

    def run_full_pipeline(self):
        self.load_and_reshape()
        self.add_cyclical_time_features()
        self.aggregate_per_zone()
        self.split_by_region()
        self.clean_final_data()
        self.save_regions()


if __name__ == "__main__":
    processor = Preprocess_power_sarima(
        "/home2/s5549329/windAI_rug/WindAi/given_datasets/wind_power_per_bidzone.parquet"
    )
    processor.run_full_pipeline()