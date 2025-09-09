import os
from preprocessing_raw_data import Preprocessing_raw
from preprocessing_dl_data_region import Preprocessing_DL_region


RAW_INPUTS = {
    "met_forecast" : "windAI_rug/WindAi/given_datasets/met_forecast.parquet",
    "met_nowcast" : "windAI_rug/WindAi/given_datasets/met_nowcast.parquet",
    "power" : "windAI_rug/WindAi/given_datasets/wind_power_per_bidzone.parquet",
    "meta" : "windAI_rug/WindAi/given_datasets/windparks_bidzone.csv"
}

RAW_OUT_DIR = "windAI_rug/WindAi/deep_learning/created_datasets"
DL_OUT_DIR = RAW_OUT_DIR

def run_raw_preprocess():
    preproc = Preprocessing_raw(**RAW_INPUTS)
    preproc.read_datasets()
    preproc.time_cropping()
    preproc.filter_common_windparks()
    df_regions = preproc.create_region_dataset()
    preproc.save_datasets(**df_regions)
    return list(df_regions.keys())

def run_dl_preprocess(region_names):
    os.makedirs(DL_OUT_DIR, exist_ok=True)
    for region in region_names:
        in_path = os.path.join(RAW_OUT_DIR, f"{region}.parquet")
        if not os.path.exists(in_path):
            print(f"[WARN] Missing input parquet for region {region}: {in_path}")
            continue
        preproc_dl = Preprocessing_DL_region(in_path, region_name=region, scale=True)
        preproc_dl.fit_transform(save_path=DL_OUT_DIR)

if __name__ == "__main__":
    regions = run_raw_preprocess()
    run_dl_preprocess(regions)
    print("\n Preprocessing pipeline complete.")