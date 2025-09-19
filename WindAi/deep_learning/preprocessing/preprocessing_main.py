# preprocessing_main.py
import os
from pathlib import Path
from preprocessing_raw_data import Preprocessing_raw
from preprocessing_dl_data_region import Preprocessing_DL_region

# Resolve repo/layout regardless of CWD
THIS = Path(__file__).resolve()
WIND_ROOT = THIS.parents[2]              # .../WindAi
REPO_ROOT = THIS.parents[3]              # .../windAI_rug
GIVEN_DIR = Path(os.environ.get("WIND_DATA_DIR", WIND_ROOT / "given_datasets")).resolve()
RAW_OUT_DIR = str((WIND_ROOT / "deep_learning" / "created_datasets").resolve())
DL_OUT_DIR  = RAW_OUT_DIR

RAW_INPUTS = {
    "met_forecast": str(GIVEN_DIR / "met_forecast.parquet"),
    "met_nowcast":  str(GIVEN_DIR / "met_nowcast.parquet"),
    "power":        str(GIVEN_DIR / "wind_power_per_bidzone.parquet"),
    "meta":         str(GIVEN_DIR / "windparks_bidzone.csv"),
}

def _assert_inputs_exist(d):
    missing = [f"{k}: {v}" for k, v in d.items() if not Path(v).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing input files:\n  " + "\n  ".join(missing) +
            f"\n\nCWD: {Path.cwd()}\nExpected GIVEN_DIR: {GIVEN_DIR}"
        )

def run_raw_preprocess():
    _assert_inputs_exist(RAW_INPUTS)
    preproc = Preprocessing_raw(**RAW_INPUTS)
    preproc.read_datasets()
    preproc.time_cropping()
    preproc.filter_common_windparks()
    df_regions = preproc.create_region_dataset(save_dir=RAW_OUT_DIR, save_splits=True, also_save_full=True)
    # don’t call save_datasets() here — splits are already saved
    return list(df_regions.keys())

def run_dl_preprocess(region_names):
    os.makedirs(DL_OUT_DIR, exist_ok=True)
    for region in region_names:
        preproc_dl = Preprocessing_DL_region(base_dir=DL_OUT_DIR, region_name=region, scale=True, max_lag=7)
        preproc_dl.fit_transform_splits()

if __name__ == "__main__":
    print("CWD:", Path.cwd())
    print("GIVEN_DIR:", GIVEN_DIR)
    print("RAW_OUT_DIR:", RAW_OUT_DIR)
    regions = run_raw_preprocess()
    run_dl_preprocess(regions)
    print("\nPreprocessing pipeline complete.")
