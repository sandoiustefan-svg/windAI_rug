from GRU_deep import GruDeep
from GRU_weak import GruWeak
from LSTM_main import LSTM
from transformer import TransformerForecast
import pandas as pd
import os, re, sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from WindAi.deep_learning.preprocessing.preprocess_windowing_region import WindowGenerator

MODEL_REGISTRY = {
    # "Gru Deep": GruDeep,
    # "Gru Weak": GruWeak,
    # "LSTM": LSTM,
    "Transformer_Masked": TransformerForecast,
}

HERE     = Path(__file__).resolve()
DL_ROOT  = HERE.parents[1]                           # .../WindAi/deep_learning
DATA_DIR = Path(os.environ.get("WIND_CREATED_DIR", DL_ROOT / "created_datasets")).resolve()
WEIGHT_DIR = (DL_ROOT / "weights").resolve()
PLOT_DIR   = (DL_ROOT / "results").resolve()

def _region_dirs(base_dir=DATA_DIR):
    """Return region folders that contain processed train/val/test."""
    out = []
    if not os.path.isdir(base_dir):
        return out
    for name in sorted(os.listdir(base_dir)):
        rdir = os.path.join(base_dir, name)
        proc = os.path.join(rdir, "processed")
        if os.path.isdir(proc):
            need = [os.path.join(proc, f"{s}_scaled.parquet") for s in ("train", "val", "test")]
            if all(os.path.exists(p) for p in need):
                out.append((name, rdir))
    return out

def _load_processed(region_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(region_dir, "processed", f"{split}_scaled.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path).drop(columns=["time"], errors="ignore")

def _ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def run(model_class, model_name, input_width=336, label_width=61, epochs=100, shift=0):
    regions = _region_dirs(DATA_DIR)
    if not regions:
        raise RuntimeError(f"No processed regions found in {DATA_DIR}")

    _ensure(WEIGHT_DIR, PLOT_DIR)

    for region_name, region_dir in regions:
        print(f"\n========== Training Region {region_name} ==========")

        # Load preprocessed splits
        train_df = _load_processed(region_dir, "train")
        val_df   = _load_processed(region_dir, "val")
        test_df  = _load_processed(region_dir, "test")

        # Windowed datasets
        window = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["power_MW"],
        )

        # Region number if name like "NO1"
        m = re.search(r"(\d+)$", region_name)
        region_number = int(m.group(1)) if m else None

        # Init model
        num_features = window.train.element_spec[0].shape[-1]
        model = model_class(
            input_width=input_width,
            label_width=label_width,
            num_features=num_features,
            region_number=region_number,
            name=model_name,
        )

        # Per-model/region output dirs
        out_w = os.path.join(WEIGHT_DIR, model_name, region_name)
        out_p = os.path.join(PLOT_DIR,   model_name, region_name)
        _ensure(out_w, out_p)

        model.summary()
        history = model.fit(window, out_w, epochs=epochs)

        # Forecast plot
        pred, y_true = model.predict_test(window, first_batch_only=False)
        model.plot_prediction(
            pred, y_true,
            os.path.join(out_p, f"forecast_{region_name}_{model.name}.png"),
        )

        # Learning curves
        model.plot_learning_curves(
            history,
            os.path.join(out_p, f"learning_{region_name}_{model.name}.png"),
        )

        # Evaluate
        model.evaluate_model(window.test, dataset_name="Test")


if __name__ == "__main__":
    for model_name, model_class in MODEL_REGISTRY.items():
        run(model_class, model_name)
