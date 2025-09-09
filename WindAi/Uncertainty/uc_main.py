# uc_infer_and_plot.py
import os
import sys
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR    = "/home4/s5539099/test/windAI_rug/WindAi/deep_learning/created_dataset"
WEIGHTS_DIR = "/home4/s5539099/test/windAI_rug/WindAi/deep_learning/weights"
RESULTS_DIR = "/home4/s5539099/test/windAI_rug/WindAi/deep_learning/results"

REGION      = 3        # 1..4
NUM_SEEDS   = 10
INPUT_WIDTH = 336
LABEL_WIDTH = 61
SHIFT       = 0
DEVICE      = "/GPU:0"  # or "/CPU:0" if you prefer

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # .../windAI_rug
assert (PROJECT_ROOT / "WindAi").is_dir(), f"'WindAi' not found under {PROJECT_ROOT}"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from WindAi.deep_learning.preprocessing.preprocess_windowing_region import WindowGenerator
from WindAi.deep_learning.models.transformer_seed import TransformerForecast


# --------------------- data/window helpers ---------------------
def build_window_for_region(region: int):
    import pandas as pd
    path = os.path.join(DATA_DIR, f"scaled_features_power_MW_NO{region}.parquet")
    df = pd.read_parquet(path).drop(columns=["time"], errors="ignore")

    test_df   = df[-(INPUT_WIDTH + LABEL_WIDTH):]
    usable_df = df[:-(INPUT_WIDTH + LABEL_WIDTH)]
    n = len(usable_df)
    train_df = usable_df[:int(n * 0.7)]
    val_df   = usable_df[int(n * 0.7):]

    win = WindowGenerator(
        input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=SHIFT,
        train_df=train_df, val_df=val_df, test_df=test_df,
        label_columns=["power_MW"]
    )
    num_features = int(win.train.element_spec[0].shape[-1])
    return win, num_features


def map_teacher_forced(x, y):
    zero   = tf.zeros_like(y[:, :1, :])
    dec_in = tf.concat([zero, y[:, :-1, :]], axis=1)
    return ({"encoder_input": x, "decoder_input": dec_in}, y)




# --------------------- model factory ---------------------
def make_model_build(region: int, num_features: int):
    """Returns a callable that builds the exact Transformer (no compile, no fit)."""
    def _build():
        model = TransformerForecast(
            input_width=INPUT_WIDTH,
            label_width=LABEL_WIDTH,
            num_features=num_features,
            region_number=region,
            name="Ensemble",
            d_model=128, num_heads=4, ff_dim=256, num_layers=2
        ).model
        return model
    return _build


# --------------------- weights collector ---------------------
def get_seed_weight_paths(weights_root: str, region: int, num_seeds: int = 10):
    """
    Finds one .h5 per seed under:
      weights/region{region}/seed{idx}/  (preferred)
    and also supports the underscore variant:
      weights/region_{region}/seed_{idx:02d}/
    Picks *.weights.h5 if present, else any *.h5.
    """
    # prefer region1
    base = os.path.join(weights_root, f"region{region}")
    if not os.path.isdir(base):
        alt = os.path.join(weights_root, f"region_{region}")
        if os.path.isdir(alt):
            base = alt
        else:
            raise FileNotFoundError(f"Region directory not found: {base} or {alt}")

    paths = []
    for s in range(num_seeds):
        seed_dir = os.path.join(base, f"seed{s}")
        if not os.path.isdir(seed_dir):
            seed_dir = os.path.join(base, f"seed_{s:02d}")
        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(f"Missing seed dir: {seed_dir}")

        cands = (sorted(glob.glob(os.path.join(seed_dir, "*.weights.h5"))) or
                 sorted(glob.glob(os.path.join(seed_dir, "*.h5"))))
        if not cands:
            raise FileNotFoundError(f"No .h5 weights in {seed_dir}")
        paths.append(cands[-1]) 
    return paths


# --------------------- ensemble wrapper ---------------------
class UcEnsemble:
    """
    Loads N seed models, predicts ensemble mean & std (optionally calibrated).
    Works with dict inputs: {"encoder_input": ..., "decoder_input": ...}.
    """
    def __init__(self, model_build, paths, device="/CPU:0", eps=1e-8):
        self.model_build = model_build
        self.paths = paths
        self.device = device
        self.eps = eps
        self.temp = 1.0
        self.models = []
        self._load_all()

    def _load_all(self):
        self.models = []
        with tf.device(self.device):
            for p in self.paths:
                m = self.model_build()
                m.trainable = False
                m.load_weights(p)
                self.models.append(m)

    def predict(self, inputs, return_raw=False):
        with tf.device(self.device):
            if isinstance(inputs, dict):
                outs = [m(inputs, training=False) for m in self.models]
            else:
                x = tf.convert_to_tensor(inputs)
                outs = [m(x, training=False) for m in self.models]
            stacked = tf.stack([tf.convert_to_tensor(o) for o in outs], axis=0)  # [M,B,T,1]
        if return_raw:
            return stacked
        mean = tf.reduce_mean(stacked, axis=0)
        std  = tf.math.reduce_std(stacked, axis=0)
        std  = tf.maximum(std * self.temp, self.eps)
        return mean, std

    def calibrate(self, val_ds, steps=41, rng=(0.5, 2.0)):
        """Grid-search a scalar temperature to minimize Gaussian NLL on val."""
        mus, sigs, trues = [], [], []
        for (inp, y) in val_ds:
            raw = self.predict(inp, return_raw=True)
            m = tf.reduce_mean(raw, axis=0)
            s = tf.math.reduce_std(raw, axis=0)
            mus.append(tf.reshape(m, [-1]))
            sigs.append(tf.reshape(s, [-1]))
            trues.append(tf.reshape(y, [-1]))
        mu = tf.concat(mus, 0).numpy()
        sd = np.maximum(tf.concat(sigs, 0).numpy(), self.eps)
        yy = tf.concat(trues, 0).numpy()

        best_t, best = 1.0, np.inf
        for t in np.linspace(rng[0], rng[1], steps):
            sdt = np.maximum(sd * t, self.eps)
            nll = np.mean(0.5 * ((yy - mu) / sdt) ** 2 + np.log(sdt))
            if nll < best:
                best, best_t = nll, t
        self.temp = float(best_t)
        return self.temp


# --------------------- plotting ---------------------
def plot_forecast_with_unc(mean, std, y_true, raw_ens=None, title="", save_path=None):
    """
    mean,std,y_true: tensors with shape [B,T,1]; plot first sample in batch.
    raw_ens: optional ndarray [M,B,T,1] to draw quantile bands.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mu  = mean.numpy()[0, :, 0]
    sd  = std.numpy()[0, :, 0]
    ygt = y_true.numpy()[0, :, 0]
    T   = mu.shape[0]
    xs  = np.arange(T)

    plt.figure(figsize=(12, 5))
    # Quantile bands from ensemble (nonparametric)
    if raw_ens is not None:
        yM = raw_ens[:, 0, :, 0]  # [M,T]
        ql95, qh95 = np.quantile(yM, [0.025, 0.975], axis=0)
        ql90, qh90 = np.quantile(yM, [0.05,  0.95 ], axis=0)
        plt.fill_between(xs, ql95, qh95, alpha=0.15, label="95% (quantiles)")
        plt.fill_between(xs, ql90, qh90, alpha=0.20, label="90% (quantiles)")

    # Parametric ±kσ bands
    plt.fill_between(xs, mu - 2*sd, mu + 2*sd, alpha=0.12, label="±2σ")
    plt.fill_between(xs, mu - 1*sd, mu + 1*sd, alpha=0.18, label="±1σ")

    plt.plot(xs, ygt, lw=2, label="Ground truth")
    plt.plot(xs, mu,  lw=2, label="Ensemble mean")
    plt.title(title)
    plt.xlabel("Horizon step")
    plt.ylabel("Power (MW)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, frameon=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ------------------------------ main ------------------------------
if __name__ == "__main__":
    print(f"[UC] Region NO{REGION} | building window…")
    window, num_features = build_window_for_region(REGION)

    # Choose the decoding style you want for inference:
    #   - teacher_forced (matches training)
    val_ds  = window.val.map(map_teacher_forced)
    test_ds = window.test.map(map_teacher_forced)

    print("[UC] Recreating model & collecting weights…")
    model_build = make_model_build(REGION, num_features)
    paths = get_seed_weight_paths(WEIGHTS_DIR, REGION, NUM_SEEDS)

    print(f"[UC] Loading {len(paths)} seed models on {DEVICE}…")
    ens = UcEnsemble(model_build=model_build, paths=paths, device=DEVICE)

    # Optional but recommended: calibrate std on validation set
    t_star = ens.calibrate(val_ds, steps=41, rng=(0.5, 2.0))
    print(f"[UC] Calibrated temperature = {t_star:.3f}")

    # Take the last-window batch from test and plot
    for (inputs_dict, y_true) in test_ds.take(1):
        mean, std = ens.predict(inputs_dict)                    # [B,T,1]
        raw = ens.predict(inputs_dict, return_raw=True).numpy() # [M,B,T,1]
        out_png = os.path.join(RESULTS_DIR, f"uncertainty_NO{REGION}_lastwindow.png")
        plot_forecast_with_unc(
            mean, std, y_true, raw_ens=raw,
            title=f"Region {REGION}: forecast ± uncertainty",
            save_path=out_png
        )
        print(f"[UC] Saved plot -> {out_png}")
