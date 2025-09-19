import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle


def add_lag_features(df, target_col='power_MW', max_lag=7):
    df = df.sort_values("time").copy()
    for lag in range(1, max_lag + 1):
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df

class Preprocessing_DL_region:
    """
    Aligns with the region folder structure produced by create_region_dataset().
    Expects:
      <base_dir>/<region_name>/
        ├─ train.parquet
        ├─ val.parquet
        └─ test.parquet

    It writes:
      <base_dir>/<region_name>/processed/
        ├─ train_scaled.parquet
        ├─ val_scaled.parquet
        ├─ test_scaled.parquet
        └─ scaler.pkl
    """
    def __init__(self, base_dir, region_name, scale=True, max_lag=7):
        self.base_dir = base_dir
        self.region_name = region_name
        self.scale = scale
        self.max_lag = max_lag
        self.scaler = StandardScaler() if scale else None

        # learned on train
        self.cols_to_scale_ = None
        self.cyclic_cols_ = ["Day sin", "Day cos", "Year sin", "Year cos"]

    def _convert_all_wind_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts all matched pairs: ws10m_<suffix>, wd10m_<suffix> -> Wx_<suffix>, Wy_<suffix>
        Only for suffix in {"now", "mean", "std"} (as in your engineered columns).
        """
        df = df.copy()
        wind_pairs = []
        for col in df.columns:
            if col.startswith("ws10m_"):
                suffix = col.replace("ws10m_", "")
                dir_col = f"wd10m_{suffix}"
                if dir_col in df.columns:
                    wind_pairs.append((col, dir_col, suffix))

        for ws_col, wd_col, suffix in wind_pairs:
            if suffix in ["now", "mean", "std"]:
                wv = df.pop(ws_col)
                wd_rad = df.pop(wd_col) * np.pi / 180.0
                df[f"Wx_{suffix}"] = wv * np.cos(wd_rad)
                df[f"Wy_{suffix}"] = wv * np.sin(wd_rad)
        return df

    def _add_cyclic_time_features(self, df: pd.DataFrame, time_col: str = "time", drop_original: bool = False) -> pd.DataFrame:
        """
        Adds daily/yearly sinusoidal encodings. Keeps 'time' by default.
        """
        df = df.copy()
        t = pd.to_datetime(df[time_col], errors="coerce")
        ts = t.view("int64") / 1e9  # seconds

        day = 24 * 60 * 60
        year = 365.2425 * day

        df["Day sin"]  = np.sin(ts * (2 * np.pi / day))
        df["Day cos"]  = np.cos(ts * (2 * np.pi / day))
        df["Year sin"] = np.sin(ts * (2 * np.pi / year))
        df["Year cos"] = np.cos(ts * (2 * np.pi / year))

        if drop_original:
            df = df.drop(columns=[time_col], errors="ignore")
        return df

    def _prepare_basic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Wind components + cyclic time features; sort by time."""
        df = df.sort_values("time").copy()
        df = self._convert_all_wind_components(df)
        df = self._add_cyclic_time_features(df, drop_original=False)
        return df

    def _lag_with_history(self, curr: pd.DataFrame, prev_tail: pd.DataFrame | None) -> pd.DataFrame:
        """
        Compute lags on `concat(prev_tail, curr)` then drop the prepended rows
        so that lags at the split boundary are valid and causal.
        """
        if prev_tail is not None and len(prev_tail) > 0:
            # ensure same columns
            missing_in_prev = [c for c in curr.columns if c not in prev_tail.columns]
            for c in missing_in_prev:
                prev_tail[c] = np.nan
            prev_tail = prev_tail[curr.columns]
            work = pd.concat([prev_tail, curr], ignore_index=True)
            work = add_lag_features(work, target_col="power_MW", max_lag=self.max_lag)
            # drop the prepended rows
            work = work.iloc[len(prev_tail):].reset_index(drop=True)
        else:
            work = add_lag_features(curr, target_col="power_MW", max_lag=self.max_lag)
        return work

    def _region_dir(self):
        return os.path.join(self.base_dir, self.region_name)

    def _load_split(self, split: str) -> pd.DataFrame:
        path = os.path.join(self._region_dir(), f"{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}")
        df = pd.read_parquet(path)
        df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time")

    def _save_split(self, df: pd.DataFrame, split: str):
        out_dir = os.path.join(self._region_dir(), "processed")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{split}_scaled.parquet")
        df.to_parquet(path, index=False)
        print(f"[{self.region_name}] saved {split} -> {path} ({len(df)} rows)")

    # ---------- main API ----------
    def fit_transform_splits(self):
        """
        Full pipeline with leakage-safe scaling:
        - builds features on train/val/test (with causal lags)
        - fits scaler on train features ONLY (excluding target + cyclic)
        - transforms val/test with the same scaler
        - saves 3 parquet files + scaler.pkl
        Returns dict of (X, y) tuples for each split.
        """
        # 1) Load raw splits
        train_raw = self._load_split("train")
        val_raw   = self._load_split("val")
        test_raw  = self._load_split("test")

        # 2) Basic prep (wind components + cyclic features)
        tr_prep = self._prepare_basic(train_raw)
        va_prep = self._prepare_basic(val_raw)
        te_prep = self._prepare_basic(test_raw)

        # 3) Causal lags with boundary history
        tr = self._lag_with_history(tr_prep, prev_tail=None)
        va = self._lag_with_history(va_prep, prev_tail=tr_prep.tail(self.max_lag))
        te = self._lag_with_history(te_prep, prev_tail=va_prep.tail(self.max_lag))

        # 4) Drop rows with NA (from lag creation)
        tr = tr.dropna().reset_index(drop=True)
        va = va.dropna().reset_index(drop=True)
        te = te.dropna().reset_index(drop=True)

        # 5) Build X/y and scale (fit on train only)
        def split_xy(df: pd.DataFrame):
            df_num = df.select_dtypes(include=["number"]).copy()
            y = df_num["power_MW"].to_numpy()
            X = df_num.drop(columns=["power_MW"])
            return X, y

        Xtr, ytr = split_xy(tr)
        Xva, yva = split_xy(va)
        Xte, yte = split_xy(te)

        # keep cyclic columns unscaled
        self.cyclic_cols_ = [c for c in self.cyclic_cols_ if c in Xtr.columns]
        cols_to_scale = [c for c in Xtr.columns if c not in self.cyclic_cols_]
        self.cols_to_scale_ = cols_to_scale

        if self.scale and len(cols_to_scale) > 0:
            # fit on train
            Xtr_scaled = Xtr.copy()
            Xtr_scaled[cols_to_scale] = self.scaler.fit_transform(Xtr[cols_to_scale])

            # transform val/test
            Xva_scaled = Xva.copy()
            Xte_scaled = Xte.copy()
            # align columns defensively
            for cols, Xd in [(cols_to_scale, Xva_scaled), (cols_to_scale, Xte_scaled)]:
                missing = [c for c in cols if c not in Xd.columns]
                for m in missing:
                    Xd[m] = 0.0
                Xd[cols] = self.scaler.transform(Xd[cols])

        else:
            Xtr_scaled, Xva_scaled, Xte_scaled = Xtr, Xva, Xte

        # 6) Reattach non-numerics for saving (time, etc.)
        def for_save(df_src: pd.DataFrame, X_scaled: pd.DataFrame, y: np.ndarray):
            keep = df_src[["time"]].reset_index(drop=True) if "time" in df_src.columns else None
            out = X_scaled.reset_index(drop=True).copy()
            out["power_MW"] = y
            if keep is not None:
                out = pd.concat([keep, out], axis=1)
            return out

        tr_out = for_save(tr, Xtr_scaled, ytr)
        va_out = for_save(va, Xva_scaled, yva)
        te_out = for_save(te, Xte_scaled, yte)

        # 7) Save splits + scaler
        self._save_split(tr_out, "train")
        self._save_split(va_out, "val")
        self._save_split(te_out, "test")
        if self.scale:
            with open(os.path.join(self._region_dir(), "processed", "scaler.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "scaler": self.scaler,
                        "cols_to_scale": self.cols_to_scale_,
                        "cyclic_cols": self.cyclic_cols_
                    },
                    f
                )

        return {
            "train": (Xtr_scaled, ytr),
            "val":   (Xva_scaled, yva),
            "test":  (Xte_scaled, yte),
        }
