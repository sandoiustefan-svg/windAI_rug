from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class SARIMAXModel:
    def __init__(
        self,
        file_path,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        forecast_horizon=61,
        exog_cols=None  
    ):
        self.file_path = file_path
        self.order = order
        self.seasonal_order = seasonal_order
        self.forecast_horizon = forecast_horizon
        self.exog_cols = exog_cols or []

        self.model_fit = None
        self.train_y = None
        self.test_y = None
        self.train_exog = None
        self.test_exog = None
        self.forecast = None

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = pd.read_parquet(self.file_path)

        y = df['power_MW'].values
        exog = df[self.exog_cols].values if self.exog_cols else None

        if len(y) <= self.forecast_horizon:
            raise ValueError(f"Not enough data ({len(y)} points) for horizon {self.forecast_horizon}")
        self.train_y, self.test_y = y[:-self.forecast_horizon], y[-self.forecast_horizon:]
        if exog is not None:
            self.train_exog, self.test_exog = exog[:-self.forecast_horizon], exog[-self.forecast_horizon:]

    def fit(self):
        print(f"Training SARIMAX{self.order}x{self.seasonal_order} on {os.path.basename(self.file_path)}")
        model = SARIMAX(
            endog=self.train_y,
            exog=self.train_exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_fit = model.fit(disp=False)
        if not self.model_fit.mle_retvals.get('converged', False):
            print(f"Warning: model did not converge for {os.path.basename(self.file_path)}")

    def evaluate(self):
        self.forecast = self.model_fit.forecast(
            steps=len(self.test_y), exog=self.test_exog
        )
        if np.isnan(self.forecast).any():
            raise ValueError(f"Forecast contains NaNs for {os.path.basename(self.file_path)}")
        rmse = np.sqrt(mean_squared_error(self.test_y, self.forecast))
        print(f"Test RMSE: {rmse:.2f}")
        return rmse

    def save_model(self, save_path):
        if self.model_fit:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model_fit.save(save_path)
            print(f"Model saved to: {save_path}")

    def plot_forecast(self, save_path=None):
        if self.forecast is None or self.test_y is None:
            raise ValueError("You must run evaluate() before plotting.")
        steps = min(self.forecast_horizon, len(self.test_y))
        plt.figure(figsize=(10,5))
        plt.plot(range(steps), self.test_y[:steps], marker='o', label='Actual')
        plt.plot(range(steps), self.forecast[:steps], marker='x', label='Forecast')
        plt.title(f"First {steps}-step Forecast vs Actual")
        plt.xlabel('Step')
        plt.ylabel('Power (MW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def summary(self):
        if self.model_fit:
            print(self.model_fit.summary())

    def run_all(self, model_save_path=None, plot_save_path=None):
        try:
            self.load_data()
            self.fit()
            self.evaluate()
            self.plot_forecast(save_path=plot_save_path)
            if model_save_path:
                self.save_model(model_save_path)
        except Exception as e:
            print(f"Error in {os.path.basename(self.file_path)}: {e}")


if __name__ == '__main__':

    exogs = ['Day sin', 'Day cos', 'Year sin', 'Year cos']
    for i in tqdm(range(1, 5), desc='Training zones'):
        file_path = f"/home4/s5539099/test/windAI_rug/created_datasets/arima_power_no{i}.parquet"
        plot_save_path = f"C:/competition/windAI_rug/datasets_figures/forecast_vs_actual_sarimax_{i}.png"

        model = SARIMAXModel(
            file_path=file_path,
            order=(2, 1, 2),
            seasonal_order=(1, 1, 1, 24),
            forecast_horizon=61,
            exog_cols=exogs
        )
        model.run_all(model_save_path=None, plot_save_path=plot_save_path)
