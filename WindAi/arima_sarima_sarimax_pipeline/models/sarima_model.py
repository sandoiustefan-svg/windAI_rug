from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class SARIMA:
    def __init__(self, file_path, order=(1,1,1), seasonal_order=(1,1,1,24), forecast_horizon=61):
        self.file_path = file_path
        self.order = order
        self.seasonal_order = seasonal_order
        self.forecast_horizon = forecast_horizon
        self.model_fit = None
        self.train = None
        self.test = None
        self.forecast = None

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = pd.read_parquet(self.file_path)
        series = df["power_MW"].values

        if len(series) <= self.forecast_horizon:
            raise ValueError(f"Not enough data for forecast horizon: {len(series)} points")

        self.train = series[:-self.forecast_horizon]
        self.test = series[-self.forecast_horizon:]
    
    def fit(self):
        print(f"Training SARIMA{self.order}x{self.seasonal_order} on {os.path.basename(self.file_path)}")
        model = SARIMAX(
            self.train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_fit = model.fit(disp=False)

        if not self.model_fit.mle_retvals.get("converged", False):
            print(f"Model did not converge for {os.path.basename(self.file_path)}")

    def evaluate(self):
        self.forecast = self.model_fit.forecast(steps=len(self.test))

        if np.isnan(self.forecast).any():
            raise ValueError(f"Model failed to converge for {os.path.basename(self.file_path)}")

        rmse = np.sqrt(mean_squared_error(self.test, self.forecast))
        print(f"Test RMSE: {rmse:.2f}")
        return rmse
    
    def save_model(self, save_path):
        if self.model_fit:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model_fit.save(save_path)
            print(f"Model saved to: {save_path}")
        
    def plot_forecast(self, save_path=None):
        if self.forecast is None or self.test is None:
            raise ValueError("You must run evaluate() before plotting.")

        steps = min(self.forecast_horizon, len(self.test))
        true_values = self.test[:steps]
        forecast_values = self.forecast[:steps]

        plt.figure(figsize=(10, 5))
        plt.plot(range(steps), true_values, label="True", marker='o')
        plt.plot(range(steps), forecast_values, label="Forecast", marker='x')
        plt.title(f"First {steps}-hour Forecast vs Actual")
        plt.xlabel("Hour")
        plt.ylabel("Power (MW)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to: {save_path}")
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
        except Exception as e:
            print(f"{os.path.basename(self.file_path)}: {e}")


if __name__ == "__main__":
    for i in range(1, 5):
        file_path = f"/home2/s5549329/windAI_rug/WindAi/arima_sarima_sarimax_pipeline/created_datasets/arima_power_elspot_no{i}.parquet"
        plot_save_path = f"/home2/s5549329/windAI_rug/WindAi/arima_sarima_sarimax_pipeline/results/forecast_vs_actual_61h_sarima_{i}.png"

        sarima_model = SARIMA(
            file_path=file_path,
            order=(2, 1, 2),
            seasonal_order=(1, 1, 1, 24),
            forecast_horizon=61
        )

        sarima_model.run_all(plot_save_path=plot_save_path)


        