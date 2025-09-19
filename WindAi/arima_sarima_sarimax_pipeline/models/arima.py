from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
import numpy as np
import re

import matplotlib.pyplot as plt

class Arima:
    def __init__(self, file_path, order=(24, 1, 2), forecast_horizon=61):
        self.file_path = file_path
        self.region_number = self.extract_region_name(file_path)
        self.order = order
        self.forecast_horizon = forecast_horizon
        self.model_fit = None
        self.train = None
        self.test = None
        self.forecast = None

    def extract_region_name(self, path):        

        filename = os.path.basename(path) 
        
        match = re.search(r'no(\d+)', filename)
        if match:
            return f"no_{match.group(1)}"
        else:
            return "unknown"

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        df = pd.read_parquet(self.file_path)
        series = df["power_MW"].values

        self.train = series[:-self.forecast_horizon]
        self.test = series[-self.forecast_horizon:]

    def fit(self):
        model = ARIMA(
            self.train, 
            order=self.order
            )
        
        self.model_fit = model.fit()

    def evaluate(self):
        self.forecast = self.model_fit.forecast(steps=len(self.test))
        rmse = np.sqrt(mean_squared_error(self.test, self.forecast))
        print(f"Test MSE: {rmse:.2f}")
        return rmse
    
    def save_model(self):
        pass

    def plot_forecast_custom(self, save_path=None):
        plt.figure(figsize=(10, 5))
        plt.plot(self.test[:61], label="True", marker="o")
        plt.plot(self.forecast[:61], label="Forecast", marker="x")
        plt.legend()
        plt.title(f"First 61-hour Forecast vs Actual for {self.region_number}")
        plt.xlabel("Hour")
        plt.ylabel("Power (MW)")
        plt.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def summary(self):
        if self.model_fit:
            print(self.model_fit.summary())

    def run_all(self, plot_save_path=None):
        try:
            self.load_data()
            self.fit()
            self.evaluate()
            self.plot_forecast_custom(plot_save_path)
            self.summary()
        except Exception as e:
            print(f"Error in {os.path.basename(self.file_path)}: {e}")

