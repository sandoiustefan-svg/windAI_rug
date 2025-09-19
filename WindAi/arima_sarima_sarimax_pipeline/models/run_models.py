from arima import Arima
from sarima_model import SARIMA
from sarimax import SARIMAXModel

model = "Sarimax"
exogs = ['Day sin', 'Day cos', 'Year sin', 'Year cos']
for i in range(1, 5):
        file_path = f"/home2/s5549329/windAI_rug/WindAi/arima_sarima_sarimax_pipeline/created_datasets/arima_power_elspot_no{i}.parquet"
        plot_save_path_sarima = f"/home2/s5549329/windAI_rug/WindAi/arima_sarima_sarimax_pipeline/results/forecast_vs_actual_61h_sarima_{i}.png"
        plot_save_path_sarimax = f"/home2/s5549329/windAI_rug/WindAi/arima_sarima_sarimax_pipeline/results/forecast_vs_actual_61h_sarimax_{i}.png"
        plot_save_path_arima = f"/home2/s5549329/windAI_rug/WindAi/arima_sarima_sarimax_pipeline/results/forecast_vs_actual_61h_arima_{i}.png"

        if model == "Arima":
            arima_model = Arima(
                file_path=file_path,
                order=(2, 1, 2),
                forecast_horizon=61
            )
            arima_model.run_all(plot_save_path=plot_save_path_arima)

        elif model == "Sarima":
            sarima_model = SARIMA(
                file_path=file_path,
                order=(2, 1, 2),
                seasonal_order=(1, 1, 1, 24),
                forecast_horizon=61
            )
            sarima_model.run_all(plot_save_path=plot_save_path_sarima)

        elif model == "Sarimax":
            sarimax_model = SARIMAXModel(
                file_path=file_path,
                order=(2, 1, 2),
                seasonal_order=(1, 1, 1, 24),
                forecast_horizon=61,
                exog_cols=exogs
            )
            sarimax_model.run_all(model_save_path=None, plot_save_path=plot_save_path_sarimax)

        
