import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from WindAi.deep_learning.preprocessing.preprocess_windowing_region import WindowGenerator
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LSTM:
    def __init__(self, input_width, label_width, num_features, region_number):
        self.input_width = input_width
        self.label_width = label_width
        self.num_features = num_features
        self.region_number = region_number
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_width, self.num_features)),
            tf.keras.layers.GaussianNoise(0.05),

            tf.keras.layers.LSTM(64, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.RepeatVector(self.label_width),

            tf.keras.layers.LSTM(32, return_sequences=True, kernel_regularizer = tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
        ])

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss = tf.keras.losses.Huber(delta=1.0),
            metrics=['mae']
        )
        return model

    def fit(self, window, weights_dir, epochs=20):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
        checkpoint_path = os.path.join(weights_dir, f"best_lstm_model_region_{self.region_number}.h5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        history = self.model.fit(
            window.train,
            validation_data=window.val,
            epochs=epochs,
            callbacks=[early_stop, lr_schedule, checkpoint]
        )
        return history
    
    def summary(self):
        self.model.summary()

    def predict_last_window(self, window):
        for x, y in window.test.take(1):
            prediction = self.model.predict(x)
        return prediction, y
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
        print(f"Model weights saved to: {filepath}")

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        print(f"Model weights loaded from: {filepath}")

    def plot_prediction(self, prediction, y_true, save_path):
        plt.figure(figsize=(12, 4))
        plt.plot(y_true[0, :, 0], label="Actual")
        plt.plot(prediction[0, :, 0], label="Predicted")
        plt.title("Power_MW Forecast: Next 61 Hours")
        plt.xlabel("Hour")
        plt.ylabel("Power_MW")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def plot_learning_curves(self, history, save_path):
        history_df = pd.DataFrame(history.history)

        plt.figure(figsize=(10, 5))
        plt.plot(history_df["loss"], label="Train Loss (MSE)")
        plt.plot(history_df["val_loss"], label="Val Loss (MSE)")
        if "mae" in history_df and "val_mae" in history_df:
            plt.plot(history_df["mae"], label="Train MAE", linestyle="--")
            plt.plot(history_df["val_mae"], label="Val MAE", linestyle="--")

        plt.title(f"LSTM Learning Curves - Region {self.region_number}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / MAE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Learning curves saved to: {save_path}")

    def evaluate_model(self, dataset, dataset_name):
        y_trues = []
        y_preds = []

        for x, y_true in dataset:
            y_pred = self.model.predict(x, verbose=0)
            y_trues.append(y_true.numpy().reshape(-1))
            y_preds.append(y_pred.reshape(-1))

        y_trues = np.concatenate(y_trues)
        y_preds = np.concatenate(y_preds)

        mse = mean_squared_error(y_trues, y_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_trues, y_preds)

        print(f"\n {dataset_name} Set Evaluation (Region {self.region_number}):")
        print(f"   - RMSE: {rmse:.2f}")
        print(f"   - MSE:  {mse:.2f}")
        print(f"   - MAE:  {mae:.2f}")

if __name__ == "__main__":
    input_width = 72
    label_width = 61
    shift = 0
    total_window = input_width + label_width + shift

    data_dir = "C:/competition/windAI_rug/WindAi/deep_learning/created_datasets"
    weight_dir = "C:/competition/windAI_rug/WindAi/deep_learning/weights"
    plot_dir = "C:/competition/windAI_rug/WindAi/deep_learning/results"

    for region_number in range(1, 5):  # NO1 → NO4
        region_name = f"ELSPOT NO{region_number}"
        print(f"\n--- Training for {region_name} ---")

        # Load dataset
        path = os.path.join(data_dir, f"scaled_features_power_MW_{region_name}.parquet")
        df = pd.read_parquet(path).drop(columns=["time", "bidding_area"], errors="ignore")

        # Split into train/val/test
        test_df = df[-total_window:]
        usable_df = df[:-total_window]
        n_usable = len(usable_df)
        train_df = usable_df[:int(n_usable * 0.7)]
        val_df   = usable_df[int(n_usable * 0.7):]

        # Window
        window = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_columns=["power_MW"]
        )

        for x_batch, _ in window.train.take(1):
            input_shape = x_batch.shape[1:]

        # Model
        rnn = LSTM(input_width=input_width, label_width=label_width, num_features=input_shape[-1], region_number=region_number)
        rnn.summary()

        # Training
        history = rnn.fit(window, weight_dir, epochs=100)

        # Save weights
        rnn.plot_learning_curves(history, save_path=os.path.join(plot_dir, f"lstm_learning_curves_elspot_no{region_number}.png"))

        # Evaluate
        rnn.evaluate_model(window.train, "Train")
        rnn.evaluate_model(window.val, "Validation")

        # Forecast
        pred, y_true = rnn.predict_last_window(window)
        rnn.plot_prediction(pred, y_true, save_path=os.path.join(plot_dir, f"lstm_forecast_plot_elspot_no{region_number}.png"))
    