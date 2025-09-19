import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from WindAi.deep_learning.preprocessing.preprocess_windowing_region import WindowGenerator
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna 

class RNN_trial:
    def __init__(self, input_width, label_width, num_features, region_number):
        self.input_width = input_width
        self.label_width = label_width
        self.num_features = num_features
        self.model = None
        self.region_number = region_number

    def _build_model(self, trial):
        n_layers = trial.suggest_int("n_layers", 2, 7)
        units = trial.suggest_categorical("gru_units", [32, 64, 128, 256, 512])
        dropout_rate = 0.0
        recurrent_dropout = 0.0
        dense_units = trial.suggest_categorical("dense_units", [8, 16, 32])
        l2_strength = trial.suggest_float("l2_strength", 1e-5, 1e-3, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        loss_function = trial.suggest_categorical("loss", ["mse", "mae"])
        #activation = trial.suggest_categorical("activation", ["tanh", "relu", "leaky_relu"])

        activation = "tanh"
        recurrent_activation = "sigmoid"


        if loss_function == "mae":
            loss = tf.keras.losses.MeanAbsoluteError()
        else:
            loss = tf.keras.losses.MeanSquaredError()

        regularizer = tf.keras.regularizers.l2(l2_strength)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_width, self.num_features)))
        model.add(tf.keras.layers.GaussianNoise(0.1))

        for i in range(n_layers):
            model.add(tf.keras.layers.GRU(
                units,
                return_sequences=True,
                activation=activation,
                recurrent_activation=recurrent_activation,
                reset_after=True,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=regularizer,
                name=f"gru_{i}"
            ))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Lambda(lambda x: x[:, -61:, :]))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dense_units, activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['mae']
        )
        return model
    
    def build_with_trial(self, trial):
        self.model = self._build_model(trial)

    
    def fit(self, window, weights_dir, epochs=100):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        )
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5
        )
        checkpoint_path = os.path.join(weights_dir, f"best_model_region_{self.region_number}.h5")
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
            callbacks=[early_stop, lr_schedule]
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
        print(f" Model weights saved to: {filepath}")
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        print(f" Model weights loaded from: {filepath}")
    
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

        plt.title(f"Learning Curves - Region {self.region_number}")
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

def objective(trial):
    region_number = 1  # You can loop over multiple later
    input_width = 168
    label_width = 61
    shift = 0

    data_dir = "/home2/s5549329/windAI_rug/WindAi/deep_learning/created_datasets"
    weight_dir = "/home2/s5549329/windAI_rug/WindAi/deep_learning/weights"
    plot_dir = "/home2/s5549329/windAI_rug/WindAi/deep_learning/results"

    path = f"/home2/s5549329/windAI_rug/WindAi/deep_learning/created_datasets//scaled_features_power_MW_NO{region_number}.parquet"
    df = pd.read_parquet(path).drop(columns=["time"], errors="ignore")

    test_df = df[-(input_width + 61):]
    usable_df = df[:-(input_width + 61)]
    n_usable = len(usable_df)
    train_df = usable_df[:int(n_usable * 0.7)]
    val_df = usable_df[int(n_usable * 0.7):]

    window = WindowGenerator(
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=["power_MW"]
    )

    for x_batch, y_batch in window.train.take(1):
            input_shape = x_batch.shape[1:]
            output_shape = y_batch.shape[1:]
            print(f"Input shape (X): {input_shape}")
            print(f"Output shape (Y): {output_shape}")
            print(f"Number of input features: {input_shape[-1]}")
            print("\nFirst input timestep (x[0, 0, :]):")
            print(x_batch[0, 0, :].numpy())

            print("\nFirst label timestep (y[0, 0, :]):")
            print(y_batch[0, 0, :].numpy())

    rnn = RNN_trial(input_width, label_width, input_shape[-1], region_number)
    rnn.build_with_trial(trial)
    rnn.summary()

    history = rnn.fit(window, weight_dir, epochs=100)

    pred, y_true = rnn.predict_last_window(window)
    forecast_plot_path = os.path.join(plot_dir, f"forecast_plot_trial_{trial.number}.png")
    rnn.plot_prediction(pred, y_true, save_path=forecast_plot_path)

    learning_plot_path = os.path.join(plot_dir, f"learning_curve_trial_{trial.number}.png")
    rnn.plot_learning_curves(history, save_path=learning_plot_path)

    #rnn.plot_learning_curves(history, save_path=os.path.join(plot_dir, f"learning_curves_elspot_no{region_number}.png"))

    # Evaluate
    #rnn.evaluate_model(window.train, "Train")
    #rnn.evaluate_model(window.val, "Validation")

    #pred, y_true = rnn.predict_last_window(window)
    #rnn.plot_prediction(pred, y_true, save_path=os.path.join(plot_dir, f"forecast_plot_elspot_no{region_number}.png"))

    return history.history['val_loss'][-1]
    
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best hyperparameters:", study.best_params)

    log_dir = "/home2/s5549329/windAI_rug/WindAi/deep_learning"
    os.makedirs(log_dir, exist_ok=True) 
    csv_path = os.path.join(log_dir, "optuna_trials_results.csv")

    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    print("Saved Optuna results to {csv_path}")
    
