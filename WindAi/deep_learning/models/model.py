from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import os


#This is a custom loss function, that we use in some of the models
def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    huber = tf.keras.losses.huber(y_true, y_pred, delta=1.0)

    pred_diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    smoothness = tf.reduce_mean(tf.square(pred_diff))

    return (0.4 * tf.reduce_mean(mse) + 
            0.3 * tf.reduce_mean(mae) + 
            0.25 * tf.reduce_mean(huber) + 
            0.05 * smoothness)


class Model(ABC):

    @abstractmethod
    def _build_model(self):
        pass
    
    def fit(self, window, weights_dir, epochs=100):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=1e-6
        )

        callbacks = [early_stop]
            
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.7,  
            patience=8,   
            min_lr=1e-7,
            verbose=1,
            cooldown=2
        )
        callbacks.append(reduce_lr)

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(weights_dir, f"best_weights_region_{self.region_number}_{self.name}.h5"),
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint_cb)

        history = self.model.fit(
                window.train,
                validation_data=window.val,
                epochs=epochs,
                callbacks=callbacks
            )
        return history

    def summary(self):
        self.model.summary()

    def predict_test(self, window, first_batch_only=False):

        if first_batch_only:
            for x, y in window.test.take(1):
                prediction = self.model.predict(x)
            return prediction, y
        else:
            preds, trues = [], []
            for x, y in window.test:
                pred = self.model.predict(x)
                preds.append(pred)
                trues.append(y.numpy())
            return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)
        print(f" Model weights saved to: {filepath}")
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        print(f" Model weights loaded from: {filepath}")

    def plot_prediction(self, prediction, y_true, save_path):
        pred = prediction[:, -self.label_width:, :]
        true = y_true[:, -self.label_width:, :]

        plt.figure(figsize=(12, 4))
        plt.plot(true[0, :, 0], label="Actual")
        plt.plot(pred[0, :, 0], label="Predicted")
        plt.title("Power_MW Forecast: Next {} Hours".format(self.label_width))
        plt.xlabel("Hour")
        plt.ylabel("Power_MW")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Forecast prediction saved to: {save_path}")

    def plot_learning_curves(self, history, save_path):
        df = pd.DataFrame(history.history)
        plt.figure(figsize=(10, 5))
        plt.plot(df["loss"],   label="Train Loss")
        plt.plot(df["val_loss"], label="Val Loss")
        if "mae" in df:
            plt.plot(df["mae"],      label="Train MAE",   linestyle="--")
            plt.plot(df["val_mae"],  label="Val MAE",     linestyle="--")
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


