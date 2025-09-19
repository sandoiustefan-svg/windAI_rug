import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from WindAi.deep_learning.preprocessing.preprocess_windowing_region import WindowGenerator
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from model import Model, combined_loss

class GruDeep(Model):
    def __init__(self, input_width, label_width, num_features, region_number, name="GRU_CNN"):
        self.input_width = input_width
        self.label_width = label_width
        self.num_features = num_features
        self.region_number = region_number
        self.name = name

        self.model = self._build_model()

    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_width, self.num_features))
        x = tf.keras.layers.GaussianNoise(0.1)(inputs)

        x = tf.keras.layers.GRU(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.GRU(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        shared_features = tf.keras.layers.GRU(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        shared_features = tf.keras.layers.Dropout(0.2)(shared_features)
        shared_features = tf.keras.layers.LayerNormalization()(shared_features)

        shared_features = tf.keras.layers.GRU(256, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        shared_features = tf.keras.layers.Dropout(0.2)(shared_features)
        shared_features = tf.keras.layers.LayerNormalization()(shared_features)

        shared_features = tf.keras.layers.GRU(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        shared_features = tf.keras.layers.Dropout(0.2)(shared_features)
        shared_features = tf.keras.layers.LayerNormalization()(shared_features)
        #(batch_size, 336, 128)
        shared_features = tf.keras.layers.Lambda(lambda x: x[:, -self.label_width:, :])(shared_features)
        #(batchh_size, 61)
        main_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(16, activation='relu'))(shared_features)
        main_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1), name="main_output")(main_output)

        model = model = tf.keras.Model(inputs=inputs, outputs=main_output)

        model.compile(
            optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3,
                        weight_decay=1e-5,
                        clipnorm=1.0
                        ),
            loss=combined_loss,
            metrics=["mae", "mse"]
        )

        return model
    
    def fit(self, window, weights_dir, epochs=100):
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
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

        BASE_LR = 1e-4
        MAX_LR  = 1e-3
        PCT_START = 0.3  # fraction of epochs spent increasing

        def one_cycle_lr(epoch):
            """Linear ramp up then ramp down."""
            pct = epoch / epochs
            if pct <= PCT_START:
                return BASE_LR + (MAX_LR - BASE_LR) * (pct / PCT_START)
            else:
                return MAX_LR - (MAX_LR - BASE_LR) * ((pct - PCT_START) / (1 - PCT_START))

        lr_callback = tf.keras.callbacks.LearningRateScheduler(one_cycle_lr, verbose=1)

        callbacks.append(lr_callback)

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
    