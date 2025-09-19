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

class LSTM(Model):
    def __init__(self, input_width, label_width, num_features, region_number, name="LSTM"):
        self.input_width = input_width
        self.label_width = label_width
        self.num_features = num_features
        self.region_number = region_number
        self.name = name

        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.input_width, self.num_features)))
        model.add(tf.keras.layers.GaussianNoise(0.1))

        model.add(tf.keras.layers.LSTM(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.LSTM(512, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.LSTM(512, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
        model.add(tf.keras.layers.Dropout(0.4))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.LSTM(256, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.LSTM(128, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l2(1e-3)))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Lambda(lambda x: x[:, -self.label_width:, :]))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu')))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4,
                        beta_1=0.9, 
                        beta_2=0.999,
                        epsilon=1e-7,  
                        clipnorm=1.0
                        ),
            loss=combined_loss,
            metrics=["mae", "mse"]
        )

        return model
    


    