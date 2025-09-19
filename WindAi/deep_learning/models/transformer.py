import tensorflow as tf
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from WindAi.deep_learning.preprocessing.preprocess_windowing_region import WindowGenerator
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import Model


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i   = tf.cast(tf.range(d_model)[tf.newaxis, :], tf.float32)
        angle = pos / tf.pow(10000.0, (2*(i//2)) / d_model)
        pe = tf.where(tf.equal(i % 2, 0), tf.sin(angle), tf.cos(angle))
        self.pos_encoding = pe[tf.newaxis, ...]  # (1, seq_len, d_model)

    def call(self, x):
        return x + self.pos_encoding[:, : tf.shape(x)[1], :]


def TransformerBlock(embed_dim, num_heads, ff_dim, rate=0.1, causal=False, name=None):
    """Transformer block with optional causal self-attention (for decoder)."""
    inputs = tf.keras.Input(shape=(None, embed_dim), name=None if name is None else f"{name}_in")
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads,
        name=None if name is None else f"{name}_mha"
    )(inputs, inputs, use_causal_mask=causal)
    attn = tf.keras.layers.Dropout(rate, name=None if name is None else f"{name}_drop1")(attn)
    x1   = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=None if name is None else f"{name}_ln1")(inputs + attn)

    ff   = tf.keras.layers.Dense(ff_dim, activation="relu", name=None if name is None else f"{name}_ff1")(x1)
    ff   = tf.keras.layers.Dense(embed_dim, name=None if name is None else f"{name}_ff2")(ff)
    ff   = tf.keras.layers.Dropout(rate, name=None if name is None else f"{name}_drop2")(ff)
    x2   = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=None if name is None else f"{name}_ln2")(x1 + ff)

    return tf.keras.Model(inputs, x2, name=None if name is None else f"{name}_block")


class TransformerForecast(Model):
    def __init__(self,
                 input_width,
                 label_width,
                 num_features,
                 region_number,
                 name="Transformer_0.15",
                 d_model=128,
                 num_heads=4,
                 ff_dim=256,
                 num_layers=2,
                 decoder_causal=True  # <-- choose masked (True) or unmasked (False)
                 ):
        self.input_width   = input_width
        self.label_width   = label_width
        self.num_features  = num_features
        self.region_number = region_number
        self.name          = name
        self.d_model       = d_model
        self.num_heads     = num_heads
        self.ff_dim        = ff_dim
        self.num_layers    = num_layers
        self.decoder_causal = decoder_causal  # <-- store choice
        self.model         = self._build_model()

    def _build_model(self):
        enc_in = tf.keras.Input((self.input_width, self.num_features), name="encoder_input")
        dec_in = tf.keras.Input((self.label_width, 1),                name="decoder_input")

        proj_enc = tf.keras.layers.Dense(self.d_model, name="proj_enc")
        proj_dec = tf.keras.layers.Dense(self.d_model, name="proj_dec")
        x_enc = proj_enc(enc_in)
        x_dec = proj_dec(dec_in)

        x_enc = PositionalEncoding(self.input_width, self.d_model)(x_enc)
        x_dec = PositionalEncoding(self.label_width, self.d_model)(x_dec)

        # Encoder stack (no mask)
        for i in range(self.num_layers):
            x_enc = TransformerBlock(self.d_model, self.num_heads, self.ff_dim,
                                     rate=0.1, causal=False, name=f"enc_{i}")(x_enc)

        # Decoder stack
        x = x_dec
        for i in range(self.num_layers):
            # --- THIS is the explicit if/else you asked for ---
            if self.decoder_causal:
                # masked (causal) decoder self-attention
                x = TransformerBlock(self.d_model, self.num_heads, self.ff_dim,
                                     rate=0.1, causal=True, name=f"dec_{i}_self_masked")(x)
            else:
                # unmasked decoder self-attention
                x = TransformerBlock(self.d_model, self.num_heads, self.ff_dim,
                                     rate=0.1, causal=False, name=f"dec_{i}_self_nomask")(x)
            # cross-attention (no causal mask)
            cross = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.d_model // self.num_heads,
                name=f"dec_{i}_cross_mha"
            )(x, x_enc)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"dec_{i}_cross_ln")(x + cross)

        out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1), name="power_MW")(x)
        model = tf.keras.Model([enc_in, dec_in], out, name="TransformerForecast")

        # compile
        try:
            optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3, weight_decay=1e-5)
        except Exception:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        return model

    def fit(self, window, weights_dir, epochs=100):
        # teacher forcing: decoder input = <0> + y[:-1]
        def map_fn(x, y):
            zero   = tf.zeros_like(y[:, :1, :])
            dec_in = tf.concat([zero, y[:, :-1, :]], axis=1)
            return ({"encoder_input": x, "decoder_input": dec_in}, y)

        train_ds = window.train.map(map_fn)
        val_ds   = window.val  .map(map_fn)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                             restore_best_weights=True, min_delta=1e-6),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7,
                                                 patience=8, min_lr=1e-7, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    weights_dir, f"best_weights_transformer_NO{self.region_number}_{self.name}"
                    f"{'_masked' if self.decoder_causal else '_nomask'}.h5"
                ),
                save_weights_only=True, monitor="val_loss", mode="min", save_best_only=True, verbose=1
            ),
        ]

        return self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    def predict_test(self, window, first_batch_only=False):
        def map_fn(x, y):
            zero   = tf.zeros_like(y[:, :1, :])
            dec_in = tf.concat([zero, y[:, :-1, :]], axis=1)
            return ({"encoder_input": x, "decoder_input": dec_in}, y)

        test_ds = window.test.map(map_fn)

        if first_batch_only:
            for (inputs, y_true) in test_ds.take(1):
                pred = self.model.predict(inputs, verbose=0)
                return pred[:, -self.label_width:, :], y_true.numpy()
            raise ValueError("Test dataset is empty; cannot predict last window.")
        else:
            preds, trues = [], []
            for (inputs, y_true) in test_ds:
                pred = self.model.predict(inputs, verbose=0)
                preds.append(pred[:, -self.label_width:, :])
                trues.append(y_true.numpy())
            if not preds:
                raise ValueError("Test dataset is empty; cannot predict all the windows.")
            return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

    def evaluate_model(self, dataset, dataset_name):
        y_trues, y_preds = [], []
        for x, y_true in dataset:
            dec0 = tf.zeros((tf.shape(x)[0], self.label_width, 1))
            y_pred = self.model.predict({"encoder_input": x, "decoder_input": dec0}, verbose=0)
            y_trues.append(y_true.numpy().reshape(-1))
            y_preds.append(y_pred.reshape(-1))

        y_trues = np.concatenate(y_trues)
        y_preds = np.concatenate(y_preds)
        mse  = mean_squared_error(y_trues, y_preds)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_trues, y_preds)

        print(f"\n{dataset_name} Set Evaluation (Region {self.region_number}):")
        print(f"   - RMSE: {rmse:.2f}")
        print(f"   - MSE:  {mse:.2f}")
        print(f"   - MAE:  {mae:.2f}")
