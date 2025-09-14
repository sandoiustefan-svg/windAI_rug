"""
TransformerXAI: Explainability utilities for Transformer-based forecasting models.

This module extends the TransformerForecast model with post-hoc explainability tools,
focusing on mechanistic interpretability of the attention mechanism and feature attribution.

Implemented methods:
--------------------
1. Attention Probes
   - Extracts encoder self-attention, decoder self-attention, and cross-attention maps.
   - Supports per-head visualization and averaged attention.

2. Attention Rollout
   - Computes layer-wise aggregated attention across encoder layers.
   - Reveals how information flows through the model’s depth.

3. Integrated Gradients (IG)
   - Estimates feature importance over time by integrating gradients
     along a baseline -> input path.
   - Can target specific steps (e.g., last prediction step).

4. Head Importance
   - Quantifies contribution of individual attention heads via gradient norms.

Usage:
------
- Import and wrap a trained TransformerForecast model:
    from WindAi.deep_learning.models.transformer import TransformerForecast
    from WindAi.deep_learning.explain.transformer_xai import TransformerXAI

    forecast = TransformerForecast(...)
    forecast.model.load_weights("path/to/weights.h5")
    xai = TransformerXAI(forecast.model)

- Run analyses:
    attn = xai.attention_probes(enc_input, dec_input)
    rollout = xai.attention_rollout(attn["encoder"])
    ig = xai.integrated_gradients(enc_input, dec_input, target="last_step")
    hi = xai.head_importance(enc_input, dec_input)

- Save visualizations:
    xai.save_plots(attn, rollout, ig, outdir="xai_outputs/")

Notes:
------
- No SHAP values are used.
- Mechanistic interpretability is emphasized (inspecting attention heads, flows).
- Integrated Gradients is used for feature attribution.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Literal, Any

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import os, sys
HERE = os.path.dirname(__file__)
DL_DIR = os.path.abspath(os.path.join(HERE, ".."))         
MODELS_DIR = os.path.join(DL_DIR, "models")
for p in (DL_DIR, MODELS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.transformer import TransformerForecast, PositionalEncoding


TargetType = Literal["last_step", "mean_last_step", "mean_all"]


def _find_layers_recursive(model: tf.keras.Model, cls) -> List[tf.keras.layers.Layer]:
    """
    Recursively collect layers of type `cls` within (nested) models.
    """
    found = []
    for layer in model.layers:
        if isinstance(layer, cls):
            found.append(layer)
        if isinstance(layer, tf.keras.Model):
            found.extend(_find_layers_recursive(layer, cls))
    return found


def _get_block_mha(block_model: tf.keras.Model) -> tf.keras.layers.MultiHeadAttention:
    """
    Return the first MultiHeadAttention layer inside a TransformerBlock submodel.
    """
    for layer in block_model.layers:
        if isinstance(layer, tf.keras.layers.MultiHeadAttention):
            return layer
        if isinstance(layer, tf.keras.Model):
            sub = _get_block_mha(layer)
            if sub is not None:
                return sub
    raise ValueError("No MultiHeadAttention found inside the provided block model.")


class TransformerXAI:
    """
    XAI helper bound to a compiled Transformer (Keras Model).
    It reconstructs the forward pass using the existing (trained) layers to expose attention scores.
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model
        self._inspect_graph()

    # graph parsing
    def _inspect_graph(self) -> None:
        """
        Parse the top-level model to locate:
          - projection Dense layers (proj_enc, proj_dec),
          - positional encoding layers (two instances of PositionalEncoding),
          - encoder & decoder TransformerBlock submodels,
          - cross-attention MultiHeadAttention layers and their following LayerNorms,
          - TimeDistributed(Dense(1)) head.
        """
        # named projections
        self.proj_enc = self.model.get_layer("proj_enc")
        self.proj_dec = self.model.get_layer("proj_dec")

        # positional encodings (two instances; order in model.layers is encoder then decoder)
        pos_layers = [l for l in self.model.layers if isinstance(l, PositionalEncoding)]
        if len(pos_layers) != 2:
            raise RuntimeError(
                f"Expected exactly 2 PositionalEncoding layers, found {len(pos_layers)}."
            )
        self.pos_enc_enc, self.pos_enc_dec = pos_layers[0], pos_layers[1]

        # Collect TransformerBlock submodels. The first N belong to encoder, next N to decoder.
        block_models = [l for l in self.model.layers
                        if isinstance(l, tf.keras.Model) and _find_layers_recursive(l, tf.keras.layers.MultiHeadAttention)]
        if len(block_models) % 2 != 0:
            raise RuntimeError(
                "Uneven number of TransformerBlock submodels; expected encoder and decoder stacks of equal depth."
            )
        half = len(block_models) // 2
        self.encoder_blocks: List[tf.keras.Model] = block_models[:half]
        self.decoder_blocks: List[tf.keras.Model] = block_models[half:]

        # Top-level cross-attention layers (not inside submodels)
        self.cross_attn_layers: List[tf.keras.layers.MultiHeadAttention] = [
            l for l in self.model.layers if isinstance(l, tf.keras.layers.MultiHeadAttention)
        ]
        if len(self.cross_attn_layers) != len(self.decoder_blocks):
            # filter to likely top-level cross-attn layers
            top_level = []
            for l in self.cross_attn_layers:
                if getattr(l, "_inbound_nodes", None):
                    top_level.append(l)
            self.cross_attn_layers = top_level

        # LayerNorms after each cross-attn (residual + norm)
        ln_after_cross = []
        record = False
        for l in self.model.layers:
            if l in self.cross_attn_layers:
                record = True
                continue
            if record:
                if isinstance(l, tf.keras.layers.LayerNormalization):
                    ln_after_cross.append(l)
                    record = False
        if len(ln_after_cross) != len(self.cross_attn_layers):
            lns = [l for l in self.model.layers if isinstance(l, tf.keras.layers.LayerNormalization)]
            ln_after_cross = lns[-len(self.cross_attn_layers):]

        self.ln_after_cross: List[tf.keras.layers.LayerNormalization] = ln_after_cross

        # Output head (TimeDistributed(Dense(1)))
        td_layers = [l for l in self.model.layers if isinstance(l, tf.keras.layers.TimeDistributed)]
        self.out_head = td_layers[-1] if td_layers else None

    # helpers

    def _forward_encoder(self, enc_in: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Run encoder stream and collect self-attention maps for each layer.
        """
        x = self.proj_enc(enc_in, training=training)
        x = self.pos_enc_enc(x)
        enc_attns = []
        for block in self.encoder_blocks:
            mha = _get_block_mha(block)
            _, scores = mha(x, x, return_attention_scores=True, training=training)
            enc_attns.append(scores)  # (B, H, L_enc, L_enc)
            x = block(x, training=training)
        return x, enc_attns

    def _forward_decoder(self, dec_in: tf.Tensor, enc_ctx: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, List[tf.Tensor], List[tf.Tensor]]:
        """
        Run decoder stream and collect (self-attn, cross-attn) scores for each layer.
        """
        x = self.proj_dec(dec_in, training=training)
        x = self.pos_enc_dec(x)
        dec_self_attns = []
        cross_attns = []
        for block, cross_attn, ln in zip(self.decoder_blocks, self.cross_attn_layers, self.ln_after_cross):
            # decoder self-attn
            mha_dec = _get_block_mha(block)
            _, dec_scores = mha_dec(x, x, return_attention_scores=True, training=training)
            dec_self_attns.append(dec_scores)  # (B, H, L_dec, L_dec)
            # block forward
            x = block(x, training=training)
            # cross-attn on the main path
            y, cross_scores = cross_attn(x, enc_ctx, return_attention_scores=True, training=training)
            cross_attns.append(cross_scores)   # (B, H, L_dec, L_enc)
            x = ln(x + y, training=training)
        return x, dec_self_attns, cross_attns

    def attention_probes(self, enc_in: np.ndarray | tf.Tensor, dec_in: np.ndarray | tf.Tensor, training: bool = False) -> Dict[str, List[np.ndarray]]:
        """
        Return attention score tensors for all layers as numpy arrays.
        """
        enc_in = tf.convert_to_tensor(enc_in, dtype=tf.float32)
        dec_in = tf.convert_to_tensor(dec_in, dtype=tf.float32)
        enc_ctx, enc_scores = self._forward_encoder(enc_in, training=training)
        _, dec_self, cross = self._forward_decoder(dec_in, enc_ctx, training=training)
        return {
            "encoder": [e.numpy() for e in enc_scores],
            "decoder": [d.numpy() for d in dec_self],
            "cross":   [c.numpy() for c in cross],
        }

    def attention_rollout(self, attn_list: List[np.ndarray], residual: bool = True) -> np.ndarray:
        """
        Compose attention across layers (head-averaged), optionally with residual identity.
        Returns (B, L, L).
        """
        assert len(attn_list) > 0, "No attention maps provided."
        A = [a.mean(axis=1) for a in attn_list]  # (B, L, L)
        R = None
        for a in A:
            a = np.maximum(a, 0.0)
            if residual:
                B, L, _ = a.shape
                a = a + np.eye(L)[None, :, :]
            a = a / (a.sum(axis=-1, keepdims=True) + 1e-8)
            R = a if R is None else np.matmul(a, R)
        return R

    def head_importance(self, enc_in: np.ndarray | tf.Tensor, dec_in: np.ndarray | tf.Tensor, target: TargetType = "last_step") -> Dict[str, List[np.ndarray]]:
        """
        Gradient-based head importance. Robust to self-attn scores not being on the forward path.
        For encoder/decoder self-attn, if gradients are None, returns zeros of shape (B, H) and prints a note.
        For cross-attn (on-path), returns true gradient norms.
        """
        enc_in = tf.convert_to_tensor(enc_in, dtype=tf.float32)
        dec_in = tf.convert_to_tensor(dec_in, dtype=tf.float32)

        notices = []

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(enc_in)
            tape.watch(dec_in)
            enc_ctx, enc_attn = self._forward_encoder(enc_in, training=False)
            dec_out, dec_attn, cross_attn = self._forward_decoder(dec_in, enc_ctx, training=False)
            # Map to model outputs
            if self.out_head is not None:
                y = self.out_head(dec_out)
            else:
                y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1), name="tmp_head")(dec_out)

            if target == "last_step":
                scalar = y[:, -1, 0]
            elif target == "mean_last_step":
                scalar = tf.reduce_mean(y[:, -1, 0], axis=0)
            elif target == "mean_all":
                scalar = tf.reduce_mean(y)
            else:
                raise ValueError(f"Unknown target spec: {target}")

        def per_head_norm(grads: Optional[tf.Tensor], ref_scores: tf.Tensor, label: str) -> np.ndarray:
            if grads is None:
                # Return zeros and record a note
                B, H = ref_scores.shape[0], ref_scores.shape[1]
                notices.append(f"{label}: gradients unavailable for a self-attention layer; returned zeros (scores not on output path).")
                return np.zeros((B, H), dtype=np.float32)
            # grads shape matches attn: (B, H, Lq, Lk) -> reduce over tokens
            g = tf.norm(grads, ord="euclidean", axis=[-1, -2])
            return g.numpy()

        enc_hi = [per_head_norm(tape.gradient(scalar, a), a, "encoder") for a in enc_attn]
        dec_hi = [per_head_norm(tape.gradient(scalar, a), a, "decoder") for a in dec_attn]
        cross_hi = [per_head_norm(tape.gradient(scalar, a), a, "cross") for a in cross_attn]
        del tape

        # Deduplicate notices and print once
        if notices:
            print("\\n[head_importance] Notes:")
            for n in sorted(set(notices)):
                print(" -", n)

        return {"encoder": enc_hi, "decoder": dec_hi, "cross": cross_hi}

    # Integrated Gradients

    def integrated_gradients(
        self,
        enc_in: np.ndarray | tf.Tensor,
        dec_in: np.ndarray | tf.Tensor,
        target: TargetType = "last_step",
        which: Literal["encoder", "decoder", "both"] = "encoder",
        m_steps: int = 64,
        baseline: Literal["zero", "mean"] = "zero",
    ) -> Dict[str, np.ndarray]:
        """
        Compute Integrated Gradients for inputs.
        Returns dict with 'encoder' and/or 'decoder' keys, each (B, L, F) attributions.
        """
        enc_in = tf.convert_to_tensor(enc_in, dtype=tf.float32)
        dec_in = tf.convert_to_tensor(dec_in, dtype=tf.float32)

        B = tf.shape(enc_in)[0]
        if baseline == "zero":
            enc_base = tf.zeros_like(enc_in)
            dec_base = tf.zeros_like(dec_in)
        elif baseline == "mean":
            enc_mu = tf.reduce_mean(enc_in, axis=0, keepdims=True)
            dec_mu = tf.reduce_mean(dec_in, axis=0, keepdims=True)
            enc_base = tf.repeat(enc_mu, B, axis=0)
            dec_base = tf.repeat(dec_mu, B, axis=0)
        else:
            raise ValueError("baseline must be 'zero' or 'mean'")

        alphas = tf.linspace(0.0, 1.0, m_steps + 1)[1:]  # exclude 0
        enc_attrs = tf.zeros_like(enc_in)
        dec_attrs = tf.zeros_like(dec_in)

        for a in tf.unstack(alphas):
            enc_step = enc_base + a * (enc_in - enc_base)
            dec_step = dec_base + a * (dec_in - dec_base)
            with tf.GradientTape() as tape:
                tape.watch(enc_step)
                tape.watch(dec_step)
                y = self.model([enc_step, dec_step], training=False)  # (B, T, 1)

                if target == "last_step":
                    scalar = y[:, -1, 0]
                elif target == "mean_last_step":
                    scalar = tf.reduce_mean(y[:, -1, 0], axis=0)
                elif target == "mean_all":
                    scalar = tf.reduce_mean(y)
                else:
                    raise ValueError(f"Unknown target spec: {target}")

            if which in ("encoder", "both"):
                g_enc = tape.gradient(scalar, enc_step)
                enc_attrs += g_enc
            if which in ("decoder", "both"):
                g_dec = tape.gradient(scalar, dec_step)
                dec_attrs += g_dec

        results = {}
        if which in ("encoder", "both"):
            results["encoder"] = ((enc_in - enc_base) * enc_attrs) / float(len(alphas))
            results["encoder"] = results["encoder"].numpy()
        if which in ("decoder", "both"):
            results["decoder"] = ((dec_in - dec_base) * dec_attrs) / float(len(alphas))
            results["decoder"] = results["decoder"].numpy()
        return results
    
# Convenience utilities
    def summarize_cross_attention(self, att, layer=0, head="avg", dec_step=-1, topk=5, batch=0):
        """
        Print top-k encoder steps that a given decoder step attends to.
        """
        cross = att["cross"][layer][batch]  # (H, L_dec, L_enc)
        if head == "avg":
            cross_map = cross.mean(axis=0)  # (L_dec, L_enc)
        else:
            cross_map = cross[head]
        top_enc_idx = np.argsort(cross_map[dec_step])[-topk:][::-1]
        print(f"\nTop {topk} encoder steps for decoder step {dec_step} (layer {layer}, head {head}): {top_enc_idx.tolist()}")
        print("Their weights:", cross_map[dec_step, top_enc_idx].tolist())

    def summarize_head_importance(self, hi, batch=0, layer=0):
        """
        Print head importance ranking for one cross-attention layer.
        """
        cross_hi = hi["cross"][layer][batch]  # (H,)
        order = np.argsort(cross_hi)[::-1]
        print("\nCross-attention head importance (layer {layer}) high -> low:", cross_hi[order].tolist())
        print("Head order (indices):", order.tolist())

    def summarize_ig(self, ig, batch=0, topk=5):
        """
        Print top features and timesteps by IG absolute value.
        """
        E = ig["encoder"][batch]  # (L_enc, F_enc)
        feat_scores = np.abs(E).sum(axis=0)
        time_scores = np.abs(E).sum(axis=1)
        print("\nTop features by |IG|:", np.argsort(feat_scores)[-topk:][::-1].tolist())
        print("Top time steps by |IG|:", np.argsort(time_scores)[-topk:][::-1].tolist())

    def save_plots(self, att, roll, ig, outdir="xai_outputs", batch=0):
        """
        Save heatmaps to disk for quick inspection.
        """
        outdir = Path(outdir); outdir.mkdir(exist_ok=True)

        # Cross-attention (head-avg, layer 0)
        cross_avg = att["cross"][0][batch].mean(axis=0)
        plt.imshow(cross_avg, aspect="auto"); plt.title("Cross-attention (layer 0, head-avg)")
        plt.xlabel("Encoder time"); plt.ylabel("Decoder time")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(outdir / "cross_attn_l0_avg.png"); plt.close()

        # Encoder rollout
        plt.imshow(roll[batch], aspect="auto"); plt.title("Encoder attention rollout")
        plt.xlabel("Source (enc)"); plt.ylabel("Target (enc)")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(outdir / "encoder_rollout.png"); plt.close()

        # IG encoder
        E = ig["encoder"][batch]
        plt.imshow(E.T, aspect="auto"); plt.title("IG (encoder features × time)")
        plt.ylabel("Feature index"); plt.xlabel("Encoder time")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(outdir / "ig_encoder.png"); plt.close()

        print("\nSaved plots to:", outdir.resolve())


# Quick sanity check

if __name__ == "__main__":
    import os
    import tensorflow as tf

    # Tiny toy config mirroring model interface
    cfg = dict(
        input_width=16,     # encoder window length
        label_width=4,      # decoder/horizon length
        num_features=8,     # encoder feature count
        region_number=1,
        d_model=16,
        num_heads=2,
        ff_dim=32,
        num_layers=1,
    )

    # 1) Build a toy TransformerForecast
    forecast = TransformerForecast(**cfg)
    model = forecast.model
    print("Toy Transformer built with positional encodings.")

    # 2) Wrap with XAI
    xai = TransformerXAI(model)

    # 3) Safety check: ensure encoder PE is assigned to the long sequence (input_width)
    #    If not, swap encoder/decoder PE references once.
    try:
        _probe_enc = tf.random.normal([1, cfg["input_width"], cfg["num_features"]])  # (1,16,8)
        _ = xai.pos_enc_enc(tf.random.normal([1, cfg["input_width"], xai.proj_enc.units]))  # (1,16,d_model)
    except Exception as e:
        msg = str(e)
        if "Incompatible shapes" in msg and hasattr(xai, "pos_enc_enc") and hasattr(xai, "pos_enc_dec"):
            print("PositionalEncoding order looks reversed; swapping enc/dec PE for demo...")
            xai.pos_enc_enc, xai.pos_enc_dec = xai.pos_enc_dec, xai.pos_enc_enc
        else:
            raise

    # 4) Dummy inputs with correct shapes
    enc = tf.random.normal([1, cfg["input_width"],  cfg["num_features"]])  # (1,16,8)
    dec = tf.random.normal([1, cfg["label_width"], 1])                     # (1,4,1)

    # 5) Run XAI pipeline
    att = xai.attention_probes(enc, dec)
    print("Encoder attention L0:", att["encoder"][0].shape)   # (1, H, 16, 16)
    print("Cross   attention L0:", att["cross"][0].shape)     # (1, H, 4, 16)

    roll = xai.attention_rollout(att["encoder"])
    print("Rollout shape:", roll.shape)                       # (1, 16, 16)

    ig = xai.integrated_gradients(enc, dec, target="mean_last_step", which="encoder", m_steps=8)
    print("IG encoder shape:", ig["encoder"].shape)           # (1, 16, 8)

    hi = xai.head_importance(enc, dec, target="last_step")
    print("Head importance (encoder L0):", hi["encoder"][0].shape)  # (1, H)

    # 6) Save demo plots
    outdir = os.path.join(os.path.dirname(__file__), "xai_outputs_demo")
    xai.save_plots(att, roll, ig, outdir=outdir)
    print("Demo XAI run finished. Plots saved to:", outdir)



