from pathlib import Path
import tensorflow as tf

import os, sys
HERE = os.path.dirname(__file__)                           
DL_DIR = os.path.abspath(os.path.join(HERE, ".."))         
MODELS_DIR = os.path.join(DL_DIR, "models")
for p in (DL_DIR, MODELS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.transformer import TransformerForecast
from explain.transformer_xai import TransformerXAI

BASE = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE / "weights"

CANDIDATES = [
    WEIGHTS_DIR / "best_weights_transformer_NO1_Transformer_0.15.h5",
    WEIGHTS_DIR / "best_weights_transformer_NO1_Transformer.h5",
]

WEIGHTS = next((p for p in CANDIDATES if p.exists()), None)
if WEIGHTS is None:
    raise FileNotFoundError(
        f"No suitable weights found in {WEIGHTS_DIR}\n"
        f"Found: {[p.name for p in WEIGHTS_DIR.glob('best_weights_transformer_*.h5')]}"
    )
print("Using weights:", WEIGHTS.name)

# Build the SAME architecture used in training (defaults from transformer.py)
CFG = dict(
    input_width=168,       # encoder length
    label_width=61,        # decoder length
    num_features=44,       # encoder features (decoder has 1)
    region_number=1,       # NO1
    name="Transformer_0.15",
    d_model=128,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
)

model = TransformerForecast(**CFG).model
try:
    model.load_weights(str(WEIGHTS))
    print("Weights loaded with exact graph match.")
except Exception as e:
    print("Exact match failed:", e)
    print("Trying by_name=True, skip_mismatch=True (NOT for final analysis).")
    model.load_weights(str(WEIGHTS), by_name=True, skip_mismatch=True)

# Wrap with XAI and smoke test
xai = TransformerXAI(model)

enc_in = tf.random.normal([1, CFG["input_width"], CFG["num_features"]])
dec_in = tf.random.normal([1, CFG["label_width"], 1])

att = xai.attention_probes(enc_in, dec_in)
print("Encoder L0 attention:", att["encoder"][0].shape)
print("Cross   L0 attention:", att["cross"][0].shape)

roll = xai.attention_rollout(att["encoder"])
print("Rollout shape:", roll.shape)

ig = xai.integrated_gradients(enc_in, dec_in, target="mean_last_step", which="encoder", m_steps=8)
print("IG encoder shape:", ig["encoder"].shape)

hi = xai.head_importance(enc_in, dec_in, target="last_step")
print("Head importance (encoder L0):", hi["encoder"][0].shape)

print("XAI pipeline ran.")

# Summaries & Plots

xai.summarize_cross_attention(att, layer=0, head="avg", dec_step=-1, topk=5)
xai.summarize_head_importance(hi, layer=0)
xai.summarize_ig(ig, topk=5)
xai.save_plots(att, roll, ig, outdir="xai_outputs")

