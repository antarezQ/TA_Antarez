"""
Mengubah model BiLSTM-CRF (tensorflow-addons) menjadi format inferensi tanpa tfa:
  models/bilstm_crf_encoder_weights.npz  -> Embedding, BiLSTM, Dense, + Dense internal CRF
  models/bilstm_crf_transitions.npz      -> chain_kernel, left_boundary, right_boundary
Jalankan setelah train_export.py (butuh tensorflow-addons, Python <= 3.11).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ner_core import MODEL_DIR, build_bilstm_crf, build_bilstm_crf_encoder

v = json.load(open(os.path.join(MODEL_DIR, "bilstm_crf_vocab.json")))
V, T = len(v["word2idx"]), len(v["tag2idx"])
model, _, crf = build_bilstm_crf(V, T)
model.load_weights(os.path.join(MODEL_DIR, "bilstm_crf.weights.h5"))
enc = build_bilstm_crf_encoder(V, T)

src = [l for l in model.layers if l.weights]
dst = [l for l in enc.layers if l.weights]
for a, b in zip(src[:-1], dst[:-1]):          # embedding, bidirectional, time_distributed
    b.set_weights(a.get_weights())
w = {x.name.split("/")[-1].split(":")[0]: x.numpy() for x in crf.weights}
dst[-1].set_weights([w["kernel"], w["bias"]])  # Dense internal CRF
np.savez(os.path.join(MODEL_DIR, "bilstm_crf_encoder_weights.npz"), *enc.get_weights())
np.savez(os.path.join(MODEL_DIR, "bilstm_crf_transitions.npz"),
         chain_kernel=w["chain_kernel"], left_boundary=w["left_boundary"],
         right_boundary=w["right_boundary"])
print("OK")
