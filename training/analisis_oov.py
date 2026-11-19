"""
Analisis: seberapa baik tiap model mendeteksi token nama orang yang
(a) pernah muncul di data latih, vs (b) tidak pernah muncul (OOV).
Hasil disimpan ke models/analisis_oov.json dan ditampilkan di tab "Hasil Penelitian".

    python training/analisis_oov.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ner_core import (BASE_DIR, MAX_LEN, MODEL_DIR, BiLSTMCRFPredictor, BiLSTMPredictor,
                      CRFPredictor, sent2features)
from train_export import read_data

train = read_data(os.path.join(BASE_DIR, "dataset", "train.txt"))
test = read_data(os.path.join(BASE_DIR, "dataset", "test.txt"))
freq = Counter(t[0] for s in train for t in s)

crf, bil, bic = CRFPredictor(), BiLSTMPredictor(), BiLSTMCRFPredictor()
names = {"crf": "CRF", "bilstm": "BiLSTM", "bilstm_crf": "BiLSTM-CRF"}
stats = {k: {"dikenal": [0, 0], "oov": [0, 0]} for k in names}

for s in test:
    s = s[:MAX_LEN]  # model neural memakai max_len 75
    toks = [t[0] for t in s]
    gold = [t[3] for t in s]
    pred = {
        # CRF memakai POS tag asli dari data uji, sama seperti evaluasi di notebook
        "crf": list(crf.model.predict([sent2features([(t[0], t[1]) for t in s])])[0]),
        "bilstm": bil.predict(toks),
        "bilstm_crf": bic.predict(toks),
    }
    for i, (w, g) in enumerate(zip(toks, gold)):
        if not g.endswith("PER"):
            continue
        grp = "dikenal" if freq[w] > 0 else "oov"
        for k in names:
            stats[k][grp][1] += 1
            stats[k][grp][0] += int(pred[k][i].endswith("PER"))

contoh = ["Yeltsin", "Wayne", "Boris", "Kohl", "Helmut", "Rooney",
          "Barack", "Obama", "Yamal", "Raphinha"]
out = {
    "recall_per": {names[k]: {g: {"benar": v[0], "total": v[1]} for g, v in d.items()}
                   for k, d in stats.items()},
    "frekuensi_contoh": {w: freq[w] for w in contoh},
}
json.dump(out, open(os.path.join(MODEL_DIR, "analisis_oov.json"), "w"), indent=1)
for k, d in out["recall_per"].items():
    print(k, {g: f"{v['benar']}/{v['total']} = {v['benar'] / v['total']:.1%}" for g, v in d.items()})
