"""
Melatih ulang & menyimpan 3 model TA (CRF, BiLSTM, BiLSTM-CRF) untuk dipakai di Streamlit.

Arsitektur, hyperparameter, dan preprocessing diambil PERSIS dari:
  - CRF_TA_FIX.ipynb
  - BiLSTM_TA_FIX.ipynb
  - BiLSTM_CRF_FIX.ipynb
Satu-satunya perubahan: daftar kata/tag di-sort agar word2idx deterministik,
lalu disimpan ke JSON (di notebook asli BiLSTM, urutan `set()` berubah tiap run).

Pemakaian:
    python training/train_export.py --data dataset --out models --model all
"""
import argparse
import json
import os
import time

import numpy as np

MAX_LEN = 75
EVAL_LABELS = ["B-PER", "I-PER", "O"]


# ---------------------------------------------------------------- data
def read_data(file_path):
    sentences, sentence = [], []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                splits = line.strip().split()
                if len(splits) == 4:
                    sentence.append(tuple(splits))
    if sentence:
        sentences.append(sentence)
    return sentences


def label_summary(y_true_flat, y_pred_flat):
    """Ringkasan untuk halaman analisis error di app."""
    from collections import Counter
    err_true, err_pred, ok_true = Counter(), Counter(), Counter()
    for t, p in zip(y_true_flat, y_pred_flat):
        if t == p:
            ok_true[t] += 1
        else:
            err_true[t] += 1
            err_pred[p] += 1
    return {"error_by_true": dict(err_true), "error_by_pred": dict(err_pred),
            "correct_by_true": dict(ok_true)}


def report_dict(y_true_flat, y_pred_flat):
    from sklearn.metrics import classification_report
    rep = classification_report(y_true_flat, y_pred_flat, labels=EVAL_LABELS,
                                digits=4, output_dict=True, zero_division=0)
    print(classification_report(y_true_flat, y_pred_flat, labels=EVAL_LABELS,
                                digits=4, zero_division=0))
    return rep


# ---------------------------------------------------------------- CRF
def train_crf(train, valid, test, out):
    import joblib
    from sklearn_crfsuite import CRF
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ner_core import sent2features

    X_train = [sent2features([(w, p) for w, p, _, _ in s]) for s in train]
    y_train = [[l for _, _, _, l in s] for s in train]
    X_test = [sent2features([(w, p) for w, p, _, _ in s]) for s in test]
    y_test = [[l for _, _, _, l in s] for s in test]

    crf = CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100,
              all_possible_transitions=True)
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)

    joblib.dump(crf, os.path.join(out, "crf_model.joblib"))
    yt = [t for s in y_test for t in s]
    yp = [t for s in y_pred for t in s]
    return {"report": report_dict(yt, yp), **label_summary(yt, yp)}


# ---------------------------------------------------------------- vocab
def build_vocab(train):
    words = sorted({w for s in train for w, _, _, _ in s})
    tags = sorted({l for s in train for _, _, _, l in s})
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0
    tag2idx = {t: i for i, t in enumerate(tags)}
    return word2idx, tag2idx


def flat_eval(X, y_true, y_pred, idx2tag, pad_idx):
    yt, yp = [], []
    for i in range(len(X)):
        for j in range(MAX_LEN):
            if X[i][j] != pad_idx:
                yt.append(idx2tag[int(y_true[i][j])])
                yp.append(idx2tag[int(y_pred[i][j])])
    return yt, yp


# ---------------------------------------------------------------- BiLSTM
def train_bilstm(train, valid, test, out, epochs):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ner_core import build_bilstm, capitalization_feature

    word2idx, tag2idx = build_vocab(train)
    idx2tag = {i: t for t, i in tag2idx.items()}

    def prep(sents):
        X = [[word2idx.get(w, 1) for w, _, _, _ in s] for s in sents]
        y = [[tag2idx[l] for _, _, _, l in s] for s in sents]
        c = [[capitalization_feature(w) for w, _, _, _ in s] for s in sents]
        X = pad_sequences(X, maxlen=MAX_LEN, padding="post")
        y = pad_sequences(y, maxlen=MAX_LEN, padding="post")
        c = pad_sequences(c, maxlen=MAX_LEN, padding="post")
        return X, y, c

    X_tr, y_tr, c_tr = prep(train)
    X_va, y_va, c_va = prep(valid)
    X_te, y_te, c_te = prep(test)
    n_tags = len(tag2idx)
    y_tr_oh = np.array([to_categorical(i, num_classes=n_tags) for i in y_tr])
    y_va_oh = np.array([to_categorical(i, num_classes=n_tags) for i in y_va])

    model = build_bilstm(len(word2idx), n_tags)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit([X_tr, c_tr], y_tr_oh, validation_data=([X_va, c_va], y_va_oh),
                     batch_size=32, epochs=epochs, verbose=2)

    y_pred = np.argmax(model.predict([X_te, c_te], verbose=0), axis=-1)
    model.save_weights(os.path.join(out, "bilstm.weights.h5"))
    np.savez(os.path.join(out, "bilstm_weights.npz"), *model.get_weights())  # dipakai app
    with open(os.path.join(out, "bilstm_vocab.json"), "w") as f:
        json.dump({"word2idx": word2idx, "tag2idx": tag2idx, "max_len": MAX_LEN}, f)

    yt, yp = flat_eval(X_te, y_te, y_pred, idx2tag, 0)
    return {"report": report_dict(yt, yp), **label_summary(yt, yp),
            "history": {k: [float(v) for v in vs] for k, vs in hist.history.items()}}


# ---------------------------------------------------------------- BiLSTM-CRF
def train_bilstm_crf(train, valid, test, out, epochs):
    import tensorflow as tf
    import tensorflow_addons as tfa
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.metrics import SparseCategoricalAccuracy
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ner_core import build_bilstm_crf

    word2idx, tag2idx = build_vocab(train)
    tags = sorted(tag2idx)
    idx2tag = {i: t for t, i in tag2idx.items()}
    pad_tag = tag2idx[tags[0]]  # sama seperti notebook: tags[0] dipakai sbg nilai padding

    def prep(sents):
        X = [[word2idx.get(w, 1) for w, _, _, _ in s] for s in sents]
        y = [[tag2idx[l] for _, _, _, l in s] for s in sents]
        X = pad_sequences(X, maxlen=MAX_LEN, padding="post", value=0)
        y = pad_sequences(y, maxlen=MAX_LEN, padding="post", value=pad_tag)
        return X, y

    X_tr, y_tr = prep(train)
    X_va, y_va = prep(valid)
    X_te, y_te = prep(test)

    model, inference_model, crf = build_bilstm_crf(len(word2idx), len(tag2idx))
    transition_params = crf.trainable_variables[0]

    def crf_loss(y_true, y_pred):  # persis seperti di notebook
        mask = tf.cast(tf.not_equal(y_true, pad_tag), tf.int32)
        seq_lengths = tf.reduce_sum(mask, axis=-1)
        ll, _ = tfa.text.crf_log_likelihood(inputs=y_pred, tag_indices=y_true,
                                            sequence_lengths=seq_lengths,
                                            transition_params=transition_params)
        return -tf.reduce_mean(ll)

    model.compile(optimizer="adam", loss=crf_loss, metrics=[SparseCategoricalAccuracy()])
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
                     batch_size=32, verbose=2)

    y_pred = inference_model.predict(X_te, verbose=0)
    model.save_weights(os.path.join(out, "bilstm_crf.weights.h5"))
    with open(os.path.join(out, "bilstm_crf_vocab.json"), "w") as f:
        json.dump({"word2idx": word2idx, "tag2idx": tag2idx, "max_len": MAX_LEN}, f)

    yt, yp = flat_eval(X_te, y_te, y_pred, idx2tag, 0)
    return {"report": report_dict(yt, yp), **label_summary(yt, yp),
            "history": {k: [float(v) for v in vs] for k, vs in hist.history.items()}}


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="models")
    ap.add_argument("--model", default="all", choices=["all", "crf", "bilstm", "bilstm_crf"])
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    train = read_data(os.path.join(a.data, "train.txt"))
    valid = read_data(os.path.join(a.data, "valid.txt"))
    test = read_data(os.path.join(a.data, "test.txt"))
    print(f"train={len(train)} valid={len(valid)} test={len(test)} kalimat")

    metrics_path = os.path.join(a.out, "retrain_metrics.json")
    jobs = {"crf": lambda: train_crf(train, valid, test, a.out),
            "bilstm": lambda: train_bilstm(train, valid, test, a.out, a.epochs),
            "bilstm_crf": lambda: train_bilstm_crf(train, valid, test, a.out, a.epochs)}
    for name, fn in jobs.items():
        if a.model in ("all", name):
            t0 = time.time()
            print(f"\n===== {name} =====")
            result = fn()
            result["train_seconds"] = round(time.time() - t0)
            metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
            metrics[name] = result
            json.dump(metrics, open(metrics_path, "w"), indent=1)
    print("Selesai. Model tersimpan di", a.out)
