"""
Inti NER: preprocessing, arsitektur model, dan fungsi prediksi.
Dipakai bersama oleh app.py (Streamlit) dan training/train_export.py
supaya preprocessing saat training dan saat prediksi dijamin sama.
"""
import json
import os
import re

import numpy as np

MAX_LEN = 75
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")

MODEL_NAMES = {"crf": "CRF", "bilstm": "BiLSTM", "bilstm_crf": "BiLSTM-CRF"}


# ================================================================ tokenisasi
_TOKEN_RE = re.compile(r"""
    (?:[A-Za-z]\.){2,}                 # singkatan: U.S. U.N.
  | (?:Mr|Mrs|Ms|Dr|Jr|Sr|St|Inc|Corp|Ltd|Co|Gen|Gov|Sen|Rep|Prof)\.   # gelar/badan
  | \d+(?:st|nd|rd|th)\b              # bilangan urutan: 1st 11th
  | \d+(?:[.,:/-]\d+)*                 # angka, tanggal, skor 2-1
  | \w+(?:-\w+)*                       # kata, termasuk kata-berhubung
  | '\w+                               # 's  've
  | \S                                 # tanda baca lain
""", re.VERBOSE)


def tokenize(text, mode="auto"):
    """mode='auto' -> pisahkan tanda baca (gaya CoNLL-2003);
    mode='split' -> text.split() persis seperti fungsi predict_manual_input di notebook."""
    if mode == "split":
        return text.split()
    return _TOKEN_RE.findall(text)


# ================================================================ fitur CRF (CRF_TA_FIX.ipynb)
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "postag": postag,
        "postag[:2]": postag[:2],
    }
    if i > 0:
        word1, postag1 = sent[i - 1][0], sent[i - 1][1]
        features.update({"-1:word.lower()": word1.lower(),
                         "-1:postag": postag1, "-1:postag[:2]": postag1[:2]})
    else:
        features["BOS"] = True
    if i < len(sent) - 1:
        word1, postag1 = sent[i + 1][0], sent[i + 1][1]
        features.update({"+1:word.lower()": word1.lower(),
                         "+1:postag": postag1, "+1:postag[:2]": postag1[:2]})
    else:
        features["EOS"] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


_TAGGER = None
_TAGGER_DIR = os.path.join(NLTK_DIR, "taggers", "averaged_perceptron_tagger_eng")


def _load_tagger():
    """Muat POS tagger perceptron NLTK langsung dari file JSON di folder nltk_data/.
    Tidak memakai pencarian path NLTK (nltk.data.find), yang bisa gagal di server
    seperti Streamlit Cloud. Bila gagal, coba cara standar NLTK."""
    global _TAGGER
    if _TAGGER is not None:
        return _TAGGER
    from nltk.tag.perceptron import PerceptronTagger
    try:
        tagger = PerceptronTagger(load=False)
        params = []
        for fname in tagger.param_files("eng"):           # weights, tagdict, classes
            with open(os.path.join(_TAGGER_DIR, fname), encoding="utf-8") as f:
                params.append(json.load(f))
        tagger.decode_json_params(params)
    except Exception:
        import nltk
        if NLTK_DIR not in nltk.data.path:
            nltk.data.path.insert(0, NLTK_DIR)
        try:
            tagger = PerceptronTagger()
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            tagger = PerceptronTagger()
    _TAGGER = tagger
    return _TAGGER


def pos_tag(tokens, mode="nltk"):
    """mode='nltk' -> POS tagger Penn Treebank (tagset sama dg CoNLL-2003);
    mode='dummy' -> semua 'NN' seperti di notebook."""
    if mode == "dummy" or not tokens:
        return ["NN"] * len(tokens)
    return [p for _, p in _load_tagger().tag(tokens)]


# ================================================================ fitur kapitalisasi (BiLSTM_TA_FIX.ipynb)
def capitalization_feature(word):
    if word[0].isupper():
        return 1
    elif word.islower():
        return 0
    return 2


# ================================================================ arsitektur
def build_bilstm(vocab_size, n_tags):
    """Sama dengan sel 'Membuat Model BiLSTM' di BiLSTM_TA_FIX.ipynb."""
    from tensorflow.keras.layers import (LSTM, Bidirectional, Concatenate, Dense,
                                         Embedding, Input)
    from tensorflow.keras.models import Model
    input_word = Input(shape=(MAX_LEN,), name="word_input")
    input_caps = Input(shape=(MAX_LEN,), name="caps_input")
    embedding = Embedding(input_dim=vocab_size, output_dim=50)(input_word)
    caps_embedding = Embedding(input_dim=3, output_dim=10)(input_caps)
    merged = Concatenate()([embedding, caps_embedding])
    bilstm = Bidirectional(LSTM(units=150, return_sequences=True,
                                dropout=0.1, recurrent_dropout=0.1))(merged)
    output = Dense(n_tags, activation="softmax")(bilstm)
    return Model(inputs=[input_word, input_caps], outputs=output)


def build_bilstm_crf(vocab_size, n_tags):
    """Sama dengan BiLSTM_CRF_FIX.ipynb. Return (model_training, model_inferensi, layer_crf)."""
    import tensorflow_addons as tfa
    from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Dropout,
                                         Embedding, Input, TimeDistributed)
    from tensorflow.keras.models import Model
    input_layer = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(input_layer)
    x = Bidirectional(LSTM(units=64, return_sequences=True,
                           dropout=0.5, recurrent_dropout=0.5))(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(n_tags))(x)
    crf = tfa.layers.CRF(n_tags)
    decoded_sequence, potentials, _, _ = crf(x)
    model = Model(inputs=input_layer, outputs=potentials)
    model.build(input_shape=(None, MAX_LEN))
    inference_model = Model(inputs=input_layer, outputs=decoded_sequence)
    return model, inference_model, crf


def build_bilstm_crf_encoder(vocab_size, n_tags):
    """Bagian BiLSTM-CRF tanpa tensorflow-addons: menghasilkan 'potentials' (emission score).
    Lapisan Dense terakhir = proyeksi internal milik lapisan CRF tfa. Decoding Viterbi
    dilakukan dengan NumPy di viterbi_decode(), identik dengan tfa.layers.CRF."""
    from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Dropout,
                                         Embedding, Input, TimeDistributed)
    from tensorflow.keras.models import Model
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(inp)
    x = Bidirectional(LSTM(units=64, return_sequences=True,
                           dropout=0.5, recurrent_dropout=0.5))(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(n_tags))(x)
    x = Dense(n_tags, name="crf_dense")(x)
    return Model(inp, x)


def viterbi_decode(potentials, chain_kernel, left_boundary, right_boundary):
    """potentials: (L, n_tags) untuk token asli saja. Sama dengan tfa CRF (use_boundary=True)."""
    pot = potentials.astype(np.float64).copy()
    pot[0] += left_boundary
    pot[-1] += right_boundary
    L, n = pot.shape
    score = pot[0]
    back = np.zeros((L, n), dtype=np.int64)
    for t in range(1, L):
        cand = score[:, None] + chain_kernel          # (prev, cur)
        back[t] = cand.argmax(0)
        score = cand.max(0) + pot[t]
    path = [int(score.argmax())]
    for t in range(L - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return path[::-1]


# ================================================================ predictor
def _chunks(seq, n=MAX_LEN):
    """Kalimat > 75 token dipotong per 75 (model dilatih dengan max_len=75)."""
    return [seq[i:i + n] for i in range(0, len(seq), n)] or [[]]


class CRFPredictor:
    name = "CRF"

    def __init__(self, model_dir=MODEL_DIR):
        import joblib
        self.model = joblib.load(os.path.join(model_dir, "crf_model.joblib"))

    def predict(self, tokens, pos_mode="nltk"):
        if not tokens:
            return []
        tags = pos_tag(tokens, pos_mode)
        feats = sent2features(list(zip(tokens, tags)))
        return list(self.model.predict([feats])[0])


def load_npz_weights(model, path):
    """Muat bobot dari .npz berdasarkan URUTAN (bukan nama layer), aman lintas OS & versi Keras."""
    data = np.load(path)
    model.set_weights([data[f"arr_{i}"] for i in range(len(data.files))])


class _KerasBase:
    def _load_vocab(self, path):
        v = json.load(open(path))
        self.word2idx = v["word2idx"]
        self.tag2idx = v["tag2idx"]
        self.idx2tag = {i: t for t, i in self.tag2idx.items()}

    def _encode(self, tokens):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        ids = [self.word2idx.get(w, self.word2idx["UNK"]) for w in tokens]
        return pad_sequences([ids], maxlen=MAX_LEN, padding="post", value=0)

    def unknown_tokens(self, tokens):
        return [w for w in tokens if w not in self.word2idx]


class BiLSTMPredictor(_KerasBase):
    name = "BiLSTM"

    def __init__(self, model_dir=MODEL_DIR):
        self._load_vocab(os.path.join(model_dir, "bilstm_vocab.json"))
        self.model = build_bilstm(len(self.word2idx), len(self.tag2idx))
        load_npz_weights(self.model, os.path.join(model_dir, "bilstm_weights.npz"))

    def predict(self, tokens, **_):
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        out = []
        for ch in _chunks(tokens):
            if not ch:
                continue
            X = self._encode(ch)
            caps = pad_sequences([[capitalization_feature(w) for w in ch]],
                                 maxlen=MAX_LEN, padding="post")
            probs = self.model.predict([X, caps], verbose=0)[0]
            out += [self.idx2tag[int(i)] for i in probs.argmax(-1)[:len(ch)]]
        return out


class BiLSTMCRFPredictor(_KerasBase):
    """Inferensi BiLSTM-CRF tanpa tensorflow-addons (agar bisa jalan di Windows / Python baru).
    Bobot diekspor dari model tfa oleh training/export_bilstm_crf_inference.py."""
    name = "BiLSTM-CRF"

    def __init__(self, model_dir=MODEL_DIR):
        self._load_vocab(os.path.join(model_dir, "bilstm_crf_vocab.json"))
        self.encoder = build_bilstm_crf_encoder(len(self.word2idx), len(self.tag2idx))
        load_npz_weights(self.encoder, os.path.join(model_dir, "bilstm_crf_encoder_weights.npz"))
        crf = np.load(os.path.join(model_dir, "bilstm_crf_transitions.npz"))
        self.chain, self.left, self.right = crf["chain_kernel"], crf["left_boundary"], crf["right_boundary"]

    def predict(self, tokens, **_):
        out = []
        for ch in _chunks(tokens):
            if not ch:
                continue
            pot = self.encoder.predict(self._encode(ch), verbose=0)[0][:len(ch)]
            out += [self.idx2tag[i] for i in viterbi_decode(pot, self.chain, self.left, self.right)]
        return out


PREDICTORS = {"crf": CRFPredictor, "bilstm": BiLSTMPredictor, "bilstm_crf": BiLSTMCRFPredictor}


def extract_persons(tokens, tags):
    """Gabungkan B-PER + I-PER berurutan menjadi nama orang utuh."""
    persons, cur = [], []
    for w, t in zip(tokens, tags):
        if t == "B-PER" or (t == "I-PER" and not cur):
            if cur:
                persons.append(" ".join(cur))
            cur = [w]
        elif t == "I-PER":
            cur.append(w)
        else:
            if cur:
                persons.append(" ".join(cur))
            cur = []
    if cur:
        persons.append(" ".join(cur))
    return persons
