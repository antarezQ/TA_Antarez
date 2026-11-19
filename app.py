"""
Aplikasi Streamlit — NER Entitas Person: CRF vs BiLSTM vs BiLSTM-CRF (CoNLL-2003)
Jalankan:  streamlit run app.py
"""
import html
import json
import os
import random

import altair as alt
import pandas as pd
import streamlit as st

from ner_core import (BASE_DIR, MODEL_DIR, MODEL_NAMES, PREDICTORS, extract_persons,
                      tokenize)

# ------------------------------------------------------------------ konfigurasi
JUDUL = "Identifikasi Entitas Orang dalam Dataset CoNLL-2003"
SUBJUDUL = "versi bahasa Inggris menggunakan model CRF, BiLSTM, dan BiLSTM-CRF"
JUDUL_EN = ("Identification of the person entity in English version of CoNLL-2003 Dataset "
            "using the CRF, BiLSTM and BiLSTM-CRF models")
PENULIS = "Muhammad Antarez Qhadafi"
NIM = "1301210572"
PEMBIMBING = ["Dr. Moch. Arif Bijaksana, Ir. M. Tech.", "Bunyamin, S.Kom., M.Kom."]
INSTANSI = "Program Studi Sarjana Informatika · Fakultas Informatika · Universitas Telkom · 2024"

# warna kategorikal tetap per model (urutan tidak berubah meski ada model yang tidak dipilih)
MODEL_COLORS = {"CRF": "#2a78d6", "BiLSTM": "#eb6834", "BiLSTM-CRF": "#1baf7a"}
ENTITY_STYLE = {
    "PER":  ("#fde7d9", "#b2410f"),
    "ORG":  ("#eceae4", "#52514e"),
    "LOC":  ("#eceae4", "#52514e"),
    "MISC": ("#eceae4", "#52514e"),
}

CONTOH = [
    "Barack Obama, the former president of the United States, visited Apple Inc. headquarters in Cupertino, California last week.",
    "Wayne Rooney scored twice as Manchester United beat Arsenal 3-1 on Saturday.",
    "German Chancellor Helmut Kohl met Russian President Boris Yeltsin in Moscow.",
    # contoh input dari Tabel 3.3 buku TA
    "Teenager Yamal then found his captain with a sublime cross using the outside of his foot 18 minutes later, which Raphinha slotted in for his 11th league goal of the season.",
]

st.set_page_config(page_title="NER Person — TA", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.ent{padding:2px 6px;border-radius:6px;margin:0 1px;line-height:2.1;white-space:nowrap}
.ent small{font-size:.68em;font-weight:600;margin-left:4px;letter-spacing:.02em}
.sent{font-size:1.06rem;line-height:2.1}
.diff{background:#fff4cc;box-shadow:inset 0 -3px #eda100}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
def available_models():
    need = {"crf": ["crf_model.joblib"],
            "bilstm": ["bilstm_weights.npz", "bilstm_vocab.json"],
            "bilstm_crf": ["bilstm_crf_encoder_weights.npz", "bilstm_crf_transitions.npz", "bilstm_crf_vocab.json"]}
    return [k for k, fs in need.items()
            if all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in fs)]


@st.cache_resource(show_spinner="Memuat model…")
def load_predictor(key):
    return PREDICTORS[key]()


@st.cache_data
def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}


@st.cache_data
def dataset_stats():
    """Jumlah kalimat, token, dan label per bagian dataset (tanpa baris -DOCSTART-)."""
    from collections import Counter
    out = {}
    for part in ("train", "valid", "test"):
        path = os.path.join(BASE_DIR, "dataset", f"{part}.txt")
        if not os.path.exists(path):
            continue
        n_sent, in_sent, labels = 0, False, Counter()
        for line in open(path):
            p = line.split()
            if len(p) == 4 and p[0] != "-DOCSTART-":
                labels[p[3]] += 1
                in_sent = True
            elif not p and in_sent:
                n_sent += 1
                in_sent = False
        n_sent += in_sent
        out[part] = {"kalimat": n_sent, "token": sum(labels.values()), "label": dict(labels)}
    return out


@st.cache_data
def load_test_sentences():
    path = os.path.join(BASE_DIR, "dataset", "test.txt")
    if not os.path.exists(path):
        return []
    sents, cur = [], []
    for line in open(path):
        p = line.split()
        if not p:
            if cur:
                sents.append(cur)
            cur = []
        elif len(p) == 4 and p[0] != "-DOCSTART-":
            cur.append((p[0], p[1], p[3]))
    if cur:
        sents.append(cur)
    # pilih kalimat yang mengandung entitas PERSON & cukup panjang, agar menarik untuk demo
    return [s for s in sents if any(t[2].endswith("PER") for t in s) and 6 <= len(s) <= 40]


# ------------------------------------------------------------------ render
def render_tagged(tokens, tags, show_all, diff_mask=None):
    parts = []
    for i, (w, t) in enumerate(zip(tokens, tags)):
        w_html = html.escape(w)
        ent = t.split("-", 1)[1] if "-" in t else None
        cls = " diff" if diff_mask and diff_mask[i] else ""
        if ent and (ent == "PER" or show_all):
            bg, fg = ENTITY_STYLE.get(ent, ENTITY_STYLE["MISC"])
            parts.append(f'<span class="ent{cls}" style="background:{bg};color:{fg}">'
                         f'{w_html}<small>{t}</small></span>')
        elif cls:
            parts.append(f'<span class="ent{cls}">{w_html}</span>')
        else:
            parts.append(w_html)
    out = ""
    for i, (w, part) in enumerate(zip(tokens, parts)):
        nospace = i == 0 or w in {",", ".", ":", ";", "!", "?", ")", "'s", "%"} or tokens[i - 1] in {"(", "$"}
        out += ("" if nospace else " ") + part
    return f'<div class="sent">{out}</div>'


def run_models(tokens, keys, pos_mode):
    return {MODEL_NAMES[k]: load_predictor(k).predict(tokens, pos_mode=pos_mode) for k in keys}


# ------------------------------------------------------------------ sidebar
avail = available_models()
with st.sidebar:
    st.header("⚙️ Pengaturan")
    pilih = st.multiselect("Model", options=avail, default=avail,
                           format_func=lambda k: MODEL_NAMES[k])
    show_all = st.toggle("Tampilkan juga entitas ORG / LOC / MISC", value=False,
                         help="Fokus TA adalah PERSON, tapi model dilatih dengan 9 label CoNLL-2003.")
    with st.expander("Lanjutan"):
        tok_mode = st.radio("Tokenisasi", ["auto", "split"], format_func=lambda m: {
            "auto": "Pisahkan tanda baca (disarankan)",
            "split": "Split spasi (seperti notebook)"}[m])
        pos_mode = st.radio("POS tag untuk CRF", ["nltk", "dummy"], format_func=lambda m: {
            "nltk": "NLTK Penn Treebank (disarankan)",
            "dummy": "Semua 'NN' (seperti notebook)"}[m])
    missing = [MODEL_NAMES[k] for k in MODEL_NAMES if k not in avail]
    if missing:
        st.warning("File model belum ada: " + ", ".join(missing)
                   + ". Lihat README bagian *Melatih ulang model*.")
    st.divider()
    st.caption(f"**Tugas Akhir**  \n{PENULIS}  \n{NIM}")
    st.caption("**Pembimbing**  \n" + "  \n".join(PEMBIMBING))
    st.caption(INSTANSI.replace(" · ", "  \n"))

st.title(JUDUL)
st.markdown(f"#### {SUBJUDUL}")
st.caption(f"{PENULIS} ({NIM}) · {INSTANSI}")

tab_demo, tab_uji, tab_eval, tab_info = st.tabs(
    ["Coba Prediksi", "Contoh Data Uji", "Hasil Penelitian", "Tentang Penelitian"])

# ------------------------------------------------------------------ tab 1: demo
with tab_demo:
    if "teks" not in st.session_state:
        st.session_state.teks = CONTOH[0]
    st.write("**Contoh kalimat:**")
    cols = st.columns(len(CONTOH))
    for c, ex in zip(cols, CONTOH):
        if c.button(ex[:38] + "…", width="stretch", key=ex):
            st.session_state.teks = ex
    teks = st.text_area("Masukkan kalimat berbahasa Inggris", key="teks", height=90)
    st.caption("Dataset CoNLL-2003 berbahasa Inggris (berita Reuters), jadi model paling akurat untuk teks berita berbahasa Inggris.")

    if st.button("Deteksi entitas", type="primary") and teks.strip():
        if not pilih:
            st.error("Pilih minimal satu model di sidebar.")
        else:
            tokens = tokenize(teks, tok_mode)
            hasil = run_models(tokens, pilih, pos_mode)
            names = list(hasil)
            diff = [len({hasil[n][i] for n in names}) > 1 for i in range(len(tokens))]

            for n in names:
                persons = extract_persons(tokens, hasil[n])
                with st.container(border=True):
                    a, b = st.columns([1, 3])
                    a.markdown(f"<span style='color:{MODEL_COLORS[n]}'>●</span> **{n}**",
                               unsafe_allow_html=True)
                    a.caption(f"{len(persons)} nama orang")
                    b.markdown(render_tagged(tokens, hasil[n], show_all,
                                             diff if len(names) > 1 else None),
                               unsafe_allow_html=True)
                    if persons:
                        b.markdown("**Person:** " + " · ".join(f"`{p}`" for p in persons))

            if len(names) > 1 and any(diff):
                st.caption("Latar kuning = token yang prediksinya berbeda antar model.")

            with st.expander("Tabel per token"):
                df = pd.DataFrame({"Token": tokens, **hasil})
                for k in pilih:
                    if k in ("bilstm", "bilstm_crf"):
                        unk = set(load_predictor(k).unknown_tokens(tokens))
                        df[f"OOV ({MODEL_NAMES[k]})"] = ["✓" if t in unk else "" for t in tokens]
                        break
                st.dataframe(df, width="stretch", hide_index=True)
                st.caption("OOV = kata yang tidak ada di kosakata training (dipetakan ke UNK oleh model BiLSTM/BiLSTM-CRF).")

# ------------------------------------------------------------------ tab 2: data uji
with tab_uji:
    test = load_test_sentences()
    if not test:
        st.info("File dataset/test.txt tidak ditemukan.")
    else:
        st.write("Ambil kalimat acak dari **data uji CoNLL-2003**, lalu bandingkan label asli dengan prediksi tiap model.")
        if st.button("🎲 Kalimat acak") or "uji_idx" not in st.session_state:
            st.session_state.uji_idx = random.randrange(len(test))
        s = test[st.session_state.uji_idx]
        tokens = [t[0] for t in s]
        gold = [t[2] for t in s]
        # data uji sudah punya POS tag asli → pakai itu untuk CRF
        hasil = {}
        for k in pilih:
            p = load_predictor(k)
            if k == "crf":
                from ner_core import sent2features
                hasil[MODEL_NAMES[k]] = list(p.model.predict(
                    [sent2features([(t[0], t[1]) for t in s])])[0])
            else:
                hasil[MODEL_NAMES[k]] = p.predict(tokens)

        with st.container(border=True):
            st.markdown("**Label asli (gold)**")
            st.markdown(render_tagged(tokens, gold, show_all), unsafe_allow_html=True)
        for n, pred in hasil.items():
            wrong = [g != p for g, p in zip(gold, pred)]
            n_err = sum(wrong)
            with st.container(border=True):
                st.markdown(f"<span style='color:{MODEL_COLORS[n]}'>●</span> **{n}** — "
                            + ("✅ semua benar" if n_err == 0 else f"❌ {n_err} token salah"),
                            unsafe_allow_html=True)
                st.markdown(render_tagged(tokens, pred, show_all, wrong), unsafe_allow_html=True)
        st.caption("Latar kuning = prediksi berbeda dari label asli.")

# ------------------------------------------------------------------ tab 3: evaluasi
with tab_eval:
    ta = load_json(os.path.join(BASE_DIR, "config", "hasil_evaluasi_ta.json"))
    order = list(MODEL_NAMES.values())
    keys = [k for k in MODEL_NAMES if k in ta]

    st.subheader("Perbandingan performa model")
    st.caption("Tabel 4.1 buku TA · data uji CoNLL-2003 · nilai entitas orang = rata-rata B-PER dan I-PER")

    # kartu ringkas: F1-score entitas orang per model
    best = max(keys, key=lambda k: ta[k]["PER"]["f1-score"])
    cols = st.columns(len(keys))
    for c, k in zip(cols, keys):
        r = ta[k]["PER"]
        c.metric(f"{MODEL_NAMES[k]}{'  🏆' if k == best else ''}", f"{r['f1-score'] * 100:.2f}%",
                 help="F1-score entitas orang (rata-rata B-PER & I-PER)")
        c.caption(f"Precision {r['precision'] * 100:.2f}% · Recall {r['recall'] * 100:.2f}%")

    t41 = pd.DataFrame([{"Model": MODEL_NAMES[k], "Precision": ta[k]["PER"]["precision"] * 100,
                         "Recall": ta[k]["PER"]["recall"] * 100,
                         "F1-Score": ta[k]["PER"]["f1-score"] * 100} for k in keys]).set_index("Model")
    st.dataframe(t41.style.format("{:.2f}%").highlight_max(axis=0, color="#dbeafb"), width="stretch")
    st.caption("Sel biru = nilai tertinggi pada kolom tersebut.")

    st.info(f"**Kesimpulan TA:** model **{MODEL_NAMES[best]}** paling efektif mengidentifikasi entitas orang "
            "(F1-score tertinggi dan recall 90%). BiLSTM unggul di precision (91,9%), "
            "sedangkan BiLSTM-CRF memiliki recall terendah (67,59%).")

    # ---- grafik per label
    st.subheader("Rincian per label")
    rows = []
    for k in keys:
        for lab, nama in [("B-PER", "B-PER"), ("I-PER", "I-PER"), ("PER", "Entitas orang (rata-rata)")]:
            r = ta[k][lab]
            rows.append({"Model": MODEL_NAMES[k], "Label": nama, "Precision": r["precision"],
                         "Recall": r["recall"], "F1-score": r["f1-score"]})
    df = pd.DataFrame(rows)
    lab_order = ["B-PER", "I-PER", "Entitas orang (rata-rata)"]
    metrik = st.radio("Metrik", ["F1-score", "Precision", "Recall"], horizontal=True)
    chart = (alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=30)
             .encode(x=alt.X("Model:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
                     y=alt.Y(f"{metrik}:Q", scale=alt.Scale(domain=[0, 1]), title=metrik,
                             axis=alt.Axis(format="%")),
                     color=alt.Color("Model:N", sort=order,
                                     scale=alt.Scale(domain=order, range=[MODEL_COLORS[m] for m in order]),
                                     legend=alt.Legend(orient="top", title=None)),
                     column=alt.Column("Label:N", sort=lab_order, title=None,
                                       header=alt.Header(labelOrient="bottom")),
                     tooltip=["Model", "Label", alt.Tooltip(f"{metrik}:Q", format=".2%")])
             .properties(width=170, height=280)
             .configure_view(stroke=None)
             .configure_axis(grid=True, gridColor="#eceae4", domainColor="#c3c2b7"))
    st.altair_chart(chart)

    # ---- Tabel 4.2
    st.subheader("TP, FP, dan FN per label")
    st.caption("Tabel 4.2 buku TA · dasar perhitungan precision = TP/(TP+FP) dan recall = TP/(TP+FN)")
    t42 = []
    for k in keys:
        for lab in ["B-PER", "I-PER"]:
            r = ta[k][lab]
            t42.append({"Model": MODEL_NAMES[k], "Label": lab, "TP": r["TP"], "FP": r["FP"], "FN": r["FN"],
                        "Precision": r["TP"] / (r["TP"] + r["FP"]), "Recall": r["TP"] / (r["TP"] + r["FN"])})
    st.dataframe(pd.DataFrame(t42).style.format({"Precision": "{:.2%}", "Recall": "{:.2%}"}),
                 hide_index=True, width="stretch")

    with st.expander("Sumber angka"):
        for k in keys:
            st.write(f"**{MODEL_NAMES[k]}**: {ta[k].get('sumber', '-')}")

    # ---- kenapa BiLSTM-CRF lebih rendah
    oov = load_json(os.path.join(MODEL_DIR, "analisis_oov.json"))
    if oov:
        st.divider()
        st.subheader("Kenapa BiLSTM-CRF lebih rendah?")
        rp = oov["recall_per"]
        tot_iv = rp["CRF"]["dikenal"]["total"]
        tot_oov = rp["CRF"]["oov"]["total"]
        ribuan = lambda n: f"{n:,}".replace(",", ".")
        pct = lambda m, g: rp[m][g]["benar"] / rp[m][g]["total"]
        st.markdown(
            f"Penyebab utamanya adalah **nama yang tidak pernah muncul di data latih** "
            f"(*out-of-vocabulary* / OOV). Dari {ribuan(tot_iv + tot_oov)} token nama orang di data uji, "
            f"**{ribuan(tot_oov)} token ({tot_oov / (tot_iv + tot_oov):.0%}) adalah kata yang belum pernah dilihat model**. "
            "Grafik di bawah membandingkan persentase token nama yang berhasil dideteksi (recall) "
            "untuk nama yang sudah dikenal dan nama baru.")

        rows = []
        for m in order:
            for g, lab in [("dikenal", "Nama ada di data latih"), ("oov", "Nama baru (OOV)")]:
                rows.append({"Model": m, "Kelompok": lab, "Recall": pct(m, g),
                             "Detail": f'{rp[m][g]["benar"]}/{rp[m][g]["total"]} token'})
        odf = pd.DataFrame(rows)
        grp_order = ["Nama ada di data latih", "Nama baru (OOV)"]
        bars = (alt.Chart(odf).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=56)
                .encode(x=alt.X("Model:N", sort=order, title=None, axis=alt.Axis(labels=False, ticks=False)),
                        y=alt.Y("Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                                title="Token nama terdeteksi"),
                        color=alt.Color("Model:N", sort=order,
                                        scale=alt.Scale(domain=order, range=[MODEL_COLORS[m] for m in order]),
                                        legend=alt.Legend(orient="top", title=None)),
                        tooltip=["Model", "Kelompok", alt.Tooltip("Recall:Q", format=".1%"), "Detail"]))
        labels = (alt.Chart(odf).mark_text(dy=-9, fontSize=13, color="#52514e")
                  .encode(x=alt.X("Model:N", sort=order), y="Recall:Q",
                          text=alt.Text("Recall:Q", format=".0%")))
        oc = (alt.layer(bars, labels, data=odf).properties(width=280, height=300)
              .facet(column=alt.Column("Kelompok:N", sort=grp_order, title=None,
                                       header=alt.Header(labelOrient="bottom", labelFontSize=13)))
              .configure_view(stroke=None)
              .configure_axis(grid=True, gridColor="#eceae4", domainColor="#c3c2b7"))
        st.altair_chart(oc)

        idp = lambda x: f"{x * 100:.1f}%".replace(".", ",")
        c1, c2 = st.columns(2)
        c1.markdown(
            f"**Nama yang dikenal:** ketiga model setara (sekitar {pct('CRF', 'dikenal'):.0%}). "
            f"BiLSTM-CRF malah **paling tinggi** ({idp(pct('BiLSTM-CRF', 'dikenal'))}).")
        c2.markdown(
            f"**Nama baru:** BiLSTM-CRF jatuh ke **{idp(pct('BiLSTM-CRF', 'oov'))}**, "
            f"sedangkan CRF {idp(pct('CRF', 'oov'))} dan BiLSTM {idp(pct('BiLSTM', 'oov'))}.")

        st.markdown("##### Apa yang dilihat model saat bertemu kata baru?")
        st.markdown("""
| Model | Informasi untuk kata yang belum pernah dilihat | Contoh: *Obama* |
|---|---|---|
| **CRF** | Bentuk kata (huruf kapital, sufiks), POS tag, dan kata di sekitarnya | huruf kapital ✓, POS = NNP ✓, kata berikutnya "," |
| **BiLSTM** | Input tambahan **fitur kapitalisasi** (kapital / kecil / lainnya) | `UNK` + huruf kapital ✓ |
| **BiLSTM-CRF** | **Hanya embedding kata.** Semua kata baru menjadi satu vektor `UNK` yang sama | `UNK`, sama persis dengan kata baru lain seperti *sublime* |
""")
        st.caption("BiLSTM-CRF tidak memakai fitur kapitalisasi, sehingga kata baru tidak bisa dibedakan dari kata biasa.")

        st.markdown("##### Contoh: berapa kali nama muncul di data latih?")
        fq = oov["frekuensi_contoh"]
        fdf = pd.DataFrame([{"Kata": w, "Muncul di data latih": n,
                             "Status": "dikenal" if n else "baru (OOV)"} for w, n in fq.items()])
        a, b = st.columns([2, 3])
        a.dataframe(fdf, hide_index=True, width="stretch")
        b.markdown("""
- **Helmut Kohl, Boris Yeltsin:** tokoh besar di berita Reuters 1996–1997, sehingga ketiga model berhasil.
- **Barack Obama, Yamal, Raphinha:** baru terkenal setelah dataset dibuat. BiLSTM-CRF gagal, sedangkan model lain masih punya petunjuk dari bentuk kata.
- **Wayne Rooney:** *Rooney* juga kata baru, tapi BiLSTM-CRF **tetap berhasil**. *Wayne* dikenal sebagai nama depan
  (`B-PER`), lalu lapisan **CRF** memakai aturan transisi bahwa setelah `B-PER` biasanya datang `I-PER`.
  Ini menunjukkan lapisan CRF-nya sendiri **membantu**, bukan sumber masalah.
""")
        st.markdown("##### Faktor pendukung")
        st.markdown("""
- **Regularisasi lebih berat:** dropout 0,5 di tiga tempat dan LSTM 64 unit, dibanding BiLSTM dengan dropout 0,1 dan 150 unit.
  Model menjadi lebih "hati-hati" dan cenderung menebak `O`, sehingga **recall rendah tetapi precision tetap tinggi** (88,68%).
- **Masking pada loss:** label padding memakai `B-LOC` (`tags[0]`). Panjang kalimat pada loss dihitung dari token
  yang bukan `B-LOC`, sehingga pada kalimat yang memuat lokasi, beberapa token terakhir tidak ikut dipelajari.
""")
        st.success(
            "**Kesimpulan:** rendahnya BiLSTM-CRF disebabkan oleh **konfigurasi input**: tanpa fitur kapitalisasi "
            "dan tanpa informasi bentuk kata, sehingga model lemah pada nama baru. Arsitektur CRF-nya sendiri "
            "tidak bermasalah. Hal ini sejalan dengan saran TA untuk mencoba parameter lain. Di literatur, "
            "BiLSTM-CRF biasanya diberi embedding karakter atau embedding pretrained (misalnya GloVe) "
            "agar tetap kuat pada kata baru.")
        st.caption("Analisis memakai model yang dilatih ulang untuk aplikasi ini (kode sama dengan TA); "
                   "lihat training/analisis_oov.py.")

    # ---- analisis error (dari hasil latih ulang untuk app)
    rm = load_json(os.path.join(MODEL_DIR, "retrain_metrics.json"))
    if rm:
        st.divider()
        st.subheader("Analisis error per label")
        st.caption("Dari model yang dipakai aplikasi ini (dilatih ulang dengan kode yang sama). "
                   "Jumlah token data uji yang salah diprediksi, dikelompokkan menurut label aslinya.")
        er = []
        for k, name in MODEL_NAMES.items():
            for lab, n in rm.get(k, {}).get("error_by_true", {}).items():
                er.append({"Model": name, "Label asli": lab, "Jumlah error": n})
        if er:
            edf = pd.DataFrame(er)
            labs = sorted(edf["Label asli"].unique(), key=lambda l: (l[2:], l))
            ch = (alt.Chart(edf).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=7)
                  .encode(y=alt.Y("Label asli:N", sort=labs, title=None),
                          yOffset=alt.YOffset("Model:N", sort=order),
                          x=alt.X("Jumlah error:Q", title="Jumlah token salah"),
                          color=alt.Color("Model:N", sort=order,
                                          scale=alt.Scale(domain=order, range=[MODEL_COLORS[m] for m in order]),
                                          legend=alt.Legend(orient="top", title=None)),
                          tooltip=["Model", "Label asli", "Jumlah error"])
                  .properties(height=420)
                  .configure_view(stroke=None)
                  .configure_axis(gridColor="#eceae4", domainColor="#c3c2b7"))
            st.altair_chart(ch, width="stretch")

        hist = [(MODEL_NAMES[k], rm[k]["history"]) for k in ("bilstm", "bilstm_crf")
                if k in rm and "history" in rm[k]]
        if hist:
            st.subheader("Kurva training (model neural)")
            cols = st.columns(len(hist))
            for c, (name, h) in zip(cols, hist):
                lrows = []
                for split, key in [("train", "loss"), ("validasi", "val_loss")]:
                    for ep, v in enumerate(h.get(key, []), 1):
                        lrows.append({"Epoch": ep, "Loss": v, "Data": split})
                lc = (alt.Chart(pd.DataFrame(lrows)).mark_line(point=True, strokeWidth=2)
                      .encode(x=alt.X("Epoch:O", axis=alt.Axis(labelAngle=0)), y="Loss:Q",
                              color=alt.Color("Data:N", scale=alt.Scale(
                                  domain=["train", "validasi"], range=["#2a78d6", "#eb6834"]),
                                  legend=alt.Legend(orient="top", title=None)),
                              tooltip=["Epoch", "Data", alt.Tooltip("Loss:Q", format=".4f")])
                      .properties(title=name, height=260))
                c.altair_chart(lc, width="stretch")

# ------------------------------------------------------------------ tab 4: info
with tab_info:
    st.markdown(f"""
### {JUDUL} {SUBJUDUL}
*{JUDUL_EN}*

**{PENULIS}** ({NIM})  
Pembimbing I: {PEMBIMBING[0]}  
Pembimbing II: {PEMBIMBING[1]}  
{INSTANSI}
""")

    st.divider()

    # ---------------- Apa itu NER
    st.markdown("""
#### Apa itu Named Entity Recognition (NER)?
**Named Entity Recognition** (pengenalan entitas bernama) adalah tugas dalam *Natural Language Processing*
untuk **menemukan dan mengelompokkan nama-nama penting di dalam teks**, seperti nama orang, organisasi,
lokasi, dan nama lain (kebangsaan, acara, produk, dan sebagainya).

NER dikerjakan sebagai **pelabelan urutan** (*sequence labeling*): setiap kata dalam kalimat diberi satu label.
Model harus memperhatikan kata itu sendiri sekaligus konteks di sekitarnya. Misalnya, *Washington* bisa berarti
orang (*George Washington*) atau lokasi (*Washington D.C.*).

NER dipakai di banyak aplikasi, misalnya mesin pencari, ekstraksi informasi dari berita, chatbot,
dan analisis dokumen. Penelitian ini berfokus pada satu jenis entitas, yaitu **orang (PER)**.
""")
    contoh_tok = ["German", "Chancellor", "Helmut", "Kohl", "met", "Boris", "Yeltsin", "in",
                  "Moscow", "before", "the", "European", "Union", "summit", "."]
    contoh_tag = ["B-MISC", "O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O",
                  "B-LOC", "O", "O", "B-ORG", "I-ORG", "O", "O"]
    with st.container(border=True):
        st.caption("Contoh kalimat yang sudah diberi label NER")
        st.markdown(render_tagged(contoh_tok, contoh_tag, True), unsafe_allow_html=True)

    # ---------------- Dataset
    st.markdown("""
#### Dataset CoNLL-2003
**CoNLL-2003 versi bahasa Inggris** adalah dataset standar untuk NER, dibuat untuk *shared task* CoNLL tahun 2003.
Teksnya diambil dari **korpus berita Reuters** (Agustus 1996 – Agustus 1997) dan sudah dibagi menjadi tiga bagian:
- **train**: untuk melatih model
- **dev / valid**: untuk memantau pelatihan
- **test**: untuk evaluasi akhir
""")
    ds = dataset_stats()
    if ds:
        rb = lambda n: f"{n:,}".replace(",", ".")
        st.markdown("| Bagian | Kalimat | Token | Token nama orang (B-PER + I-PER) |\n|---|---:|---:|---:|\n" + "".join(
            f'| **{k}** | {rb(v["kalimat"])} | {rb(v["token"])} | '
            f'{rb(v["label"].get("B-PER", 0) + v["label"].get("I-PER", 0))} |\n' for k, v in ds.items()))

    st.markdown("""
##### Format file
Setiap baris berisi **satu kata dan tiga anotasinya**, dipisahkan spasi. Baris kosong menandai akhir kalimat.

| Kata | POS tag | Chunk tag | NER tag |
|---|---|---|---|
| `EU` | `NNP` | `B-NP` | `B-ORG` |
| `rejects` | `VBZ` | `B-VP` | `O` |
| `German` | `JJ` | `B-NP` | `B-MISC` |
| `call` | `NN` | `I-NP` | `O` |

- **POS tag** (*part-of-speech*): kelas kata menurut Penn Treebank, misalnya `NNP` = kata benda nama diri,
  `NN` = kata benda, `VBZ` = kata kerja, `JJ` = kata sifat. Dipakai sebagai fitur oleh model CRF.
- **Chunk tag**: penanda frasa, misalnya `B-NP` = awal frasa nomina, `B-VP` = awal frasa verba. Tidak dipakai di penelitian ini.
- **NER tag**: label entitas yang ingin diprediksi model.

##### Skema label BIO
Label NER memakai skema **BIO**, sehingga entitas yang terdiri dari beberapa kata bisa ditandai utuh:
- **B-** (*Begin*): kata **pertama** dari sebuah entitas
- **I-** (*Inside*): kata **lanjutan** di dalam entitas yang sama
- **O** (*Outside*): kata yang **bukan** entitas

Contohnya, *Helmut Kohl* dilabeli `Helmut/B-PER Kohl/I-PER`. Dengan begitu, dua nama yang berdampingan tetap bisa dipisahkan,
karena setiap nama baru selalu diawali `B-PER`.
""")
    LABEL_INFO = [
        ("B-PER", "Kata pertama nama **orang**", "**Helmut** Kohl, **Wayne** Rooney"),
        ("I-PER", "Kata lanjutan nama **orang**", "Helmut **Kohl**, Wayne **Rooney**"),
        ("B-ORG", "Kata pertama nama **organisasi** (perusahaan, klub, partai, lembaga)", "**European** Union, **Manchester** United"),
        ("I-ORG", "Kata lanjutan nama **organisasi**", "European **Union**, Manchester **United**"),
        ("B-LOC", "Kata pertama nama **lokasi** (negara, kota, wilayah, tempat)", "**Moscow**, **United** States"),
        ("I-LOC", "Kata lanjutan nama **lokasi**", "United **States**, New **York**"),
        ("B-MISC", "Kata pertama **entitas lain**: kebangsaan, bahasa, acara, produk, dll.", "**German**, **British**, **World** Cup"),
        ("I-MISC", "Kata lanjutan **entitas lain**", "World **Cup**"),
        ("O", "**Bukan entitas** (kata biasa)", "met, the, scored, in"),
    ]
    rows = []
    for lab, arti, contoh in LABEL_INFO:
        r = {"Label": lab, "Arti": arti, "Contoh": contoh}
        for part in ("train", "valid", "test"):
            if ds and part in ds:
                r[f"Jumlah ({part})"] = f'{ds[part]["label"].get(lab, 0):,}'.replace(",", ".")
        rows.append(r)
    cols_md = list(rows[0].keys())
    md = "| " + " | ".join(cols_md) + " |\n|---|---|---|" + "---:|" * (len(cols_md) - 3) + "\n"
    for r in rows:
        lab = r["Label"]
        if lab.endswith("PER"):
            r = {**r, "Arti": "⭐ " + r["Arti"]}
        r = {**r, "Label": f"`{lab}`"}
        md += "| " + " | ".join(str(r[c]) for c in cols_md) + " |\n"
    st.markdown(md)
    st.caption("⭐ = label yang menjadi fokus penelitian. Jumlah dihitung per token. "
               "Sebagian besar token berlabel O, sehingga data tidak seimbang.")

    st.markdown("""
#### Alur sistem
1. **Tokenization:** `read_data` membaca file menjadi daftar kalimat berisi tuple *(kata, POS, chunk, label)*.
2. **Pelatihan** tiga model: CRF, BiLSTM, dan BiLSTM-CRF.
3. **Evaluasi** dengan precision, recall, dan F1-score untuk entitas orang.

#### 1. CRF (Conditional Random Field)
`sklearn-crfsuite`, algoritma L-BFGS, `c1=0.1`, `c2=0.1`, 100 iterasi.
Fitur dari `word2features`: bentuk huruf kecil, sufiks 2–3 huruf, status kapital/judul/angka,
POS-tag, serta kata dan POS-tag token sebelum/sesudahnya, ditambah penanda awal (BOS) dan akhir (EOS) kalimat.

#### 2. BiLSTM
Kata diubah menjadi indeks dengan `word2idx` (`PAD`=0, `UNK`=1) lalu dipadatkan ke 75 token.
Embedding kata (50 dimensi) digabung (*Concatenate*) dengan embedding fitur kapitalisasi (10 dimensi),
lalu masuk ke BiLSTM 150 unit per arah (`dropout=0.1`, `recurrent_dropout=0.1`) dan Dense softmax.
Pelatihan: 10 epoch, batch 32.

#### 3. BiLSTM-CRF
Lapisan BiLSTM digabung dengan lapisan CRF: embedding 256 → BiLSTM 64 unit (dropout 0.5)
→ Dropout → TimeDistributed Dense → CRF (`tensorflow-addons`), dengan loss *negative log-likelihood* CRF.
Pelatihan: 10 epoch, batch 32.

#### Saran penelitian lanjutan
Parameter BiLSTM-CRF perlu dieksplorasi lagi, tidak hanya memakai parameter dari model BiLSTM dan CRF.
Penelitian berikutnya juga dapat mencoba model dengan arsitektur yang lebih modern.
""")
