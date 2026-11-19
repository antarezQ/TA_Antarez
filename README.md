# Identifikasi Entitas Orang dalam Dataset CoNLL-2003 versi bahasa Inggris menggunakan model CRF, BiLSTM, dan BiLSTM-CRF

Aplikasi Streamlit untuk Tugas Akhir Muhammad Antarez Qhadafi (1301210572), S1 Informatika, Universitas Telkom, 2024.

**Isi aplikasi**
| Tab | Isi |
|---|---|
| Coba Prediksi | Ketik kalimat, lalu lihat hasil ketiga model berdampingan. Token yang hasilnya berbeda antar model ditandai. |
| Contoh Data Uji | Kalimat acak dari test set: label asli vs prediksi tiap model. Berguna untuk demo. |
| Hasil Penelitian | Tabel 4.1 (precision/recall/F1 entitas orang), Tabel 4.2 (TP/FP/FN), **penjelasan kenapa BiLSTM-CRF lebih rendah** (analisis nama baru/OOV), analisis error per label, dan kurva loss. |
| Tentang Penelitian | Identitas TA, dataset, alur sistem, arsitektur ketiga model, dan saran. |

## Struktur folder
```
ner_app/
├── app.py                      # aplikasi Streamlit
├── ner_core.py                 # preprocessing, arsitektur, dan fungsi prediksi (dipakai app & training)
├── requirements.txt
├── .streamlit/config.toml
├── config/hasil_evaluasi_ta.json   # angka Tabel 4.1 & 4.2 buku TA (bisa diedit)
├── models/                     # model hasil training (siap pakai)
│   ├── crf_model.joblib
│   ├── bilstm_weights.npz + bilstm_vocab.json
│   ├── bilstm_crf_encoder_weights.npz + bilstm_crf_transitions.npz + bilstm_crf_vocab.json   (dipakai app)
│   ├── bilstm_crf.weights.h5   # bobot asli model tfa (untuk ekspor ulang)
│   └── retrain_metrics.json
├── nltk_data/                  # POS tagger untuk fitur CRF (offline, tidak perlu download)
├── dataset/                    # CoNLL-2003 train/valid/test
└── training/
    ├── train_export.py                 # melatih ulang & menyimpan ketiga model
    ├── export_bilstm_crf_inference.py  # konversi BiLSTM-CRF ke format tanpa tfa
    └── analisis_oov.py                 # recall nama dikenal vs nama baru -> models/analisis_oov.json
```

## Menjalankan di laptop (Windows, PowerShell)
> **Wajib Python 3.11** (TensorFlow 2.15). Aplikasi **tidak** butuh `tensorflow-addons`: decoding CRF untuk
> BiLSTM-CRF dijalankan dengan NumPy (`viterbi_decode` di `ner_core.py`), dan hasilnya sudah diverifikasi identik
> dengan lapisan CRF `tensorflow-addons` pada seluruh 46.610 token data uji.

```bash
cd "C:\path\ke\ner_app"      # pakai tanda kutip jika path mengandung spasi
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```
Aplikasi terbuka di http://localhost:8501.

## Deploy online (Streamlit Community Cloud, gratis)
1. Push folder ini ke repo GitHub. Semua file di bawah 100 MB, jadi tidak perlu Git LFS.
2. Buka https://share.streamlit.io → **Create app** → pilih repo, branch, lalu isi *Main file path* dengan `app.py`.
3. Klik **Advanced settings** → **Python version: 3.11**. Langkah ini penting; tanpanya instalasi gagal.
4. Klik **Deploy**. Build pertama butuh sekitar 5–10 menit karena memasang TensorFlow.

## Melatih ulang model
Training BiLSTM-CRF masih butuh `tensorflow-addons` (hanya ada untuk Linux/Mac, atau WSL di Windows):
```bash
pip install -r requirements-training.txt
python training/train_export.py --data dataset --out models --model all     # atau: crf / bilstm / bilstm_crf
python training/export_bilstm_crf_inference.py   # ubah BiLSTM-CRF ke format inferensi tanpa tfa
```
Arsitektur, hyperparameter, dan preprocessing **sama persis** dengan notebook. Bedanya hanya:
- **Daftar kata/tag di-sort sebelum dibuat `word2idx`.** Di notebook BiLSTM, `list(set(...))` membuat
  urutan indeks berubah setiap kali dijalankan, sehingga model tidak bisa dipakai ulang tanpa vocab yang sama.
  Sekarang vocab disimpan ke JSON bersama model.
- **Bobot model Keras disimpan sebagai array NumPy (`.npz`), lalu arsitekturnya dibangun ulang di `ner_core.py`.**
  Format ini dimuat berdasarkan urutan layer, sehingga aman dipakai lintas OS. File `.weights.h5` yang disimpan
  di Linux gagal dimuat di Windows.

## Perbedaan dengan fungsi `predict_manual_input` di notebook
Keduanya bisa diatur di sidebar → *Lanjutan* (pilihan "seperti notebook" tetap tersedia):
- **Tokenisasi.** Notebook memakai `sentence.split()`, sehingga `Obama,` dianggap satu kata dan menjadi
  UNK. App memisahkan tanda baca seperti format CoNLL-2003.
- **POS tag untuk CRF.** Notebook memberi POS dummy `'NN'` ke semua kata, padahal fitur POS dipakai CRF.
  App memakai POS tagger NLTK (tagset Penn Treebank, sama dengan CoNLL-2003).

## Catatan untuk sidang
- Angka di tab *Hasil Penelitian* **sama dengan Tabel 4.1 dan 4.2 di buku TA**. Nilai "entitas orang" adalah
  rata-rata B-PER dan I-PER. Contohnya CRF: precision (82,79% + 86,58%) / 2 = 84,67%.
  Untuk BiLSTM, angka di buku berasal dari run di **sel 6** notebook.
- Model yang dipakai untuk prediksi adalah hasil latih ulang. **CRF** identik dengan TA karena L-BFGS deterministik.
  **BiLSTM / BiLSTM-CRF** sedikit berbeda karena inisialisasi bobot acak
  (F1 entitas orang hasil latih ulang: 85,75% dan 75,12%). Urutan peringkat ketiga model tetap sama.
- **Support CRF dan model neural berbeda** (B-PER 1617 vs 1610) karena model neural memotong kalimat di 75 token.
- **Masking pada loss BiLSTM-CRF.** Padding label memakai `tags[0]` = `B-LOC`. Akibatnya `seq_lengths`
  pada `crf_loss` dihitung sebagai jumlah token yang **bukan** `B-LOC`. Untuk kalimat yang mengandung
  entitas lokasi, token-token terakhir ikut terpotong dari perhitungan loss. Ini mungkin salah satu penyebab
  BiLSTM-CRF tidak unggul dari BiLSTM. Temuan ini sejalan dengan Saran di Bab 5 buku TA. Kode ini sengaja **tidak diubah** di app supaya tetap sama dengan TA.
