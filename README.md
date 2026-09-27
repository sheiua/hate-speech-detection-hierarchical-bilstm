# Deteksi Ujaran Kebencian Multi-Label pada Twitter Menggunakan Hierarchical BiLSTM dengan Attention Mechanism

Sistem klasifikasi teks cerdas berbasis Deep Learning untuk mendeteksi ujaran kebencian (*hate speech*) multi-label pada data Twitter berbahasa Indonesia. Model menggunakan arsitektur **Hierarchical Attention Network (HAN)** yang menggabungkan **Word-level BiLSTM + Attention** dan **Sentence-level BiLSTM + Attention**.

---

## 🎯 4 Kategori Target Multi-Label

Aplikasi ini mampu mendeteksi lebih dari satu kategori ujaran kebencian secara bersamaan dalam satu tweet:
1. **`HS_Religion`** : Ujaran kebencian yang menyasar agama / keyakinan.
2. **`HS_Physical`** : Ujaran kebencian terkait kondisi fisik / disabilitas.
3. **`HS_Individual`** : Ujaran kebencian yang ditargetkan kepada perorangan / individu.
4. **`HS_Group`** : Ujaran kebencian yang ditargetkan kepada kelompok / golongan tertentu.

---

## 🧠 Arsitektur Model: Hierarchical Attention Network (HAN)

1. **Word-Level Encoder:**
   - Input: Kata-kata dalam satu kalimat (maksimal 35 kata)
   - Embedding: Trainable Embedding 128 dimensi
   - Word BiLSTM: 64 units
   - Word Attention: Custom Attention Layer dengan *trainable context vector* ($u_w$)
   - Output: Vektor representasi kalimat (128 dimensi)
2. **Sentence-Level Encoder:**
   - Input: Rangkaian kalimat dalam satu tweet (maksimal 5 kalimat)
   - `TimeDistributed(Sentence_Encoder)`
   - Sentence BiLSTM: 64 units
   - Sentence Attention: Custom Attention Layer dengan *trainable context vector* ($u_s$)
   - Output: Vektor representasi tweet (128 dimensi)
3. **Multi-Label Classifier Head:**
   - Dense(64, ReLU) + Dropout(0.3)
   - Dense(4, Sigmoid) dengan loss *Binary Crossentropy*

---

## 🚀 Fitur Aplikasi Web Streamlit

- **Deteksi Multi-Label Real-time:** Menampilkan status deteksi dan nilai probabilitas untuk seluruh kategori.
- **Explainability (Attention Heatmap):** Menampilkan visualisasi kata mana yang paling diperhatikan oleh model saat mengambil keputusan.
- **Sentence Breakdown:** Menampilkan proses pemecahan kalimat (*sentence splitting*) dari tweet mentah.
- **Slider Threshold:** Fleksibilitas mengubah batas ambang (*threshold*) deteksi langsung di antarmuka web.

---

## 💻 Cara Menjalankan Secara Lokal

1. Clone repositori ini:
   ```bash
   git clone https://github.com/USERNAME/NAMA_REPO.git
   cd NAMA_REPO
   ```
2. Pasang dependensi pustaka:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

---

## 📚 Dataset & Referensi
Dataset yang digunakan berasal dari penelitian:
> Muhammad Okky Ibrohim and Indra Budi. 2019. *Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter*. In ALW3: 3rd Workshop on Abusive Language Online, 46-57.
