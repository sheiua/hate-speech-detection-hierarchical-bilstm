import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Deteksi Ujaran Kebencian Multi-Label",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# 2. STYLE
# ==============================================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.15rem;
            font-weight: 800;
            color: #1E3A8A;
            margin-bottom: 0.35rem;
        }
        .sub-title {
            color: #4B5563;
            font-size: 1.02rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }
        .result-box {
            padding: 1.1rem 1.25rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .danger {
            background: #FEF2F2;
            border-left: 6px solid #EF4444;
        }
        .safe {
            background: #ECFDF5;
            border-left: 6px solid #10B981;
        }
        .category-card {
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 0.9rem;
            margin-bottom: 0.7rem;
            background: #FFFFFF;
        }
        .detected {
            color: #B91C1C;
            font-weight: 700;
        }
        .not-detected {
            color: #047857;
            font-weight: 700;
        }
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. CUSTOM LAYER
# ==============================================================================
class HierarchicalAttention(Layer):
    """Attention layer yang digunakan oleh model Hierarchical BiLSTM."""

    def __init__(self, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.return_attention = return_attention

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        self.W = self.add_weight(
            name="att_weight",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="context_vector",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.squeeze(ait, -1)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            ait -= (1.0 - mask) * 1e9

        ait = tf.nn.softmax(ait, axis=-1)
        ait_expanded = tf.expand_dims(ait, -1)
        output = tf.reduce_sum(inputs * ait_expanded, axis=1)

        if self.return_attention:
            return output, ait
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"return_attention": self.return_attention})
        return config


# ==============================================================================
# 4. ASSET MODEL, TOKENIZER, DAN KONFIGURASI
# ==============================================================================
@st.cache_resource(show_spinner="Memuat model deteksi...")
def load_assets():
    model_path = "best_hierarchical_bilstm_model.keras"

    if not os.path.exists(model_path):
        return None, None, {}, None

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"HierarchicalAttention": HierarchicalAttention},
        compile=False,
    )

    tokenizer = None
    if os.path.exists("tokenizer.pickle"):
        with open("tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)

    # Kamus alay bersifat opsional; jika tidak ada, sistem tetap berjalan.
    alay_dict = {}
    if os.path.exists("new_kamusalay.csv"):
        df_alay = pd.read_csv(
            "new_kamusalay.csv",
            encoding="latin-1",
            header=None,
        )
        if df_alay.shape[1] >= 2:
            alay_dict = dict(zip(df_alay.iloc[:, 0], df_alay.iloc[:, 1]))

    config = {
        "max_sentences": 5,
        "max_words": 35,
        "target_labels": [
            "HS_Religion",
            "HS_Physical",
            "HS_Individual",
            "HS_Group",
        ],
    }

    if os.path.exists("config_model.json"):
        with open("config_model.json", "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        config.update(loaded_config)

    return model, tokenizer, alay_dict, config


model, tokenizer, alay_dict, config = load_assets()

# ==============================================================================
# 5. THRESHOLD FINAL - TETAP / TIDAK DAPAT DIUBAH PENGGUNA
# ==============================================================================
FIXED_THRESHOLDS = {
    "HS_Individual": 0.45,
    "HS_Group": 0.45,
    "HS_Religion": 0.35,
    "HS_Physical": 0.25,
}

LABEL_NAMES = {
    "HS_Individual": "Individu",
    "HS_Group": "Kelompok",
    "HS_Religion": "Agama",
    "HS_Physical": "Fisik",
}

# Urutan tampilan tidak mengubah urutan output model.
DISPLAY_ORDER = [
    "HS_Individual",
    "HS_Group",
    "HS_Religion",
    "HS_Physical",
]

# ==============================================================================
# 6. PREPROCESSING DAN ENCODING
# ==============================================================================
def clean_and_split_sentences(text, alay_map):
    """Membersihkan tweet dan memecahnya menjadi unit kalimat."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+|url\b", " ", text)
    text = re.sub(r"@\w+|user\b", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    raw_sentences = re.split(r"[\.\!\?\n;]+|\\\\n|\\n", text)
    clean_sentences = []

    for sentence in raw_sentences:
        words = re.findall(r"[a-zA-Z]+", sentence)
        if words:
            normalized_words = [alay_map.get(word, word) for word in words]
            clean_sentences.append(" ".join(normalized_words))

    return clean_sentences if clean_sentences else [""]


def encode_hierarchical_data(sentences, tok, max_sentences, max_words):
    """Mengubah kalimat menjadi tensor 3D sesuai input Hierarchical BiLSTM."""
    data_tensor = np.zeros(
        (1, max_sentences, max_words),
        dtype="int32",
    )

    sentences = sentences[:max_sentences]
    sequences = tok.texts_to_sequences(sentences)
    padded = pad_sequences(
        sequences,
        maxlen=max_words,
        padding="post",
        truncating="post",
    )

    for idx, sequence in enumerate(padded):
        data_tensor[0, idx, :] = sequence

    return data_tensor


# ==============================================================================
# 7. SIDEBAR: INFORMASI SISTEM SAJA
# ==============================================================================
with st.sidebar:
    st.markdown("### 🛡️ Informasi Sistem")
    st.markdown("**Model:** Hierarchical BiLSTM + Attention Mechanism")
    st.markdown("**Jenis:** Multi-label classification")
    st.markdown("**Bahasa:** Bahasa Indonesia")
    st.markdown("**Kategori:** Individu, Kelompok, Agama, Fisik")
    st.markdown("---")
    st.caption("Threshold sistem telah ditetapkan dan tidak dapat diubah oleh pengguna.")

# ==============================================================================
# 8. HALAMAN UTAMA
# ==============================================================================
st.markdown(
    '<div class="main-title">🛡️ Deteksi Ujaran Kebencian Multi-Label pada Twitter</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Sistem deteksi ujaran kebencian pada tweet berbahasa Indonesia menggunakan <b>Hierarchical BiLSTM dengan Attention Mechanism</b>.</div>',
    unsafe_allow_html=True,
)

if model is None:
    st.error("Berkas model `best_hierarchical_bilstm_model.keras` tidak ditemukan.")
    st.stop()

if tokenizer is None:
    st.error("Berkas `tokenizer.pickle` tidak ditemukan.")
    st.stop()

# ==============================================================================
# 9. INPUT TWEET
# ==============================================================================
st.markdown("### Masukkan Tweet")
tweet_input = st.text_area(
    "",
    placeholder="Ketik atau tempel tweet yang ingin dianalisis...",
    height=150,
    label_visibility="collapsed",
)

analyze_btn = st.button(
    "🔍 Deteksi Ujaran Kebencian",
    type="primary",
    use_container_width=True,
)

# ==============================================================================
# 10. INFERENSI
# ==============================================================================
if analyze_btn:
    if not tweet_input.strip():
        st.warning("Silakan masukkan tweet terlebih dahulu.")
        st.stop()

    # Preprocessing yang digunakan aplikasi.
    sentences = clean_and_split_sentences(tweet_input, alay_dict)

    tensor_input = encode_hierarchical_data(
        sentences,
        tokenizer,
        int(config.get("max_sentences", 5)),
        int(config.get("max_words", 35)),
    )

    with st.spinner("Menganalisis tweet..."):
        probabilities = model.predict(tensor_input, verbose=0)[0]

    target_labels = config.get(
        "target_labels",
        ["HS_Religion", "HS_Physical", "HS_Individual", "HS_Group"],
    )

    if len(probabilities) != len(target_labels):
        st.error(
            "Jumlah output model tidak sesuai dengan jumlah target label pada konfigurasi."
        )
        st.stop()

    # Mapping probabilitas berdasarkan NAMA LABEL, bukan posisi tampilan.
    probability_by_label = {
        label: float(probability)
        for label, probability in zip(target_labels, probabilities)
    }

    detected = []
    results = []

    for label in DISPLAY_ORDER:
        if label not in probability_by_label:
            continue

        probability = probability_by_label[label]
        threshold = FIXED_THRESHOLDS[label]
        is_detected = probability >= threshold

        results.append(
            {
                "label": label,
                "name": LABEL_NAMES[label],
                "probability": probability,
                "status": is_detected,
            }
        )

        if is_detected:
            detected.append(label)

    # ===========================================================================
    # 11. HASIL
    # ===========================================================================
    st.markdown("---")
    st.markdown("### Hasil Deteksi")

    if detected:
        detected_names = [LABEL_NAMES[label] for label in detected]
        st.markdown(
            f"""
            <div class="result-box danger">
                <h3 style="margin:0; color:#B91C1C;">🚨 Ujaran Kebencian Terdeteksi</h3>
                <p style="margin:0.4rem 0 0 0;">
                    Kategori yang terdeteksi: <b>{', '.join(detected_names)}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="result-box safe">
                <h3 style="margin:0; color:#047857;">✅ Tidak Terdeteksi Ujaran Kebencian</h3>
                <p style="margin:0.4rem 0 0 0;">
                    Tweet tidak terdeteksi pada kategori ujaran kebencian yang digunakan sistem.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Probabilitas Prediksi")

    for result in results:
        probability_percent = result["probability"] * 100

        if result["status"]:
            status_text = "Terdeteksi"
            status_class = "detected"
        else:
            status_text = "Tidak terdeteksi"
            status_class = "not-detected"

        st.markdown(
            f"""
            <div class="category-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <b>{result['name']}</b>
                    <span class="{status_class}">{status_text}</span>
                </div>
                <div style="margin-top:0.35rem; color:#4B5563;">
                    Probabilitas model: <b>{probability_percent:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# 12. FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9CA3AF; font-size:0.82rem;'>"
    "Sistem Deteksi Ujaran Kebencian Multi-Label Twitter | Hierarchical BiLSTM + Attention Mechanism"
    "</div>",
    unsafe_allow_html=True,
)
