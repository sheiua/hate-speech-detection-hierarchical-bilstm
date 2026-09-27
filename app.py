import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Deteksi Ujaran Kebencian Twitter (Hierarchical BiLSTM)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-box-safe {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .status-box-danger {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DEFINISI CUSTOM LAYER: HIERARCHICAL ATTENTION
# ==============================================================================
class HierarchicalAttention(Layer):
    """
    Attention Mechanism dengan Trainable Context Vector (Yang et al., 2016)
    """
    def __init__(self, return_attention=False, **kwargs):
        super(HierarchicalAttention, self).__init__(**kwargs)
        self.return_attention = return_attention

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.W = self.add_weight(
            name="att_weight",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="context_vector",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super(HierarchicalAttention, self).build(input_shape)

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
        config = super(HierarchicalAttention, self).get_config()
        config.update({"return_attention": self.return_attention})
        return config

# ==============================================================================
# 3. RESOURCE CACHE: MEMUAT MODEL, TOKENIZER, & ASSET
# ==============================================================================
@st.cache_resource(show_spinner="Sedang memuat model dan aset Deep Learning...")
def load_all_assets():
    candidate_paths = [
        "model_deploy.keras",
        "best_hierarchical_bilstm_model.keras",
        "hierarchical_bilstm_attention_model.keras"
    ]
    model_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        return None, None, None, None

    # Load Model Keras
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'HierarchicalAttention': HierarchicalAttention}
    )

    # Load Tokenizer
    tokenizer = None
    if os.path.exists("tokenizer.pickle"):
        with open("tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)

    # Load Kamus Alay
    alay_dict = {}
    if os.path.exists("new_kamusalay.csv"):
        df_alay = pd.read_csv("new_kamusalay.csv", encoding="latin-1", header=None)
        alay_dict = dict(zip(df_alay[0], df_alay[1]))

    # Load Config jika ada
    config = {
        'max_sentences': 5,
        'max_words': 35,
        'target_labels': ['HS_Religion', 'HS_Physical', 'HS_Individual', 'HS_Group']
    }
    if os.path.exists("config_model.json"):
        with open("config_model.json", "r") as f:
            config = json.load(f)

    return model, tokenizer, alay_dict, config

model, tokenizer, alay_dict, config = load_all_assets()

# ==============================================================================
# 4. FUNGSI PREPROCESSING & ENCODING
# ==============================================================================
def clean_and_split_sentences(text, alay_map):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+|url\b', ' ', text)
    text = re.sub(r'@\w+|user\b', ' ', text)
    text = re.sub(r'#(\w+)', r' \1 ', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    raw_sentences = re.split(r'[\.\!\?\n;]+|\\\\n|\\n', text)
    clean_sentences = []
    
    for sent in raw_sentences:
        words = re.findall(r'[a-zA-Z]+', sent)
        if words:
            norm_words = [alay_map.get(w, w) for w in words]
            clean_sentences.append(' '.join(norm_words))
            
    if not clean_sentences:
        clean_sentences = ['']
    return clean_sentences

def encode_hierarchical_data(sentences_list, tok, max_sents, max_words):
    data_tensor = np.zeros((1, max_sents, max_words), dtype='int32')
    sents_to_process = sentences_list[:max_sents]
    seqs = tok.texts_to_sequences(sents_to_process)
    padded_seqs = pad_sequences(seqs, maxlen=max_words, padding='post', truncating='post')
    
    for j, sent_seq in enumerate(padded_seqs):
        data_tensor[0, j, :] = sent_seq
    return data_tensor

# ==============================================================================
# 5. SIDEBAR: PENGATURAN & INFORMASI
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/shield.png", width=70)
    st.title("Pengaturan Model")
    st.markdown("**Arsitektur:** Hierarchical BiLSTM + Attention Mechanism (HAN)")
    
    st.markdown("---")
    st.subheader("⚙️ Batas Ambang (Threshold)")
    st.caption("Atur nilai batas probabilitas untuk menandai suatu kategori terdeteksi:")
    
    th_religion = st.slider("HS_Religion (Agama)", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
    th_physical = st.slider("HS_Physical (Fisik)", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
    th_individual = st.slider("HS_Individual (Individu)", min_value=0.10, max_value=0.90, value=0.45, step=0.05)
    th_group = st.slider("HS_Group (Kelompok)", min_value=0.10, max_value=0.90, value=0.45, step=0.05)
    
    thresholds = {
        'HS_Religion': th_religion,
        'HS_Physical': th_physical,
        'HS_Individual': th_individual,
        'HS_Group': th_group
    }
    
    st.markdown("---")
    st.markdown("### ℹ️ Tentang 4 Target Label:")
    st.markdown("""
    - **HS_Religion:** Ujaran kebencian menyasar agama/keyakinan.
    - **HS_Physical:** Ujaran kebencian terkait fisik/disabilitas.
    - **HS_Individual:** Hinaan ditargetkan ke perorangan.
    - **HS_Group:** Hinaan ditargetkan ke kelompok/golongan.
    """)

# ==============================================================================
# 6. KONTEN UTAMA APLIKASI
# ==============================================================================
st.markdown('<div class="main-title">🛡️ Deteksi Ujaran Kebencian Multi-Label pada Twitter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem cerdas berbasis <b>Hierarchical BiLSTM dengan Attention Mechanism</b> untuk mendeteksi berbagai kombinasi ujaran kebencian dalam satu tweet berbahasa Indonesia.</div>', unsafe_allow_html=True)

if model is None or tokenizer is None:
    st.error("⚠️ Berkas model (`best_hierarchical_bilstm_model.keras`) atau `tokenizer.pickle` belum ditemukan di folder aplikasi!")
    st.info("Silakan salin atau unduh berkas hasil pelatihan dari Google Colab ke dalam folder kerja aplikasi ini.")
    st.stop()

# Pilihan Preset Contoh Tweet
st.markdown("##### 💡 Pilih Contoh Tweet atau Ketik Tweet Anda Sendiri:")
preset_options = [
    "-- Ketik Sendiri di Bawah --",
    "Selamat pagi semuanya! Semoga hari ini penuh berkah dan produktif untuk kita semua. URL",
    "Dasar kaum kafir perusak negara! Semua agama kalian sesat dan tidak pantas hidup di sini!",
    "Heh cebol cacat, jalan aja masih pincang gak usah sok jagoan lu di Twitter!",
    "Dasar kamu manusia buta budek idiot! Muka kamu jelek banget gak punya otak USER!",
    "Kumpulan orang-orang idiot dan sakit jiwa emang mereka itu, satu gerombolan otaknya miring semua!",
    "USER kamu memang dasar manusia bodoh tolol penipu, hidup lu gak berguna!"
]

selected_preset = st.selectbox("Pilih Tweet Contoh:", preset_options, index=0)
default_text = "" if selected_preset == preset_options[0] else selected_preset

tweet_input = st.text_area(
    "Masukkan Teks Tweet:",
    value=default_text,
    placeholder="Contoh: Dasar kaum sesat tidak punya otak bikin onar terus di negeri ini...",
    height=100
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    analyze_btn = st.button("🚀 Analisis Tweet", type="primary", use_container_width=True)

# ==============================================================================
# 7. PROSES INFERENSI & VISUALISASI HASIL
# ==============================================================================
if analyze_btn:
    if not tweet_input.strip():
        st.warning("Silakan masukkan teks tweet terlebih dahulu sebelum menganalisis!")
    else:
        # Preprocessing & Sentence Splitting
        sentences = clean_and_split_sentences(tweet_input, alay_dict)
        tensor_input = encode_hierarchical_data(
            sentences, tokenizer, config.get('max_sentences', 5), config.get('max_words', 35)
        )
        
        # Prediksi Probabilitas
        with st.spinner("Model sedang menganalisis struktur hierarki kata dan kalimat..."):
            probs = model.predict(tensor_input, verbose=0)[0]
            
        target_labels = config.get('target_labels', ['HS_Religion', 'HS_Physical', 'HS_Individual', 'HS_Group'])
        
        # Evaluasi dengan Threshold
        detected_categories = []
        res_rows = []
        for i, label in enumerate(target_labels):
            th = thresholds.get(label, 0.5)
            is_pos = (probs[i] >= th)
            if is_pos:
                detected_categories.append(label)
            res_rows.append({
                'Kategori': label,
                'Probabilitas': probs[i],
                'Persentase': f"{probs[i]*100:.2f}%",
                'Threshold': f"{th*100:.0f}%",
                'Status': is_pos
            })
            
        # Banner Status Utama
        st.markdown("---")
        if detected_categories:
            st.markdown(f"""
            <div class="status-box-danger">
                <h3 style="color:#B91C1C; margin:0;">🚨 PERINGATAN: Ujaran Kebencian Terdeteksi!</h3>
                <p style="margin:5px 0 0 0; font-size:1.05rem;">
                    Tweet ini terdeteksi memiliki <b>{len(detected_categories)} Kategori Multi-Label</b>: 
                    <span style="color:#DC2626; font-weight:bold;">{', '.join(detected_categories)}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-box-safe">
                <h3 style="color:#047857; margin:0;">🟢 AMAN: Tidak Terdeteksi Ujaran Kebencian</h3>
                <p style="margin:5px 0 0 0; font-size:1.05rem;">
                    Tweet ini tidak melebihi batas ambang kategori ujaran kebencian manapun.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Tampilkan 4 Kartu Metrik
        st.markdown("#### 📊 Hasil Deteksi Tiap Kategori:")
        cols = st.columns(4)
        for i, row in enumerate(res_rows):
            with cols[i]:
                status_icon = "⚠️ Terdeteksi" if row['Status'] else "✅ Bersih"
                badge_color = "#DC2626" if row['Status'] else "#059669"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-weight:700; font-size:1.1rem; color:#1F2937;">{row['Kategori']}</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{badge_color}; margin:6px 0;">{row['Persentase']}</div>
                    <div style="font-size:0.85rem; color:#6B7280;">Batas: {row['Threshold']}</div>
                    <div style="font-size:0.9rem; font-weight:600; color:{badge_color}; margin-top:4px;">{status_icon}</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Tab Detail Analisis
        st.markdown("<br>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["📈 Grafik Probabilitas", "🔍 Pemisahan Kalimat (Hierarki)", "🧠 Interpretasi Attention"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            labels = [r['Kategori'] for r in res_rows]
            values = [r['Probabilitas'] * 100 for r in res_rows]
            thresh_vals = [thresholds[l] * 100 for l in labels]
            colors = ['#EF4444' if r['Status'] else '#3B82F6' for r in res_rows]
            
            bars = ax.bar(labels, values, color=colors, width=0.5)
            # Garis threshold
            for idx, th_val in enumerate(thresh_vals):
                ax.plot([idx - 0.28, idx + 0.28], [th_val, th_val], color='red', linestyle='--', linewidth=2)
                
            ax.set_ylim(0, 100)
            ax.set_ylabel("Probabilitas (%)")
            ax.set_title("Probabilitas Prediksi vs Garis Threshold (Merah)", fontweight="bold")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
            sns.despine()
            st.pyplot(fig)
            st.caption("Keterangan: Batang merah berarti probabilitas melewati garis threshold (Terdeteksi Positif).")
            
        with tab2:
            st.markdown("##### 📝 Struktur Hierarki Kalimat yang Diekstrak:")
            st.write(f"Tweet asli dipecah menjadi **{len(sentences)} kalimat** untuk diproses oleh Hierarchical BiLSTM:")
            for s_idx, s in enumerate(sentences, 1):
                st.info(f"**Kalimat #{s_idx}:** {s}")
                
        with tab3:
            st.markdown("##### 🔎 Word-Level Attention Heatmap (Kata Paling Disorot Model):")
            try:
                # Ekstraksi attention weights
                time_dist = model.get_layer('time_distributed_sentences')
                sent_enc = time_dist.layer
                word_bilstm_layer = sent_enc.get_layer('word_bilstm')
                word_att_layer = sent_enc.get_layer('word_attention')
                
                extractor = Model(inputs=sent_enc.input, outputs=word_bilstm_layer.output)
                W, b, u = word_att_layer.get_weights()
                
                # Visualisasi untuk kalimat pertama yang memiliki kata
                target_sent = sentences[0] if sentences else ""
                words = target_sent.split()[:35]
                
                if words:
                    seq = tokenizer.texts_to_sequences([' '.join(words)])
                    padded = pad_sequences(seq, maxlen=35, padding='post', truncating='post')
                    h = extractor.predict(padded, verbose=0)
                    
                    uit = np.tanh(np.dot(h, W) + b)
                    ait = np.dot(uit, u).squeeze(-1)
                    valid_ait = ait[0][:len(words)]
                    exp_ait = np.exp(valid_ait - np.max(valid_ait))
                    alphas = exp_ait / np.sum(exp_ait)
                    
                    fig_att, ax_att = plt.subplots(figsize=(max(8, len(words) * 0.9), 1.8))
                    sns.heatmap(
                        alphas.reshape(1, -1), annot=True, fmt=".2f", cmap='YlOrRd',
                        xticklabels=words, yticklabels=['Perhatian'], cbar=False, ax=ax_att
                    )
                    plt.title("Visualisasi Bobot Perhatian Kata (Attention Mechanism)", fontweight='bold', fontsize=11)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig_att)
                    st.caption("Semakin pekat warna merah/oranye, semakin tinggi pengaruh kata tersebut terhadap keputusan model.")
                else:
                    st.write("Tidak ada kata yang valid untuk ditampilkan pada heatmap.")
            except Exception as e:
                st.warning(f"Informasi ekstraksi bobot attention tidak dapat ditampilkan: {e}")

# ==============================================
# 8. FOOTER
# ==============================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9CA3AF; font-size:0.85rem;'>"
    "Aplikasi Sistem Deteksi Ujaran Kebencian Multi-Label Twitter | Hierarchical BiLSTM + Attention Mechanism (HAN)"
    "</div>",
    unsafe_allow_html=True
)
