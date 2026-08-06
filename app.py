import streamlit as st
import re
import joblib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

st.set_page_config(
    page_title="Analisis Sentimen",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.main .block-container {
    max-width: 1100px;
    padding: 0 2.5rem 4rem 2.5rem;
}

/* ── Top Header Banner ── */
.site-header {
    background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(59,130,246,0.15));
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 1.4rem 2.5rem;
    margin: 0 -2.5rem 2.5rem -2.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.site-header-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, #8b5cf6, #3b82f6);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}

.site-header-text h2 {
    color: #ffffff;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 2px 0;
    line-height: 1;
}

.site-header-text p {
    color: rgba(255,255,255,0.45);
    font-size: 0.8rem;
    margin: 0;
}

/* ── Section title ── */
.section-label {
    color: rgba(255,255,255,0.5);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
}

/* ── Textarea ── */
label, .stTextArea label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

textarea {
    background: rgba(255,255,255,0.07) !important;
    color: #ffffff !important;
    border: 1.5px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
}

textarea:focus {
    border-color: rgba(139,92,246,0.7) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}

textarea::placeholder {
    color: rgba(255,255,255,0.22) !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #8b5cf6, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.72rem 1.5rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.02em;
    margin-top: 0.75rem;
    box-shadow: 0 4px 18px rgba(139,92,246,0.35) !important;
    transition: opacity 0.2s ease, transform 0.2s ease !important;
}

.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(139,92,246,0.5) !important;
}

/* ── Divider between columns ── */
.col-divider {
    width: 1px;
    background: rgba(255,255,255,0.08);
    min-height: 260px;
    margin: 0 auto;
}

/* ── Result panel ── */
.result-panel {
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-radius: 18px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    animation: fadeIn 0.4s ease forwards;
    min-height: 260px;
}

@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.96); }
    to   { opacity: 1; transform: scale(1); }
}

.result-empty {
    background: rgba(255,255,255,0.03);
    border: 1.5px dashed rgba(255,255,255,0.1);
}

.result-positif {
    background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(52,211,153,0.07));
    border: 1.5px solid rgba(52,211,153,0.4);
}

.result-negatif {
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(252,165,165,0.07));
    border: 1.5px solid rgba(252,165,165,0.4);
}

.result-netral {
    background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(253,230,138,0.07));
    border: 1.5px solid rgba(253,230,138,0.4);
}

.result-emoji {
    font-size: 3.5rem;
    line-height: 1;
    margin-bottom: 1rem;
    display: block;
}

.result-waiting-icon {
    font-size: 2.5rem;
    opacity: 0.2;
    margin-bottom: 0.75rem;
    display: block;
}

.result-waiting-text {
    color: rgba(255,255,255,0.2);
    font-size: 0.85rem;
}

.result-badge {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.4);
    margin-bottom: 0.4rem;
}

.result-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    margin: 0;
}

.result-value-positif { color: #34d399; }
.result-value-negatif { color: #f87171; }
.result-value-netral  { color: #fbbf24; }

/* hide streamlit default padding for columns */
[data-testid="column"] {
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load Resources ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    model      = joblib.load("data/SVM/model/svm_linear_C10.0.pkl")
    vectorizer = joblib.load("data/tf-idf/tfidf_vectorizer.pkl")

    try:
        url_slang   = ("https://raw.githubusercontent.com/nasalsabila/kamus-alay/"
                       "master/colloquial-indonesian-lexicon.csv")
        kamus_slang = pd.read_csv(url_slang)
        slang_dict  = dict(zip(kamus_slang["slang"], kamus_slang["formal"]))
    except Exception:
        slang_dict = {}

    try:
        kamus_jawa = pd.read_csv("../Dataset/norm_java.csv")
        jawa_dict  = dict(zip(kamus_jawa["java"], kamus_jawa["formal"]))
    except Exception:
        jawa_dict = {}

    norm_dict = {**slang_dict, **jawa_dict}

    factory_sw  = StopWordRemoverFactory()
    stop_words  = set(factory_sw.get_stop_words())
    kata_dipertahankan = {
        "tidak", "bukan", "jangan", "belum", "kurang", "tak", "tiada",
        "gak", "nggak", "enggak", "blm", "ora", "ra",
        "juara", "apik", "bagus", "menang", "kalah", "ajur", "modyar",
        "pekok", "parah", "hancur", "suog", "cok", "jancok", "sialan",
        "paling", "banget", "tenan", "pisan", "terlalu",
    }
    stop_words_final = stop_words - kata_dipertahankan

    stemmer = StemmerFactory().create_stemmer()
    return model, vectorizer, norm_dict, stop_words_final, stemmer


# ── Preprocessing ───────────────────────────────────────────────────────────────
def preprocess(raw_text, norm_dict, stop_words_final, stemmer):
    text = str(raw_text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"RT[\s]+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    tokens = word_tokenize(text)
    tokens = [norm_dict.get(str(w).lower(), w) for w in tokens]
    tokens = [t for t in tokens if t.lower() not in stop_words_final]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


SENTIMENT_CONFIG = {
    "positif": {
        "emoji": "😊",
        "label": "Positif",
        "card":  "result-positif",
        "val":   "result-value-positif",
    },
    "negatif": {
        "emoji": "😠",
        "label": "Negatif",
        "card":  "result-negatif",
        "val":   "result-value-negatif",
    },
    "netral": {
        "emoji": "😐",
        "label": "Netral",
        "card":  "result-netral",
        "val":   "result-value-netral",
    },
}


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    # ── Top Header Banner
    st.markdown("""
    <div class="site-header">
        <div class="site-header-icon">💬</div>
        <div class="site-header-text">
            <h2>Analisis Sentimen Pelita Jaya IBL</h2>
            <p>Deteksi sentimen teks secara otomatis menggunakan machine learning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model
    with st.spinner("Memuat model..."):
        try:
            model, vectorizer, norm_dict, stop_words_final, stemmer = load_resources()
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            st.stop()

    # ── Two-column layout: Input | Result
    col_input, col_gap, col_result = st.columns([5, 0.3, 4])

    with col_input:
        st.markdown('<div class="section-label">Masukkan Teks</div>', unsafe_allow_html=True)
        user_input = st.text_area(
            "teks",
            placeholder="Ketik atau tempel teks di sini...\n\nContoh: Pelita Jaya tampil luar biasa musim ini!",
            height=200,
            key="input_text",
            label_visibility="collapsed",
        )
        analyze_btn = st.button("Analisis Sentimen", use_container_width=True)

    with col_gap:
        st.markdown('<div class="col-divider"></div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="section-label">Hasil Sentimen</div>', unsafe_allow_html=True)

        if analyze_btn:
            if not user_input.strip():
                st.warning("Silakan masukkan teks terlebih dahulu.")
            else:
                with st.spinner("Memproses..."):
                    text_final = preprocess(user_input, norm_dict, stop_words_final, stemmer)
                    vec        = vectorizer.transform([text_final])
                    sentiment  = str(model.predict(vec)[0]).lower()

                cfg = SENTIMENT_CONFIG.get(sentiment, SENTIMENT_CONFIG["netral"])
                st.markdown(f"""
                <div class="result-panel {cfg['card']}">
                    <span class="result-emoji">{cfg['emoji']}</span>
                    <div class="result-badge">Sentimen Terdeteksi</div>
                    <div class="result-value {cfg['val']}">{cfg['label']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-panel result-empty">
                <span class="result-waiting-icon">💡</span>
                <div class="result-waiting-text">Hasil analisis akan<br>muncul di sini</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
