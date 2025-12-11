import streamlit as st
import pdfplumber
import numpy as np
from groq import Groq

# -------------------------
# 🌈 Page Settings
# -------------------------
st.set_page_config(
    page_title="AskMyPDF",
    page_icon="📘",
    layout="wide"
)

# -------------------------
# 🌟 Custom CSS for Premium UI
# -------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-container {
    background: linear-gradient(135deg, #dff1ff, #f3e8ff);
    padding: 40px 0;
}

.card {
    background: rgba(255,255,255,0.55);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(18px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
}

.button {
    background: linear-gradient(90deg, #4A90E2, #8E44AD);
    color: white !important;
    padding: 12px 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 18px;
    transition: 0.3s;
    cursor: pointer;
}

.button:hover {
    background: linear-gradient(90deg, #357ABD, #732D91);
    transform: translateY(-3px);
}

.answer-box {
    background:white;
    padding:18px;
    border-radius:12px;
    box-shadow:0 4px 15px rgba(0,0,0,0.1);
}

h1 {
    background: -webkit-linear-gradient(45deg, #4A90E2, #9B59B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    text-align:center;
    font-size:46px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# 🌟 Beautiful Header
# -------------------------
st.markdown("""
<div class="main-container">
    <h1>📘 AskMyPDF</h1>
    <h3 style="text-align:center; color:#4b4b4b; margin-top:-10px;">
        Your AI Assistant to Understand Any PDF Instantly
    </h3>
</div>
""", unsafe_allow_html=True)


# -------------------------
# 🔒 Secure API Key
# -------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

if "requests_made" not in st.session_state:
    st.session_state.requests_made = 0

MAX_REQUESTS = 100


# -------------------------
# 📄 Upload Card
# -------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded = st.file_uploader("📄 Upload your PDF", type="pdf")

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# 🧠 Extract + Process PDF
# -------------------------
def extract_text(pdf_bytes):
    text = ""
    with pdfplumber.open(pdf_bytes) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def simple_embed(text, vocab=None):
    words = text.lower().split()
    if vocab is None:
        vocab = list(set(words))
    vec = np.array([words.count(w) for w in vocab], dtype=np.float32)
    if np.linalg.norm(vec) > 0:
        vec /= np.linalg.norm(vec)
    return vec, vocab


def search(q_vec, doc_vecs, top_k=5):
    sims = [np.dot(q_vec, dv) for dv in doc_vecs]
    return np.argsort(sims)[-top_k:][::-1]


# -------------------------
# 💬 Question Section
# -------------------------
if uploaded:

    with st.spinner("✨ Reading your PDF..."):
        text = extract_text(uploaded)

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    vocab = None
    doc_vecs = []
    for c in chunks:
        v, vocab = simple_embed(c, vocab)
        doc_vecs.append(v)
    doc_vecs = np.array(doc_vecs)

    st.success("✨ PDF Successfully Processed!")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    question = st.text_input(
        "💬 Ask a question from your PDF",
        placeholder="Example: What is the conclusion mentioned in the document?"
    )

    ask = st.button("🔍 Get Answer", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if ask:
        if st.session_state.requests_made >= MAX_REQUESTS:
            st.error("⚠️ You reached your daily limit of 100 questions!")
        elif question.strip() == "":
            st.error("Please type a question!")
        else:
            st.session_state.requests_made += 1

            q_vec, _ = simple_embed(question, vocab)
            top_idx = search(q_vec, doc_vecs)

            context = "\n\n---\n\n".join([chunks[i] for i in top_idx])

            with st.spinner("🤖 Thinking… generating best possible answer…"):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Use ONLY the PDF content provided."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                    ]
                )

            st.markdown("### 📝 Answer")

            st.markdown(
                f"<div class='answer-box'>{response.choices[0].message.content}</div>",
                unsafe_allow_html=True
            )

            st.info(f"📊 Request count: {st.session_state.requests_made}/100")


# -------------------------
# 🌙 Footer
# -------------------------
st.markdown("""
<br><br>
<div style="text-align:center; color:#777;">
Made with ❤️ by <b>Henil</b> | AskMyPDF Premium UI
</div>
""", unsafe_allow_html=True)
