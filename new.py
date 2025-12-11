import streamlit as st
import pdfplumber
from groq import Groq
import numpy as np

# -------------------------
# 🔧 App Configuration
# -------------------------
st.set_page_config(
    page_title="AskMyPDF",
    page_icon="📘",
    layout="centered"
)

# -------------------------
# 🎨 Stylish Header
# -------------------------
st.markdown("""
<div style="text-align:center; padding: 10px 0;">
    <h1 style="color:#2E86C1; font-size: 42px; font-weight: bold;">📘 AskMyPDF</h1>
    <h3 style="color:#555;">Ask questions directly from your PDF and get instant answers!</h3>
</div>
""", unsafe_allow_html=True)


# -------------------------
# 🔑 Secure API Key
# -------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]     # never shown publicly
client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# 🧮 Daily Request Counter
# -------------------------
if "requests_made" not in st.session_state:
    st.session_state.requests_made = 0
MAX_REQUESTS = 100


# -------------------------
# 📤 Upload PDF
# -------------------------
st.markdown("### 📄 Upload your PDF")
uploaded = st.file_uploader("Select your PDF file", type="pdf")


# -------------------------
# 📘 Extract PDF Text
# -------------------------
def extract_text(pdf_bytes):
    text = ""
    with pdfplumber.open(pdf_bytes) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


# -------------------------
# 🔍 Lightweight Embedding
# -------------------------
def simple_embed(text, vocab=None):
    words = text.lower().split()
    if vocab is None:
        vocab = list(set(words))
    vec = np.array([words.count(w) for w in vocab], dtype=np.float32)
    if np.linalg.norm(vec) > 0:
        vec = vec / np.linalg.norm(vec)
    return vec, vocab


def search(query_vec, doc_vecs, top_k=5):
    sims = [np.dot(query_vec, dv) for dv in doc_vecs]
    idx = np.argsort(sims)[-top_k:][::-1]
    return idx


# -------------------------
# 🧠 Main App Logic
# -------------------------
if uploaded:
    with st.spinner("📖 Reading your PDF..."):
        text = extract_text(uploaded)

    chunk_size = 1000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    doc_vecs = []
    vocab = None

    for c in chunks:
        v, vocab = simple_embed(c, vocab)
        doc_vecs.append(v)
    doc_vecs = np.array(doc_vecs)

    st.success("✅ PDF loaded successfully!")

    st.markdown("### 🤔 Ask any question from your PDF")
    question = st.text_input(
        "Ask here...",
        placeholder="Example: Summarize the main points of section 3."
    )

    if st.button("🔍 Get Answer", use_container_width=True):

        # Limit check
        if st.session_state.requests_made >= MAX_REQUESTS:
            st.error("⚠️ Daily limit of 100 requests reached! Try again tomorrow.")
        elif question.strip() == "":
            st.error("Please enter a question.")
        else:
            st.session_state.requests_made += 1

            q_vec, _ = simple_embed(question, vocab)
            top_idx = search(q_vec, doc_vecs)
            context = "\n\n-----\n\n".join([chunks[i] for i in top_idx])

            with st.spinner("🤖 Groq is thinking..."):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Answer ONLY using the provided PDF context."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                    ]
                )

            st.markdown("### 📝 Answer")
            st.text_area(
                "Answer",
                value=response.choices[0].message.content,
                height=250
            )

            st.info(f"📊 Requests used today: {st.session_state.requests_made}/100")


# -------------------------
# 🔚 Footer
# -------------------------
st.markdown("""
<hr>
<div style="text-align:center; color:gray;">
Made with ❤️ by <b>Henil</b> | AskMyPDF v1.0
</div>
""", unsafe_allow_html=True)
