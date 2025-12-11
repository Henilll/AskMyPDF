import streamlit as st
import pdfplumber
import numpy as np
from groq import Groq

# --------------------------------
# Basic Page Setup
# --------------------------------
st.set_page_config(
    page_title="AskMyPDF",
    page_icon="📄",
    layout="centered"
)

# --------------------------------
# Clean Minimal Styles
# --------------------------------
st.markdown("""
<style>
body {
    font-family: 'Inter', sans-serif;
}
.header {
    text-align:center;
    margin-top:10px;
}
.box {
    background:#ffffff;
    padding:20px;
    border-radius:12px;
    border:1px solid #e5e5e5;
}
.answer-box {
    background:#f8f9fa;
    padding:18px;
    border-radius:10px;
    border:1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Header
# --------------------------------
st.markdown("""
<div class='header'>
    <h1>AskMyPDF</h1>
    <p style="color:#555;font-size:15px;">
        Upload your PDF and ask any question about its content.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------
# API KEY (Hidden in Deployment)
# --------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# --------------------------------
# Limit 100 requests/day
# --------------------------------
if "requests" not in st.session_state:
    st.session_state.requests = 0
MAX_REQ = 100

# --------------------------------
# PDF Upload
# --------------------------------
st.markdown("<div class='box'>", unsafe_allow_html=True)
uploaded = st.file_uploader("📄 Upload PDF", type="pdf")
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# Extract PDF Text
# --------------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


# Simple lightweight embeddings
def simple_embed(text, vocab=None):
    words = text.lower().split()
    if vocab is None:
        vocab = list(set(words))
    vector = np.array([words.count(w) for w in vocab], dtype=np.float32)
    if np.linalg.norm(vector) > 0:
        vector = vector / np.linalg.norm(vector)
    return vector, vocab


def search(q_vec, doc_vecs, top_k=5):
    sims = [np.dot(q_vec, d) for d in doc_vecs]
    return np.argsort(sims)[-top_k:][::-1]


# --------------------------------
# Main Logic
# --------------------------------
if uploaded:

    with st.spinner("Extracting text…"):
        text = extract_text(uploaded)

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]

    vocab = None
    doc_vectors = []
    for c in chunks:
        v, vocab = simple_embed(c, vocab)
        doc_vectors.append(v)
    doc_vectors = np.array(doc_vectors)

    st.success("PDF uploaded and processed!")

    # Ask question
    question = st.text_input("Ask something about your PDF:")

    if st.button("Get Answer"):
        if st.session_state.requests >= MAX_REQ:
            st.error("You reached your daily limit of 100 questions.")
        elif question.strip() == "":
            st.error("Please enter a question.")
        else:
            st.session_state.requests += 1

            q_vec, _ = simple_embed(question, vocab)
            idx = search(q_vec, doc_vectors)

            context = "\n---\n".join([chunks[i] for i in idx])

            with st.spinner("Thinking…"):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Answer ONLY from the PDF context."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                    ]
                )

            st.markdown("### Answer")
            st.markdown(
                f"<div class='answer-box'>{response.choices[0].message.content}</div>",
                unsafe_allow_html=True
            )

            st.caption(f"Used {st.session_state.requests}/{MAX_REQ} requests today.")


# --------------------------------
# Footer
# --------------------------------
st.markdown("<br><center style='color:#888;'>Made by Henil</center>", unsafe_allow_html=True)
