import streamlit as st
import pdfplumber
import numpy as np
from groq import Groq

# -------------------------
# Page setup
# -------------------------
st.set_page_config(
    page_title="AskMyPDF",
    page_icon="📘",
    layout="centered"
)

# -------------------------
# CSS for clean UI
# -------------------------
st.markdown("""
<style>
body {
    font-family: 'Inter', sans-serif;
}






button.stButton > button {
    background-color: #4A90E2;
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    border: none;
}
button.stButton > button:hover {
    background-color: #357ABD;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown("<h1 style='text-align:center;'>📘 AskMyPDF</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;'>Upload a PDF and get instant answers</p>", unsafe_allow_html=True)

# -------------------------
# Groq API
# -------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

if "requests_made" not in st.session_state:
    st.session_state.requests_made = 0
MAX_REQUESTS = 100

# -------------------------
# Upload PDF
# -------------------------
st.markdown('<div class="pdf-card">', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload PDF", type="pdf")
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# PDF extraction
# -------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
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
        vec = vec / np.linalg.norm(vec)
    return vec, vocab

def search(query_vec, doc_vecs, top_k=5):
    sims = [np.dot(query_vec, dv) for dv in doc_vecs]
    return np.argsort(sims)[-top_k:][::-1]

# -------------------------
# Main Logic
# -------------------------
if uploaded:
    with st.spinner("Processing PDF..."):
        text = extract_text(uploaded)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    
    doc_vecs = []
    vocab = None
    for c in chunks:
        v, vocab = simple_embed(c, vocab)
        doc_vecs.append(v)
    doc_vecs = np.array(doc_vecs)

    st.success("PDF uploaded and processed!")

    st.markdown('<div class="question-card">', unsafe_allow_html=True)
    question = st.text_input("Ask something about your PDF:")
    ask = st.button("Get Answer")
    st.markdown('</div>', unsafe_allow_html=True)

    if ask:
        if st.session_state.requests_made >= MAX_REQUESTS:
            st.error("⚠️ Daily limit reached (100 requests).")
        elif question.strip() == "":
            st.error("Please enter a question.")
        else:
            st.session_state.requests_made += 1
            q_vec, _ = simple_embed(question, vocab)
            top_idx = search(q_vec, doc_vecs)
            context = "\n---\n".join([chunks[i] for i in top_idx])

            with st.spinner("Generating answer..."):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role":"system", "content":"Answer ONLY using the PDF content."},
                        {"role":"user", "content":f"Context:\n{context}\n\nQuestion:{question}"}
                    ]
                )

            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(f"**Answer:**\n{response.choices[0].message.content}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.info(f"Requests used: {st.session_state.requests_made}/{MAX_REQUESTS}")

# -------------------------
# Footer
# -------------------------
st.markdown("<center style='color:#888;'>Made with ❤️ by Henil</center>", unsafe_allow_html=True)
