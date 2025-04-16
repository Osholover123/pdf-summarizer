import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import tempfile
import os

# Title
st.title("📄 PDF Summarizer AI")
st.write("Upload a PDF file and get a smart summary powered by Hugging Face Transformers.")

# Upload file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

def chunk_text(text, max_tokens=1000):
    sentences = text.split(". ")
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_tokens:
            chunk += sentence + ". "
        else:
            chunks.append(chunk)
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk)
    return chunks

def summarize_text(text):
    chunks = chunk_text(text)
    summaries = []
    for i, chunk in enumerate(chunks):
        with st.spinner(f"Summarizing chunk {i+1}/{len(chunks)}..."):
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
            summaries.append(summary)
    return "\n\n".join(summaries)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("📖 Reading and summarizing the PDF..."):
        text = extract_text_from_pdf(tmp_path)
        if text.strip():
            summary = summarize_text(text)
            st.subheader("🧾 Summary")
            st.write(summary)
        else:
            st.warning("No readable text found in the PDF.")

    os.remove(tmp_path)
