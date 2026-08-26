"""
app.py
-------
Step 6 of the pipeline: Streamlit web interface.

Run locally with:
    streamlit run app.py

The user pastes text, picks Extractive / Abstractive / Both, and
gets the summary/summaries back.
"""

import time

import streamlit as st

from extractive_summarizer import tfidf_summarize, textrank_summarize
from abstractive_summarizer import bart_summarize, t5_summarize


st.set_page_config(page_title="AI Text Summarizer", page_icon="📝", layout="centered")

st.title("📝 AI Text Summarizer")
st.caption(
    "Paste a long text below and get a short summary that preserves "
    "the main idea — no training or fine-tuning, just pretrained models."
)

# ---------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Summarization mode",
        options=["Extractive", "Abstractive", "Both"],
        index=2,
    )

    st.subheader("Extractive options")
    extractive_method = st.selectbox(
        "Method", options=["TextRank (recommended)", "TF-IDF (baseline)"]
    )
    num_sentences = st.slider("Number of sentences", 1, 10, 3)

    st.subheader("Abstractive options")
    abstractive_model = st.selectbox(
        "Model", options=["BART (facebook/bart-large-cnn)", "T5 (t5-base)"]
    )
    max_length = st.slider("Max summary length (tokens)", 30, 300, 130)
    min_length = st.slider("Min summary length (tokens)", 5, 100, 30)

    st.divider()
    st.caption(
        "First run will download the selected model(s) from Hugging "
        "Face — this can take a minute or two depending on your "
        "connection."
    )


# ---------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------
text_input = st.text_area(
    "Paste your text here",
    height=280,
    placeholder="Paste a long article, report, or document...",
)

col1, col2 = st.columns([1, 3])
with col1:
    run_button = st.button("Summarize", type="primary", use_container_width=True)
with col2:
    word_count = len(text_input.split()) if text_input else 0
    st.caption(f"{word_count} words in input")


# ---------------------------------------------------------------------
# Run summarization
# ---------------------------------------------------------------------
if run_button:
    if not text_input or not text_input.strip():
        st.warning("Please paste some text first.")
    else:
        show_extractive = mode in ("Extractive", "Both")
        show_abstractive = mode in ("Abstractive", "Both")

        if show_extractive:
            st.subheader("📌 Extractive Summary")
            with st.spinner("Selecting the most important sentences..."):
                start = time.time()
                if extractive_method.startswith("TextRank"):
                    try:
                        summary = textrank_summarize(
                            text_input, num_sentences=num_sentences
                        )
                    except Exception as e:
                        st.error(f"TextRank failed ({e}); falling back to TF-IDF.")
                        summary = tfidf_summarize(
                            text_input, num_sentences=num_sentences
                        )
                else:
                    summary = tfidf_summarize(
                        text_input, num_sentences=num_sentences
                    )
                elapsed = time.time() - start
            st.write(summary)
            st.caption(f"Generated in {elapsed:.1f}s")

        if show_abstractive:
            st.subheader("🧠 Abstractive Summary")
            with st.spinner("Reading the text and writing a new summary..."):
                start = time.time()
                try:
                    if abstractive_model.startswith("BART"):
                        summary = bart_summarize(
                            text_input, max_length=max_length, min_length=min_length
                        )
                    else:
                        summary = t5_summarize(
                            text_input, max_length=max_length, min_length=min_length
                        )
                except Exception as e:
                    st.error(f"Abstractive summarization failed: {e}")
                    summary = None
                elapsed = time.time() - start
            if summary:
                st.write(summary)
                st.caption(f"Generated in {elapsed:.1f}s")

st.divider()
st.caption(
    "Built with TF-IDF, TextRank (Sentence-Transformers), BART, and T5 — "
    "all pretrained, no fine-tuning."
)
