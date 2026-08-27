"""
app.py
-------
Streamlit web interface: runs ALL FOUR summarization methods
(TF-IDF, TextRank, BART, T5) at the same time and shows them
side by side for comparison.

Run locally with:
    streamlit run app.py
"""

import time

import streamlit as st

from extractive_summarizer import tfidf_summarize, textrank_summarize
from abstractive_summarizer import bart_summarize, t5_summarize


st.set_page_config(page_title="AI Text Summarizer", page_icon="📝", layout="wide")

st.title("📝 AI Text Summarizer — Model Comparison")
st.caption(
    "Paste a long text below and compare TF-IDF, TextRank, BART, and T5 "
    "summaries side by side — all pretrained, no fine-tuning."
)

# ---------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    st.subheader("Which methods to run")
    run_tfidf = st.checkbox("TF-IDF (baseline)", value=True)
    run_textrank = st.checkbox("TextRank", value=True)
    run_bart = st.checkbox("BART", value=True)
    run_t5 = st.checkbox("T5", value=True)

    st.subheader("Extractive options")
    num_sentences = st.slider("Number of sentences", 1, 10, 3)

    st.subheader("Abstractive options")
    max_length = st.slider("Max summary length (tokens)", 30, 300, 130)
    min_length = st.slider("Min summary length (tokens)", 5, 100, 30)

    st.divider()
    st.caption(
        "First run will download BART/T5/sentence-transformer models "
        "from Hugging Face — this can take a minute or two."
    )


# ---------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------
text_input = st.text_area(
    "Paste your text here",
    height=220,
    placeholder="Paste a long article, report, or document...",
)

col1, col2 = st.columns([1, 3])
with col1:
    run_button = st.button("Compare all models", type="primary", use_container_width=True)
with col2:
    word_count = len(text_input.split()) if text_input else 0
    st.caption(f"{word_count} words in input")


# ---------------------------------------------------------------------
# Run every selected method and collect results
# ---------------------------------------------------------------------
def run_method(name, func):
    """Run one summarizer, catching errors so one failure doesn't stop the rest."""
    start = time.time()
    try:
        summary = func()
        elapsed = time.time() - start
        return {"name": name, "summary": summary, "time": elapsed, "error": None}
    except Exception as e:
        elapsed = time.time() - start
        return {"name": name, "summary": None, "time": elapsed, "error": str(e)}


if run_button:
    if not text_input or not text_input.strip():
        st.warning("Please paste some text first.")
    else:
        jobs = []
        if run_tfidf:
            jobs.append(("TF-IDF (baseline)", lambda: tfidf_summarize(
                text_input, num_sentences=num_sentences
            )))
        if run_textrank:
            jobs.append(("TextRank", lambda: textrank_summarize(
                text_input, num_sentences=num_sentences
            )))
        if run_bart:
            jobs.append(("BART", lambda: bart_summarize(
                text_input, max_length=max_length, min_length=min_length
            )))
        if run_t5:
            jobs.append(("T5", lambda: t5_summarize(
                text_input, max_length=max_length, min_length=min_length
            )))

        if not jobs:
            st.warning("Select at least one method in the sidebar.")
        else:
            results = []
            progress = st.progress(0.0, text="Running models...")
            for i, (name, func) in enumerate(jobs):
                progress.progress((i) / len(jobs), text=f"Running {name}...")
                results.append(run_method(name, func))
            progress.progress(1.0, text="Done")
            progress.empty()

            st.subheader("Comparison")
            cols = st.columns(len(results))
            for col, result in zip(cols, results):
                with col:
                    st.markdown(f"**{result['name']}**")
                    if result["error"]:
                        st.error(f"Failed: {result['error']}")
                    else:
                        st.write(result["summary"])
                        st.caption(f"{result['time']:.1f}s")

            with st.expander("Show as table"):
                st.table(
                    [
                        {
                            "Method": r["name"],
                            "Summary": r["summary"] or f"ERROR: {r['error']}",
                            "Time (s)": round(r["time"], 1),
                        }
                        for r in results
                    ]
                )

st.divider()
st.caption(
    "Methods: TF-IDF & TextRank (extractive, pick existing sentences) "
    "vs. BART & T5 (abstractive, generate new text) — all pretrained, no fine-tuning."
)
