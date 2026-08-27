# AI Text Summarizer

An automatic text summarization system that takes a long text and
produces a short summary preserving the main idea — using pretrained
models only, no training or fine-tuning.

## Pipeline

```
Text
 │
 ├─▶ Preprocessing (clean + split into sentences)
 │
 ├─▶ Extractive Summarization
 │      ├─ TF-IDF (baseline)
 │      └─ TextRank (Sentence-Transformers embeddings + PageRank)
 │
 ├─▶ Abstractive Summarization
 │      ├─ BART (facebook/bart-large-cnn)
 │      └─ T5 (t5-base)  — optional comparison
 │
 └─▶ Evaluation (ROUGE-1 / ROUGE-2 / ROUGE-L) — optional, needs
       a small dataset of reference summaries
```

## Project structure

| File                        | Purpose                                            |
|-----------------------------|-----------------------------------------------------|
| `preprocessing.py`          | Clean text, split into sentences                    |
| `extractive_summarizer.py`  | TF-IDF baseline + TextRank extractive methods       |
| `abstractive_summarizer.py` | BART and T5 abstractive summarization (with chunking for long text) |
| `evaluation.py`             | ROUGE scoring helpers                               |
| `compare_models.py`         | Compare all 4 methods on a small labeled dataset    |
| `app.py`                    | Streamlit web interface                             |
| `requirements.txt`          | Python dependencies                                 |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first time you run BART, T5, or TextRank, the underlying models
will be downloaded automatically from Hugging Face (a few hundred MB
for BART/T5, ~80MB for the sentence-transformer). This needs an
internet connection but only happens once — the models are cached
locally afterward.

## Usage

### As a library

```python
from extractive_summarizer import tfidf_summarize, textrank_summarize
from abstractive_summarizer import bart_summarize, t5_summarize

text = "... your long text ..."

print(tfidf_summarize(text, num_sentences=3))
print(textrank_summarize(text, num_sentences=3))
print(bart_summarize(text, max_length=130, min_length=30))
print(t5_summarize(text, max_length=130, min_length=30))
```

### Web app

```bash
streamlit run app.py
```

This opens a browser UI where you paste text, choose
Extractive / Abstractive / Both, and get the summary.

### Comparing methods with ROUGE (optional)

Once you've collected a handful of (text, reference_summary) pairs,
edit `DATASET` at the top of `compare_models.py` and run:

```bash
python compare_models.py
```

This prints the average ROUGE-1/2/L score for each method so you can
see which one performs best on your data — no assumption is made in
advance about whether TextRank, BART, or T5 will "win".

## Notes on long input text

BART and T5 have limited input windows (~1024 and ~512 tokens
respectively). `abstractive_summarizer.py` automatically splits long
text into word-count-based chunks, summarizes each chunk, and (if
needed) does a final pass to merge the partial summaries into one
coherent result — so you don't need to worry about truncation on
long documents.

## Notes on language

The default models (`bart-large-cnn`, `t5-base`,
`all-MiniLM-L6-v2`) are trained primarily on **English** text. If
your input text is mostly **Arabic**, swap in Arabic-capable models
instead, e.g.:

- Abstractive: `csebuetnlp/mT5_multilingual_XLSum` or `moussaKam/AraBART`
- Sentence embeddings for TextRank: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Just change the `model_name` argument in the relevant function call —
the rest of the pipeline (chunking, TextRank graph, Streamlit UI)
works the same regardless of language.

## Deployment (Streamlit Community Cloud)

1. Push this project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub.
3. Click **New app**, select the repository and branch, and set the
   main file path to `app.py`.
4. Deploy — Streamlit Cloud will install `requirements.txt`
   automatically and give you a public URL.

**Tip:** BART/T5/PyTorch are large; Streamlit Cloud's free tier has
limited memory, so if the app runs out of memory, either use smaller
model variants (`t5-small`, `sshleifer/distilbart-cnn-12-6`) or
enable the "Extractive only" mode by default.

## Suggested execution order

1. Get TextRank extractive working.
2. Get BART abstractive working.
3. Wire both into the Streamlit app.
4. Add T5 as a second abstractive option for comparison.
5. Add ROUGE evaluation once you have reference summaries.
6. Deploy to Streamlit Community Cloud.

Don't try to build everything at once — get extractive + abstractive
working first, then layer on comparison, evaluation, and deployment.
