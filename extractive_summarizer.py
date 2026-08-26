"""
extractive_summarizer.py
--------------------------
Step 2 of the pipeline: pick the most important sentences from the
original text without generating any new text.

Two methods are provided:

1. tfidf_summarize   -> simple baseline using TF-IDF sentence scores
2. textrank_summarize -> Sentence-Transformers embeddings + TextRank
                          (graph-based ranking via networkx PageRank)

Both are unsupervised: no training or fine-tuning is needed.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess


# ---------------------------------------------------------------------
# 1) TF-IDF baseline
# ---------------------------------------------------------------------
def tfidf_summarize(text: str, num_sentences: int = 3) -> str:
    """
    Baseline extractive summarizer.

    Scores each sentence by the sum of its words' TF-IDF weights,
    then returns the top-N highest scoring sentences in their
    original order.
    """
    sentences = preprocess(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Score = sum of TF-IDF weights in that sentence's row
    scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

    top_indices = sorted(
        np.argsort(scores)[-num_sentences:], key=lambda i: i
    )
    summary = " ".join(sentences[i] for i in top_indices)
    return summary


# ---------------------------------------------------------------------
# 2) TextRank using Sentence-Transformer embeddings
# ---------------------------------------------------------------------
_EMBEDDING_MODEL_CACHE: dict[str, object] = {}


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load and cache the sentence-transformer model."""
    if model_name not in _EMBEDDING_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL_CACHE[model_name]


def textrank_summarize(
    text: str,
    num_sentences: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
) -> str:
    """
    Extractive summarizer using TextRank over sentence embeddings.

    Pipeline:
        sentences -> Sentence-Transformer embeddings
                  -> cosine similarity matrix
                  -> graph where edge weight = similarity
                  -> PageRank over the graph
                  -> top-N sentences (in original order)
    """
    import networkx as nx

    sentences = preprocess(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    model = _get_embedding_model(model_name)
    embeddings = model.encode(sentences)

    similarity_matrix = cosine_similarity(embeddings)
    # Zero out self-similarity so it doesn't dominate the graph
    np.fill_diagonal(similarity_matrix, 0)

    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    ranked_indices = sorted(scores, key=scores.get, reverse=True)
    top_indices = sorted(ranked_indices[:num_sentences])

    summary = " ".join(sentences[i] for i in top_indices)
    return summary


if __name__ == "__main__":
    sample = (
        "Artificial intelligence is transforming many industries today. "
        "Machine learning models can now understand language, images, "
        "and audio at a very high level. Companies are investing heavily "
        "in AI research and deployment. However, there are also concerns "
        "about job displacement and ethical use of AI. Governments around "
        "the world are starting to draft regulations for AI systems. "
        "Despite the challenges, AI continues to unlock new possibilities "
        "in healthcare, education, and science."
    )
    print("TF-IDF baseline summary:")
    print(tfidf_summarize(sample, num_sentences=2))
    print()
    print("TextRank summary (requires sentence-transformers installed):")
    try:
        print(textrank_summarize(sample, num_sentences=2))
    except ImportError as e:
        print(f"[skipped - missing dependency: {e}]")
