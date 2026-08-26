"""
compare_models.py
--------------------
Step 4 & 5 of the pipeline: once you have a small dataset of
(text, reference_summary) pairs, run all four methods on it and
compare their ROUGE scores to decide which one performs best.

You do NOT need this to use the summarizer — it's only for a
formal, numbers-based comparison once you've collected a handful of
example articles with human-written reference summaries.

Usage:
    python compare_models.py

Edit the `DATASET` list below with your own (text, reference) pairs,
or load them from a CSV/JSON file instead.
"""

from extractive_summarizer import tfidf_summarize, textrank_summarize
from abstractive_summarizer import bart_summarize, t5_summarize
from evaluation import compare_methods


# Replace this with your real dataset: a list of
# (full_text, human_reference_summary) tuples.
DATASET = [
    (
        "Artificial intelligence is transforming many industries today. "
        "Machine learning models can now understand language, images, "
        "and audio at a very high level. Companies are investing heavily "
        "in AI research and deployment. However, there are also concerns "
        "about job displacement and ethical use of AI. Governments around "
        "the world are starting to draft regulations for AI systems.",
        "AI is transforming industries through advanced machine learning, "
        "prompting heavy investment as well as concerns over jobs and "
        "the need for government regulation.",
    ),
    # Add more (text, reference_summary) pairs here...
]


def run_comparison(dataset=DATASET, num_sentences: int = 2):
    results_by_method = {"tfidf": [], "textrank": [], "bart": [], "t5": []}

    for text, reference in dataset:
        results_by_method["tfidf"].append(
            (tfidf_summarize(text, num_sentences=num_sentences), reference)
        )
        try:
            results_by_method["textrank"].append(
                (textrank_summarize(text, num_sentences=num_sentences), reference)
            )
        except Exception as e:
            print(f"[textrank skipped: {e}]")

        try:
            results_by_method["bart"].append(
                (bart_summarize(text, max_length=60, min_length=15), reference)
            )
        except Exception as e:
            print(f"[bart skipped: {e}]")

        try:
            results_by_method["t5"].append(
                (t5_summarize(text, max_length=60, min_length=15), reference)
            )
        except Exception as e:
            print(f"[t5 skipped: {e}]")

    # Drop methods with no successful results (e.g. missing dependency)
    results_by_method = {k: v for k, v in results_by_method.items() if v}

    scores = compare_methods(results_by_method)

    print("\n=== Average ROUGE scores across dataset ===")
    print(f"{'Method':<12}{'ROUGE-1':>10}{'ROUGE-2':>10}{'ROUGE-L':>10}")
    for method, score in scores.items():
        print(
            f"{method:<12}{score['rouge1']:>10}{score['rouge2']:>10}{score['rougeL']:>10}"
        )

    return scores


if __name__ == "__main__":
    run_comparison()
