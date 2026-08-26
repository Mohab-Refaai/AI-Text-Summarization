"""
evaluation.py
---------------
Step 4 of the pipeline: evaluate generated summaries against
human-written reference summaries using ROUGE.

This step is OPTIONAL and only meaningful once you have a small
dataset of (text, reference_summary) pairs. It is not required to
get the summarizer working.
"""

from __future__ import annotations


def compute_rouge(candidate_summary: str, reference_summary: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for a single
    candidate summary against a single reference summary.

    Returns a dict like:
        {
            "rouge1": 0.42,
            "rouge2": 0.21,
            "rougeL": 0.38,
        }
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference_summary, candidate_summary)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def evaluate_dataset(pairs: list[tuple[str, str]]) -> dict:
    """
    Evaluate a full dataset of (candidate_summary, reference_summary)
    pairs and return the AVERAGE ROUGE-1 / ROUGE-2 / ROUGE-L scores
    across the whole dataset.

    Args:
        pairs: list of (candidate_summary, reference_summary) tuples.
    """
    if not pairs:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for candidate, reference in pairs:
        scores = compute_rouge(candidate, reference)
        for key in totals:
            totals[key] += scores[key]

    n = len(pairs)
    return {key: round(value / n, 4) for key, value in totals.items()}


def compare_methods(results_by_method: dict[str, list[tuple[str, str]]]) -> dict:
    """
    Compare several summarization methods on the same dataset.

    Args:
        results_by_method: e.g.
            {
                "tfidf":    [(candidate, reference), ...],
                "textrank": [(candidate, reference), ...],
                "bart":     [(candidate, reference), ...],
                "t5":       [(candidate, reference), ...],
            }

    Returns:
        {"tfidf": {...avg rouge...}, "textrank": {...}, ...}
    """
    return {
        method: evaluate_dataset(pairs)
        for method, pairs in results_by_method.items()
    }


if __name__ == "__main__":
    candidate = "AI is changing many industries and raising new questions."
    reference = "Artificial intelligence is transforming industries and raising ethical questions."
    try:
        print(compute_rouge(candidate, reference))
    except ImportError as e:
        print(f"[skipped - missing dependency: {e}]")
