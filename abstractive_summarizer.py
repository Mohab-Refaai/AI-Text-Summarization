"""
abstractive_summarizer.py
---------------------------
Step 3 of the pipeline: generate a *new* summary that captures the
meaning of the text, using pretrained sequence-to-sequence models.
No training or fine-tuning is performed here.

Two models are supported:
    - BART  (facebook/bart-large-cnn) -> primary abstractive model
    - T5    (t5-base / t5-small)      -> optional comparison model

Both models have a limited input token window (BART ~1024 tokens,
T5-base ~512 tokens). Since user input can be much longer than that,
long text is automatically split into chunks, each chunk is
summarized independently, and the partial summaries are joined.
This is a simple, effective strategy that avoids truncating and
losing the end of long documents.
"""

from __future__ import annotations

from preprocessing import preprocess

# Cache loaded pipelines so repeated calls (e.g. from the Streamlit
# app) don't reload the model from disk every time.
_PIPELINE_CACHE: dict[str, object] = {}


def _get_pipeline(model_name: str):
    if model_name not in _PIPELINE_CACHE:
        from transformers import pipeline
        _PIPELINE_CACHE[model_name] = pipeline(
            "summarization", model=model_name
        )
    return _PIPELINE_CACHE[model_name]


def _chunk_text_by_sentences(
    sentences: list[str], max_words_per_chunk: int = 350
) -> list[str]:
    """
    Group sentences into chunks that stay under a rough word-count
    budget, so each chunk fits comfortably inside the model's token
    limit (word count is a safe, tokenizer-free proxy for token count).
    """
    chunks = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if current and current_words + words_in_sentence > max_words_per_chunk:
            chunks.append(" ".join(current))
            current = [sentence]
            current_words = words_in_sentence
        else:
            current.append(sentence)
            current_words += words_in_sentence

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [""]


def _summarize_long_text(
    text: str,
    model_name: str,
    max_length: int,
    min_length: int,
    max_words_per_chunk: int,
) -> str:
    sentences = preprocess(text)
    if not sentences:
        return ""

    chunks = _chunk_text_by_sentences(sentences, max_words_per_chunk)
    summarizer = _get_pipeline(model_name)

    partial_summaries = []
    for chunk in chunks:
        chunk_word_count = len(chunk.split())
        # Don't ask for a summary longer than the chunk itself
        effective_max = min(max_length, max(20, chunk_word_count // 2))
        effective_min = min(min_length, max(10, effective_max - 10))
        result = summarizer(
            chunk,
            max_length=effective_max,
            min_length=effective_min,
            do_sample=False,
        )
        partial_summaries.append(result[0]["summary_text"].strip())

    combined = " ".join(partial_summaries)

    # If we had to summarize multiple chunks, do one final pass to
    # tighten the combined result into a single coherent summary.
    if len(chunks) > 1:
        combined_word_count = len(combined.split())
        if combined_word_count > max_length:
            final = summarizer(
                combined,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
            combined = final[0]["summary_text"].strip()

    return combined


def bart_summarize(
    text: str,
    max_length: int = 130,
    min_length: int = 30,
    model_name: str = "facebook/bart-large-cnn",
) -> str:
    """Abstractive summary using pretrained BART (fine-tuned on CNN/DailyMail)."""
    return _summarize_long_text(
        text,
        model_name=model_name,
        max_length=max_length,
        min_length=min_length,
        max_words_per_chunk=700,  # BART handles ~1024 tokens comfortably
    )


def t5_summarize(
    text: str,
    max_length: int = 130,
    min_length: int = 30,
    model_name: str = "t5-base",
) -> str:
    """
    Abstractive summary using pretrained T5.

    Note: T5 expects a task prefix ("summarize: ") before the input,
    which the transformers 'summarization' pipeline for T5 checkpoints
    handles automatically.
    """
    return _summarize_long_text(
        text,
        model_name=model_name,
        max_length=max_length,
        min_length=min_length,
        max_words_per_chunk=350,  # T5-base handles ~512 tokens
    )


if __name__ == "__main__":
    sample = (
        "Artificial intelligence is transforming many industries today. "
        "Machine learning models can now understand language, images, "
        "and audio at a very high level. Companies are investing heavily "
        "in AI research and deployment."
    )
    try:
        print("BART summary:")
        print(bart_summarize(sample, max_length=40, min_length=10))
    except Exception as e:
        print(f"[skipped - {e}]")
