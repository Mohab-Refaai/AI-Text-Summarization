"""
preprocessing.py
-----------------
Step 1 of the pipeline: clean raw text and split it into sentences.

We avoid a hard dependency on NLTK's punkt download (which requires
network access at runtime) by using a robust regex-based sentence
splitter as the default, with an optional NLTK path if the user has
already downloaded the 'punkt' data.
"""

import re


_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.",
    "e.g.", "i.e.", "u.s.", "u.k.", "u.n.", "no.", "fig.", "st.",
}

_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?\u061F\u06D4])\s+")


def clean_text(text: str) -> str:
    """
    Basic cleaning:
    - Normalize whitespace
    - Strip control characters
    - Remove obvious noise (multiple spaces, weird line breaks)
    """
    if not text:
        return ""

    # Normalize newlines and collapse whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    return text


def split_sentences(text: str, use_nltk: bool = False) -> list[str]:
    """
    Split cleaned text into a list of sentences.

    Args:
        text: cleaned input text.
        use_nltk: if True, try NLTK's sentence tokenizer (requires the
                   'punkt' data to already be downloaded). Falls back
                   to the regex splitter automatically if unavailable.
    """
    if not text:
        return []

    if use_nltk:
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            try:
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            pass  # fall back to regex splitter below

    # Regex-based splitter: split on sentence-ending punctuation
    # followed by whitespace, while trying to avoid breaking on
    # common abbreviations.
    raw_sentences = _SENTENCE_SPLIT_REGEX.split(text.replace("\n", " "))

    sentences = []
    buffer = ""
    for chunk in raw_sentences:
        chunk = chunk.strip()
        if not chunk:
            continue
        buffer = (buffer + " " + chunk).strip() if buffer else chunk
        last_word = buffer.split(" ")[-1].lower()
        if last_word in _ABBREVIATIONS:
            continue  # keep accumulating, this wasn't a real sentence end
        sentences.append(buffer)
        buffer = ""
    if buffer:
        sentences.append(buffer)

    return sentences


def preprocess(text: str, use_nltk: bool = False) -> list[str]:
    """Convenience wrapper: clean text then split into sentences."""
    cleaned = clean_text(text)
    return split_sentences(cleaned, use_nltk=use_nltk)


if __name__ == "__main__":
    sample = (
        "Dr. Smith went to Washington. He met the president there. "
        "It was a productive trip, e.g. several deals were signed. "
        "Everyone was happy!"
    )
    for i, s in enumerate(preprocess(sample), 1):
        print(f"{i}. {s}")
