import re

STOPWORDS = set([
    "is", "are", "the", "a", "an", "of", "to", "and", "in", "for", "on", "with"
])

def extract_keywords(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return set([w for w in words if w not in STOPWORDS and len(w) > 3])

def concept_overlap_score(answer: str, chunks: list[str]):
    answer_keywords = extract_keywords(answer)
    context_keywords = set()

    for chunk in chunks:
        context_keywords.update(extract_keywords(chunk))

    if len(answer_keywords) == 0:
        return 0.0

    overlap = answer_keywords.intersection(context_keywords)
    return len(overlap) / len(answer_keywords)
