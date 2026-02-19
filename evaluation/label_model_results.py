import os
import re
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ABSTAIN_PAT = re.compile(
    r"\b(I\s+don'?t\s+know|I\s+do\s+not\s+know|cannot\s+say|can'?t\s+say|not\s+sure|insufficient\s+information)\b",
    re.I
)

def is_abstention_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(ABSTAIN_PAT.search(text))

def semantic_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf = vectorizer.transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def label_df(df):
    correctness = []
    hallucination = []
    abstention = []

    for _, row in df.iterrows():
        gold = row.get("gold_answer") or row.get("gold")
        ans = row.get("answer")

        abst = "yes" if is_abstention_text(ans) else "no"

        if isinstance(gold, str) and isinstance(ans, str):
            sim = semantic_similarity(gold, ans)
        else:
            sim = 0.0

        # Relaxed thresholds
        if sim >= 0.4:
            corr = "correct"
        elif sim >= 0.25:
            corr = "partial"
        else:
            corr = "incorrect"

        hall = "yes" if (corr == "incorrect" and abst == "no") else "no"

        correctness.append(corr)
        hallucination.append(hall)
        abstention.append(abst)

    df["correctness"] = correctness
    df["hallucination"] = hallucination
    df["abstention"] = abstention

    return df

def main():
    files = [
        "evaluation/results/results_meta-llama_llama-4-maverick-17b-128e-instruct.csv",
        "evaluation/results/results_moonshotai_kimi-k2-instruct-0905.csv",
        "evaluation/results/results_qwen_qwen3-32b.csv"
    ]
    if not files:
        print("No result files found.")
        return

    for path in files:
        if path.endswith("_labeled.csv"):
            continue

        print("Labeling:", path)
        df = pd.read_csv(path)
        labeled = label_df(df)
        out = path.replace(".csv", "_labeled.csv")
        labeled.to_csv(out, index=False)
        print("Wrote:", out)

if __name__ == "__main__":
    main()
