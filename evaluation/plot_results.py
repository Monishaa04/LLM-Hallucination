import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def find_input_path(provided: str | None) -> str | None:
    candidates = [provided] if provided else []
    candidates += [
        os.path.join("evaluation", "results_rag_labeled.csv"),
        os.path.join("evaluation", "results_rag.csv"),
        "results_rag_labeled.csv",
        "results_rag.csv",
    ]

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def detect_trust_column(df: pd.DataFrame) -> str | None:
    for candidate in ("trust", "final_trust", "final_trust_score"):
        if candidate in df.columns:
            return candidate
    return None


def main(path: str | None = None):
    input_path = find_input_path(path)
    if input_path is None:
        print("No results file found. Expected evaluation/results_rag_labeled.csv or results_rag_labeled.csv")
        sys.exit(1)

    df = pd.read_csv(input_path)

    if "correctness" not in df.columns:
        print("Column 'correctness' not found. Run evaluation/label_results.py first.")
        sys.exit(1)

    trust_col = detect_trust_column(df)
    if trust_col is None:
        print("No trust column found ('trust' or 'final_trust').")
        sys.exit(1)

    plots_dir = os.path.join("evaluation", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Average trust per correctness
    grouped = df.groupby("correctness")[trust_col].mean()
    print("\nAverage Trust per Correctness:")
    print(grouped)

    ax = grouped.plot(kind="bar", rot=0)
    ax.set_title("Trust Score vs Correctness")
    ax.set_ylabel("Average Trust Score")
    plt.tight_layout()
    out1 = os.path.join(plots_dir, "avg_trust_by_correctness.png")
    plt.savefig(out1)
    print("Saved:", out1)
    plt.clf()

    # Correctness counts
    counts = df["correctness"].value_counts()
    print("\nCorrectness counts:")
    print(counts)
    ax = counts.plot(kind="bar", rot=0)
    ax.set_title("Correctness Distribution")
    ax.set_ylabel("Count")
    plt.tight_layout()
    out2 = os.path.join(plots_dir, "correctness_counts.png")
    plt.savefig(out2)
    print("Saved:", out2)
    plt.clf()

    # Trust distribution histogram
    ax = df[trust_col].plot(kind="hist", bins=20)
    ax.set_title("Trust Score Distribution")
    ax.set_xlabel("Trust Score")
    plt.tight_layout()
    out3 = os.path.join(plots_dir, "trust_histogram.png")
    plt.savefig(out3)
    print("Saved:", out3)
    plt.clf()

    print("All plots saved to:", plots_dir)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
