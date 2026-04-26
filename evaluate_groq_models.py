import os
import pandas as pd
from groq import Groq

# Load API key from environment or .env (if python-dotenv is installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv not available or failed to load; rely on environment variables
    pass

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not set. Create a .env file with GROQ_API_KEY=your_key or set the environment variable."
    )

client = Groq(api_key=api_key)

# Read benchmark questions (expects columns: question,gold_answer)
bench_path = "benchmarks/benchmarks.csv"
df = pd.read_csv(bench_path)

models = [
    # "llama-3.1-8b-instant",
    # "openai/gpt-oss-20b",
    # "qwen/qwen3-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct"
]

import os
os.makedirs("evaluation", exist_ok=True)

for model in models:
    results = []

    for idx, row in df.iterrows():
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                 "You are an AI assistant for Machine Learning concepts. "
                 "Answer concisely in 3–5 sentences. "
                 "If unsure, explicitly say you do not know."},
                {"role": "user", "content": row["question"]}
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        results.append({
            "id": int(row.get("id", idx)),
            "question": row["question"],
            "gold_answer": row.get("gold_answer") if "gold_answer" in row else row.get("gold"),
            "answer": answer
        })

    safe_model = model.replace('/', '_')
    out_path = os.path.join("evaluation", f"results_{safe_model}.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
