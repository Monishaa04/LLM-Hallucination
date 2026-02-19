def build_prompt(context, question):
    prompt = f"""
You are an AI assistant for Machine Learning concepts.

Answer the question using ONLY the context below.
Give a concise and clear answer in 3–5 sentences.
Provide a complete explanation covering all key aspects mentioned in the context.
Do NOT give examples unless asked.
If the answer is not present in the context, say:
"I do not have sufficient information to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt
