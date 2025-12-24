from transformers import pipeline
from prompt_template import build_prompt

# Load local LLM
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=200,
    do_sample=False
)

def generate_answer(context, question):
    prompt = build_prompt(context, question)
    output = generator(prompt)
    return output[0]["generated_text"]
