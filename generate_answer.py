# from transformers import pipeline
# from prompt_template import build_prompt

# # Load local LLM
# generator = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-large",
#     max_length=300,
#     do_sample=False
# )

# def generate_answer(context, question):
#     prompt = build_prompt(context, question)
#     output = generator(prompt)
#     return output[0]["generated_text"]

from transformers import pipeline
import streamlit as st
from prompt_template import build_prompt


@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=200,
        do_sample=False
    )


generator = load_model()


def generate_answer(context, question=None):
    """Generate an answer.

    If `question` is provided, `context` is treated as retrieval context
    and the prompt is built via `build_prompt(context, question)`.
    If `question` is None, `context` is treated as a raw prompt string.
    """
    if question is not None:
        prompt = build_prompt(context, question)
    else:
        prompt = context

    output = generator(prompt)
    # generator may return a list of dicts or list-like; handle common shapes
    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
        return output[0].get("generated_text") or output[0].get("text")
    # fallback: stringify
    return str(output)
