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
from prompt_template import build_prompt
from functools import lru_cache


@lru_cache(maxsize=1)
def get_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=200,
        do_sample=False
    )


def generate_answer(context, question=None):
    """Generate an answer.

    Model is loaded lazily on first call to avoid heavy import-time side effects
    (and to prevent circular/import-time Streamlit interactions).
    """
    if question is not None:
        prompt = build_prompt(context, question)
    else:
        prompt = context

    generator = get_generator()
    output = generator(prompt)
    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
        return output[0].get("generated_text") or output[0].get("text")
    return str(output)
