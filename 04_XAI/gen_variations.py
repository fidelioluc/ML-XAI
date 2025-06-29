import requests

def generate_variations_with_llm(text, num_variants=5, model="llama3.2"):
    prompt = (
        f"Please generate {num_variants} meaningful variations of the following text. "
        f"Each variation should rephrase parts of the text while keeping its meaning similar. "
        f"Do not just reword every sentence — introduce small, realistic changes. \n\n"
        f"The format of the ouput should just be variation1: ... variation2:    nothing else "
        f"Text:\n{text}"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }

    )

    return response.json().get("response", "[No response]")


import re
import pandas as pd


def parse_ollama_variations(text_response, base_text_id=0):
    """
    Parses a multi-variation LLM response into a list of dictionaries.
    """
    # Split by Variation N: and extract clean blocks
    matches = re.split(r"Variation\s*(\d+):\s*", text_response.strip())


    variants = []
    for i in range(1, len(matches), 2):
        variant_text = matches[i + 1].strip()

        variants.append({
            "base_text_id": base_text_id,
            #"variant_index": idx - 1,  # start from 0
            "text": variant_text,
            "model_prediction": None,
            "model_confidence": None,
            "llm_prediction": None,
            "llm_confidence": None
        })

    return pd.DataFrame(variants)