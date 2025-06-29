import numpy as np
import requests
import pandas as pd


def classify_with_model(pipeline, target_names, text):
    """Predict class and confidence and return class name using the pipeline."""
    prediction = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    return prediction, target_names[prediction], np.max(proba)

def explain_with_llm(target_labels, doc_text, nr_features=5):
    """Send document and prediction to Ollama for explanation."""
    model = "llama3.2"
    ollama_prompt = (
        f"Please classify the following document into one of the following categories:\n"
        f"{', '.join(target_labels)}\n\n"
        f"Document:\n{doc_text}\n\n"
        f"Instructions:\n"
        f"1. Return the predicted category.\n"
        f"2. Include a confidence score between 0 and 1.\n"
        f"3. List the top {nr_features} most influential words from the text that impacted the classification, ranked from most to least influential.\n\n"
        f"Format your answer as:\n"
        f"Category: <predicted_category>\n"
        f"Confidence: <confidence_score>\n"
        f"Top Words: [word1, word2, ..., word{nr_features}]\n"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": ollama_prompt,
            "stream": False
        }
    )

    return response.json().get("response", "[No response received]")


def compare_text_variants(pipeline, target_labels, text_variants, nr_features):
    """Loop through variants, print results, and collect prediction data."""
    results = []

    for i, variant in enumerate(text_variants):
        print(f"\n=== Variant {i} ===")

        # Model prediction
        class_index, predicted_label, model_confidence = classify_with_model(pipeline, target_labels, variant)
        print(f"[Model Prediction]: {predicted_label}")

        # LLM explanation & prediction
        print("[Ollama LLM Explanation]:")
        explanation = explain_with_llm(target_labels, variant, nr_features)
        print(explanation)

        # Parse LLM explanation if formatted properly
        pred_class, confidence, top_words = None, None, None
        try:
            lines = explanation.splitlines()
            for line in lines:
                if line.lower().startswith("category:"):
                    pred_class = line.split(":", 1)[1].strip()
                elif line.lower().startswith("confidence:"):
                    confidence = float(line.split(":", 1)[1].strip())
                elif line.lower().startswith("top words:"):
                    top_words = line.split(":", 1)[1].strip().strip("[]")
        except Exception as e:
            print(f"[Warning] Could not parse LLM explanation: {e}")

        results.append({
            "variant_index": i,
            "model_prediction": predicted_label,
            "model_confidence": model_confidence,
            "llm_prediction": pred_class,
            "llm_confidence": confidence,
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Optional: compute summary stats
    pred_counts = df_results["llm_prediction"].value_counts()
    avg_conf = df_results["llm_confidence"].mean()
    conf_range = df_results["llm_confidence"].max() - df_results["llm_confidence"].min()

    print("\n=== Summary ===")
    print("Prediction Counts:\n", pred_counts)
    print("Average Confidence:", round(avg_conf, 3))
    print("Confidence Range:", round(conf_range, 3))

    return df_results

def update_predictions(df, pipeline, target_names, nr_features=7):
    """Adds model and LLM predictions to a DataFrame with text variants."""
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=target_names)

    model_preds = []
    model_confs = []
    lime_words = []

    llm_preds = []
    llm_confs = []

    for i, row in df.iterrows():
        text = row["text"]

        # --- Model prediction ---
        class_index = pipeline.predict([text])[0]
        class_name = target_names[class_index]
        confidence = max(pipeline.predict_proba([text])[0])

        # --- LIME explanation (optional, top words) ---
        explanation = explainer.explain_instance(text, pipeline.predict_proba, num_features=nr_features)
        top_words = [w for w, _ in explanation.as_list()]

        # --- LLM prediction via Ollama ---
        try:
            prompt = (
                f"Please classify the following document into one of the following categories:\n"
                f"{', '.join(target_names)}\n\n"
                f"Document:\n{text[:1000]}\n\n"
                f"Instructions:\n"
                f"1. Return the predicted category.\n"
                f"2. Include a confidence score between 0 and 1.\n"
                f"3. List the top {nr_features} most influential words from the text that impacted the classification, ranked from most to least influential.\n\n"
                f"Format your answer as:\n"
                f"Category: <predicted_category>\n"
                f"Confidence: <confidence_score>\n"
                f"Top Words: [word1, word2, ..., word{nr_features}]\n"
            )

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False},
                timeout=30
            )
            result = response.json().get("response", "")

            # Parse result
            pred_line = next((line for line in result.splitlines() if "Category:" in line), None)
            conf_line = next((line for line in result.splitlines() if "Confidence:" in line), None)

            llm_label = pred_line.split(":")[1].strip() if pred_line else None
            llm_conf = float(conf_line.split(":")[1].strip()) if conf_line else None

        except Exception as e:
            llm_label = None
            llm_conf = None
            print(f"[Error parsing Ollama response for row {i}]: {e}")

        # --- Append all results ---
        model_preds.append(class_name)
        model_confs.append(confidence)
        lime_words.append(top_words)
        llm_preds.append(llm_label)
        llm_confs.append(llm_conf)

    # --- Add columns to DataFrame ---
    df["model_prediction"] = model_preds
    df["model_confidence"] = model_confs
    df["lime_top_words"] = lime_words
    df["llm_prediction"] = llm_preds
    df["llm_confidence"] = llm_confs

    return df

from itertools import combinations
from collections import Counter


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 1.0

def compute_robustness_scores(df):
    """Compute robustness scores for each base_text_id."""
    robustness = []

    for base_id, group in df.groupby("base_text_id"):
        model_var = group["model_prediction"].nunique()
        llm_var = group["llm_prediction"].nunique()

        model_conf_range = group["model_confidence"].max() - group["model_confidence"].min()
        llm_conf_range = group["llm_confidence"].max() - group["llm_confidence"].min()

        # LIME keyword variation (pairwise Jaccard distance)
        lime_sets = [set(words) for words in group["lime_top_words"] if isinstance(words, list)]

        if len(lime_sets) > 1:
            jaccard_scores = []
            for a, b in combinations(lime_sets, 2):
                score = jaccard_similarity(a, b)
                jaccard_scores.append(score)
            avg_jaccard = np.mean(jaccard_scores)
            lime_drift = 1 - avg_jaccard  # 1 = max drift
        else:
            lime_drift = 0

        robustness.append({
            "base_text_id": base_id,
            "model_prediction_variability": model_var,
            "llm_prediction_variability": llm_var,
            "model_confidence_range": round(model_conf_range, 3),
            "llm_confidence_range": round(llm_conf_range, 3),
            "lime_keyword_drift": round(lime_drift, 3)
        })

    return pd.DataFrame(robustness)


