from lime.lime_text import LimeTextExplainer


def lime_explanation(pipeline, class_names, text_instance, num_features=10):
    """
    Generate a LIME explanation for a given text input and pipeline.

    Parameters:
    - pipeline: sklearn Pipeline with a vectorizer + classifier
    - class_names: list of class names (e.g. ["spam", "not spam"])
    - text_instance: a single string to explain
    - num_features: number of most important features to display

    Returns:
    - LIME explanation object
    """
    # Create LIME explainer
    explainer = LimeTextExplainer(class_names=class_names)

    # Define a predict function that outputs class probabilities
    predict_fn = lambda x: pipeline.predict_proba(x)

    # Generate explanation
    explanation = explainer.explain_instance(
        text_instance,
        predict_fn,
        num_features=num_features
    )

    return explanation

