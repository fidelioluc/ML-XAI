from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


def train_val_test_split(
    X, y, test_size=0.15, val_size=0.20, random_state=42, shuffle=True, stratify=None
):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    - X: Features
    - y: Labels
    - test_size: Proportion of the dataset to include in the test split
    - val_size: Proportion of the training set to include in the validation split
    - random_state: Controls the shuffling applied to the data before applying the split

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    # Adjust val_size relative to the remaining data
    val_size_adjusted = round(val_size / (1 - test_size), 2)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_classification_results(results_dict, X_test_dict, y_test_dict, save_path=None):
    """
    Plot classification results including confusion matrices and performance metrics.

    Parameters:
    -----------
    results_dict : dict
        Dictionary with vectorization method names as keys and results dictionaries as values
    X_test_dict : dict
        Dictionary with vectorization method names as keys and test features as values
    y_test_dict : dict
        Dictionary with vectorization method names as keys and test labels as values
    save_path : str, optional
        Path to save the plot. If None, plot is displayed instead
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Classification Results", fontsize=16)

    # Plot confusion matrices
    for idx, (name, results) in enumerate(results_dict.items()):
        # Get predictions for confusion matrix
        y_true = y_test_dict[name]
        y_pred = results["model"].predict(X_test_dict[name])

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, idx])
        axes[0, idx].set_title(f"{name} Confusion Matrix")
        axes[0, idx].set_xlabel("Predicted")
        axes[0, idx].set_ylabel("True")

    # Extract metrics from classification reports
    metrics = {"Accuracy": [], "Macro F1": [], "Method": []}
    for name, results in results_dict.items():
        # Parse classification report
        report_lines = results["test_report"].split("\n")
        accuracy = float(report_lines[-2].split()[-2])
        macro_f1 = float(report_lines[-4].split()[-2])

        metrics["Accuracy"].append(accuracy)
        metrics["Macro F1"].append(macro_f1)
        metrics["Method"].append(name)

    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics)

    # Plot accuracy and F1 scores
    metrics_df.plot(
        x="Method", y=["Accuracy", "Macro F1"], kind="bar", ax=axes[1, 1], width=0.8
    )
    axes[1, 1].set_title("Performance Metrics Comparison")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].grid(True, alpha=0.3)

    # Remove empty subplots
    axes[1, 0].remove()
    axes[1, 2].remove()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    return metrics_df
