# evaluation/evaluation.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Define Class Labels
CLASS_NAMES = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Benign Keratosis",
    "Psoriasis",
    "Seborrheic Keratosis",
    "Tinea Ringworm",
    "Molluscum"
]

# Calculate Classification Metrics
def compute_metrics(y_true, y_pred):
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
    }
    return metrics

# Generate and Save Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Confusion Matrix saved → {save_path}")

# Save Metrics to JSON File
def save_metrics(metrics, save_path):
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f" Metrics saved → {save_path}")

# Main Execution: Evaluation Pipeline
if __name__ == "__main__":

    # Generate Dummy Data for Testing
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.randint(0, 10, n_samples)
    y_pred = np.random.randint(0, 10, n_samples)

    # Calculate Performance Metrics
    metrics = compute_metrics(y_true, y_pred)

    print("\n Evaluation Results:")
    print(f"  Accuracy  : {metrics['accuracy']  * 100:.2f}%")
    print(f"  Precision : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall    : {metrics['recall']    * 100:.2f}%")
    print(f"  F1 Score  : {metrics['f1_score']  * 100:.2f}%")

    # Save Metrics File
    os.makedirs("outputs", exist_ok=True)
    save_metrics(metrics, "outputs/metrics.json")

   # Generate Visualization
    plot_confusion_matrix(y_true, y_pred, "outputs/confusion_matrix.png")
    # ─── MLflow ───
    mlflow.set_experiment("Skin Condition Detection")
    with mlflow.start_run(run_name="Evaluation Run"):
        mlflow.log_metric("accuracy",  metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall",    metrics["recall"])
        mlflow.log_metric("f1_score",  metrics["f1_score"])
        mlflow.log_artifact("outputs/confusion_matrix.png")
        mlflow.log_artifact("outputs/metrics.json")
        print("\n MLflow run logged successfully!")