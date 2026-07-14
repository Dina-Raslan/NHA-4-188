# evaluation/evaluation.py
import os
import sys
import json
import numpy as np
import torch
from PIL import Image
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

# Make src/ importable (config.py, model.py, transforms.py live there)
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.append(SRC_DIR)

import config
from model import SkinLesionClassifier
from transforms import test_transform

# Define Class Labels (must match config.CLASS_NAMES / ImageFolder order)
CLASS_NAMES = config.CLASS_NAMES

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


# Run the Real Trained Model on the Real Test Set
def get_real_predictions():
    """
    Loads the trained checkpoint (config.BEST_MODEL_PATH) and runs it over
    every image in config.TEST_DIR, returning true/predicted label indices.
    Replaces the old dummy-data placeholder with real inference.
    """
    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.BEST_MODEL_PATH}. "
            f"Train the CV model first (src/trainer.py or trainer_mlflow.py)."
        )
    if not os.path.isdir(config.TEST_DIR):
        raise FileNotFoundError(
            f"Test directory not found at {config.TEST_DIR}. "
            f"Expected an ImageFolder-style structure with one subfolder per class."
        )

    device = config.DEVICE
    model = SkinLesionClassifier()
    state_dict = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    y_true, y_pred = [], []
    total = 0
    with torch.no_grad():
        for true_class in sorted(os.listdir(config.TEST_DIR)):
            class_folder = os.path.join(config.TEST_DIR, true_class)
            if not os.path.isdir(class_folder) or true_class not in class_to_idx:
                continue
            true_idx = class_to_idx[true_class]

            for filename in os.listdir(class_folder):
                image_path = os.path.join(class_folder, filename)
                try:
                    img = Image.open(image_path).convert("RGB")
                    input_tensor = test_transform(img).unsqueeze(0).to(device)
                    logits = model(input_tensor)
                    pred_idx = logits.argmax(dim=1).item()
                except Exception as e:
                    print(f"Skipped {image_path}: {e}")
                    continue

                y_true.append(true_idx)
                y_pred.append(pred_idx)
                total += 1
                if total % 200 == 0:
                    print(f"  Evaluated {total} images so far...")

    print(f"Finished evaluating {total} real test images.")
    return np.array(y_true), np.array(y_pred)

# Main Execution: Evaluation Pipeline
if __name__ == "__main__":

    # Run the real trained model on the real test set (no more dummy data)
    y_true, y_pred = get_real_predictions()

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