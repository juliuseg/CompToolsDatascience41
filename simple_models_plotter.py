import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# =========================
#   Accuracy Bar Plot
# =========================
def plot_accuracy(results):
    models = [r["model"] for r in results]
    accuracies = [r["accuracy"] for r in results]

    plt.figure(figsize=(7, 4))
    sns.barplot(x=models, y=accuracies)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


# =========================
#   MAE Bar Plot
# =========================
def plot_mae(results):
    models = [r["model"] for r in results]
    maes = [r["mae"] for r in results]

    plt.figure(figsize=(7, 4))
    sns.barplot(x=models, y=maes)
    plt.ylabel("MAE")
    plt.title("Model MAE Comparison")
    plt.tight_layout()
    plt.show()


# =========================
#   Confusion Matrices
# =========================
def plot_confusion_matrices(results):
    labels = ["Neg", "Neu", "Pos"]

    for res in results:
        model_name = res["model"]
        cm = np.array(res["confusion_matrix"])

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix: {model_name}")
        plt.tight_layout()
        plt.show()


# --- Load JSON results ---
script_dir = Path(__file__).parent
json_path = script_dir / "simple_models_results.json"
with open(json_path, "r") as f:
    results = json.load(f)

plot_accuracy(results)
plot_mae(results)
#plot_confusion_matrices(results)
