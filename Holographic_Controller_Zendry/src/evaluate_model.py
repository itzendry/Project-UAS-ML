import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluasi model gesture terbaik.")
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--output-dir", default="reports/evaluation")
    return parser


def main():
    args = build_parser().parse_args()
    artifact = joblib.load(args.model)
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]
    x_test = artifact["x_test"]
    y_test = artifact["y_test"]

    y_pred = model.predict(x_test)
    labels = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title("Confusion Matrix Gesture Controller")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    print(f"Hasil evaluasi tersimpan di {output_dir}")


if __name__ == "__main__":
    main()
