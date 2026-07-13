import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.data_preprocessing import load_dataset, prepare_features


def build_parser():
    parser = argparse.ArgumentParser(description="Training model klasifikasi gesture tangan.")
    parser.add_argument("--data", default="data/raw/gestures.csv")
    parser.add_argument("--models-dir", default="models")
    return parser


def train_with_grid_search(name, estimator, param_grid, x_train, y_train):
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=1,
        error_score=0.0,
    )
    search.fit(x_train, y_train)
    return name, search.best_estimator_, search.best_params_


def main():
    args = build_parser().parse_args()
    data = load_dataset(args.data)
    x_train, x_val, x_test, y_train, y_val, y_test, scaler, label_encoder = prepare_features(data)

    candidates = [
        (
            "knn",
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        ),
        (
            "random_forest",
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [None, 8, 16]},
        ),
        (
            "xgboost",
            XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=1, tree_method="exact"),
            {"n_estimators": [100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        ),
    ]

    rows = []
    trained = {}
    for name, estimator, grid in candidates:
        model_name, model, params = train_with_grid_search(name, estimator, grid, x_train, y_train)
        y_pred = model.predict(x_val)
        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_val, y_pred),
                "f1_macro": f1_score(y_val, y_pred, average="macro"),
                "best_params": params,
            }
        )
        trained[model_name] = model

    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    best_name = comparison.iloc[0]["model"]
    artifact = {
        "model": trained[best_name],
        "model_name": best_name,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "x_test": x_test,
        "y_test": y_test,
        "comparison": comparison,
    }

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, models_dir / "best_model.joblib")
    comparison.to_csv(models_dir / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print(f"Model terbaik tersimpan: {models_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
