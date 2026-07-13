from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.feature_extraction import feature_columns


def load_dataset(path="data/raw/gestures.csv"):
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset belum ada: {dataset_path}")
    data = pd.read_csv(dataset_path)
    required = {"label", "hand", *feature_columns()}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Kolom dataset belum lengkap: {sorted(missing)}")
    return data


def describe_dataset(data):
    return {
        "rows": int(len(data)),
        "features": int(len(feature_columns())),
        "labels": data["label"].value_counts().to_dict(),
        "missing_values": data.isna().sum().to_dict(),
        "duplicates": int(data.duplicated().sum()),
    }


def prepare_features(data, test_size=0.2, val_size=0.2, random_state=42):
    cleaned = data.drop_duplicates().dropna().copy()
    x = cleaned[feature_columns()]
    y = cleaned["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x_scaled,
        y_encoded,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    relative_test_size = test_size / (test_size + val_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test, scaler, label_encoder
