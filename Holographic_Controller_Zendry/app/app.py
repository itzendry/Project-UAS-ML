from pathlib import Path
import sys

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_preprocessing import describe_dataset
from src.feature_extraction import feature_columns

DATA_PATH = ROOT / "data" / "raw" / "gestures.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"

st.set_page_config(page_title="Holographic Controller ML", layout="wide")
st.title("Holographic Controller ML")
st.caption("Capstone machine learning untuk klasifikasi gesture tangan kanan berbasis MediaPipe landmarks.")

tabs = st.tabs(["Dataset & EDA", "Model Demo", "Evaluasi Model", "Dokumentasi"])

with tabs[0]:
    st.subheader("Dataset Gesture")
    if not DATA_PATH.exists():
        st.warning("Dataset belum tersedia. Rekam dataset dengan `python -m src.collect_dataset`.")
    else:
        data = pd.read_csv(DATA_PATH)
        summary = describe_dataset(data)
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Sampel", summary["rows"])
        col2.metric("Jumlah Fitur", summary["features"])
        col3.metric("Duplikat", summary["duplicates"])

        st.dataframe(data.head(20), use_container_width=True)
        label_counts = data["label"].value_counts().reset_index()
        label_counts.columns = ["label", "count"]
        st.plotly_chart(px.bar(label_counts, x="label", y="count", title="Distribusi Label"), use_container_width=True)

        missing = pd.Series(summary["missing_values"]).sort_values(ascending=False).head(10).reset_index()
        missing.columns = ["column", "missing_count"]
        st.plotly_chart(px.bar(missing, x="column", y="missing_count", title="Top Missing Values"), use_container_width=True)

with tabs[1]:
    st.subheader("Demo Prediksi Gesture")
    if not MODEL_PATH.exists():
        st.warning("Model belum tersedia. Jalankan `python -m src.train_model` setelah dataset direkam.")
    else:
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
        scaler = artifact["scaler"]
        label_encoder = artifact["label_encoder"]
        st.info(f"Model aktif: {artifact['model_name']}")

        uploaded = st.file_uploader("Unggah CSV landmark dengan kolom fitur x0-y20-z20", type=["csv"])
        if uploaded is not None:
            input_data = pd.read_csv(uploaded)
            missing_columns = set(feature_columns()).difference(input_data.columns)
            if missing_columns:
                st.error(f"Kolom kurang: {sorted(missing_columns)[:10]}")
            else:
                predictions = model.predict(scaler.transform(input_data[feature_columns()]))
                input_data["prediction"] = label_encoder.inverse_transform(predictions)
                st.dataframe(input_data, use_container_width=True)

with tabs[2]:
    st.subheader("Evaluasi Model")
    if COMPARISON_PATH.exists():
        comparison = pd.read_csv(COMPARISON_PATH)
        st.dataframe(comparison, use_container_width=True)
        st.plotly_chart(px.bar(comparison, x="model", y="f1_macro", title="Perbandingan F1-Macro Model"), use_container_width=True)
    else:
        st.warning("Tabel perbandingan model belum ada. Jalankan training terlebih dahulu.")

    report_path = ROOT / "reports" / "evaluation" / "classification_report.csv"
    cm_path = ROOT / "reports" / "evaluation" / "confusion_matrix.png"
    if report_path.exists():
        st.dataframe(pd.read_csv(report_path), use_container_width=True)
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix")

with tabs[3]:
    st.subheader("Dokumentasi Singkat")
    st.markdown(
        """
        **Problem statement:** membangun sistem kontrol game berbasis kamera yang mengenali gesture tangan kanan
        untuk cursor, klik, arrow kiri, arrow kanan, dan idle.

        **Dataset:** data dikumpulkan mandiri dari webcam menggunakan MediaPipe. Setiap sampel berisi 63 fitur
        numerik dari 21 landmark tangan kanan yang dinormalisasi terhadap pergelangan tangan.

        **Metodologi:** perekaman dataset, pembersihan duplikat/missing values, scaling fitur,
        train-validation-test split, training beberapa model, tuning Grid Search, evaluasi, lalu deployment.

        **Cara pakai:** rekam data, latih model, evaluasi model, lalu jalankan aplikasi Streamlit.
        """
    )
