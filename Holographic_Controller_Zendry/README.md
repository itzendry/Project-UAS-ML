# Holographic Controller ML

Holographic Controller ML adalah capstone project Pembelajaran Mesin untuk membuat sistem kontrol game berbasis kamera. Project ini memakai MediaPipe untuk mengambil 21 landmark tangan, lalu menambahkan pipeline machine learning untuk mengklasifikasikan gesture tangan kanan seperti cursor, klik, arrow kiri, arrow kanan, dan idle. Tangan kiri tetap dipakai sebagai joystick virtual rule-based karena gerakannya bersifat analog.

## Pembuat Game & Machine Learning

- Zendry - A11.2024.15881

## Problem Statement

Kontrol game umumnya membutuhkan keyboard, mouse, atau joystick fisik. Pada beberapa skenario, perangkat fisik kurang fleksibel, kurang menarik untuk demonstrasi interaktif, atau tidak sesuai dengan konsep game berbasis gesture. Project ini menyelesaikan masalah tersebut dengan membuat kontrol game berbasis computer vision yang membaca gerakan tangan dari webcam. Tangan kiri digunakan sebagai analog movement untuk menggerakkan player ke atas, bawah, kiri, kanan, dan diagonal. Tangan kanan digunakan untuk aksi yang lebih diskrit, yaitu mengarahkan cursor dengan telunjuk, klik kiri dengan gesture cubit, arrow kiri dengan dua jari, arrow kanan dengan tiga jari, serta idle ketika tidak ada gesture aksi.

Tujuan analisis adalah membangun dataset gesture tangan kanan dari webcam, melatih model klasifikasi gesture, mengevaluasi beberapa algoritma, lalu menyediakan aplikasi Streamlit untuk menjelaskan dataset, performa model, dan demo prediksi. Metrik kesuksesan utama adalah F1-score macro karena setiap kelas gesture perlu dikenali dengan seimbang, bukan hanya kelas mayoritas. Model dianggap layak jika mampu membedakan gesture utama secara stabil pada data uji dan dapat dipakai sebagai komponen tambahan untuk controller realtime.

## Dataset

Dataset menggunakan data pribadi yang dikumpulkan dari webcam.

- Sumber: rekaman mandiri menggunakan `python -m src.collect_dataset`
- Bentuk data: CSV landmark tangan kanan
- Lokasi data mentah: `data/raw/gestures.csv`
- Jumlah fitur: 63 fitur numerik dari 21 landmark MediaPipe (`x`, `y`, `z`)
- Label target: `right_cursor`, `right_click`, `right_arrow_left`, `right_arrow_right`, `idle`

Dataset tidak disimpan di repository karena dapat berisi data rekaman pribadi dan ukuran dapat bertambah. Formatnya didokumentasikan di `data/README.md`.

## Struktur Repository

```text
Holographic_Controller_Zendry/
├── app/
│   ├── app.py
│   ├── pages/
│   └── assets/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── ai_tangan.py
│   ├── collect_dataset.py
│   ├── controls.py
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Metodologi

1. Akuisisi data: webcam membaca tangan kanan, MediaPipe menghasilkan 21 landmark, lalu data disimpan ke CSV.
2. EDA: analisis jumlah sampel, distribusi label, missing values, duplikat, dan pola fitur landmark.
3. Preprocessing: hapus duplikat/missing values, normalisasi landmark terhadap titik pergelangan tangan, scaling fitur, dan train-validation-test split.
4. Modeling: training minimal dua model. Script saat ini menyiapkan KNN, Random Forest, dan XGBoost.
5. Tuning: Grid Search dengan metrik `f1_macro`.
6. Evaluasi: accuracy, F1-score macro, classification report, dan confusion matrix.
7. Deployment: aplikasi Streamlit untuk dashboard dataset, demo prediksi, evaluasi model, dan dokumentasi.

## Cara Menjalankan

Aktifkan virtual environment dan install dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Rekam dataset gesture:

```powershell
python -m src.collect_dataset
```

Kontrol perekaman:

- `1`: `right_cursor`
- `2`: `right_click`
- `3`: `right_arrow_left`
- `4`: `right_arrow_right`
- `5`: `idle`
- `Space`: simpan sampel
- `Q` atau `Esc`: keluar

Latih model:

```powershell
python -m src.train_model
```

Evaluasi model:

```powershell
python -m src.evaluate_model
```

Jalankan aplikasi Streamlit:

```powershell
streamlit run app/app.py
```

Jalankan controller game realtime:

```powershell
python main.py
```

Jalankan controller dan game sekaligus:

```powershell
python launch_game.py
```

Launcher akan membuka `main.py` untuk kamera/controller, menunggu sebentar, lalu membuka game di `../NecroDungeon_5/Necro Dungeon.exe`.