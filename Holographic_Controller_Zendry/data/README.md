# Data

Dataset project ini adalah dataset pribadi berbasis MediaPipe hand landmarks.

Struktur:

- `raw/gestures.csv`: data mentah hasil perekaman webcam.
- `processed/`: data yang sudah dibersihkan atau ditransformasi.
- `external/`: referensi data eksternal jika nanti dibutuhkan.

Format `gestures.csv`:

```text
label,hand,x0,y0,z0,...,x20,y20,z20
right_cursor,Right,...
right_click,Right,...
right_arrow_left,Right,...
right_arrow_right,Right,...
idle,Right,...
```

Cara merekam dataset:

```powershell
python -m src.collect_dataset
```
