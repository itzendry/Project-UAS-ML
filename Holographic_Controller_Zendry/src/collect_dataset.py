import argparse
import csv
from pathlib import Path

import cv2

from src.ai_tangan import HandTracker
from src.feature_extraction import feature_columns, landmarks_to_features


DEFAULT_LABELS = (
    "right_cursor",
    "right_click",
    "right_arrow_left",
    "right_arrow_right",
    "idle",
)


def build_parser():
    parser = argparse.ArgumentParser(description="Rekam dataset landmark tangan dari webcam.")
    parser.add_argument("--output", default="data/raw/gestures.csv")
    parser.add_argument("--label", choices=DEFAULT_LABELS, help="Label gesture yang sedang direkam.")
    parser.add_argument("--camera", type=int, default=0)
    return parser


def ensure_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "hand"] + feature_columns())


def main():
    args = build_parser().parse_args()
    output = Path(args.output)
    ensure_header(output)

    label = args.label or DEFAULT_LABELS[0]
    tracker = HandTracker(max_hands=2)
    cap = cv2.VideoCapture(args.camera)

    print("Kontrol: 1-5 pilih label, SPACE simpan sampel, Q/ESC keluar.")
    print("Label:", ", ".join(f"{idx + 1}={name}" for idx, name in enumerate(DEFAULT_LABELS)))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.process(rgb_frame)

        selected = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                tracker.draw(frame, hand_landmarks)
                if hand_label == "Right":
                    selected = (hand_landmarks, hand_label)

        cv2.putText(frame, f"Label: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "SPACE=simpan | 1-5=ganti label | Q=keluar", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imshow("Dataset Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in [ord(str(i)) for i in range(1, len(DEFAULT_LABELS) + 1)]:
            label = DEFAULT_LABELS[int(chr(key)) - 1]
        if key == ord(" ") and selected is not None:
            hand_landmarks, hand_label = selected
            row = [label, hand_label] + landmarks_to_features(hand_landmarks)
            with output.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(row)
            print(f"Sampel tersimpan: {label}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
