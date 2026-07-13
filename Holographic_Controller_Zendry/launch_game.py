import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_GAME_EXE = PROJECT_DIR.parent / "NecroDungeon_5" / "Necro Dungeon.exe"


def build_parser():
    parser = argparse.ArgumentParser(description="Buka Holographic Controller dan game sekaligus.")
    parser.add_argument(
        "--game",
        default=str(DEFAULT_GAME_EXE),
        help="Path ke file .exe game.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Jeda detik sebelum membuka game setelah controller kamera dimulai.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    game_path = Path(args.game)
    controller_path = PROJECT_DIR / "main.py"

    if not controller_path.exists():
        raise FileNotFoundError(f"Controller tidak ditemukan: {controller_path}")
    if not game_path.exists():
        raise FileNotFoundError(f"Game tidak ditemukan: {game_path}")

    print("Membuka controller kamera...")
    controller_process = subprocess.Popen([sys.executable, str(controller_path)], cwd=PROJECT_DIR)

    time.sleep(args.delay)

    print(f"Membuka game: {game_path}")
    game_process = subprocess.Popen([str(game_path)], cwd=game_path.parent)

    print("Launcher aktif. Tutup window controller atau tekan Ctrl+C untuk menghentikan controller.")
    try:
        controller_process.wait()
    except KeyboardInterrupt:
        controller_process.terminate()
        controller_process.wait(timeout=5)

    if game_process.poll() is None:
        print("Game masih berjalan. Tutup game dari window game jika sudah selesai.")


if __name__ == "__main__":
    main()
