import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
INPUT_IMAGE = BASE_DIR / "text.bmp"
OUTPUT_DIR = BASE_DIR / "symbols"
PROFILE_DIR = BASE_DIR / "profiles"
ALPHABET_PROFILE_DIR = BASE_DIR / "alphabet_profiles"
LAB5_SYMBOLS_DIR = BASE_DIR.parent / "lab5" / "symbols"

# Вариант 1: арабский алфавит из ЛР5.
ALPHABET = "أ ب ج د ﻩ و ز ح ط ي ك ل م ن س ع ف ص ق ر ش ت ث خ ذ ض ظ غ".split()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)
os.makedirs(ALPHABET_PROFILE_DIR, exist_ok=True)

def clear_pngs(path):
    path = Path(path)
    for name in os.listdir(path):
        if name.endswith(".png"):
            os.remove(path / name)

def preprocess(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {path}")
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
    return img, binary

def crop_binary(binary):
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return binary
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return binary[y_min:y_max + 1, x_min:x_max + 1]

def get_profiles(img):
    profile_x = img.sum(axis=0)
    profile_y = img.sum(axis=1)
    return profile_x, profile_y

def save_profile(profile, name, axis, profile_dir=PROFILE_DIR):
    plt.figure()
    if axis == "X":
        plt.bar(range(len(profile)), profile)
        plt.xlabel("X")
        plt.ylabel("Сумма черных пикселей")
    else:
        plt.barh(range(len(profile)), profile)
        plt.gca().invert_yaxis()
        plt.ylabel("Y")
        plt.xlabel("Сумма черных пикселей")

    plt.title(f"{name} profile {axis}")
    plt.savefig(Path(profile_dir) / f"{name}_{axis}.png")
    plt.close()

def get_segments(profile, threshold=0):
    segments = []
    in_segment = False
    for i, val in enumerate(profile):
        if val > threshold and not in_segment:
            start = i
            in_segment = True
        elif val <= threshold and in_segment:
            segments.append((start, i))
            in_segment = False
    if in_segment:
        segments.append((start, len(profile)))
    return segments

def split_segment(profile_x, start, end):
    if end - start < 90:
        return [(start, end)]

    low_profile = (profile_x[start:end] <= 25).astype(np.uint8)
    gaps = [
        (gap_start + start, gap_end + start)
        for gap_start, gap_end in get_segments(low_profile)
        if gap_end - gap_start >= 20
    ]
    gaps = [
        (gap_start, gap_end)
        for gap_start, gap_end in gaps
        if gap_start - start >= 8 and end - gap_end >= 8
    ]
    if not gaps:
        return [(start, end)]

    center = (start + end) / 2
    gap_start, gap_end = min(gaps, key=lambda gap: abs(((gap[0] + gap[1]) / 2) - center))
    cut = (gap_start + gap_end) // 2
    return [(start, cut), (cut, end)]

def segment_symbols(binary):
    profile_x = binary.sum(axis=0)
    segments = []
    for start, end in get_segments(profile_x):
        segments.extend(split_segment(profile_x, start, end))
    return segments

def save_alphabet_profiles():
    for char in ALPHABET:
        path = LAB5_SYMBOLS_DIR / f"{char}.png"
        if path.exists():
            _, binary = preprocess(path)
            px, py = get_profiles(binary)
            save_profile(px, char, "X", ALPHABET_PROFILE_DIR)
            save_profile(py, char, "Y", ALPHABET_PROFILE_DIR)

def main():
    clear_pngs(OUTPUT_DIR)
    clear_pngs(PROFILE_DIR)
    clear_pngs(ALPHABET_PROFILE_DIR)

    _, binary = preprocess(INPUT_IMAGE)
    binary = crop_binary(binary)
    px, py = get_profiles(binary)
    save_profile(px, "text", "X")
    save_profile(py, "text", "Y")

    segments = segment_symbols(binary)
    for i, (x1, x2) in enumerate(segments, start=1):
        symbol = crop_binary(binary[:, x1:x2])
        name = f"symbol{i}"
        cv2.imwrite(str(OUTPUT_DIR / f"{name}.png"), 255 - symbol * 255)
        px_s, py_s = get_profiles(symbol)
        save_profile(px_s, name, "X")
        save_profile(py_s, name, "Y")

    save_alphabet_profiles()
    print("Найдено символов:", len(segments))

if __name__ == "__main__":
    main()
