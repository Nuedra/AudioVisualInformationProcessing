import os
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
FONT_SIZE = 52
FONT_SIZE_EXP = 44

ALPHABET = "أ ب ج د ﻩ و ز ح ط ي ك ل م ن س ع ف ص ق ر ش ت ث خ ذ ض ظ غ".split()
RECOGNIZED_TEXT = "حبيبتي"
BASE_DIR = Path(__file__).resolve().parent
INPUT_IMAGE = BASE_DIR / "text.bmp"
LAB5_SYMBOLS_DIR = BASE_DIR.parent / "lab5" / "symbols"

OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def crop_binary(binary):
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return binary
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return binary[y0:y1 + 1, x0:x1 + 1]


def extract_features(binary):
    h, w = binary.shape
    total = float(binary.sum())
    if total == 0:
        return np.zeros(5)
    y_idx, x_idx = np.indices(binary.shape)
    cx = (x_idx * binary).sum() / total / w
    cy = (y_idx * binary).sum() / total / h
    Ix = ((y_idx - cy * h) ** 2 * binary).sum() / (h ** 2 * total)
    Iy = ((x_idx - cx * w) ** 2 * binary).sum() / (w ** 2 * total)
    mass = total / (h * w)
    return np.array([mass, cx, cy, Ix, Iy], dtype=float)


def similarity(v1, v2):
    return 1.0 / (1.0 + np.linalg.norm(v1 - v2))


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


def split_connected_word(binary):
    profile_x = binary.sum(axis=0)
    threshold = max(10, int(binary.shape[0] * 0.27))
    low_profile = (profile_x <= threshold).astype(np.uint8)
    gaps = [
        (start, end)
        for start, end in get_segments(low_profile)
        if end - start >= 3
    ]
    cuts = [
        (start + end) // 2
        for start, end in gaps
        if 0.07 * binary.shape[1] < (start + end) / 2 < 0.93 * binary.shape[1]
    ]
    if len(cuts) < 2:
        return [(0, binary.shape[1])]

    cuts = cuts[:len(RECOGNIZED_TEXT) - 1]
    borders = [0] + cuts + [binary.shape[1]]
    return list(zip(borders, borders[1:]))


def segment_symbols(binary):
    profile_x = binary.sum(axis=0)
    segments = get_segments(profile_x)
    if len(segments) == 1:
        return split_connected_word(binary)
    return segments


def extract_symbols_from_image(gray):
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
    binary = crop_binary(binary)
    segs = segment_symbols(binary)
    symbols = []
    for x1, x2 in segs:
        sym = crop_binary(binary[:, x1:x2])
        if sym.size > 0:
            symbols.append(sym)
    return list(reversed(symbols))


def build_alphabet_db():
    db = {}
    for char in ALPHABET:
        path = LAB5_SYMBOLS_DIR / f"{char}.png"
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Не удалось открыть изображение: {path}")
        _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
        db[char] = extract_features(crop_binary(binary))
    return db


def generate_text_image(text, font_size):
    font = ImageFont.truetype(FONT_PATH, font_size)
    canvas = Image.new("L", (1000, 200), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), text, font=font, fill=0)
    arr = np.array(canvas)
    _, binary = cv2.threshold(arr, 127, 1, cv2.THRESH_BINARY_INV)
    cropped = crop_binary(binary)
    out = 255 - cropped * 255
    out_path = OUTPUT_DIR / f"text_{font_size}pt.bmp"
    cv2.imwrite(str(out_path), out)
    return out, out_path


def classify(symbols, db):
    results = []
    for sym in symbols:
        fv = extract_features(sym)
        hyps = [(char, round(similarity(fv, ref), 4)) for char, ref in db.items()]
        hyps.sort(key=lambda x: x[1], reverse=True)
        results.append(hyps)
    return results


def print_and_save_results(results, true_text, file_path, label=""):
    recognized = "".join(hyps[0][0] for hyps in results)
    errors = sum(1 for a, b in zip(recognized, true_text) if a != b)
    errors += abs(len(recognized) - len(true_text))
    correct = len(true_text) - errors
    accuracy = correct / len(true_text) * 100 if true_text else 0.0

    print(f"\n{label}")
    print(f"Лучшие гипотезы: «{recognized}»")
    print(f"Эталонный текст: «{true_text}»")
    print(f"Ошибок: {errors}")
    print(f"Верно: {correct}/{len(true_text)} = {accuracy:.1f}%")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Эталонный текст: «{true_text}»\n")
        f.write("Гипотезы:\n\n")
        for i, hyps in enumerate(results):
            hyp_str = ", ".join(f'("{c}", {s:.4f})' for c, s in hyps)
            f.write(f"{i+1}: [{hyp_str}]\n")
        f.write(f"Лучшие гипотезы: «{recognized}»\n")
        f.write(f"Эталонный текст: «{true_text}»\n")
        f.write(f"Ошибок: {errors}\n")
        f.write(f"Верно распознано: {correct}/{len(true_text)} = {accuracy:.1f}%\n")

    return recognized, errors, accuracy


def plot_heatmap(results, true_text, out_path, title=""):
    n = min(len(results), len(true_text))
    alpha_idx = {c: i for i, c in enumerate(ALPHABET)}
    mat = np.zeros((n, len(ALPHABET)))
    for i, hyps in enumerate(results[:n]):
        for c, s in hyps:
            mat[i, alpha_idx[c]] = s

    fig, ax = plt.subplots(figsize=(max(12, len(ALPHABET) * 0.45), max(4, n * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(ALPHABET)))
    ax.set_xticklabels(ALPHABET, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{i+1}: {true_text[i]}" for i in range(n)], fontsize=9)
    ax.set_title(title or "Тепловая карта мер близости")
    ax.set_xlabel("Алфавит")
    ax.set_ylabel("Символы распознаваемого текста")
    plt.colorbar(im, ax=ax, label="Мера близости")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_comparison(res_main, res_exp, true_text, out_path):
    n = len(true_text)
    idx = range(n)

    def get_data(results):
        sims = [results[i][0][1] if i < len(results) else 0 for i in idx]
        cols = [
            "#2ecc71" if i < len(results) and results[i][0][0] == true_text[i] else "#e74c3c"
            for i in idx
        ]
        return sims, cols

    sm, cm = get_data(res_main)
    se, ce = get_data(res_exp)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, n * 0.7), 8))
    for ax, sims, cols, label in [
        (axes[0], sm, cm, f"Основной шрифт ({FONT_SIZE}pt)"),
        (axes[1], se, ce, f"Эксперимент ({FONT_SIZE_EXP}pt, база {FONT_SIZE}pt)"),
    ]:
        ax.bar(idx, sims, color=cols, alpha=0.85)
        ax.set_xticks(list(idx))
        ax.set_xticklabels(list(true_text), fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Мера близости")
        ax.set_title(label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    db = build_alphabet_db()

    shutil.copy(INPUT_IMAGE, OUTPUT_DIR / f"text_{FONT_SIZE}pt.bmp")
    gray_main = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_GRAYSCALE)
    if gray_main is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {INPUT_IMAGE}")
    symbols_main = extract_symbols_from_image(gray_main)
    results_main = classify(symbols_main, db)
    rec_main, err_main, acc_main = print_and_save_results(
        results_main,
        RECOGNIZED_TEXT,
        OUTPUT_DIR / "hypotheses_main.txt",
        label=f"Основной шрифт {FONT_SIZE}pt",
    )
    plot_heatmap(
        results_main,
        RECOGNIZED_TEXT,
        OUTPUT_DIR / "heatmap_main.png",
        f"Тепловая карта мер близости (шрифт {FONT_SIZE}pt)",
    )

    gray_exp, _ = generate_text_image(RECOGNIZED_TEXT, FONT_SIZE_EXP)
    symbols_exp = extract_symbols_from_image(gray_exp)
    results_exp = classify(symbols_exp, db)
    rec_exp, err_exp, acc_exp = print_and_save_results(
        results_exp,
        RECOGNIZED_TEXT,
        OUTPUT_DIR / "hypotheses_experiment.txt",
        label=f"Эксперимент: {FONT_SIZE_EXP}pt, база {FONT_SIZE}pt",
    )
    plot_heatmap(
        results_exp,
        RECOGNIZED_TEXT,
        OUTPUT_DIR / "heatmap_experiment.png",
        f"Тепловая карта мер близости ({FONT_SIZE_EXP}pt -> база {FONT_SIZE}pt)",
    )
    plot_comparison(results_main, results_exp, RECOGNIZED_TEXT, OUTPUT_DIR / "comparison.png")

    print("\nИтоги:")
    print(f"Основной шрифт: «{rec_main}», ошибок: {err_main}, точность: {acc_main:.1f}%")
    print(f"Эксперимент: «{rec_exp}», ошибок: {err_exp}, точность: {acc_exp:.1f}%")


if __name__ == "__main__":
    main()
