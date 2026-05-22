import os
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALPHABET = "أ ب ج د ﻩ و ز ح ط ي ك ل م ن س ع ف ص ق ر ش ت ث خ ذ ض ظ غ".split()
RECOGNIZED_TEXT = "حبيبتي"

LAB5_SYMBOLS_DIR = os.path.join(BASE_DIR, "..", "lab5", "symbols")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FONT_SIZE = 52
FONT_SIZE_EXP = 44

EXPERIMENTS = [
    {
        "label": f"Основной шрифт {FONT_SIZE}pt",
        "input": os.path.join(BASE_DIR, "text_52.bmp"),
        "image_out": os.path.join(OUTPUT_DIR, "text_52pt.bmp"),
        "hypotheses_out": os.path.join(OUTPUT_DIR, "hypotheses_main.txt"),
        "heatmap_out": os.path.join(OUTPUT_DIR, "heatmap_main.png"),
    },
    {
        "label": f"Эксперимент: {FONT_SIZE_EXP}pt, база {FONT_SIZE}pt",
        "input": os.path.join(BASE_DIR, "text_44.bmp"),
        "image_out": os.path.join(OUTPUT_DIR, "text_44pt.bmp"),
        "hypotheses_out": os.path.join(OUTPUT_DIR, "hypotheses_experiment.txt"),
        "heatmap_out": os.path.join(OUTPUT_DIR, "heatmap_experiment.png"),
    },
]


def spaced(text):
    return " ".join(text)


def crop_binary(binary):
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return binary
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return binary[y0:y1 + 1, x0:x1 + 1]


def normalize_profile(profile, length=32):
    profile = profile.astype(float)
    if profile.max() > 0:
        profile /= profile.max()
    return cv2.resize(profile.reshape(1, -1), (length, 1), interpolation=cv2.INTER_AREA).ravel()


def extract_features(binary):
    h, w = binary.shape
    total = float(binary.sum())
    if total == 0:
        return np.zeros(69)

    y_idx, x_idx = np.indices(binary.shape)
    cx = (x_idx * binary).sum() / total / w
    cy = (y_idx * binary).sum() / total / h
    ix = ((y_idx - cy * h) ** 2 * binary).sum() / (h ** 2 * total)
    iy = ((x_idx - cx * w) ** 2 * binary).sum() / (w ** 2 * total)
    mass = total / (h * w)

    profile_x = normalize_profile(binary.sum(axis=0))
    profile_y = normalize_profile(binary.sum(axis=1))

    return np.concatenate(([mass, cx, cy, ix, iy], profile_x, profile_y))


def similarity(v1, v2):
    dist = np.linalg.norm(v1 - v2) / np.sqrt(len(v1))
    return 1.0 / (1.0 + dist)


def preprocess(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {path}")
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
    return gray, crop_binary(binary)


def segment_symbols(binary):
    profile_x = binary.sum(axis=0)
    segments = []
    in_symbol = False

    for i, val in enumerate(profile_x):
        if val > 0 and not in_symbol:
            start = i
            in_symbol = True
        elif val == 0 and in_symbol:
            segments.append((start, i))
            in_symbol = False

    if in_symbol:
        segments.append((start, len(profile_x)))

    return segments


def extract_symbols_from_image(path):
    _, binary = preprocess(path)
    symbols = []

    for x1, x2 in segment_symbols(binary):
        symbol = crop_binary(binary[:, x1:x2])
        if symbol.size > 0:
            symbols.append(symbol)

    return list(reversed(symbols))


def build_alphabet_db():
    db = {}
    for char in ALPHABET:
        path = os.path.join(LAB5_SYMBOLS_DIR, f"{char}.png")
        _, binary = preprocess(path)
        db[char] = extract_features(binary)
    return db


def classify(symbols, db):
    results = []
    for symbol in symbols:
        fv = extract_features(symbol)
        hyps = [(char, round(similarity(fv, ref), 4)) for char, ref in db.items()]
        hyps.sort(key=lambda x: x[1], reverse=True)
        results.append(hyps)
    return results


def print_and_save_results(results, true_text, file_path, label=""):
    recognized = "".join(hyps[0][0] for hyps in results)
    errors = sum(1 for a, b in zip(recognized, true_text) if a != b)
    errors += abs(len(recognized) - len(true_text))
    correct = max(0, len(true_text) - errors)
    accuracy = correct / len(true_text) * 100 if true_text else 0.0

    print("\n")
    if label:
        print(f"  {label}")
    print(f"\n{'№':>3}  {'Ист':^5}  {'Лучш':^7}  {'Топ-5 гипотез'}")
    print("-" * 70)

    for i, hyps in enumerate(results):
        true_ch = true_text[i] if i < len(true_text) else "?"
        best = hyps[0][0]
        mark = "OK" if best == true_ch else "!!"
        top5 = ", ".join(f"({c},{s:.3f})" for c, s in hyps[:5])
        print(f"{i + 1:>3}  {true_ch:^5}  [{mark}] {best:^3}  {top5}")

    print()
    print(f"  Лучшие гипотезы : «{spaced(recognized)}»")
    print(f"  Эталонный текст : «{spaced(true_text)}»")
    print(f"  Ошибок          : {errors}")
    print(f"  Верно           : {correct}/{len(true_text)} = {accuracy:.1f}%")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Эталонный текст: «{spaced(true_text)}»\n")
        file.write("Гипотезы:\n\n")
        for i, hyps in enumerate(results):
            hyp_str = ", ".join(f'("{c}", {s:.4f})' for c, s in hyps)
            file.write(f"{i + 1}: [{hyp_str}]\n")
        file.write(f"Лучшие гипотезы: «{spaced(recognized)}»\n")
        file.write(f"Эталонный текст: «{spaced(true_text)}»\n")
        file.write(f"Ошибок: {errors}\n")
        file.write(f"Верно распознано: {correct}/{len(true_text)} = {accuracy:.1f}%\n")

    return recognized, errors, accuracy


def plot_heatmap(results, true_text, out_path, title=""):
    n = min(len(results), len(true_text))
    alpha_idx = {char: i for i, char in enumerate(ALPHABET)}
    mat = np.zeros((n, len(ALPHABET)))

    for i, hyps in enumerate(results[:n]):
        for char, score in hyps:
            mat[i, alpha_idx[char]] = score

    fig, ax = plt.subplots(figsize=(max(12, len(ALPHABET) * 0.45), max(4, n * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(ALPHABET)))
    ax.set_xticklabels(ALPHABET, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{i + 1}: {true_text[i]}" for i in range(n)], fontsize=9)
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
        colors = [
            "#2ecc71" if i < len(results) and results[i][0][0] == true_text[i] else "#e74c3c"
            for i in idx
        ]
        return sims, colors

    sm, cm = get_data(res_main)
    se, ce = get_data(res_exp)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, n * 0.6), 9))
    labels = [
        f"Основной шрифт ({FONT_SIZE}pt)",
        f"Эксперимент ({FONT_SIZE_EXP}pt, база {FONT_SIZE}pt)",
    ]

    for ax, sims, colors, label in [
        (axes[0], sm, cm, labels[0]),
        (axes[1], se, ce, labels[1]),
    ]:
        bars = ax.bar(idx, sims, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xticks(list(idx))
        ax.set_xticklabels(list(true_text), fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Мера близости")
        ax.set_title(f"{label}  (зелёный - верно, красный - ошибка)")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        for bar, score in zip(bars, sims):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.01,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    db = build_alphabet_db()
    all_results = []

    print(f"    Текст: «{spaced(RECOGNIZED_TEXT)}»")

    for experiment in EXPERIMENTS:
        shutil.copyfile(experiment["input"], experiment["image_out"])
        symbols = extract_symbols_from_image(experiment["input"])
        print(f"\n    Изображение: {experiment['input']}")
        print(f"    Обнаружено символов: {len(symbols)}")

        results = classify(symbols, db)
        recognized, errors, accuracy = print_and_save_results(
            results,
            RECOGNIZED_TEXT,
            experiment["hypotheses_out"],
            label=experiment["label"],
        )

        plot_heatmap(
            results,
            RECOGNIZED_TEXT,
            experiment["heatmap_out"],
            f"Тепловая карта мер близости ({experiment['label']})",
        )

        all_results.append((symbols, results, recognized, errors, accuracy))

    plot_comparison(
        all_results[0][1],
        all_results[1][1],
        RECOGNIZED_TEXT,
        os.path.join(OUTPUT_DIR, "comparison.png"),
    )

    print("\n  Итоги:")
    for experiment, (_, _, recognized, errors, accuracy) in zip(EXPERIMENTS, all_results):
        print(f"  {experiment['label']}:")
        print(f"    Распознано: «{spaced(recognized)}»")
        print(f"    Ошибок: {errors}   Точность: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
