from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
SYMBOLS_DIR = BASE_DIR / "symbols"
PROFILES_DIR = BASE_DIR / "profiles"
FEATURES_PATH = BASE_DIR / "features.csv"

FONT_PATH = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
FONT_SIZE = 52

ALPHABET = "أ ب ج د ﻩ و ز ح ط ي ك ل م ن س ع ف ص ق ر ش ت ث خ ذ ض ظ غ".split()

def crop_binary(binary):
    coords = np.column_stack(np.where(binary > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return binary[y_min:y_max + 1, x_min:x_max + 1]


def generate_symbols():
    SYMBOLS_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    for char in ALPHABET:
        image = Image.new("L", (100, 100), color=255)
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = draw.textbbox((0, 0), char, font=font)

        x = (100 - (right - left)) // 2 - left
        y = (100 - (bottom - top)) // 2 - top
        draw.text((x, y), char, font=font, fill=0)

        binary = (np.array(image) < 128).astype(np.uint8)
        cropped = crop_binary(binary)
        result = (255 - cropped * 255).astype(np.uint8)
        Image.fromarray(result, mode="L").save(SYMBOLS_DIR / f"{char}.png")


def preprocess(path):
    image = Image.open(path).convert("L")
    return (np.array(image) < 128).astype(np.uint8)


def extract_features(image):
    h, w = image.shape

    q1 = image[:h // 2, :w // 2]
    q2 = image[:h // 2, w // 2:]
    q3 = image[h // 2:, :w // 2]
    q4 = image[h // 2:, w // 2:]
    quarters = (q1, q2, q3, q4)
    weights = [int(q.sum()) for q in quarters]
    rel_weights = [weight / q.size for weight, q in zip(weights, quarters)]

    y, x = np.indices(image.shape)
    weight = image.sum()

    cx = (x * image).sum() / weight
    cy = (y * image).sum() / weight
    cx_rel = cx / (w - 1) if w > 1 else 0
    cy_rel = cy / (h - 1) if h > 1 else 0

    ix = ((y - cy) ** 2 * image).sum()
    iy = ((x - cx) ** 2 * image).sum()
    ix_rel = ix / (h * w)
    iy_rel = iy / (h * w)

    profile_x = image.sum(axis=0)
    profile_y = image.sum(axis=1)

    return {
        "Верхняя левая четверть": weights[0],
        "Верхняя правая четверть": weights[1],
        "Нижняя левая четверть": weights[2],
        "Нижняя правая четверть": weights[3],
        "Удельный вес верхней левой четверти": rel_weights[0],
        "Удельный вес верхней правой четверти": rel_weights[1],
        "Удельный вес нижней левой четверти": rel_weights[2],
        "Удельный вес нижней правой четверти": rel_weights[3],
        "Центр тяжести x": cx,
        "Центр тяжести y": cy,
        "Нормированный центр тяжести x": cx_rel,
        "Нормированный центр тяжести y": cy_rel,
        "Момент инерции x": ix,
        "Момент инерции y": iy,
        "Нормированный момент инерции x": ix_rel,
        "Нормированный момент инерции y": iy_rel,
        "Профиль x": profile_x,
        "Профиль y": profile_y,
    }


def save_profile(profile, char, axis):
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()

    if axis == "X":
        plt.bar(range(len(profile)), profile)
        plt.xlabel("X")
        plt.ylabel("Сумма пикселей")
    else:
        plt.barh(range(len(profile)), profile)
        plt.gca().invert_yaxis()
        plt.ylabel("Y")
        plt.xlabel("Сумма пикселей")

    plt.title(f"{char} profile {axis}")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(PROFILES_DIR / f"{char}_{axis}.png")
    plt.close()


def save_features(rows):
    headers = [key for key in rows[0].keys()]
    with FEATURES_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main():
    generate_symbols()
    rows = []

    for char in ALPHABET:
        image = preprocess(SYMBOLS_DIR / f"{char}.png")
        features = extract_features(image)

        save_profile(features["Профиль x"], char, "X")
        save_profile(features["Профиль y"], char, "Y")

        row = {"Символ": char}
        row.update({k: v for k, v in features.items() if "Профиль" not in k})
        rows.append(row)

    save_features(rows)


if __name__ == "__main__":
    main()
