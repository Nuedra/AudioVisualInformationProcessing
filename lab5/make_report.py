from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "features.csv"
REPORT_PATH = BASE_DIR / "Report.md"


def num(row, key):
    return float(row[key])


def section(row, index):
    char = row["Символ"]

    return f"""### Символ {index} — {char}

**Изображение символа:**

<img src="./symbols/{char}.png" alt="{char}" height="70" style="border:1px solid #777; padding:6px; background:#fff;">

**Скалярные характеристики:**

| Признак | Значение |
|---|---:|
| Вес 1-й четверти | {row["Верхняя левая четверть"]} |
| Вес 2-й четверти | {row["Верхняя правая четверть"]} |
| Вес 3-й четверти | {row["Нижняя левая четверть"]} |
| Вес 4-й четверти | {row["Нижняя правая четверть"]} |
| Удельный вес 1-й четверти | {num(row, "Удельный вес верхней левой четверти"):.6f} |
| Удельный вес 2-й четверти | {num(row, "Удельный вес верхней правой четверти"):.6f} |
| Удельный вес 3-й четверти | {num(row, "Удельный вес нижней левой четверти"):.6f} |
| Удельный вес 4-й четверти | {num(row, "Удельный вес нижней правой четверти"):.6f} |
| Координата центра тяжести X | {num(row, "Центр тяжести x"):.6f} |
| Координата центра тяжести Y | {num(row, "Центр тяжести y"):.6f} |
| Нормированная координата X | {num(row, "Нормированный центр тяжести x"):.6f} |
| Нормированная координата Y | {num(row, "Нормированный центр тяжести y"):.6f} |
| Осевой момент инерции Ix | {num(row, "Момент инерции x"):.6f} |
| Осевой момент инерции Iy | {num(row, "Момент инерции y"):.6f} |
| Нормированный момент инерции Ix | {num(row, "Нормированный момент инерции x"):.6f} |
| Нормированный момент инерции Iy | {num(row, "Нормированный момент инерции y"):.6f} |

**Профиль X:**

<img src="./profiles/{char}_X.png" alt="{char} profile X" width="700">

**Профиль Y:**

<img src="./profiles/{char}_Y.png" alt="{char} profile Y" width="700">

---

"""


def main():
    with FEATURES_PATH.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file, delimiter=";"))

    report = """# Лабораторная работа №5

## Выделение признаков символов

**Вариант:** 1  
**Алфавит:** арабский  
**Шрифт:** Times New Roman  
**Кегль:** 52  

---

## Результаты

"""

    for index, row in enumerate(rows, start=1):
        report += section(row, index)

    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
