import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


GAMMA = 0.5
C_POWER = 1.0
F0 = 0.0

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "images"
IMAGE_FILES = {
    "regular": IMAGE_DIR / "regular.png",
    "low":     IMAGE_DIR / "low.png",
    "high":    IMAGE_DIR / "high.png",
}

GLCM_LEVELS = 256
GLCM_OFFSETS = [
    (0,  1),
    (-1, 0),
    (0, -1),
    (1,  0),
]


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def rgb_to_hsl(rgb):
    r = rgb[:, :, 0] / 255.0
    g = rgb[:, :, 1] / 255.0
    b = rgb[:, :, 2] / 255.0

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    L = (cmax + cmin) / 2.0

    S = np.where(delta == 0, 0.0, delta / (1.0 - np.abs(2 * L - 1)))
    S = np.clip(S, 0, 1)

    H = np.zeros_like(L)
    mask_r = (cmax == r) & (delta != 0)
    mask_g = (cmax == g) & (delta != 0)
    mask_b = (cmax == b) & (delta != 0)
    H[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    H[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    return H, S, L


def hsl_to_rgb(H, S, L):
    C = (1.0 - np.abs(2 * L - 1)) * S
    X = C * (1.0 - np.abs((H / 60.0) % 2 - 1))
    m = L - C / 2.0

    R1 = np.zeros_like(L)
    G1 = np.zeros_like(L)
    B1 = np.zeros_like(L)

    masks = [
        (H < 60),
        (H >= 60) & (H < 120),
        (H >= 120) & (H < 180),
        (H >= 180) & (H < 240),
        (H >= 240) & (H < 300),
        (H >= 300),
    ]
    vals = [
        (C, X, np.zeros_like(L)),
        (X, C, np.zeros_like(L)),
        (np.zeros_like(L), C, X),
        (np.zeros_like(L), X, C),
        (X, np.zeros_like(L), C),
        (C, np.zeros_like(L), X),
    ]
    for mask, (r, g, b) in zip(masks, vals):
        R1 = np.where(mask, r, R1)
        G1 = np.where(mask, g, G1)
        B1 = np.where(mask, b, B1)

    R = np.clip((R1 + m) * 255, 0, 255).astype(np.uint8)
    G = np.clip((G1 + m) * 255, 0, 255).astype(np.uint8)
    B = np.clip((B1 + m) * 255, 0, 255).astype(np.uint8)

    return np.stack([R, G, B], axis=-1)


def power_transform(L_channel, gamma=GAMMA, c=C_POWER, f0=F0):
    f = np.clip(L_channel + f0, 0, None)
    g = c * np.power(f, gamma)
    g_min, g_max = g.min(), g.max()
    if g_max - g_min > 1e-8:
        g = (g - g_min) / (g_max - g_min)
    return np.clip(g, 0, 1)


def compute_glcm(gray_img, levels=GLCM_LEVELS):
    img = gray_img.astype(np.uint8)
    glcm = np.zeros((levels, levels), dtype=np.float64)

    for dr, dc in GLCM_OFFSETS:
        if dr >= 0:
            r1 = slice(0, img.shape[0] - dr)
            r2 = slice(dr, img.shape[0])
        else:
            r1 = slice(-dr, img.shape[0])
            r2 = slice(0, img.shape[0] + dr)

        if dc >= 0:
            c1 = slice(0, img.shape[1] - dc)
            c2 = slice(dc, img.shape[1])
        else:
            c1 = slice(-dc, img.shape[1])
            c2 = slice(0, img.shape[1] + dc)

        first = img[r1, c1].ravel()
        second = img[r2, c2].ravel()
        np.add.at(glcm, (first, second), 1)

    if glcm.sum() > 0:
        glcm = glcm / glcm.sum()
    return glcm


def glcm_features(glcm):
    return {
        "ASM": np.sum(glcm ** 2),
        "MPR": np.max(glcm),
        "ENT": -np.sum(glcm * np.log2(glcm + 1e-12)),
        "TR":  np.trace(glcm),
    }


def glcm_for_view(glcm):
    if glcm.max() == 0:
        return glcm
    view = np.log1p(glcm / glcm.max() * 255.0)
    view = view / view.max()
    return view


def brightness_histogram(channel_0_255, n_bins=256):
    hist, edges = np.histogram(channel_0_255.ravel(), bins=n_bins, range=(0, 256))
    hist = hist / hist.sum()
    return hist, edges[:-1]


def process_image(name, path):
    print("\n")
    print(f"  Изображение: {name} ")

    rgb = load_image(path)
    H_ch, S_ch, L_ch = rgb_to_hsl(rgb)
    gray = (L_ch * 255).astype(np.uint8)

    glcm_orig = compute_glcm(gray)
    feat_orig = glcm_features(glcm_orig)

    print(f"\n Исходное ")
    for k, v in feat_orig.items():
        print(f"  {k:12s}: {v:.4f}")

    L_contrast = power_transform(L_ch, gamma=GAMMA, c=C_POWER, f0=F0)

    rgb_contrast = hsl_to_rgb(H_ch, S_ch, L_contrast)
    gray_contrast = (L_contrast * 255).astype(np.uint8)

    glcm_contr = compute_glcm(gray_contrast)
    feat_contr = glcm_features(glcm_contr)

    print(f"\n После контрастирования (γ={GAMMA})")
    for k, v in feat_contr.items():
        print(f"  {k:12s}: {v:.4f}")

    hist_br_orig,  edges_orig  = brightness_histogram(gray)
    hist_br_contr, edges_contr = brightness_histogram(gray_contrast)

    return {
        "name":           name,
        "rgb_orig":       rgb,
        "rgb_contrast":   rgb_contrast,
        "gray_orig":      gray,
        "gray_contrast":  gray_contrast,
        "glcm_orig":      glcm_orig,
        "glcm_contr":     glcm_contr,
        "feat_orig":      feat_orig,
        "feat_contr":     feat_contr,
        "hist_br_orig":   hist_br_orig,
        "hist_br_contr":  hist_br_contr,
        "edges":          edges_orig,
    }


def plot_results(results):
    name = results["name"]

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(f"GLCM + Степенное преобразование\nИзображение: «{name}»",
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax_rgb_o   = fig.add_subplot(gs[0, 0])
    ax_gray_o  = fig.add_subplot(gs[0, 1])
    ax_rgb_c   = fig.add_subplot(gs[0, 2])
    ax_gray_c  = fig.add_subplot(gs[0, 3])

    ax_rgb_o.imshow(results["rgb_orig"])
    ax_rgb_o.set_title("Исходное (RGB)", fontsize=9)
    ax_rgb_o.axis("off")

    ax_gray_o.imshow(results["gray_orig"], cmap="gray", vmin=0, vmax=255)
    ax_gray_o.set_title("Полутоновое (L)", fontsize=9)
    ax_gray_o.axis("off")

    ax_rgb_c.imshow(results["rgb_contrast"])
    ax_rgb_c.set_title(f"Контрастированное (RGB, γ={GAMMA})", fontsize=9)
    ax_rgb_c.axis("off")

    ax_gray_c.imshow(results["gray_contrast"], cmap="gray", vmin=0, vmax=255)
    ax_gray_c.set_title(f"Контрастир. полутоновое", fontsize=9)
    ax_gray_c.axis("off")

    ax_glcm_o  = fig.add_subplot(gs[1, 0:2])
    ax_glcm_c  = fig.add_subplot(gs[1, 2:4])

    im1 = ax_glcm_o.imshow(glcm_for_view(results["glcm_orig"]), cmap="gray")
    ax_glcm_o.set_title("GLCM исходного изображения", fontsize=9)
    ax_glcm_o.set_xlabel("j", fontsize=8)
    ax_glcm_o.set_ylabel("i", fontsize=8)
    plt.colorbar(im1, ax=ax_glcm_o, fraction=0.046, pad=0.04)

    im2 = ax_glcm_c.imshow(glcm_for_view(results["glcm_contr"]), cmap="gray")
    ax_glcm_c.set_title("GLCM после контрастирования", fontsize=9)
    ax_glcm_c.set_xlabel("j", fontsize=8)
    ax_glcm_c.set_ylabel("i", fontsize=8)
    plt.colorbar(im2, ax=ax_glcm_c, fraction=0.046, pad=0.04)

    ax_br_o  = fig.add_subplot(gs[2, 0:2])
    ax_br_c  = fig.add_subplot(gs[2, 2:4])

    edges = results["edges"]
    ax_br_o.bar(edges, results["hist_br_orig"], width=1.0,
                color="steelblue", alpha=0.8)
    ax_br_o.set_title("Гистограмма яркости — исходное", fontsize=9)
    ax_br_o.set_xlabel("Яркость", fontsize=8)
    ax_br_o.set_ylabel("Нормир. частота", fontsize=8)
    ax_br_o.tick_params(labelsize=7)

    ax_br_c.bar(edges, results["hist_br_contr"], width=1.0,
                color="darkorange", alpha=0.8)
    ax_br_c.set_title("Гистограмма яркости — после", fontsize=9)
    ax_br_c.set_xlabel("Яркость", fontsize=8)
    ax_br_c.set_ylabel("Нормир. частота", fontsize=8)
    ax_br_c.tick_params(labelsize=7)

    ax_br_all = fig.add_subplot(gs[3, 0:2])
    ax_feat   = fig.add_subplot(gs[3, 2:4])

    ax_br_all.bar(edges, results["hist_br_orig"],  width=1.0,
                  color="steelblue", alpha=0.6, label="До")
    ax_br_all.bar(edges, results["hist_br_contr"], width=1.0,
                  color="darkorange", alpha=0.6, label="После")
    ax_br_all.set_title("Гистограмма яркости (до / после)", fontsize=9)
    ax_br_all.set_xlabel("Яркость", fontsize=8)
    ax_br_all.set_ylabel("Нормир. частота", fontsize=8)
    ax_br_all.legend(fontsize=8)
    ax_br_all.tick_params(labelsize=7)

    feat_names = list(results["feat_orig"].keys())
    col_labels = ["Признак", "До", "После", "Δ"]
    table_data = []
    for k in feat_names:
        v_o = results["feat_orig"][k]
        v_c = results["feat_contr"][k]
        table_data.append([k, f"{v_o:.4f}", f"{v_c:.4f}", f"{v_c - v_o:+.4f}"])

    ax_feat.axis("off")
    tbl = ax_feat.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)
    ax_feat.set_title("Сравнение GLCM-признаков", fontsize=9, pad=12)

    out_path = BASE_DIR / f"glcm_{name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_combined_glcm(all_results):
    feat_names = list(all_results[0]["feat_orig"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Сравнение GLCM-признаков по всем изображениям\n"
                 "",
                 fontsize=13, fontweight="bold")

    colors = {"regular": "steelblue", "low": "seagreen", "high": "crimson"}
    x = np.arange(len(all_results))
    names = [res["name"] for res in all_results]

    for ax, feat in zip(axes.ravel(), feat_names):
        before = [res["feat_orig"][feat] for res in all_results]
        after = [res["feat_contr"][feat] for res in all_results]
        bar_colors = [colors.get(name, "gray") for name in names]

        ax.bar(x - 0.18, before, width=0.36, color=bar_colors, alpha=0.85,
               label="исходное")
        ax.bar(x + 0.18, after,  width=0.36, color=bar_colors, alpha=0.45,
               label=f"после γ={GAMMA}")
        ax.set_title(feat, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = BASE_DIR / "combined.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":

    found_images = {}
    for key, path in IMAGE_FILES.items():
        if os.path.exists(path):
            found_images[key] = path


    all_results = []
    output_files = []

    for name, path in found_images.items():
        try:
            res = process_image(name, path)
            all_results.append(res)
            out = plot_results(res)
            output_files.append(out)
        except Exception as e:
            print(f" Ошибка при обработке {name}: {e}")
            import traceback; traceback.print_exc()

    if len(all_results) > 1:
        out = plot_combined_glcm(all_results)
        output_files.append(out)

    print("\n Готово")
