"""Remove fundo claro, recorta margens vazias e exporta PNG para o header."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Lado máximo após crop (arte nítida em telas retina)
MAX_SIDE = 480
ALPHA_TRIM = 18  # pixels com alpha acima disso entram no bbox
PAD = 6


def border_samples(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    top = arr[0, :, :3].reshape(-1, 3)
    bot = arr[-1, :, :3].reshape(-1, 3)
    left = arr[:, 0, :3].reshape(-1, 3)
    right = arr[:, -1, :3].reshape(-1, 3)
    return np.vstack([top, bot, left, right])


def median_bg(samples: np.ndarray) -> np.ndarray:
    return np.median(samples.astype(np.float64), axis=0)


def smooth_alpha_from_dist(dist: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """0..255 com borda suave entre t0 e t1 (distância RGB ao fundo)."""
    a = np.zeros_like(dist, dtype=np.float64)
    a[dist >= t1] = 255.0
    mid = (dist > t0) & (dist < t1)
    a[mid] = 255.0 * (dist[mid] - t0) / (t1 - t0)
    return a


def trim_to_alpha(im: Image.Image, threshold: int = ALPHA_TRIM, pad: int = PAD) -> Image.Image:
    """Recorta faixas transparentes / quase vazias (ex.: borda branca em cima e embaixo)."""
    a = np.array(im, dtype=np.uint8)[:, :, 3]
    ys, xs = np.where(a > threshold)
    if ys.size == 0:
        return im
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(im.height - 1, y1 + pad)
    x1 = min(im.width - 1, x1 + pad)
    return im.crop((x0, y0, x1 + 1, y1 + 1))


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)

    im0 = Image.open(src).convert("RGBA")
    arr_u8 = np.asarray(im0, dtype=np.uint8)
    arr = arr_u8.astype(np.float64)
    rgb = arr[:, :, :3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)

    bg = median_bg(border_samples(arr))
    dist = np.linalg.norm(rgb - bg.reshape(1, 1, 3), axis=2)

    # Alpha principal: distância ao tom médio das bordas (papel)
    a = smooth_alpha_from_dist(dist, t0=20.0, t1=58.0)

    # Papel / branco de margem: claro e pouco saturado (remove faixas grandes em cima e embaixo)
    paper = (lum > 212.0) & (chroma < 34.0) & (dist < 92.0)
    a[paper] = np.minimum(a[paper], smooth_alpha_from_dist(dist[paper], 12.0, 48.0))

    # Reforço: quase igual ao fundo amostrado
    a[dist < 18.0] = 0.0

    # Não apagar o conteúdo escuro (tipografia marrom)
    dark_ink = lum < 120.0
    a[dark_ink] = np.maximum(a[dark_ink], 255.0)

    # Detalhes médios (ícone dourado, ondas): não zerar se saturado o bastante
    rich = chroma > 42.0
    a[rich & (lum < 205.0)] = np.maximum(a[rich & (lum < 205.0)], 255.0)

    out_arr = arr_u8.copy()
    out_arr[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)

    out = Image.fromarray(out_arr, "RGBA")
    out = trim_to_alpha(out)
    out.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    out.save(dst, optimize=True)
    print(f"OK: {dst} ({out.size[0]}x{out.size[1]})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_logo.py <input.png> <output.png>")
        sys.exit(1)
    main()
