"""画像素角色用的小工具：ASCII 网格转图、自动描边、预览图。

角色模块（src/chars/*.py）统一返回 float32 RGBA 数组，形状 (H, W, 4)，取值 0..1，
透明像素 alpha=0。
"""
from pathlib import Path

import numpy as np
from PIL import Image


def hexc(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def from_ascii(rows, palette, width=None):
    """rows: 字符串列表；palette: {字符: '#rrggbb'}；'.' 和 ' ' 视为透明。"""
    w = width or max(len(r) for r in rows)
    img = np.zeros((len(rows), w, 4), np.float32)
    for y, r in enumerate(rows):
        for x, c in enumerate(r[:w]):
            if c in '. ':
                continue
            img[y, x, :3] = hexc(palette[c])
            img[y, x, 3] = 1
    return img


def blank(w, h):
    return np.zeros((h, w, 4), np.float32)


def paint(img, mask, color):
    img[mask, :3] = hexc(color) if isinstance(color, str) else color
    img[mask, 3] = 1


def outline(img, color='#140c1c', inner=False):
    """给不透明区域外侧加 1px 描边（inner=True 时描在内侧边缘）。"""
    a = img[..., 3] > 0
    grown = a.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        grown |= np.roll(np.roll(a, dy, 0), dx, 1)
    out = img.copy()
    if inner:
        edge = a & ~(np.roll(a, 1, 0) & np.roll(a, -1, 0) & np.roll(a, 1, 1) & np.roll(a, -1, 1))
        paint(out, edge, color)
    else:
        paint(out, grown & ~a, color)
    return out


def flip(img):
    return img[:, ::-1].copy()


def preview(images, path, scale=8, bg=('#3a3a46', '#2e2e38'), gap=4, labels=None):
    """把若干张 RGBA 并排放大保存，背景为棋盘格，方便对照参考图检查。"""
    h = max(i.shape[0] for i in images)
    w = sum(i.shape[1] for i in images) + gap * (len(images) + 1)
    sheet = np.zeros((h + 2 * gap, w, 3), np.float32)
    yy, xx = np.mgrid[0:sheet.shape[0], 0:sheet.shape[1]]
    chk = ((yy // 4 + xx // 4) % 2).astype(bool)
    sheet[chk] = hexc(bg[0])
    sheet[~chk] = hexc(bg[1])
    x = gap
    for im in images:
        ih, iw = im.shape[:2]
        a = im[..., 3:4]
        reg = sheet[gap:gap + ih, x:x + iw]
        reg[:] = reg * (1 - a) + im[..., :3] * a
        x += iw + gap
    big = np.repeat(np.repeat(sheet, scale, 0), scale, 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((big * 255).astype(np.uint8)).save(path)
    return path
