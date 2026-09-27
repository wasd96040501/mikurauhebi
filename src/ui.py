"""画面上的文字 UI：轨迹风对话框、双语字幕、名牌。各场景共用。"""
from functools import lru_cache

import numpy as np

from gfx import W, blit, dilate, ease_back, ease_out, hexc, rect, text, text_mask


def face_crop(ch, expr=None):
    p = ch.portrait(expr)
    return p[2:58, 12:68]


def wrap(s, width):
    """按像素宽度折行：英文按词断，中文按字断。"""
    if text_mask(s).shape[1] <= width:
        return [s]
    toks = s.split(' ') if ' ' in s else list(s)
    sep = ' ' if ' ' in s else ''
    lines, cur = [], ''
    for tk in toks:
        cand = cur + sep + tk if cur else tk
        if cur and text_mask(cand).shape[1] > width:
            lines.append(cur)
            cur = tk
        else:
            cur = cand
    return lines + [cur]


def _reveal_lines(dst, rows, n, x, ys, col, **kw):
    """多行打字机：n 为已显示的总字数。"""
    for row, y in zip(rows, ys):
        text(dst, row, x, y, col, reveal=n, **kw)
        n -= len(row) + (1 if ' ' in row else 0)
        if n <= 0:
            break


def dialog(dst, ch, line, lt, cps=14, expr=None, y0=196, leave=0.0):
    """轨迹风对话框：左侧头像、名字标签，译文（中/英）打字机 + 下方日文原句。
    line = (译文, 日文[, 中文原长度])；有第三项时，译文按中文版的时长打完，和打字音对齐。
    一行放不下就折行，框的底边不动、向上长高。"""
    if lt < 0:
        return
    tr, ja = line[:2]
    x0, x1 = 50, W - 50
    tx = x0 + 76
    avail = x1 - tx - 14
    rows = wrap(tr, avail)
    jrows = wrap_ja(ja, avail)
    n_lines = len(rows) + len(jrows)
    h = 64 if n_lines <= 2 else 64 + 13 * (n_lines - 2)
    k = ease_out(lt / 0.15)
    yb = y0 + 64 - h + (1 - k) * 20 + leave ** 2 * 90       # leave 0..1：向下滑出画面
    rect(dst, x0, yb, x1, yb + h, hexc('#0a0612'), 0.86)
    for yy in (yb, yb + h - 1):
        rect(dst, x0, yy, x1, yy + 1, hexc('#d8c8a0'))
    rect(dst, x0, yb, x0 + 1, yb + h, hexc('#d8c8a0'))
    rect(dst, x1 - 1, yb, x1, yb + h, hexc('#d8c8a0'))
    blit(dst, face_crop(ch, expr), x0 + 4, yb + h - 60)
    rect(dst, x0 + 70, yb - 9, x0 + 70 + text_mask(ch.meta['name']).shape[1] + 12, yb + 7, hexc('#3a2410'), 0.95)
    text(dst, ch.meta['name'], tx, yb - 1, hexc('#ffe0a0'), anchor='left')
    ref = line[2] if len(line) > 2 else len(tr)
    frac = lt * cps / max(1, ref)
    if n_lines == 2:                       # 原版式：译文 26、日文 46
        ys, jys = [yb + 26], [yb + 46]
    else:
        ys = [yb + 17 + 13 * i for i in range(len(rows))]
        jys = [ys[-1] + 20 + 13 * i for i in range(len(jrows))]
    _reveal_lines(dst, rows, frac * len(tr), tx, ys, hexc('#f2eefa'), anchor='left')
    _reveal_lines(dst, jrows, frac * len(ja), tx, jys, hexc('#9a8cb8'), anchor='left', font='ja')
    if frac > 1 and int(lt * 4) % 2 == 0:
        rect(dst, x1 - 14, yb + h - 12, x1 - 8, yb + h - 8, hexc('#ffd060'))


def wrap_ja(s, width):
    """日文按字折行，尽量在「、」「。」「──」之后断开。"""
    if text_mask(s, 'ja').shape[1] <= width:
        return [s]
    cut = len(s)
    while cut > 1 and text_mask(s[:cut], 'ja').shape[1] > width:
        cut -= 1
    for p in range(cut, max(1, cut - 8), -1):
        if s[p - 1] in '、。─…，':
            cut = p
            break
    return [s[:cut]] + wrap_ja(s[cut:], width)


def subtitle(dst, line, lt, y=238, cps=16, zh_col='#f2eefa', ja_col='#9a8cb8', box=True, leave=0.0):
    """译文（中/英）+ 日文原句的双行字幕。line 同 dialog。"""
    if lt < 0 or leave >= 1:
        return
    tr, ja = line[:2]
    ref = line[2] if len(line) > 2 else len(tr)
    frac = lt * cps / max(1, ref)
    y += leave ** 2 * 60
    rows = wrap(tr, W - 48)                  # 太长就折成两行（两行尽量等长），往上长；日文位置不变
    if len(rows) == 2 and ' ' in tr:
        ws = tr.split(' ')
        k = min(range(1, len(ws)), key=lambda i: max(text_mask(' '.join(ws[:i])).shape[1], text_mask(' '.join(ws[i:])).shape[1]))
        rows = [' '.join(ws[:k]), ' '.join(ws[k:])]
    top = y - 13 * (len(rows) - 1)
    if box:
        a = 0.6 * ease_out(lt / 0.2)
        rect(dst, 0, top - 14, W, y + 30, hexc('#06030a'), a)
    n = frac * len(tr)
    for i, row in enumerate(rows):
        text(dst, row, W / 2, top + 13 * i, hexc(zh_col), reveal=n)
        n -= len(row) + (1 if ' ' in row else 0)
    text(dst, ja, W / 2, y + 18, hexc(ja_col), reveal=frac * len(ja), font='ja')


def name_tag(dst, ch, lt, x=18, y=26, col=None):
    """每一幕开头的小名牌：称号 + 名字（不写技能名）。从左侧滑入。"""
    if lt < 0:
        return
    m = ch.meta
    col = col if col is not None else ch.fx('primary')
    p = ease_back(lt / 0.35)
    x0 = x - 160 * (1 - p)
    rect(dst, x0, y - 12, x0 + 3, y + 30, col)
    sep = '' if m['title'].startswith('《') else ' · '        # 官方中文写法：执行者No.Ⅰ《劫炎》
    text(dst, m['rank'] + sep + m['title'], x0 + 10, y - 2, hexc('#c8b8e8'), anchor='left')
    text(dst, m['name'], x0 + 10, y + 18, hexc('#ffffff'), scale=2, anchor='left', glow=col * 0.45)


def fire_palette(primary):
    p = primary if isinstance(primary, str) else '#%02x%02x%02x' % tuple(int(v * 255) for v in primary)
    return ['#000000', '#1a0608', _mix(p, '#000000', 0.55), p, _mix(p, '#ffffff', 0.45), '#fff8e0', '#ffffff']


def _mix(a, b, k):
    c = hexc(a) * (1 - k) + hexc(b) * k
    return '#%02x%02x%02x' % tuple(int(np.clip(v, 0, 1) * 255) for v in c)


# ---------------------------------------------------------------- 纹章

@lru_cache(None)
def emblem_img(size=96, progress=1.0, palette='gold'):
    """结社纹章（emblem.py），progress 0..1 为描绘进度。"""
    from emblem import emblem
    return emblem(size=size, progress=progress, palette=palette)


def draw_emblem(dst, cx, cy, size=96, progress=1.0, palette='gold', scale=2, glow=0.0, glow_col=(0.6, 0.3, 1.0)):
    img = emblem_img(size, round(float(progress), 2), palette)
    if glow > 0:
        m = img[..., 3] > 0
        g = dilate(dilate(m, 1), 1) & ~m
        gi = np.zeros_like(img)
        gi[g] = (*glow_col, glow)
        blit(dst, gi, cx, cy, scale=scale, anchor='center')
    blit(dst, img, cx, cy, scale=scale, anchor='center')
