"""竖屏版（PV_LAYOUT=vertical，抖音 / Shorts 用）：原 16:9 画面放在中间，上面是标题和当前一幕的名牌，
下面是放大两倍的台词（对话框 / 字幕从画面里挪到这里）。低分辨率画布 480x854，finalize 后缩放到 1080x1920。

安全区按抖音界面留：顶部约 100 像素被状态栏和频道标签盖住，底部约 130 像素被文案盖住，右侧按钮在 x > 410。
"""
import os
from functools import lru_cache

import numpy as np

import i18n
import story
import ui
from gfx import H, W, blit, dither_mask, ease_out, fade, hexc, rect, smooth, text, text_mask

LAYOUT = os.environ.get('PV_LAYOUT', 'landscape')
assert LAYOUT in ('landscape', 'vertical'), f'PV_LAYOUT 只支持 landscape / vertical，收到 {LAYOUT!r}'
VERTICAL = LAYOUT == 'vertical'

VW, VH = W, 854                 # 9:16
FY = 172                        # 原画面顶边
BY = FY + H + 20                # 台词区顶边
TX0, TX1 = 20, 416              # 台词可用宽度（右侧按钮只会压到最长那行的末尾）
ROW = 26                        # 两倍字的行距
GOLD = hexc('#d8c8a0')

# 每一段的名牌角色（没有的段只显示日文社名）
ACT = {'prologue': 'campanella', 'mcburn': 'mcburn', 'leonhardt': 'leonhardt', 'arianrhod': 'arianrhod',
       'bleublanc': 'bleublanc', 'vita': 'vita', 'phantom': 'campanella', 'grandmaster': 'grandmaster',
       'epilogue': 'campanella'}


def _backdrop():
    y = np.linspace(0, 1, VH, dtype=np.float32)[:, None, None]
    top, mid = hexc('#06030a'), hexc('#140a22')
    k = 1 - np.abs(y - (FY + H / 2) / VH) * 2.2
    bg = top + (mid - top) * np.clip(k, 0, 1)
    return np.broadcast_to(bg, (VH, VW, 3)).astype(np.float32).copy()


_BG = None


def _bg():
    global _BG
    if _BG is None:
        plain = _backdrop()
        em = plain.copy()
        ui.draw_emblem(em, VW / 2, BY + 250, size=96, progress=1.0, palette='gold', scale=2)
        _BG = plain * 0.9 + em * 0.1           # 纹章只留一点影子
    return _BG.copy()


LOGO = ('title', 'slam')                        # 画面里本身就是大标题的段：顶部标题让开


def _header(dst, sec, t):
    a = smooth(0.3, 1.8, t)                     # 开场从黑里浮出来
    if sec.name in LOGO:
        a *= 1 - min(smooth(0, 0.25, t - sec.t0), smooth(0, 0.25, sec.t1 - t))
    if a <= 0:
        return
    tmp = dst.copy()
    text(tmp, i18n.S('org'), VW / 2, 118, hexc('#ffe0a0'), scale=3, glow=hexc('#6a2aa0'))
    key = ACT.get(sec.name)
    n = (t - sec.t0) * 30                       # 每一幕开头把名牌重新打出来
    if key:
        name, title, rank = i18n.name(key)
        sep = '' if title.startswith('《') or key == 'grandmaster' else ' · '
        label = (rank + sep + title if key != 'grandmaster' else rank) + '  ' + name
        text(tmp, label, VW / 2, 150, hexc('#c8b8e8'), reveal=n)
    else:
        text(tmp, '身喰らう蛇', VW / 2, 150, hexc('#9a8cb8'), font='ja', reveal=n)
    m = ui_mask(a, dst.shape[:2])
    dst[m] = tmp[m]


def ui_mask(level, shape):
    return dither_mask(level, shape) if level < 1 else np.ones(shape, bool)


NO_START = set('、。，．,.!！?？」』）)”’…─—ー')    # 不放在行首
NO_END = set('「『（(“‘')                       # 不放在行尾
BREAK_AFTER = set('、。，,.!！?？…─—」』”')        # 优先在这些字后面断
PARTICLE = set('がはをにでもへ')                 # 日文助词后面也可以断（次选）


@lru_cache(None)
def balanced(s, width, font='zh'):
    """按像素宽度折成尽量等长的几行：英文按词、中日文按字，优先在标点后断，避头尾。"""
    tw = lambda x: text_mask(x, font).shape[1]
    if tw(s) <= width:
        return (s,)
    words = ' ' in s
    toks = s.split(' ') if words else list(s)
    sep = ' ' if words else ''
    total, L = tw(s), len(toks)
    for n in range(2, L + 1):
        target = total / n
        best = {0: (0.0, ())}                   # 已用 j 个词 -> (代价, 断点)
        for _ in range(n):
            nxt = {}
            for i, (c0, cuts) in best.items():
                for j in range(i + 1, L + 1):
                    row = sep.join(toks[i:j])
                    w = tw(row)
                    if w > width:
                        break
                    if not words and (toks[i] in NO_START or toks[j - 1] in NO_END):
                        continue
                    c = c0 + abs(w - target)
                    last = toks[j - 1].rstrip('"\'') or toks[j - 1]
                    if j < L and last[-1] in BREAK_AFTER:
                        c -= target * (0.25 if words else 0.4)
                    elif j < L and font == 'ja' and last[-1] in PARTICLE:
                        c -= target * 0.2
                    if j not in nxt or c < nxt[j][0]:
                        nxt[j] = (c, cuts + (j,))
            best = nxt
        if L in best:
            cuts = (0,) + best[L][1]
            return tuple(sep.join(toks[a:b]) for a, b in zip(cuts, cuts[1:]))
    return tuple(ui.wrap(s, width))


def _layout(line, y, width, gap=10):
    """两倍字的台词排版：译文几行 + 日文几行，返回 (rows, ys, jrows, jys)。"""
    tr, ja = line[:2]
    rows, jrows = balanced(tr, width // 2), balanced(ja, width // 2, 'ja')
    ys = [y + ROW * i for i in range(len(rows))]
    jys = [ys[-1] + ROW + gap + ROW * i for i in range(len(jrows))]
    return rows, ys, jrows, jys


def _say_text(dst, line, frac, x, lay, anchor):
    rows, ys, jrows, jys = lay
    ui._reveal_lines(dst, rows, frac * len(line[0]), x, ys, hexc('#f2eefa'), anchor=anchor, scale=2)
    ui._reveal_lines(dst, jrows, frac * len(line[1]), x, jys, hexc('#9a8cb8'), anchor=anchor, scale=2, font='ja')


def _dialog(dst, ch, line, lt, cps=14):
    k = ease_out(lt / 0.2)
    x0, x1, y0 = 12, VW - 12, BY + (1 - k) * 16
    lay = _layout(line, y0 + 146, TX1 - TX0, gap=8)
    y1 = lay[3][-1] + 20
    rect(dst, x0, y0, x1, y1, hexc('#0a0612'), 0.8)
    for yy in (y0, y1 - 1):
        rect(dst, x0, yy, x1, yy + 1, GOLD, 0.8)
    rect(dst, x0 + 10, y0 + 10, x0 + 10 + 116, y0 + 10 + 116, GOLD, 0.8)
    rect(dst, x0 + 12, y0 + 12, x0 + 12 + 112, y0 + 12 + 112, hexc('#000000'))
    blit(dst, ui.face_crop(ch), x0 + 12, y0 + 12, scale=2)
    m = ch.meta
    text(dst, m['name'], x0 + 140, y0 + 40, hexc('#ffffff'), scale=2, anchor='left', glow=ch.fx('primary') * 0.45)
    sep = '' if m['title'].startswith('《') else ' · '
    text(dst, m['rank'] + sep + m['title'], x0 + 140, y0 + 70, hexc('#c8b8e8'), anchor='left')
    ref = line[2] if len(line) > 2 else len(line[0])
    frac = lt * cps / max(1, ref)
    _say_text(dst, line, frac, TX0, lay, 'left')
    if frac > 1 and int(lt * 4) % 2 == 0:
        rect(dst, x1 - 22, y1 - 16, x1 - 12, y1 - 10, hexc('#ffd060'))


def _subtitle(dst, line, lt, cps=16):
    ref = line[2] if len(line) > 2 else len(line[0])
    frac = lt * cps / max(1, ref)
    _say_text(dst, line, frac, (TX0 + TX1) / 2, _layout(line, BY + 40, TX1 - TX0), 'center')


def compose(frame, says, t):
    """frame: 原 480x270 画面（已含转场 / 闪光 / 震屏）；says: 导演收集的 (style, ch, line, lt, leave)。"""
    sec = story.section_at(t)
    dst = _bg()
    dst[FY:FY + H] = frame
    for yy in (FY - 2, FY + H + 1):
        rect(dst, 0, yy, VW, yy + 1, GOLD, 0.45)
    _header(dst, sec, t)
    for style, ch, line, lt, leave in says:
        tmp = dst.copy()
        if style == 'dialog':
            _dialog(tmp, ch, line, lt)
        else:
            _subtitle(tmp, line, lt)
        m = ui_mask(1 - leave, dst.shape[:2])
        dst[m] = tmp[m]
    if t > story.DURATION - 0.8:
        fade(dst, hexc('#000000'), smooth(story.DURATION - 0.8, story.DURATION - 0.1, t))
    return dst
