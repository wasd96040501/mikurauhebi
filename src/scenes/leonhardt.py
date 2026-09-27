"""第二幕：剑帝莱恩哈特 —— 哈梅尔废村，黄昏。

原作（空之轨迹 SC）：莱恩哈特（罗伟）生长在帝国南部的山村哈梅尔。「哈梅尔悲剧」中村子被毁，
他深爱的卡琳为保护约修亚而死；他因此加入结社，要「试炼人类」。他在墓前说出的那句
「俺は俺の、選んだ道がある。」是一个站在自己过去面前的人的回答。

分镜（act 内 小节.拍）
----------------------
  1.0  废墟全景：夕阳压在雪山背后，镜头从左向右缓缓横摇（三层视差），灰烬与细雪飘落；
       横摇的终点是背对镜头站在墓前的剑帝，风吹动风衣下摆与发梢（背影三帧循环），剑插在身旁的地里。
  2.0  名牌滑入。2.1 他微微侧过头（back_turn）；2.1.5 一闪而过的回忆：暖光里完好的哈梅尔
       （屋顶、亮着灯的窗、钟楼里的钟），抖动溶解进来又退回废墟。
  3.0  硬切（音乐换成跳弓弦乐 + 太鼓）：侧面中景，他已转过身，手按在插进地里的剑柄上；台词对话框。
       镜头缓慢推近（视差漂移）。
  4.3  拔剑：剑从地里拔出（火星 + 碎石，sword_draw），抡过头顶，剑身燃起鬼火，落进压低的架势；
       对话框在这一拍收起。音乐在这一拍全奏一下后停顿（stop），只剩火声与呼吸。
  4.3.5 冲刺：残影 + 拖影 + 速度线，从右向左横穿画面；剑举过右肩。
  5.0  命中：两帧反相冲击帧（白底黑剪影）→ 定帧（hit-stop）→ 震屏。
       剑的真实轨迹（slash1→slash2→slash3 的剑尖路径）放大成一道横跨全屏的火焰新月；
       背景废墟沿斩线出现一道炽热的细缝。
  5.2  余波：斩线处的废墟被切开滑落、整排燃起（火焰爆裂 + 余烬），他收势起身。
  6.0  硬切（音乐 drop）：终幕画面 —— 背影，右手垂剑，剑身拖着火焰；远处被斩断的钟楼上半截
       沿斩线滑落、在 6.1 砸进火海扬起尘烟。灰烬与火星落下，最后 0.55 秒交给扑克牌转场。
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

import chars.leonhardt as LH
import fx
import sfx as S
from gfx import BAYER, H, W, blit, ease_out, hexc, rect, smooth
import ui

# ---------------------------------------------------------------- 时间轴（act 内）
SAYS = [('leonhardt', 3, 0.0, 'dialog', (4, 3.0))]         # 台词框在拔剑那一拍收起
CUES = [
    (1, 0.0, 'wind', -7), (1, 3.0, 'wind', -10), (2, 2.0, 'wind', -10),
    (2, 1.5, 'bell_far', -8),                         # 回忆：钟楼还在时的钟声
    (3, 0.0, 'coat', -6),                             # 硬切：转身，风衣一甩
    (4, 3.0, 'sword_draw', -1),                       # 拔剑
    (4, 3.3, 'fire_whoosh', -5),                      # 剑身燃起鬼火
    (4, 3.65, 'dash', -1),                            # 冲刺
    (5, 0.0, 'slash', 0), (5, 0.0, 'big_impact', -2), (5, 0.1, 'fire_burst', -4),
    (5, 2.0, 'fire_roar', -3), (5, 2.0, 'impact', -7),   # 废墟被切开、整排燃起
    (6, 0.0, 'boom_low', -3), (6, 0.0, 'coat', -10),     # 终幕硬切
    (6, 1.0, 'rumble', -5), (6, 1.0, 'impact', -9),      # 钟楼上半截砸落
]
ACCENTS = [(4, 3.0, 'stop'), (5, 0.0, 'hit'), (6, 0.0, 'drop')]


def _sfx_sword_draw():
    """剑从石缝里拔出：碎石摩擦 -> 金属长鸣。"""
    n = int(1.3 * S.SR)
    t = S.tt(n)
    scrape = S.bp(S.noise(n), 900, 4200) * np.clip(t / 0.02, 0, 1) * np.exp(-t / 0.12)
    grit = np.zeros(n)
    rng = np.random.default_rng(3)
    for i in rng.integers(0, int(0.18 * S.SR), 24):
        m = 200
        grit[i:i + m] += S.hp(S.noise(m), 2500) * np.exp(-np.arange(m) / 30) * rng.uniform(0.3, 1)
    ring = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((1180, 0.5), (2630, 0.35), (4410, 0.22), (6020, 0.12)))
    ring = ring * np.clip((t - 0.08) / 0.03, 0, 1) * np.exp(-np.maximum(t - 0.08, 0) / 0.5)
    return 1.1 * scrape + 0.8 * grit + 0.55 * ring


def _sfx_bell_far():
    """远处的教堂钟（回忆里的哈梅尔），带一点回声。"""
    n = int(2.6 * S.SR)
    t = S.tt(n)
    x = sum(np.sin(2 * np.pi * f * t) * a * np.exp(-t / d)
            for f, a, d in ((196, 0.6, 1.6), (392.8, 0.35, 1.0), (471, 0.25, 0.7), (588, 0.2, 0.5), (932, 0.1, 0.3)))
    x = x * np.clip(t / 0.004, 0, 1)
    echo = np.zeros(n)
    d = int(0.33 * S.SR)
    echo[d:] = x[:-d] * 0.35
    return S.lp(x + echo, 2500) * 0.8


def _sfx_coat():
    """风衣被甩动 / 被风鼓起的布料声。"""
    n = int(0.45 * S.SR)
    t = S.tt(n)
    x = S.bp(S.noise(n), 300, 2400) * np.sin(np.pi * np.clip(t / 0.3, 0, 1)) ** 1.5
    flap = S.lp(S.noise(n), 900) * (np.sin(2 * np.pi * 22 * t) > 0.6) * np.exp(-t / 0.15)
    return 0.9 * x + 0.6 * flap


SFX = {'sword_draw': _sfx_sword_draw, 'bell_far': _sfx_bell_far, 'coat': _sfx_coat}

BEAT = 60 / 144
BAR = 4 * BEAT
T_B = 2 * BAR                      # 3.0 硬切
T_DRAW = 3 * BAR + 3 * BEAT        # 4.3 拔剑
T_IMP = 4 * BAR                    # 5.0 命中
T_SEC = 4 * BAR + 2 * BEAT         # 5.2 余波
T_D = 5 * BAR                      # 6.0 终幕

# ---------------------------------------------------------------- 调色板
C = {k: hexc(v) for k, v in dict(
    sky0='#140f26', sky1='#2a1a3c', sky2='#512842', sky3='#8c3a42', sky4='#c85a3e', sky5='#ec8c48', sky6='#ffc878',
    msky0='#3a2a4a', msky1='#7a4a58', msky2='#c07a5a', msky3='#eaa868', msky4='#ffd490', msky5='#fff0c0',
    sun='#fff2c8', halo='#ffc070',
    mt='#3c2a4a', mt_snow='#7c6488', mt_rim='#e09470', mt2='#261a30', mt2_rim='#8a4a50',
    st0='#1c1219', st1='#2e2229', st2='#43333b', st3='#5a4549', rim='#b86a52', rim2='#e89a6a', win='#0e080c',
    char='#140c10', snow='#8e8090', snow2='#b8aab8',
    ist1='#6a4a42', ist2='#8a6a5a', ist3='#a8886c', roof='#7a3428', roof2='#a8503a', lamp='#ffd070', bell='#c8a040',
    g0='#1e1520', g1='#2a1e28', g2='#3a2a32', grass='#6a5040', path='#3e2e30',
    wood='#6a4a34', wood2='#3a2618', fl_w='#f2eef6', fl_y='#f0d060', stem='#4a6a3a',
).items()}
YY, XX = np.mgrid[0:H, 0:W]


def _dith(level, shape=None, ox=0, oy=0):
    """Bayer 抖动遮罩（像素风的渐变 / 溶解）。"""
    if shape is None:
        return BAYER < level
    h, w = shape
    b = np.tile(BAYER[:4, :4], (h // 4 + 2, w // 4 + 2))
    return b[oy % 4:oy % 4 + h, ox % 4:ox % 4 + w] < level


def _poly(w, h, pts):
    im = Image.new('L', (w, h), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=1, outline=1)
    return np.array(im, bool)


class Layer:
    def __init__(self, w, h):
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.m = np.zeros((h, w), bool)
        self.w, self.h = w, h

    def put(self, mask, col):
        self.rgb[mask] = col
        self.m |= mask

    def screen_mask(self, x, y):
        x, y = int(round(x)), int(round(y))
        out = np.zeros((H, W), bool)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + self.w), min(H, y + self.h)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = self.m[y0 - y:y1 - y, x0 - x:x1 - x]
        return out

    def draw(self, dst, x, y, mask=None):
        x, y = int(round(x)), int(round(y))
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + self.w), min(H, y + self.h)
        if x1 <= x0 or y1 <= y0:
            return
        m = self.m[y0 - y:y1 - y, x0 - x:x1 - x]
        if mask is not None:
            m = m & mask[y0:y1, x0:x1]
        dst[y0:y1, x0:x1][m] = self.rgb[y0 - y:y1 - y, x0 - x:x1 - x][m]


# ---------------------------------------------------------------- 背景：天空
@lru_cache(None)
def _sky(memory=False):
    stops = ([(0, 'msky0'), (60, 'msky1'), (110, 'msky2'), (150, 'msky3'), (175, 'msky4'), (190, 'msky5')] if memory else
             [(0, 'sky0'), (55, 'sky1'), (100, 'sky2'), (135, 'sky3'), (158, 'sky4'), (175, 'sky5'), (188, 'sky6')])
    out = np.zeros((H, W, 3), np.float32)
    for y in range(H):
        for (ya, ca), (yb, cb) in zip(stops, stops[1:]):
            if ya <= y < yb:
                k = (y - ya) / (yb - ya)
                out[y] = np.where((BAYER[y] < k)[:, None], C[cb], C[ca])
                break
        else:
            out[y] = C[stops[-1][1]]
    # 夕阳（压在山口后面）+ 抖动光晕
    d = np.hypot(XX - 300, (YY - 170) * 1.1)
    out[(d < 22) & _dith(np.clip((22 - d) / 6, 0, 1))] = C['halo']
    out[d < 13] = C['sun']
    # 几条被夕阳照亮下缘的云
    rng = np.random.default_rng(5 if not memory else 6)
    for _ in range(9):
        cx, cy, L = rng.uniform(0, W), rng.uniform(40, 130), rng.uniform(40, 120)
        th = rng.integers(2, 4)
        m = (np.abs(YY - cy) < th) & (np.abs(XX - cx) < L * (1 - np.abs(YY - cy) / 4))
        out[m] = C['sky2'] if not memory else C['msky1']
        out[m & (np.abs(YY - cy - th + 1) < 1)] = C['sky4'] if not memory else C['msky3']
    return out


# ---------------------------------------------------------------- 背景：雪山（视差 0.25）
def _ridge(n, rng, rough, base, amp):
    h = np.zeros(n)
    step = 64
    pts = base + rng.uniform(-amp, amp, n // step + 2)
    for i in range(n):
        j, f = divmod(i, step)
        k = f / step
        h[i] = pts[j] * (1 - k) + pts[j + 1] * k
    h += np.cumsum(rng.normal(0, rough, n)) * 0.0
    for s, a in ((16, amp * 0.25), (5, amp * 0.08)):
        ph = rng.uniform(0, 6.28)
        h += a * np.sin(np.arange(n) / s + ph) * rng.uniform(0.6, 1.0)
    return h


@lru_cache(None)
def _mountains():
    w, h = 720, 120
    L = Layer(w, h)
    rng = np.random.default_rng(11)
    top = _ridge(w, rng, 1.0, 62, 22)
    top = top + 22 * np.exp(-((np.arange(w) - 360) / 50) ** 2)           # 夕阳所在的山口更低
    yy = np.arange(h)[:, None]
    m = yy >= top[None, :]
    L.put(m, C['mt'])
    snow = m & (yy < top[None, :] + 14 + 6 * np.sin(np.arange(w) / 7)[None, :])
    L.put(snow & _dith(0.7, (h, w)), C['mt_snow'])
    rimm = m & ~np.roll(m, 1, 0)
    L.put(rimm, C['mt_rim'])
    top2 = _ridge(w, np.random.default_rng(12), 1.0, 92, 10)
    m2 = yy >= top2[None, :]
    trees = np.zeros_like(m2)
    for x in range(0, w, 5):                                              # 针叶林的小三角
        t = int(top2[x]) - rng.integers(3, 8)
        for k in range(8):
            trees[max(0, t + k), max(0, x - k // 3):min(w, x + k // 3 + 1)] = True
    m2 |= trees & (yy < top2[None, :] + 2)
    L.put(m2, C['mt2'])
    L.put(m2 & ~np.roll(m2, 1, 0) & _dith(0.6, (h, w)), C['mt2_rim'])
    return L


# ---------------------------------------------------------------- 背景：哈梅尔的房屋（视差 0.6）
RUIN_W, RUIN_H = 800, 150
TOWER_X = 402                       # 钟楼在废墟层中的 x（中心）
HOUSES = [  # (x, 宽, 墙高, 屋顶高, 种子)
    (30, 44, 30, 18, 1), (92, 36, 24, 15, 2), (150, 52, 34, 20, 3), (222, 40, 26, 16, 4),
    (290, 34, 22, 14, 5), (470, 46, 30, 18, 6), (536, 38, 26, 15, 7), (600, 56, 32, 20, 8),
    (680, 40, 24, 15, 9), (740, 44, 28, 17, 10),
]
GROUND_Y = 132                      # 废墟层里的地平线


def _bricks(L, mask, dark):
    yy, xx = np.nonzero(mask)
    row = yy // 4
    hor = (yy % 4) == 0
    ver = ((xx + row * 4) % 8) == 0
    sel = hor | ver
    L.rgb[yy[sel], xx[sel]] = dark


def _house(L, x, w, wall, roof, seed, intact):
    rng = np.random.default_rng(seed)
    h, W_ = L.h, L.w
    base = GROUND_Y + rng.integers(-3, 3)
    y0 = base - wall
    body = np.zeros((h, W_), bool)
    body[y0:base, x:x + w] = True
    if intact:
        gable = _poly(W_, h, [(x - 3, y0), (x + w // 2, y0 - roof), (x + w + 3, y0)])
        L.put(body, C['ist2'])
        _bricks(L, body, C['ist1'])
        L.put(gable, C['roof'])
        L.put(gable & ~np.roll(gable, 1, 0) | gable & ~np.roll(gable, 2, 0), C['roof2'])
        for k in range(1, w // 12 + 1):
            wx = x + k * w // (w // 12 + 1) - 2
            win = np.zeros_like(body)
            win[y0 + 8:y0 + 14, wx:wx + 4] = True
            L.put(win, C['lamp'])
        L.put(body & ~np.roll(body, 1, 1), C['ist3'])
        return
    # 废墟：墙顶参差、屋顶只剩焦黑的梁
    prof = np.cumsum(rng.integers(-2, 3, w)) * 0.8
    prof = np.clip(prof - prof.min(), 0, wall * 0.6)
    gable_h = np.maximum(0, roof * (1 - np.abs(np.arange(w) - w / 2) / (w / 2))) * rng.uniform(0.2, 0.7)
    tops = (y0 + prof - gable_h).astype(int)
    for i in range(w):
        body[:, x + i] = False
        body[max(0, tops[i]):base, x + i] = True
    gap = rng.integers(w // 4, w // 2)
    body[y0 - roof:y0 + wall // 3 + rng.integers(0, 6), x + gap:x + gap + rng.integers(4, 9)] = False
    L.put(body, C['st2'])
    _bricks(L, body, C['st1'])
    for k in range(1, w // 12 + 1):                               # 黑洞洞的窗
        wx = x + k * w // (w // 12 + 1) - 2
        win = np.zeros_like(body)
        win[y0 + 8:y0 + 14, wx:wx + 4] = True
        win[y0 + 7, wx + 1:wx + 3] = True
        L.put(win & body, C['win'])
        soot = np.zeros_like(body)
        soot[y0 + 1:y0 + 8, wx - 1:wx + 5] = True
        L.put(soot & body & _dith(0.45, (h, W_)), C['st1'])
    for _ in range(2):                                            # 焦黑的梁
        bx = x + rng.integers(4, w - 4)
        by = int(tops[bx - x])
        L.put(_poly(W_, h, [(bx, by + 2), (bx + 1, by + 2), (bx + rng.integers(-8, 9), by - rng.integers(6, 12))]),
              C['char'])
    top = body & ~np.roll(body, 1, 0)
    L.put(top, C['rim'])
    L.put(top & np.roll(top, 1, 1) & _dith(0.35, (h, W_)), C['snow2'])
    L.put(body & ~np.roll(body, -1, 1) & (np.arange(h)[:, None] < base - 2), C['st1'])


def _chapel(L, intact):
    h, W_ = L.h, L.w
    base = GROUND_Y
    x0 = TOWER_X + 10
    nave = np.zeros((h, W_), bool)
    nave[base - 36:base, x0:x0 + 62] = True
    tw = np.zeros((h, W_), bool)
    tw[base - 96:base, TOWER_X - 11:TOWER_X + 11] = True
    if intact:
        L.put(nave, C['ist2'])
        _bricks(L, nave, C['ist1'])
        L.put(_poly(W_, h, [(x0 - 2, base - 36), (x0 + 31, base - 58), (x0 + 64, base - 36)]), C['roof'])
        L.put(tw, C['ist2'])
        _bricks(L, tw, C['ist1'])
        L.put(_poly(W_, h, [(TOWER_X - 13, base - 96), (TOWER_X, base - 124), (TOWER_X + 13, base - 96)]), C['roof'])
        arch = _poly(W_, h, [(TOWER_X - 5, base - 70), (TOWER_X - 5, base - 84), (TOWER_X, base - 89),
                             (TOWER_X + 5, base - 84), (TOWER_X + 5, base - 70)])
        L.put(arch, C['st1'])
        L.put(_poly(W_, h, [(TOWER_X - 3, base - 72), (TOWER_X - 2, base - 80), (TOWER_X + 2, base - 80),
                            (TOWER_X + 3, base - 72)]), C['bell'])
        for wy in (base - 26, base - 50):
            win = np.zeros_like(tw)
            win[wy:wy + 8, TOWER_X - 2:TOWER_X + 2] = True
            L.put(win, C['lamp'])
        L.put(tw & ~np.roll(tw, 1, 1), C['ist3'])
        return
    # 废墟：钟楼上部斜着断掉，钟楼的拱窗只剩半边，中殿屋顶塌空
    cut = _poly(W_, h, [(TOWER_X - 12, base - 104), (TOWER_X + 12, base - 104), (TOWER_X + 12, base - 88),
                        (TOWER_X + 5, base - 92), (TOWER_X - 1, base - 84), (TOWER_X - 7, base - 90),
                        (TOWER_X - 12, base - 80)])
    tw &= ~cut
    rng = np.random.default_rng(21)
    prof = np.clip(np.cumsum(rng.integers(-2, 3, 62)), -6, 10)
    for i in range(62):
        nave[:base - 36 + 6 + prof[i], x0 + i] = False
    nave[base - 30:base - 12, x0 + 20:x0 + 30] = False               # 塌掉的一截墙
    L.put(nave, C['st2'])
    _bricks(L, nave, C['st1'])
    L.put(tw, C['st3'])
    _bricks(L, tw, C['st2'])
    arch = _poly(W_, h, [(TOWER_X - 5, base - 62), (TOWER_X - 5, base - 76), (TOWER_X, base - 81),
                         (TOWER_X + 5, base - 76), (TOWER_X + 5, base - 62)])
    L.put(arch & tw, C['win'])
    for wy in (base - 26, base - 46):
        win = np.zeros_like(tw)
        win[wy:wy + 8, TOWER_X - 2:TOWER_X + 2] = True
        L.put(win, C['win'])
    L.put(tw & ~np.roll(tw, 1, 1), C['rim'])                       # 夕阳一侧的轮廓光
    L.put((tw | nave) & ~np.roll(tw | nave, 1, 0), C['rim2'])
    # 摔在地上的钟
    L.put(_poly(W_, h, [(x0 + 36, base), (x0 + 38, base - 7), (x0 + 43, base - 8), (x0 + 46, base)]), C['st3'])
    L.put(_poly(W_, h, [(x0 + 38, base - 6), (x0 + 42, base - 7), (x0 + 43, base - 5)]), C['rim'])


@lru_cache(None)
def _ruins(intact=False):
    L = Layer(RUIN_W, RUIN_H)
    for hs in HOUSES:
        _house(L, *hs, intact=intact)
    _chapel(L, intact)
    yy = np.arange(RUIN_H)[:, None]
    xs = np.arange(RUIN_W)
    hill = GROUND_Y - 2 + 3 * np.sin(xs / 23) + 2 * np.sin(xs / 7.3)
    g = yy >= hill[None, :]
    L.put(g, C['g0'] if not intact else hexc('#3a4a2a'))
    L.put(g & ~np.roll(g, 1, 0), C['rim'] if not intact else hexc('#7a9a4a'))
    if not intact:                                                 # 瓦砾堆
        rng = np.random.default_rng(31)
        for _ in range(40):
            cx, r = rng.uniform(0, RUIN_W), rng.uniform(2, 6)
            m = (np.hypot(xs[None, :] - cx, (yy - GROUND_Y) * 1.6) < r) & (yy <= GROUND_Y + 1)
            L.put(m, C['st1'])
            L.put(m & ~np.roll(m, 1, 0), C['st3'])
    return L


# ---------------------------------------------------------------- 近景地面（视差 1.0）+ 墓
GROUND_W, GROUND_H = 1000, 110


@lru_cache(None)
def _ground():
    L = Layer(GROUND_W, GROUND_H)
    rng = np.random.default_rng(41)
    xs = np.arange(GROUND_W)
    top = 18 + 3 * np.sin(xs / 41) + 2 * np.sin(xs / 13 + 1)
    yy = np.arange(GROUND_H)[:, None]
    m = yy >= top[None, :]
    L.put(m, C['g1'])
    L.put(m & (yy > top[None, :] + 30) & _dith(0.5, (GROUND_H, GROUND_W)), C['g0'])
    L.put(m & (yy > top[None, :] + 50), C['g0'])
    # 雪斑（上面亮、下面灰）
    for _ in range(26):
        cx, cy = rng.uniform(0, GROUND_W), rng.uniform(22, 90)
        rx, ry = rng.uniform(5, 16), rng.uniform(1.0, 2.2)
        e = (((xs[None, :] - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) < 1
        L.put(e & m & _dith(0.6, (GROUND_H, GROUND_W)), C['g2'])
        L.put(e & m & ~np.roll(e, 1, 0) & _dith(0.7, (GROUND_H, GROUND_W)), C['snow'])
    # 枯草与碎石
    for _ in range(160):
        x = int(rng.uniform(0, GROUND_W))
        y = int(top[x] + rng.uniform(1, 60))
        hgt = rng.integers(2, 5)
        L.rgb[max(0, y - hgt):y, x] = C['grass']
        L.m[max(0, y - hgt):y, x] = True
    for _ in range(50):
        cx, cy, r = rng.uniform(0, GROUND_W), rng.uniform(24, 80), rng.uniform(1.5, 3.5)
        e = np.hypot(xs[None, :] - cx, (yy - cy) * 1.5) < r
        L.put(e, C['st2'])
        L.put(e & ~np.roll(e, 1, 0), C['st3'])
    L.put(m & ~np.roll(m, 1, 0), C['rim'])
    return L


@lru_cache(None)
def _grave():
    """简单的木十字墓碑 + 几朵小花（画在 1x，贴图时放大 2 倍，与角色同一像素尺度）。"""
    img = np.zeros((24, 18, 4), np.float32)

    def p(x, y, col):
        if 0 <= y < 24 and 0 <= x < 18:
            img[y, x, :3] = C[col]
            img[y, x, 3] = 1
    for y in range(2, 20):
        p(8, y, 'wood')
        p(9, y, 'wood2')
    for x in range(4, 14):
        p(x, 6, 'wood')
        p(x, 7, 'wood2')
    p(8, 2, 'rim')
    for x in range(1, 17):                         # 小土丘 + 雪
        for y in range(19, 24):
            if abs(x - 8.5) < 8 - (23 - y) * 0.6 + 3:
                p(x, y, 'g2' if y > 20 else 'snow')
    for x, y, c in ((4, 18, 'fl_w'), (5, 17, 'fl_w'), (6, 18, 'fl_y'), (11, 18, 'fl_w'), (12, 17, 'fl_y'),
                    (13, 18, 'fl_w'), (7, 18, 'fl_w')):
        p(x, y, c)
        p(x, y + 1, 'stem')
    out = img.copy()
    a = img[..., 3] > 0
    g = a.copy()
    g[1:] |= a[:-1]
    g[:-1] |= a[1:]
    g[:, 1:] |= a[:, :-1]
    g[:, :-1] |= a[:, 1:]
    out[g & ~a, :3] = hexc('#140c10')
    out[g & ~a, 3] = 1
    return out


# ---------------------------------------------------------------- 背景合成
def _cut_line(x):
    """斩线（屏幕坐标）：从右上到左下的一条斜线，穿过钟楼中上部。"""
    return 58 + (470 - x) * 0.30


def world(dst, cam, cy=0, memory=0.0, split=0.0, glow=0.0, fire=None, tower_fall=None, lt=0.0):
    """画背景：天空 → 雪山 → 废墟（可溶解成回忆里的哈梅尔、可被斩开）→ 远处的火 → 地面。"""
    sky = _sky(False)
    if memory > 0:
        dst[:] = np.where(_dith(memory)[..., None], _sky(True), sky)
    else:
        dst[:] = sky
    _mountains().draw(dst, -cam * 0.25 - 60, 178 - 120 + cy * 0.3)
    rx, ry = -cam * 0.6, 212 - RUIN_H + cy * 0.7
    R = _ruins(False)
    if memory > 0:
        m_old = ~_dith(memory)
        R.draw(dst, rx, ry, m_old)
        _ruins(True).draw(dst, rx, ry, ~m_old)
    elif split > 0 or tower_fall is not None:
        above = YY < _cut_line(XX)
        tx = TOWER_X + rx
        tower_box = above & (np.abs(XX - tx) < 16)
        # 斩线以下原地不动；以上沿斩线方向滑开 split 像素（钟楼单独处理）
        R.draw(dst, rx, ry, ~above)
        d = np.array([-1.0, 0.30]) / np.hypot(1, 0.3) * split
        R.draw(dst, rx + d[0], ry + d[1], above & ~tower_box
               if tower_fall is not None else above)
        if tower_fall is not None:
            fx_, fy_ = tower_fall
            sh = np.roll(np.roll(tower_box, int(round(fy_ + d[1])), 0), int(round(fx_ + d[0])), 1)
            R.draw(dst, rx + d[0] + fx_, ry + d[1] + fy_, sh & (YY < 206 + cy * 0.7))
    else:
        R.draw(dst, rx, ry)
    if glow > 0:                                                     # 斩线上的炽热细缝
        ruin_band = R.screen_mask(rx, ry)
        line = (np.abs(YY - _cut_line(XX)) < 1.0) & ruin_band
        dst[line] = dst[line] * (1 - glow) + hexc('#fff0b0') * glow
        halo = (np.abs(YY - _cut_line(XX)) < 3.5) & ruin_band & _dith(glow * 0.6)
        dst[halo] = dst[halo] * 0.4 + hexc('#ff7a2a') * 0.6
    if fire is not None:
        fire.add_to(dst, 0, int(206 + cy * 0.7) - fire.h, gain=0.8)
    _ground().draw(dst, -cam, 180 + cy)


def _tint(dst, col, k):
    if k > 0:
        dst *= (1 - 0.45 * k)
        dst += np.asarray(col, np.float32) * 0.45 * k


# ---------------------------------------------------------------- 角色
def _rim(img, col, k=1.0, sides='lrt'):
    """给精灵加一圈朝向光源的轮廓光（逆光 / 火光）。"""
    a = img[..., 3] > 0
    e = np.zeros_like(a)
    if 'l' in sides:
        e |= a & ~np.roll(a, 1, 1)
    if 'r' in sides:
        e |= a & ~np.roll(a, -1, 1)
    if 't' in sides:
        e |= a & ~np.roll(a, 1, 0)
    # 只描真正的外缘（不描描边像素本身，而是往里一格）
    inner = a & ~e
    e2 = inner & ~(np.roll(inner, 1, 1) & np.roll(inner, -1, 1) & np.roll(inner, 1, 0))
    out = img.copy()
    out[e2, :3] = out[e2, :3] * (1 - 0.7 * k) + np.asarray(col, np.float32) * 0.7 * k
    return out


def _char(ctx, pose, x, y, s=2, rim=None, rim_k=1.0, sides='lrt', alpha=1.0, tint=None, tint_amt=0.0, sil=None,
          dst=None, dark=1.0):
    """按脚底根点 (x, y) 贴角色；返回剑的握把 / 剑尖屏幕坐标。"""
    img = LH.body(pose)
    info = LH.pose_info(pose)
    rx, ry = info['root']
    if sil is not None:
        img = img.copy()
        img[img[..., 3] > 0, :3] = sil
    else:
        if dark != 1.0:                                           # 逆光：整体压暗，只留轮廓光
            img = img.copy()
            img[..., :3] *= dark
        if rim is not None:
            img = _rim(img, rim, rim_k, sides)
    if dst is not None:
        blit(dst, img, x - s * rx, y - s * ry, scale=s, alpha=alpha, tint=tint, tint_amt=tint_amt)

    def scr(p):
        return None if p is None else (x + s * (p[0] - rx), y + s * (p[1] - ry))
    return scr(info['hand']), scr(info['tip'])


def _blade_fire(ctx, hand, tip, n=6, k=1.0, vx=0.0):
    """鬼火：沿剑身喷出的火舌 + 从握剑的手里冒出的火。"""
    if hand is None:
        return
    P = ctx.P
    u = ctx.rng.uniform(0.15, 1.0, n)
    x = hand[0] + (tip[0] - hand[0]) * u + ctx.rng.normal(0, 1.2, n)
    y = hand[1] + (tip[1] - hand[1]) * u + ctx.rng.normal(0, 1.2, n)
    P.emit_arrays(x, y, ctx.rng.normal(vx, 14, n), -ctx.rng.uniform(30, 90, n) * k, (0.18, 0.45), 'fire',
                  size=(1, 2), drag=1.5, wob=0.2)
    P.emit(max(1, n // 3), hand[0], hand[1], (-12, 12), (-60, -20), (0.2, 0.4), 'fire', size=(1, 2), drag=1.0)


# ---------------------------------------------------------------- 斩击轨迹（来自真实的剑姿势）
SLASH_KEYS = ['slash1', 'slash2', 'slash3']
BIG_S, BIG_TO = 1.9, (262, 226)     # 巨型新月：同一条挥剑轨迹放大 1.9 倍，挪到画面中下方，横跨全屏


def _slash_path(x1, x2, x3, y, s=2, n=36):
    """把 slash1→slash2→slash3 三帧的 (握把, 剑尖) 连成一条连续的挥剑轨迹（按角度插值，经过头顶）。
    返回 [(握把x, 握把y, 角度, 剑长)]。"""
    keys = []
    for pose, x in zip(SLASH_KEYS, (x1, x2, x3)):
        info = LH.pose_info(pose)
        rx, ry = info['root']
        hx, hy = x + s * (info['hand'][0] - rx), y + s * (info['hand'][1] - ry)
        tx, ty = x + s * (info['tip'][0] - rx), y + s * (info['tip'][1] - ry)
        keys.append((hx, hy, np.arctan2(ty - hy, tx - hx), np.hypot(tx - hx, ty - hy)))
    # 角度单调递减（从右上经头顶扫到左下）
    a = [keys[0][2]]
    for k in keys[1:]:
        v = k[2]
        while v > a[-1]:
            v -= 2 * np.pi
        a.append(v)
    out = []
    for i in range(n):
        u = i / (n - 1) * 2
        j = min(int(u), 1)
        f = u - j
        lerp = lambda p, q: p + (q - p) * f
        out.append((lerp(keys[j][0], keys[j + 1][0]), lerp(keys[j][1], keys[j + 1][1]), lerp(a[j], a[j + 1]),
                    lerp(keys[j][3], keys[j + 1][3])))
    return out


def _crescent(dst, path, u1, scale=1.0, center=None, fadev=1.0, thick=0.42, sil=None, u0=0.0, to=None):
    """火焰新月：沿挥剑轨迹，在「剑身内侧 → 剑尖」之间填色；可以绕某点放大（巨型版）。"""
    n = len(path)
    i1 = max(2, int(u1 * (n - 1)) + 1)
    i0 = int(u0 * (n - 1))
    seg = path[i0:i1]
    if len(seg) < 2 or fadev <= 0:
        return
    cx, cy = center if center is not None else (0, 0)
    tx, ty = to if to is not None else (cx, cy)
    bands = [(0.0, 1.0, '#c01a1a'), (0.25, 1.0, '#ff5a1a'), (0.55, 1.0, '#ffc45a'), (0.8, 1.0, '#fff4c0')]
    for f0, f1, col in bands:
        outer, inner = [], []
        for k, (hx, hy, ang, L) in enumerate(seg):
            u = (i0 + k) / (n - 1)
            w = thick * np.sin(np.pi * np.clip(u, 0.02, 0.98)) ** 0.6
            r_out = L * (1.0 + 0.04)
            r_in = L * (1.0 - w * (1 - f0))
            ox, oy = hx + np.cos(ang) * r_out, hy + np.sin(ang) * r_out
            ix, iy = hx + np.cos(ang) * r_in, hy + np.sin(ang) * r_in
            if center is not None:
                ox, oy = tx + (ox - cx) * scale, ty + (oy - cy) * scale
                ix, iy = tx + (ix - cx) * scale, ty + (iy - cy) * scale
            outer.append((ox, oy))
            inner.append((ix, iy))
        m = fx.polygon_mask(outer + inner[::-1])
        if sil is not None:
            dst[m] = sil
            return
        m &= _dith(fadev)
        dst[m] = hexc(col)


# ---------------------------------------------------------------- 各镜头
def _shot_a(dst, ctx, lt):
    """1–2 小节：废墟横摇 → 背影 → 回头 → 回忆闪回。"""
    k = smooth(0, BAR, lt)
    cam = 300 * (k * k * (3 - 2 * k)) if lt < BAR else 300 + 40 * (lt - BAR) / BAR
    mem = 0.0
    m0 = BAR + 1.5 * BEAT
    if lt > m0:
        mem = min(1.0, (lt - m0) / 0.22) * (1 - smooth(m0 + 0.62, m0 + 0.95, lt))
    world(dst, cam, 0, memory=mem, lt=lt)
    gx = 700 - cam
    blit(dst, _grave(), gx - 18, 226 - 48, scale=2)
    if lt < BAR + BEAT:
        pose = LH.ANIM['wind'][int(lt * 6) % 4]
    else:
        pose = 'back_turn'
    rim = hexc('#ffb070') if mem < 0.5 else hexc('#fff0c0')
    _char(ctx, pose, 640 - cam, 238, 2, rim=rim, rim_k=0.6, sides='lt', dst=dst, dark=0.72 + 0.28 * mem)
    if mem > 0:                                                    # 回忆：暖光溢满
        m = _dith(mem * 0.35)
        dst[m] = dst[m] * 0.5 + hexc('#ffd890') * 0.5


def _shot_b(dst, ctx, lt):
    """3–5 小节：转身按剑 → 拔剑入架势 → 冲刺斩击 → 收势。"""
    P = ctx.P
    st = ctx.state
    cam = 420 + 10 * min(1.0, (lt - T_B) / (T_DRAW - T_B))
    cy = -14
    x_rest = 800 - cam
    fy = 194
    # ---- 姿势与位置
    if lt < T_DRAW:
        pose, x = LH.ANIM['rest'][int(lt * 5) % 4], x_rest
    elif lt < T_DRAW + 0.085:
        pose, x = 'draw1', x_rest
    elif lt < T_DRAW + 0.17:
        pose, x = 'draw2', x_rest
    elif lt < T_IMP - 0.125:
        pose, x = 'stance', x_rest + 2
    elif lt < T_IMP - 0.042:
        f = (lt - (T_IMP - 0.125)) / 0.083
        pose, x = 'dash', x_rest - (x_rest - 232) * f ** 1.6
    elif lt < T_IMP:
        pose, x = 'slash1', 222
    elif lt < T_IMP + 0.135:
        pose, x = 'slash2', 206
    elif lt < T_IMP + 0.42:
        pose, x = 'slash3', 206 - 16 * ease_out((lt - T_IMP - 0.135) / 0.28)
    else:
        pose, x = 'recover', 190
    heat = smooth(T_DRAW, T_IMP, lt) * (1 - 0.4 * smooth(T_IMP + 0.3, T_D, lt))
    # ---- 背景：斩开 + 火
    split = 6 * smooth(T_SEC, T_SEC + 0.3, lt) if lt > T_SEC else 0.0
    glow = smooth(T_IMP + 0.1, T_IMP + 0.25, lt) * (1 - 0.6 * smooth(T_SEC + 0.2, T_SEC + 0.8, lt)) if lt > T_IMP else 0
    fire = None
    if lt > T_SEC:
        fire = ctx.fire('hamel', ui.fire_palette('#ff4a2a'), W, 70, 3)
        src = np.zeros(W, np.int32)
        band = (np.arange(W) % 37 < 26)
        src[band] = int(36 * smooth(T_SEC, T_SEC + 0.2, lt))
        fire.step(src, decay=1.6, wind=-1, sub=2)
    world(dst, cam, cy, split=split, glow=glow, fire=fire, lt=lt)
    blit(dst, _grave(), 880 - cam - 18, 186 - 48, scale=2)
    _tint(dst, hexc('#ff5020'), heat * 0.55)
    # ---- 拔剑：火星与碎石
    if ctx.crossed(4, 3.0):
        hx, hy = x_rest + 2 * (14.5 - 32), fy
        P.emit(40, (hx - 4, hx + 4), hy - 2, (-90, 90), (-200, -40), (0.2, 0.6), 'gold', size=(1, 2), grav=500)
        P.emit(18, (hx - 6, hx + 6), hy, (-60, 60), (-120, -30), (0.3, 0.7), 'white', size=2, grav=400, bright=0.45)
        ctx.shake(3, 0.2)
    # ---- 冲刺：残影 + 拖影 + 速度线
    tr = st.setdefault('trail', [])
    if pose in ('dash', 'slash1'):
        fx.speed_lines(dst, lt * 3, hexc('#ff8a4a'), direction=-1, alpha=0.5)
        for i, (pp, px) in enumerate(tr[-4:][::-1]):
            _char(ctx, pp, px + 6 * (i + 1), fy, 2, alpha=0.55 - 0.12 * i, tint=(1.0, 0.35, 0.15), tint_amt=0.8,
                  dst=dst)
    tr.append((pose, x))
    # ---- 冲击帧（两帧反相）+ 定帧
    impact = T_IMP <= lt < T_IMP + 0.06
    path = _slash_path(222, 206, 190, fy)
    hand_imp = path[len(path) // 2][:2]
    if impact:
        dst[:] = hexc('#fff2d0')
        _crescent(dst, path, 1.0, scale=BIG_S, center=hand_imp, to=BIG_TO, sil=hexc('#120606'))
        _char(ctx, pose, x, fy, 2, sil=hexc('#120606'), dst=dst)
        if ctx.crossed(5, 0.0):
            ctx.shake(9, 0.5)
            P.emit_radial(160, hand_imp[0], hand_imp[1], (60, 320), (0.3, 0.9), 'fire', size=(1, 3), drag=2.2)
            P.emit_radial(60, hand_imp[0], hand_imp[1], (40, 200), (0.4, 1.0), 'white', size=1, drag=2)
        return pose, x, fy
    # ---- 火焰新月：挥剑中是真实大小的拖光，命中后放大成横跨全屏的新月
    if T_IMP - 0.042 <= lt < T_IMP:
        _crescent(dst, path, 0.5 * (lt - (T_IMP - 0.042)) / 0.042)
    if lt >= T_IMP + 0.06:
        age = lt - T_IMP - 0.06
        fv = 1 - smooth(0.12, 0.6, age)
        if fv > 0:
            _crescent(dst, path, min(1.0, 0.55 + age * 3), scale=BIG_S - 0.12 * smooth(0, 0.4, age), center=hand_imp,
                      to=BIG_TO, fadev=fv)
            _crescent(dst, path, min(1.0, 0.55 + age * 4), fadev=fv)
        if age < 0.3:
            fx.radial_lines(dst, hand_imp[0], hand_imp[1], lt, hexc('#fff0c0'), n=26, seed=4)
    # ---- 5.2 余波：斩线处爆开
    if ctx.crossed(5, 2.0):
        ctx.shake(5, 0.4)
        xs = np.linspace(0, W, 90)
        P.emit_arrays(xs, _cut_line(xs), ctx.rng.normal(-30, 40, 90),
                      -ctx.rng.uniform(40, 160, 90), (0.4, 1.1), 'fire', size=(1, 3), drag=1.2, grav=60)
    if lt > T_SEC:
        xs = ctx.rng.uniform(0, W, 3)
        P.emit_arrays(xs, np.full(3, 150.0), ctx.rng.normal(-10, 10, 3), -ctx.rng.uniform(20, 50, 3), (1.0, 2.0),
                      'fire', size=1, wob=0.3, bright=0.8)
    # ---- 角色
    rim = hexc('#ff7a3a')
    hand, tip = _char(ctx, pose, x, fy, 2, rim=rim, rim_k=0.35 + 0.65 * heat, sides='lrt', dst=dst)
    if lt > T_DRAW + 0.1:
        _blade_fire(ctx, hand, tip, n=8 if pose in ('stance', 'dash') else 5, vx=60 if pose == 'dash' else 0)
    return pose, x, fy


def _shot_d(dst, ctx, lt):
    """6 小节：终幕 —— 背影垂剑，被斩断的钟楼滑落砸进火海。"""
    P = ctx.P
    a = lt - T_D
    cam = 384 + 10 * a / BAR
    fire = ctx.fire('hamel_d', ui.fire_palette('#ff4a2a'), W, 60, 5)
    src = np.where((np.arange(W) // 14 * 7919) % 5 < 3, 34, 8).astype(np.int32)
    fire.step(src, decay=1.3, wind=-1, sub=2)
    fall = None
    if a >= 0:
        f = min(a, 0.45)
        g = max(0.0, f - 0.18)
        fall = (-26 * f - 60 * g, 12 * f + 900 * g * g)
    world(dst, cam, 8, split=6, glow=0.25, fire=fire, tower_fall=fall, lt=lt)
    _tint(dst, hexc('#ff4a1a'), 0.35)
    if ctx.crossed(6, 1.0):
        tx = TOWER_X - cam * 0.6 - 40
        P.emit(70, (tx - 30, tx + 30), (170, 190), (-80, 80), (-120, -20), (0.6, 1.4), 'white', size=2, drag=1.5,
               bright=0.35)
        P.emit(90, (tx - 30, tx + 30), (170, 190), (-100, 100), (-220, -60), (0.5, 1.2), 'fire', size=(1, 2), drag=1)
        ctx.shake(4, 0.4)
    for _ in range(2):
        P.emit(1, (0, W), (140, 200), (-10, 10), (-40, -15), (1.2, 2.2), 'fire', size=1, wob=0.3, bright=0.9)
    hand, tip = _char(ctx, 'back_sword', 250, 292, 3, rim=hexc('#ffb060'), rim_k=0.7, sides='rt', dst=dst, dark=0.58)
    _blade_fire(ctx, hand, tip, n=5, k=0.8)


# ---------------------------------------------------------------- 入口
def render(dst, ctx):
    lt = ctx.lt
    P, P2 = ctx.P, ctx.P2
    if lt < T_B:
        _shot_a(dst, ctx, lt)
        if lt > BAR:
            ui.name_tag(dst, ctx.C['leonhardt'], lt - BAR)
    elif lt < T_D:
        _shot_b(dst, ctx, lt)
    else:
        if ctx.crossed(6, 0.0):
            ctx.P.clear()
            ctx.shake(6, 0.35)
        _shot_d(dst, ctx, lt)
    # 灰烬 / 细雪（全程），技能之后夹着火星
    P2.emit(2, (-20, W + 40), -4, (-18, -6), (14, 30), (6, 9), 'white', size=1, wob=0.25, bright=0.5)
    P2.step(ctx.dt)
    P2.render(dst)
    if not (T_IMP <= lt < T_IMP + 0.06):
        P.step(ctx.dt if not (T_IMP + 0.06 <= lt < T_IMP + 0.135) else ctx.dt * 0.15)   # 定帧时粒子也几乎停住
        P.render(dst)
    np.clip(dst, 0, 1, out=dst)
    return dst
