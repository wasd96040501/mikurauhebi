"""盟主：结社「噬身之蛇」隐秘的圣所。4 小节 @144（约 6.67 s），E 小调，管风琴 + 合唱 + 管钟，全片最高潮。

分镜（小节.拍，本段内）
  1.0  SLAM 硬切：阶梯脚下的低角度仰拍，大殿一片昏暗；第一对光柱落下，照亮两把空着的「使徒」之座
  1.1–1.3  镜头沿长阶缓缓上摇，每一拍点亮一对光柱（更高处的空座），尘埃在光里浮动；
           阶顶的人影仍是一团黑色剪影
  2.0  镜头停在阶顶：玫瑰窗亮起，盟主被逆光勾出金边，随后显出本色；头发与披风在圣光里飘动；
       祭坛两侧的光柱点亮；「盟 主」字样
  2.x  缓慢推近
  3.0  台词（字幕）；继续推近到胸像，管钟每响一次玫瑰窗脉动一次
  4.0  硬切回全景：她开始举手（预备 -> 抬起 -> 翻掌），一条巨大的衔尾蛇从窗后绕出、头追着尾巴画圆；
       玫瑰窗越烧越亮，金光漫过大殿
  4.3  最后一拍：蛇咬住自己的尾巴，环闭合；手举到最高点，白金闪光 + 震屏 —— 顶点，随后硬切「总攻」

镜头与构图：静态层全部 lru_cache，按视差裁切（远墙 0.55 / 柱廊 0.8 / 阶梯 1.0 / 前景巨柱 1.3），
最后整体最近邻缩放做推近。
"""
from functools import lru_cache

import numpy as np

import fx
import i18n
from gfx import BAYER as BAYER_, H, W, XX, YY, blit, clamp01, dilate, dither_mask, ease_back, ease_out, hexc, smooth, text
import sfx as S

import chars.grandmaster as GM

LETTERBOX = 24
MUSIC_SHAKE = False        # 震屏全部由本场景控制（4.3 的顿帧要静止）
SAYS = [('grandmaster', 3, 0.0, 'sub')]

# 光柱点亮的拍点（小节, 拍）与对应的阶梯平台高度（阶梯层 y）
IGNITE = [((1, 0.0), 532), ((1, 1.0), 440), ((1, 2.0), 346), ((1, 3.0), 258)]
ALTAR_IGNITE = (2, 0.0)
BELLS = [(1, 0.0), (1, 3.0), (2, 0.0), (2, 2.0), (3, 0.0), (4, 0.0)]   # 管钟（动机扩大）落点：玫瑰窗脉动

CUES = [
    (1, 0.0, 'big_impact', 0.0), (1, 0.0, 'gm_ignite', -2.0), (1, 0.05, 'shimmer', -8.0),
    (1, 1.0, 'gm_ignite', -5.0, -0.3), (1, 2.0, 'gm_ignite', -4.0, 0.3), (1, 3.0, 'gm_ignite', -3.0, -0.2),
    (1, 1.5, 'wind', -14.0),
    (2, 0.0, 'gm_ignite', -1.0), (2, 0.0, 'gm_toll', -2.0), (2, 0.02, 'chime', -8.0),
    (3, 0.0, 'gm_toll', -6.0), (3, 2.0, 'wind', -14.0),
    (4, 0.0, 'gm_swell', -3.0), (4, 0.0, 'gm_ring', -4.0), (4, 1.0, 'swish', -10.0),
    (4, 3.0, 'big_impact', -1.0), (4, 3.0, 'gm_toll', -2.0), (4, 3.0, 'shimmer', -4.0),
]
ACCENTS = [(2, 0.0, 'hit'), (4, 0.0, 'drop'), (4, 3.0, 'hit')]

# ---------------------------------------------------------------- 世界坐标
CAM0 = 330          # 1.0 时镜头顶端在阶梯层的 y（2.0 以后为 0）
PAR = {'far': 0.55, 'mid': 0.8, 'stair': 1.0, 'fg': 1.3}
WIN = (240, 96, 98)  # 玫瑰窗（远墙层坐标）：圆心、半径
FEET = (240, 205)   # 盟主脚底（阶梯层）
TOP_Y, BOT_Y = 212, 560   # 阶梯顶 / 底


def _lh(k, cam):
    return int(round(H + CAM0 * PAR[k])) + 2


def _c(h):
    return hexc(h)


def _poly(pts, h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    px, py = xx + .5, yy + .5
    m = np.zeros((h, w), bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        c = (y1 > py) != (y2 > py)
        xi = (x2 - x1) * (py - y1) / (y2 - y1) + x1
        m ^= c & (px < xi)
    return m


def _ring(m):
    g = m.copy()
    g[1:] |= m[:-1]
    g[:-1] |= m[1:]
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    return g & ~m


def _vg(h, top, bot, w=W):
    t = np.linspace(0, 1, h)[:, None, None]
    return np.broadcast_to(_c(top) * (1 - t) + _c(bot) * t, (h, w, 3)).astype(np.float32).copy()


# ---------------------------------------------------------------- 远墙：玫瑰窗 + 尖拱 + 侧窗
GLASS = {  # 名称: (暗, 亮)
    'B': ('#1a2656', '#4a78e0'), 'D': ('#10163a', '#2a3c9a'), 'V': ('#2e1850', '#8a48d0'),
    'R': ('#4a1020', '#d83848'), 'G': ('#4a3812', '#f0c050'), 'W': ('#2a2440', '#c8d8ff'),
}


def _window(img, lit, cx, cy, R):
    """哥特玫瑰窗：中心纹章圆章、12 片花瓣、24 条外圈尖窗（每条一枚红 / 金小圆窗），石质窗棂与铅条。"""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    X, Y = xx + .5 - cx, yy + .5 - cy
    d = np.hypot(X, Y)
    a = np.arctan2(Y, X)
    col = lambda k: _c(GLASS[k][1 if lit else 0])
    inside = d < R
    img[inside] = col('D')
    # 花瓣区
    k12 = np.round(a / (2 * np.pi / 12))
    da = (a - k12 * 2 * np.pi / 12) * d            # 到花瓣中线的弧长距离
    petal = (d >= 27) & (d < 58) & (np.abs(da) < 9.5 * np.sin(np.pi * np.clip((d - 26) / 33, 0, 1)) + 1)
    img[(d >= 24) & (d < 60) & ~petal] = col('B')
    img[petal] = col('V')
    img[petal & (np.abs(da) < 3) & (d > 34) & (d < 50)] = col('R')
    img[petal & (np.abs(da) < 1.2) & (d > 40) & (d < 46)] = col('G')
    # 外圈尖窗
    k24 = np.floor((a + np.pi) / (2 * np.pi / 24)).astype(int)
    outer = (d >= 62) & (d < R - 8)
    img[outer & (k24 % 2 == 0)] = col('B')
    img[outer & (k24 % 2 == 1)] = col('D')
    ac = (k24 + .5) * 2 * np.pi / 24 - np.pi
    rx, ry = cx + (R - 21) * np.cos(ac), cy + (R - 21) * np.sin(ac)
    rnd = np.hypot(xx + .5 - rx, yy + .5 - ry)
    img[outer & (rnd < 4.2)] = np.where((k24 % 2 == 0)[..., None], col('R'), col('G'))[outer & (rnd < 4.2)]
    img[outer & (np.abs(rnd - 4.6) < 0.7)] = _c('#07040a')
    img[d < 23] = col('G') * 0.7 + col('W') * 0.3
    # 小格：每 3px 一格的明暗
    tile = ((np.floor(xx / 3) + np.floor(yy / 3)) % 2 == 0) & inside & (d >= 24)
    img[tile] *= 0.86
    # 铅条 / 窗棂
    lead = _c('#07040a')
    stone = _c('#6a5060') if lit else _c('#34283a')
    img[outer & ((np.abs(d - 78) < .6) | (np.abs(d - 70) < .5))] = lead
    img[outer & (np.abs(np.sin((a + np.pi) * 12)) * d < 1.3)] = stone
    img[petal & _ring(petal)] = lead
    img[_ring(petal) & (d < 60)] = lead
    img[(np.abs(d - 60) < 2.0) | (np.abs(d - 24) < 1.5)] = stone
    frame = (d >= R - 8) & (d < R)
    img[frame] = _c('#5a4050') if lit else _c('#2a1e2a')
    img[frame & (np.abs(np.sin(a * 24)) < 0.25)] = _c('#8a6a50') if lit else _c('#3a2c34')
    img[np.abs(d - R) < 0.8] = lead
    img[np.abs(d - (R - 8)) < 0.6] = lead


@lru_cache(None)
def far_layer(lit=False):
    h = _lh('far', 0)
    img = _vg(h, '#120a12', '#1e1420')
    yy, xx = np.mgrid[0:h, 0:W]
    # 石砖
    row = yy // 10
    brick = ((xx + (row % 2) * 12) % 24 == 0) | (yy % 10 == 0)
    img[brick] *= 0.72
    img[(yy % 10 == 1) & ~brick] *= 1.12
    cx, cy, R = WIN
    # 大尖拱（窗外的石框）
    arch = _poly([(cx - R - 26, h), (cx - R - 26, cy + 10), (cx - R + 2, cy - R + 10), (cx, cy - R - 36),
                  (cx + R - 2, cy - R + 10), (cx + R + 26, cy + 10), (cx + R + 26, h)], h, W)
    arch_in = _poly([(cx - R - 14, h), (cx - R - 14, cy + 12), (cx - R + 10, cy - R + 18), (cx, cy - R - 22),
                     (cx + R - 10, cy - R + 18), (cx + R + 14, cy + 12), (cx + R + 14, h)], h, W)
    img[arch & ~arch_in] = _c('#2e2230')
    img[_ring(arch) & (yy < h)] = _c('#0a060a')
    img[_ring(arch_in)] = _c('#4a3848')
    img[arch_in] = img[arch_in] * 0.8
    # 侧面尖顶长窗（淡蓝）
    for x0 in (38, 86, 394, 442):
        m = _poly([(x0 - 10, 250), (x0 - 10, 70), (x0, 44), (x0 + 10, 70), (x0 + 10, 250)], h, W)
        img[dilate(m, 2) & ~m] = _c('#2c2030')
        img[m] = _c('#3a58a0') if lit else _c('#18204a')
        img[m & ((xx - x0) % 7 == 0)] = _c('#07040a')
        img[m & (yy % 16 == 0)] = _c('#07040a')
        img[_ring(m)] = _c('#07040a')
    # 下方的拱廊（侧廊黑洞）
    for x0 in range(-20, W + 60, 64):
        m = _poly([(x0 - 22, h), (x0 - 22, 300), (x0, 270), (x0 + 22, 300), (x0 + 22, h)], h, W)
        m &= ~arch
        img[m] = _c('#07040a')
        img[_ring(m) & ~arch] = _c('#3a2a38')
    sub = img[:, :, :].copy()
    win = np.zeros((h, W, 3), np.float32)
    win[:] = img
    _window(win, lit, cx, cy, R)
    d = np.hypot(xx + .5 - cx, yy + .5 - cy)
    img = np.where((d < R)[..., None], win, sub)
    # 中心圆章：金色衔尾蛇纹章
    try:
        from emblem import emblem
        e = emblem(size=44, progress=1.0, palette='gold')
        if not lit:
            e = e.copy()
            e[..., :3] *= 0.45
        blit(img, e, cx - 22, cy - 22)
    except Exception:
        pass
    return img.astype(np.float32)


# ---------------------------------------------------------------- 柱廊层：成束的石柱 + 拱顶肋
@lru_cache(None)
def mid_layer():
    h = _lh('mid', 0)
    img = np.zeros((h, W, 4), np.float32)
    yy, xx = np.mgrid[0:h, 0:W]
    for side in (-1, 1):
        cx = 240 + side * 186
        body = np.abs(xx - cx) < 20
        m = body & (yy > 30)
        img[m, :3] = _c('#1c1420')
        img[m, 3] = 1
        # 成束柱的竖向亮暗
        for k, c in ((-14, '#2a2030'), (-6, '#3a2c3c'), (4, '#241a28'), (12, '#150e18')):
            mm = m & (np.abs(xx - (cx + side * k)) < 3)
            img[mm, :3] = _c(c)
        inner = m & (np.abs(xx - (cx - side * 19)) < 1)       # 朝向中央的受光边
        img[inner, :3] = _c('#8a6a48')
        # 柱头 / 柱础
        for y0 in (200, 420):
            cap = (np.abs(xx - cx) < 25) & (yy >= y0) & (yy < y0 + 7)
            img[cap, :3] = _c('#3a2c38')
            img[cap, 3] = 1
            img[cap & (yy == y0), :3] = _c('#6a5250')
        # 从柱顶弯向中央的拱肋
        arc_c = (240, 30)
        d = np.hypot((xx - arc_c[0]) / 1.0, (yy - arc_c[1]) * 1.9)
        rib = (np.abs(d - 188) < 5) & (yy < 60) & (np.sign(xx - 240) == side)
        img[rib, :3] = _c('#261c28')
        img[rib, 3] = 1
        img[rib & (np.abs(d - 188) < 1), :3] = _c('#5a4448')
    img[_ring(img[..., 3] > 0), :3] = _c('#07040a')
    img[_ring(img[..., 3] > 0), 3] = 1
    return img


# ---------------------------------------------------------------- 阶梯层：地面、长阶、地毯、空座、祭坛
def _halfw(y):
    t = np.clip((y - TOP_Y) / (BOT_Y - TOP_Y), 0, 1)
    return 72 + (196 - 72) * t


def _steps():
    """阶梯每一级的上沿 y（从上到下，越往下越高）。"""
    ys, y = [], float(TOP_Y)
    while y < BOT_Y:
        ys.append(y)
        t = (y - TOP_Y) / (BOT_Y - TOP_Y)
        y += 7 + 9 * t
    return ys + [BOT_Y]


@lru_cache(None)
def seat_img():
    """空着的使徒之座：尖拱高椅背、金框、紫色天鹅绒、酒红坐垫、石质底座。18x34。"""
    h, w = 34, 18
    img = np.zeros((h, w, 4), np.float32)
    back = _poly([(2, 23), (2, 8), (9, 0.5), (16, 8), (16, 23)], h, w)
    vel = _poly([(4.5, 22), (4.5, 9), (9, 3.5), (13.5, 9), (13.5, 22)], h, w)
    cush = _poly([(1, 22), (17, 22), (17, 26), (1, 26)], h, w)
    arms = _poly([(0, 18), (3, 18), (3, 24), (0, 24)], h, w) | _poly([(15, 18), (18, 18), (18, 24), (15, 24)], h, w)
    base = _poly([(2, 26), (16, 26), (16, 34), (2, 34)], h, w)
    for m, c in ((back, '#b08a3c'), (vel, '#3a2248'), (cush, '#5a1a30'), (arms, '#8a6a30'), (base, '#2a2030')):
        img[m, :3] = _c(c)
        img[m, 3] = 1
    yy, xx = np.mgrid[0:h, 0:w]
    img[vel & (xx >= 9), :3] = _c('#2a1834')
    img[back & (xx <= 3), :3] = _c('#e8c068')
    img[cush & (yy == 22), :3] = _c('#8a3048')
    img[base & (yy % 4 == 0), :3] = _c('#1a1420')
    img[base & (xx == 2), :3] = _c('#4a3a48')
    # 椅背上的衔尾蛇小纹（金色圆环）
    rr = np.hypot(xx + .5 - 9, yy + .5 - 12)
    img[vel & (np.abs(rr - 2.6) < 0.8), :3] = _c('#f0c860')
    a = img[..., 3] > 0
    img[_ring(a), :3] = _c('#07040a')
    img[_ring(a), 3] = 1
    return img


def seat_pos():
    out = []
    for (bb, y) in IGNITE[:3]:
        for side in (-1, 1):
            out.append((240 + side * (_halfw(y) + 24), y))
    return out


@lru_cache(None)
def stair_layer():
    h = _lh('stair', 0) + 40
    img = np.zeros((h, W, 4), np.float32)
    yy, xx = np.mgrid[0:h, 0:W]
    # 地面：暗色大理石棋盘，透视变宽
    floor = yy >= BOT_Y
    t = np.maximum(yy - BOT_Y, 0) / 40
    fx_ = (xx - 240) / (1 + t * 0.8)
    chk = ((np.floor(fx_ / 22) + np.floor((yy - BOT_Y) / (8 + 6 * t))) % 2 == 0)
    img[floor, :3] = np.where(chk[floor, None], _c('#241a24'), _c('#150e16'))
    img[floor & (yy == BOT_Y), :3] = _c('#6a5040')
    img[floor, 3] = 1
    # 阶梯两侧的暗墙 / 平台（阶梯外）
    side = (yy >= TOP_Y - 6) & (yy < BOT_Y)
    img[side, :3] = _c('#120c14')
    img[side, 3] = 1
    img[side & ((yy // 12 + (xx // 30)) % 2 == 0), :3] = _c('#170f19')
    # 台阶
    ys = _steps()
    hw_row = _halfw(yy + 0.5)
    stair = (np.abs(xx + .5 - 240) < hw_row) & (yy >= TOP_Y) & (yy < BOT_Y)
    for i in range(len(ys) - 1):
        a, b = ys[i], ys[i + 1]
        band = stair & (yy >= a) & (yy < b)
        k = (b - a)
        tread = band & (yy < a + max(2, k * 0.35))
        img[band, :3] = _c('#150e18')           # 竖面（背光，暗）
        img[tread, :3] = _c('#2c2230')          # 踏面
        img[band & (yy == int(a)), :3] = _c('#5a4648')   # 踏步前沿的受光线
    img[stair, 3] = 1
    # 中央红毯（金边）
    carpet = stair & (np.abs(xx + .5 - 240) < hw_row * 0.24)
    shade = img[..., :3].mean(-1)
    img[carpet, :3] = np.where((shade[carpet] > 0.15)[:, None], _c('#5a1a30'), _c('#300a1a'))
    for i in range(len(ys) - 1):
        img[carpet & (yy == int(ys[i])), :3] = _c('#8a3048')
    edge = stair & (np.abs(np.abs(xx + .5 - 240) - hw_row * 0.24) < 0.9)
    img[edge, :3] = _c('#c89a48')
    # 栏杆：两侧石栏 + 立柱
    for s in (-1, 1):
        rail = (np.abs(xx + .5 - (240 + s * (hw_row + 4))) < 3.5) & (yy >= TOP_Y - 4) & (yy < BOT_Y)
        img[rail, :3] = _c('#2e2230')
        img[rail & (np.abs(xx + .5 - (240 + s * (hw_row + 4 - 3 * s))) < 1), :3] = _c('#6a5048')
        img[rail, 3] = 1
        for y0 in ys[::3]:
            px = 240 + s * (_halfw(y0) + 4)
            post = (np.abs(xx + .5 - px) < 5) & (yy >= y0 - 14) & (yy < y0 + 2)
            img[post, :3] = _c('#382a38')
            img[post & (yy < y0 - 12), :3] = _c('#8a6a48')
            img[post, 3] = 1
    # 祭坛平台
    alt = (np.abs(xx + .5 - 240) < 96) & (yy >= TOP_Y - 14) & (yy < TOP_Y + 2)
    img[alt, :3] = _c('#2a1e2a')
    img[alt & (yy < TOP_Y - 10), :3] = _c('#4a3848')
    img[alt & (yy == TOP_Y - 14), :3] = _c('#a07a48')
    img[alt & (yy > TOP_Y - 6) & ((xx // 8) % 2 == 0), :3] = _c('#1e1520')
    img[alt, 3] = 1
    # 祭坛两侧的立灯（金色灯柱）
    for s in (-1, 1):
        x0 = 240 + s * 82
        pole = (np.abs(xx + .5 - x0) < 1.5) & (yy >= TOP_Y - 60) & (yy < TOP_Y - 14)
        base = (np.abs(xx + .5 - x0) < 5) & (yy >= TOP_Y - 20) & (yy < TOP_Y - 14)
        bowl = (np.abs(xx + .5 - x0) < 6 - (yy - (TOP_Y - 64)) * 0.6) & (yy >= TOP_Y - 64) & (yy < TOP_Y - 58)
        for m, c in ((pole, '#8a6a30'), (base, '#6a4a20'), (bowl, '#c89a48')):
            img[m, :3] = _c(c)
            img[m, 3] = 1
    # 空着的使徒之座
    s_img = seat_img()
    for (x, y) in seat_pos():
        blit_rgba(img, s_img, x - 9, y - 32)
    a = img[..., 3] > 0
    img[_ring(a), :3] = _c('#07040a')
    img[_ring(a), 3] = 1
    return img


def blit_rgba(dst, src, x, y):
    x, y = int(round(x)), int(round(y))
    h, w = src.shape[:2]
    y0, x0 = max(0, y), max(0, x)
    y1, x1 = min(dst.shape[0], y + h), min(dst.shape[1], x + w)
    if y1 <= y0 or x1 <= x0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4]
    d = dst[y0:y1, x0:x1]
    d[..., :3] = d[..., :3] * (1 - a) + s[..., :3] * a
    d[..., 3:4] = np.maximum(d[..., 3:4], a)


# ---------------------------------------------------------------- 前景：两根巨大的黑柱
@lru_cache(None)
def fg_layer():
    h = _lh('fg', 0) + 20
    img = np.zeros((h, W, 4), np.float32)
    yy, xx = np.mgrid[0:h, 0:W]
    for s in (-1, 1):
        cx = 240 + s * 250
        m = np.abs(xx - cx) < 34
        img[m, :3] = _c('#0c080e')
        img[m, 3] = 1
        lit = m & (np.abs(xx - (cx - s * 32)) < 1.2)
        img[lit, :3] = _c('#5a4030')
        for k in (-18, -8, 6):
            img[m & (np.abs(xx - (cx - s * k)) < 1), :3] = _c('#140e16')
    return img


# ---------------------------------------------------------------- 衔尾蛇之环
def ouroboros(dst, cx, cy, R, prog, t, glow=0.0):
    """画一条绕圆的蛇：尾巴固定在 θ0，头沿逆时针追到 θ0+2π·prog；prog=1 时咬住尾巴。"""
    if prog <= 0:
        return
    x0, x1 = int(max(0, cx - R - 14)), int(min(W, cx + R + 15))
    y0, y1 = int(max(0, cy - R - 14)), int(min(H, cy + R + 15))
    if x1 <= x0 or y1 <= y0:
        return
    X = XX[y0:y1, x0:x1] + .5 - cx
    Y = YY[y0:y1, x0:x1] + .5 - cy
    d = np.hypot(X, Y)
    th0 = np.pi / 2 + 0.25                 # 尾巴在右下方
    ang = (np.arctan2(Y, X) - th0) % (2 * np.pi)      # 0..2π，从尾巴起
    span = 2 * np.pi * min(prog, 1.0) - 0.08
    s = ang / max(span, 1e-3)                          # 0 尾 -> 1 头
    thick = 3 + 9 * np.clip(s * 1.6, 0, 1) ** 0.7
    body = (ang <= span) & (np.abs(d - R) < thick / 2)
    reg = dst[y0:y1, x0:x1]
    arc = ang * R
    scale_ = ((np.floor((arc - t * 40) / 5) + np.floor((d - R + 8) / 3)) % 2 == 0)
    base = np.where(scale_[..., None], _c('#d8a040'), _c('#a87020'))
    top = (d - R) < -thick / 2 + 2
    bot = (d - R) > thick / 2 - 2
    col = np.where(top[..., None], _c('#ffe8a0'), np.where(bot[..., None], _c('#6a4010'), base))
    col = col * (1 + glow) + glow * 0.3
    out = _ring(body)
    reg[body] = np.minimum(col[body], 1.0)
    reg[out] = _c('#1a0c04')
    # 蛇头：在 span 末端，楔形 + 张开的颚 + 发光的眼
    ha = th0 + span
    hx, hy = cx + R * np.cos(ha), cy + R * np.sin(ha)
    tx, ty = -np.sin(ha), np.cos(ha)                   # 前进方向（逆时针）
    nx, ny = np.cos(ha), np.sin(ha)
    Xh, Yh = XX[y0:y1, x0:x1] + .5 - hx, YY[y0:y1, x0:x1] + .5 - hy
    u = Xh * tx + Yh * ty
    v = Xh * nx + Yh * ny
    head = (u > -4) & (u < 12) & (np.abs(v) < 7 - u * 0.35)
    jaw = 1.5 + 2.5 * (1 - min(prog, 1.0) ** 4)
    head &= ~((u > 5) & (np.abs(v) < jaw * (u - 5) / 7))
    reg[head] = np.minimum(_c('#e8b048') * (1 + glow), 1)
    reg[_ring(head) & ~body] = _c('#1a0c04')
    eye = (np.abs(u - 4) < 1.2) & (np.abs(v + 3) < 1.2)
    reg[eye] = _c('#ff3a20') if glow < 0.5 else _c('#ffffff')


# ---------------------------------------------------------------- 工具
def _catmull(keys, x):
    xs = [k[0] for k in keys]
    ys = [k[1] for k in keys]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
    p0 = ys[max(i - 1, 0)]
    p1, p2 = ys[i], ys[i + 1]
    p3 = ys[min(i + 2, len(ys) - 1)]
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                  (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def _zoom(img, z, cx, cy):
    if abs(z - 1) < 1e-3:
        return img
    ys = np.clip(((np.arange(H) + .5 - H / 2) / z + cy).astype(int), 0, H - 1)
    xs = np.clip(((np.arange(W) + .5 - W / 2) / z + cx).astype(int), 0, W - 1)
    return img[ys][:, xs]


def _comp(dst, layer, off):
    """把 RGBA 层按纵向偏移 off 裁切叠到画布上。"""
    o = int(round(off))
    part = layer[o:o + H]
    if part.shape[0] < H:
        part = np.concatenate([part, np.zeros((H - part.shape[0], W, 4), np.float32)])
    a = part[..., 3:4]
    dst[:] = dst * (1 - a) + part[..., :3] * a


@lru_cache(512)
def _body(pose, hd, cd):
    return GM.body(GM.pose(pose, hair_dx=hd / 2, cape_dx=cd / 2))


def _rim(spr):
    a = spr[..., 3] > 0
    up = a & ~np.roll(a, 1, 0)
    side = a & (~np.roll(a, 1, 1) | ~np.roll(a, -1, 1))
    return up | side


def _pose_at(ctx):
    """4.0 起举手的关键帧（约 12 fps 的像素动画节奏）。"""
    b = ctx.since(4, 0) / ctx.sec.beat
    if b < 0:
        return 'idle'
    if b < 0.9:
        return 'raise0'          # 预备：手臂微收
    if b < 1.25:
        return 'raise1'
    if b < 1.6:
        return 'raise2'
    if b < 3.0:
        return 'raise'
    if b < 3.5:
        return 'raise_hi'        # 最后一拍：手推到最高
    return 'raise'


BEAM_LEN = 250


def _beam(dst, x, y_bot, w, level, t, seed, lean):
    """一道自高窗斜射而下的光柱（屏幕坐标）：底部最亮、向上渐隐，落点一圈光池。lean: 每上升 1px 的横移。"""
    if level <= 0.01 or y_bot < -10 or y_bot - BEAM_LEN > H:
        return
    ytop = int(max(0, y_bot - BEAM_LEN))
    yb = int(min(H, y_bot + 2))
    if yb <= ytop:
        return
    xs = [x, x + lean * BEAM_LEN]
    x0, x1 = int(max(0, min(xs) - w - 1)), int(min(W, max(xs) + w + 2))
    if x1 <= x0:
        return
    Yr = YY[ytop:yb, x0:x1]
    up = (y_bot - Yr)                                   # 离落点的高度
    X = XX[ytop:yb, x0:x1] + .5 - (x + lean * up)
    fall = np.clip(1 - up / BEAM_LEN, 0, 1) ** 1.3
    flick = 0.92 + 0.08 * np.sin(t * 7 + seed)
    lv = level * flick
    core = np.abs(X) < w * 0.3
    halo = np.abs(X) < w
    reg = dst[ytop:yb, x0:x1]
    dm = BAYER_[ytop:yb, x0:x1] < np.clip(0.7 * lv * fall, 0, 1)
    reg += (dm & halo)[..., None] * _c('#6a5028')
    reg += (core * fall)[..., None] * _c('#ffe6a0') * 0.4 * min(lv, 1.5)
    pool = (np.hypot((XX - x) / (w * 2.2), (YY - y_bot) / 5) < 1)
    dst[pool] += _c('#ffd890') * 0.3 * min(lv, 1.2)


def _light(ctx, pts, amb):
    """光照图：环境光 + 各光柱落点附近的暖光（乘到场景上，做出明暗对比）。"""
    Lm = np.full((H, W), amb, np.float32)
    for x, y, lv in pts:
        if lv <= 0 or y < -80 or y > H + 80:
            continue
        g = np.exp(-(((XX - x) / 70) ** 2 + ((YY - y) / 40) ** 2))
        Lm += g * 0.9 * min(lv, 1.3)
    return np.clip(Lm, 0, 1.25)


def _caption(dst, t_in, t_out):
    """「结社『噬身之蛇』/ 盟主」名牌：从左侧滑入，最后 0.25 s 抖动淡出。"""
    p = ease_back(t_in / 0.4)
    x0 = 22 - 190 * (1 - p)
    y0 = 36
    a = min(1.0, t_out / 0.25)
    plate = (XX >= x0 - 6) & (XX < x0 + 148) & (YY >= y0 - 4) & (YY < y0 + 46)
    m = plate & (BAYER_ < 0.82 * a)
    dst[m] = dst[m] * 0.18 + _c('#0a0610') * 0.82
    if a < 0.6:
        return
    edge = plate & ((YY == y0 - 4) | (YY == y0 + 45))
    dst[edge] = _c('#c89a48')
    bar = (XX >= x0 - 6) & (XX < x0 - 3) & (YY >= y0 - 4) & (YY < y0 + 46)
    dst[bar] = _c('#ffd870')
    text(dst, i18n.S('society'), x0 + 4, y0 + 6, _c('#d8c8a8'), anchor='left', reveal=t_in * 40)
    text(dst, i18n.name('grandmaster')[0], x0 + 4, y0 + 28, _c('#fff4d8'), scale=2, anchor='left', glow=_c('#a06a18'),
         spacing=6 if i18n.LANG == 'zh' else 0, reveal=(t_in - 0.15) * (10 if i18n.LANG == 'zh' else 30))


# ---------------------------------------------------------------- 渲染
def render(dst, ctx):
    lt, beat = ctx.lt, ctx.sec.beat
    b = lt / beat                         # 本段内的拍数 0..16
    st = ctx.state
    t = ctx.t

    # ---- 镜头：1.0 低角度仰拍 -> 沿长阶上摇 -> 2.0 抵达阶顶并缓缓落定；3.0 硬切 2x 胸像；4.0 硬切回全景
    cam = _catmull([(0, CAM0), (0.6, CAM0 - 10), (1, 296), (2, 210), (3, 112), (4, 36), (6, 10), (8, 0)], b)
    if 8 <= b < 12:
        z = 2.0                                        # 整数倍特写，像素干净
        zc = (240, 100 - 8 * smooth(8, 12, b))         # 缓慢上移：从胸口到面纱
    else:
        z, zc = 1.0, (240, 135)

    # ---- 光：玫瑰窗亮度（管钟每响一次脉动）、金光漫溢
    L = 0.06 + 0.74 * smooth(4, 5.2, b) + 0.9 * smooth(12, 15, b) ** 1.5
    for bb, kk in BELLS:
        dtb = ctx.since(bb, kk)
        if 0 <= dtb < 1.2 and (bb, kk) != (1, 0.0):
            L += 0.3 * np.exp(-dtb * 3)
    if b >= 15:
        L += 0.8 * np.exp(-max(0, b - 15.3) * 2.5)
    gold = 0.2 * smooth(12, 15, b) ** 1.3 + (0.35 * np.exp(-max(0, b - 15.3) * 3) if b >= 15 else 0)
    if 8 <= b < 12:
        L *= 0.78                                      # 特写时压暗玫瑰窗，让人物跳出来

    # ---- 远墙（玫瑰窗亮暗两版混合）
    off_far = cam * PAR['far']
    dark, lit = far_layer(False), far_layer(True)
    o = int(round(off_far))
    k = min(L, 1.0)
    img = dark[o:o + H] * (1 - k) + lit[o:o + H] * k
    wx, wy = WIN[0], WIN[1] - off_far
    dW = np.hypot(XX - wx, YY - wy)
    if L > 1:
        wm = dW < WIN[2]
        img[wm] = img[wm] + (L - 1) * 0.6 * (_c('#fff0c0') - img[wm] * 0.5)
    dst[:] = img
    if L > 0.25:                                        # 窗外一圈抖动的光晕
        halo = (dW >= WIN[2]) & (dW < WIN[2] + 26) & (BAYER_ < 0.45 * min(L, 1.6) * (1 - (dW - WIN[2]) / 26))
        dst[halo] += _c('#8a6030') * 0.7
        fx.rays(dst, wx, wy, t, _c('#ffe0a0'), n=12, strength=0.1 * min(L, 2), speed=0.2, r0=WIN[2] + 8,
                reach=300)

    # ---- 柱廊、阶梯
    _comp(dst, mid_layer(), cam * PAR['mid'])
    off = cam * PAR['stair']
    _comp(dst, stair_layer(), off)

    # ---- 光照：暗殿 + 光柱落点的暖光 + 窗光（乘到场景上）
    beams = []
    for i, ((bb, kk), y) in enumerate(IGNITE):
        dtb = ctx.since(bb, kk)
        if dtb < 0:
            continue
        lv = (1.6 * np.exp(-dtb * 4) + 0.6) * clamp01(dtb / 0.07)
        for side in (-1, 1):
            beams.append((240 + side * (_halfw(y) + 24), y - off, lv, side, i))
    dta = ctx.since(*ALTAR_IGNITE)
    if dta >= 0:
        lv = (1.8 * np.exp(-dta * 3) + 0.7) * clamp01(dta / 0.07)
        for side in (-1, 1):
            beams.append((240 + side * 82, TOP_Y - 62 - off, lv, side, 9))
    amb = 0.42 + 0.33 * smooth(4, 5.5, b) + 0.2 * smooth(12, 15, b)
    Lm = _light(ctx, [(x, y, lv) for x, y, lv, _, _ in beams] + [(wx, wy + 60, 0.5 * min(L, 1.5))], amb)
    dst *= Lm[..., None]

    # ---- 光柱（自两侧高窗斜射而下）
    for x, y, lv, side, i in beams:
        _beam(dst, x, y, 10 if i < 9 else 8, lv, t, i * 2 + side, -0.16 * side)
    for i, ((bb, kk), y) in enumerate(IGNITE):
        if ctx.crossed(bb, kk):
            for side in (-1, 1):
                x = 240 + side * (_halfw(y) + 24)
                ctx.P.emit(40, (x - 8, x + 8), (y - off - 60, y - off), (-14, 14), (-30, 10), (0.6, 1.6), 'gold',
                           size=1, drag=1.5, bright=0.8)
            ctx.shake(3 if i else 5, 0.3)
    if ctx.crossed(1, 0.14):                  # 冲击帧（两帧反相）之后再给一下白闪
        ctx.flash('#fff0d0', 0.1, 0.55)
    if ctx.crossed(*ALTAR_IGNITE):
        ctx.flash('#fff4d8', 0.14, 0.5)
        ctx.shake(4, 0.35)

    # ---- 衔尾蛇之环（在她身后）
    ring_p = ctx.prog(4, 0.0, 4, 3.0)
    if b >= 12:
        ring_p = ease_out(ring_p) * 0.35 + ring_p ** 1.6 * 0.65
        glow = 0.0 if b < 15 else 0.8 * np.exp(-(b - 15) * 1.5) + 0.3
        ouroboros(dst, wx, wy, WIN[2] + 12, ring_p if b < 15 else 1.0, t, glow)

    # ---- 盟主
    f12 = int(lt * 12)
    wind = 1.0 + 1.6 * smooth(12, 15, b)
    hd = int(round(2 * wind * 1.4 * np.sin(f12 * 0.33)))
    cd = int(round(2 * wind * 1.0 * np.sin(f12 * 0.24 + 1.1)))
    spr = _body(_pose_at(ctx), hd, cd)
    fx_, fy_ = FEET[0], FEET[1] - off
    reveal = smooth(4.0, 5.6, b)            # 2.0 起逆光剪影 -> 本色
    dark_amt = 0.94 - 0.8 * reveal
    blit(dst, spr, fx_, fy_, tint=_c('#0c0610'), tint_amt=dark_amt, anchor='bottom')
    if b >= 3.6:
        rim = _rim(spr)
        ys, xs = np.nonzero(rim)
        ys = ys + int(round(fy_ - spr.shape[0]))
        xs = xs + int(round(fx_ - spr.shape[1] / 2))
        ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        rc = _c('#ffe4a0') * (0.5 + 0.5 * smooth(3.6, 4.4, b))
        dst[ys[ok], xs[ok]] = dst[ys[ok], xs[ok]] * 0.3 + rc * 0.9
    # 举起的手里亮起的一点光（4.2 起）
    if b >= 13.6:
        hx, hy = GM.META['hand_r'].get(_pose_at(ctx), (21, 19))
        sx, sy = fx_ - spr.shape[1] / 2 + hx, fy_ - spr.shape[0] + hy
        r = 3 + 2 * np.sin(t * 20) + (6 if b >= 15 else 0) * np.exp(-max(0, b - 15) * 2)
        glowm = np.hypot(XX - sx, YY - sy) < r
        dst[glowm] = dst[glowm] * 0.3 + _c('#fff6d8') * 0.9
        if int(lt * 30) % 2 == 0:
            ctx.P.emit(2, sx, sy, (-20, 20), (-40, -5), (0.3, 0.8), 'gold', size=1, drag=2)

    # ---- 尘埃（光柱里缓缓浮动）+ 高潮的金色粒子
    P = ctx.P
    n_dust = 2 if b < 4 else 3
    P.emit(n_dust, (60, 420), (40, 250), (-4, 4), (-10, -2), (2.0, 3.5), 'gold', size=1, bright=0.35, wob=0.15)
    if b >= 12:
        P.emit(int(2 + 6 * smooth(12, 15, b)), (wx - 110, wx + 110), (wy - 90, wy + 90), (-20, 20), (-30, 10),
               (0.6, 1.4), 'gold', size=1, bright=0.8, drag=0.5)
    if ctx.crossed(4, 3.0):
        P.emit_radial(260, wx, wy, (60, 320), (0.6, 1.6), 'gold', size=(1, 2), drag=1.6)
        P.emit_radial(90, wx, wy, (30, 160), (0.8, 1.8), 'white', size=1, drag=1.2)
    if ctx.crossed(4, 3.3):                   # 顿帧结束：震屏 + 白闪
        ctx.shake(7, 0.5)
        ctx.flash('#ffffff', 0.12, 0.7)
    hitstop = 0 <= ctx.since(4, 3.0) < 0.13
    if not hitstop:
        P.step(ctx.dt)
    P.render(dst)

    # ---- 前景巨柱（视差最快）
    _comp(dst, fg_layer(), cam * PAR['fg'])

    # ---- 金光漫溢
    if gold > 0:
        dst += _c('#ffc860') * gold * (0.6 + 0.4 * np.clip(1 - np.hypot(XX - wx, YY - wy) / 300, 0, 1))[..., None]
        fx.god_rays_down(dst, t, _c('#fff0c0'), x0=wx, strength=0.6 * min(1, gold * 3))
    if b >= 15.3:                                      # 顿帧之后冲击波才扩散
        r = (b - 15.3) * beat * 560
        fx.shockwave(dst, wx, wy, r, _c('#fff4d0'), w=3)
    np.clip(dst, 0, 1, out=dst)

    # ---- 推近
    dst = _zoom(dst, z, *zc)

    # ---- 名牌（2.1 – 2.3）：左上角暗底金边牌子滑入，避开人物与阶梯的细碎纹理
    if 4.2 <= b < 8.0:
        _caption(dst, (b - 4.2) * beat, (8.0 - b) * beat)
    # ---- 冲击帧：1.0 与 4.3 各两帧高反差反相（亮处变黑、暗处变白金），随后才是闪光与震屏
    for bb, kk in ((1, 0.0), (4, 3.0)):
        if 0 <= ctx.since(bb, kk) < 0.062:
            lum = dst.mean(-1)
            thr = 0.22 if bb == 1 else 0.62
            dst = np.where((lum > thr)[..., None], _c('#140a04'), _c('#fff2cc')).astype(np.float32)
    return dst


# ---------------------------------------------------------------- 专属音效
def _sfx_ignite():
    """光柱点亮：低沉的"轰"+ 上扬的金属泛音 + 空气声。"""
    n = int(1.6 * S.SR)
    tt = S.tt(n)
    f = 38 + 60 * np.exp(-tt / 0.08)
    boom = np.sin(2 * np.pi * np.cumsum(f) / S.SR) * np.exp(-tt / 0.45)
    air = S.bp(S.noise(n), 600, 5000) * np.exp(-tt / 0.25) * np.clip(tt / 0.01, 0, 1)
    shim = sum(np.sin(2 * np.pi * fr * tt) * np.exp(-tt / (0.9 - 0.15 * i)) for i, fr in
               enumerate((659.3, 987.8, 1318.5, 1975.5))) * 0.12
    return (0.9 * boom + 0.35 * air + shim * np.clip(tt / 0.03, 0, 1)) * 0.8


def _sfx_toll():
    """管钟 / 大教堂钟（E）：非谐泛音，长余音。"""
    n = int(3.5 * S.SR)
    tt = S.tt(n)
    f0 = 164.8
    parts = ((0.5, 1.0, 2.2), (1.0, 0.8, 2.8), (1.19, 0.5, 1.6), (1.5, 0.45, 1.8), (2.0, 0.6, 1.4),
             (2.52, 0.3, 0.9), (3.01, 0.25, 0.7), (4.1, 0.15, 0.4))
    x = sum(a * np.sin(2 * np.pi * f0 * r * tt + r) * np.exp(-tt / d) for r, a, d in parts)
    strike = S.hp(S.noise(n), 2000) * np.exp(-tt / 0.01) * 0.3
    return (x * 0.35 + strike) * np.clip(tt / 0.002, 0, 1)


def _sfx_swell():
    """合唱般的气声渐强（3 拍）：带通噪声 + 缓慢上扬的共振峰。"""
    dur = 3 * 60 / 144 + 0.1
    n = int(dur * S.SR)
    tt = S.tt(n)
    x = S.sweep_lp(S.noise(n), 400, 6000, 1.5)
    vox = sum(np.sin(2 * np.pi * f * tt * (1 + 0.004 * np.sin(2 * np.pi * 5 * tt))) for f in (329.6, 493.9, 659.3))
    e = (tt / dur) ** 2
    return (0.4 * x + 0.12 * vox) * e


def _sfx_ring():
    """巨蛇绕环：旋转的嘶嘶声，左右来回。"""
    dur = 3 * 60 / 144
    n = int(dur * S.SR)
    tt = S.tt(n)
    nz = S.bp(S.noise(n), 1500, 7000)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * (1 + 3 * tt / dur) * tt)
    e = np.sin(np.pi * np.clip(tt / dur, 0, 1)) ** 0.7
    pan = np.sin(2 * np.pi * tt / dur)
    mono = nz * am * e * 0.35
    return np.stack([mono * np.sqrt(0.5 * (1 - pan)), mono * np.sqrt(0.5 * (1 + pan))], 1)


SFX = {'gm_ignite': _sfx_ignite, 'gm_toll': _sfx_toll, 'gm_swell': _sfx_swell, 'gm_ring': _sfx_ring}
