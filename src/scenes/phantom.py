"""幻焰计划：克洛斯贝尔之夜，碧之大树破城而出；肯帕雷拉在高楼边缘「见证」。4 小节 @144（约 6.67 s），E 小调半速。

原作：《碧之轨迹》终章 —— 至宝「碧之零」之力让晶体般的巨树从克洛斯贝尔市中心拔地而起，
      结社的「幻焰计划」就此展开；肯帕雷拉自称这次只是「見届け役」（见证者）。

分镜（小节.拍，本段内；从 vita 以扑克牌翻面转入）
  1.0  次低频下潜：夜的克洛斯贝尔全景（远楼、密集的亮窗、奥基斯大厦的红色航标灯），大地轰鸣，
       街道上裂开一道道青色的光缝（树根在地下蔓延），镜头微震
  1.1  碧之大树破土：晶体树干一口气冲上夜空（全奏重音），镜头随之上摇
  1.2 / 1.3 / 1.3.5  枝桠一层层在拍点上展开，晶簇在枝梢闪亮，天空被染成青色
  2.0  硬切：高楼边缘，肯帕雷拉 2x 坐在栏杆上，被身后的巨树勾出青色轮廓，只有眼睛反着金光；
       字幕「这一次，我只负责见证而已。」；2.1 把一张牌抛起、翻转、接住，2.3 反手甩向巨树
  3.0  硬切回全景（他的剪影坐在右下角）：巨树背后的夜空睁开一只巨大的蛇眼（全奏）；
       3.0.5 起城市的灯火从大树附近开始一片一片闪烁、熄灭，最后连奥基斯大厦也暗了
  4.0  硬切：他站起，2x 中景；4.1 响指（顿帧 + 全体休止），指尖炸出橙色火花与蓝色火星
  4.2  硬切全景：一圈幻焰（橙焰 + 蓝火星）从他脚下沿城市外圈两路狂奔（配乐渐强），
       4.3.5 两路火头在城市另一端相撞、环闭合，火光与巨树一起烧亮画面，急推 —— 硬切「盟主」

世界层（x -60..600，y -100..310）按视差裁切：天空 0.25 / 远楼 0.55 / 树与中景楼 0.85 / 近景屋顶 1.0。
"""
from functools import lru_cache

import numpy as np

import cast
import fx
from gfx import BAYER, H, W, XX, YY, blit, clamp01, ease_out, hexc, smooth
import sfx as S

LETTERBOX = 18
MUSIC_SHAKE = False
SAYS = [('campa_witness', 2, 0.0, 'sub')]
ACCENTS = [(1, 1.0, 'hit'), (3, 0.0, 'hit'), (4, 1.0, 'stop')]
CUES = [
    (1, 0.0, 'rumble', 0.0), (1, 0.0, 'crumble', -6.0), (1, 0.5, 'ph_crack', -6.0, -0.3),
    (1, 1.0, 'ph_erupt', 0.0), (1, 1.0, 'crumble', -4.0), (1, 1.05, 'magic_beam', -8.0),
    (1, 2.0, 'ph_crystal', -5.0, -0.4), (1, 3.0, 'ph_crystal', -5.0, 0.4), (1, 3.5, 'ph_crystal', -7.0),
    (2, 1.0, 'card_flip', -6.0, 0.3), (2, 2.5, 'tick', -10.0, 0.3), (2, 3.0, 'swish', -6.0, -0.2),
    (3, 0.0, 'ph_eye', -1.0), (3, 0.5, 'ph_off', -8.0, -0.2), (3, 1.25, 'ph_off', -8.0, 0.3),
    (3, 2.0, 'ph_off', -7.0, -0.4), (3, 2.75, 'ph_off', -6.0, 0.5),
    (4, 1.0, 'snap', 0.0), (4, 1.02, 'fire_burst', -4.0),
    (4, 2.0, 'fire_whoosh', -2.0), (4, 2.0, 'fire_roar', -3.0), (4, 3.5, 'fire_burst', -1.0),
]

TAU = 2 * np.pi
X0, Y0 = 60, 100                     # 世界坐标 -> 数组下标的偏移
WW, HW = 660, 410
PAR = {'sky': 0.25, 'far': 0.55, 'mid': 0.85, 'near': 1.0}
TREE_BASE = (240, 206)
GROUND = 206
EYE = (240, 64, 118, 40)             # 屏幕坐标：蛇眼中心、半宽、半高
RING = (240, 214, 300, 76)           # 世界（中景层）坐标：火环椭圆中心、半轴
CAMPA = cast.get('campanella')


def C(h):
    return hexc(h)


def _rng(seed):
    return np.random.default_rng(seed)


# ================================================================ 静态层
def _grid():
    yy, xx = np.mgrid[0:HW, 0:WW]
    return yy - Y0, xx - X0          # 世界坐标


@lru_cache(None)
def sky_layer():
    wy, wx = _grid()
    t = np.clip((wy + 100) / 330, 0, 1)[..., None]
    top, mid, bot = C('#03041a'), C('#0a0f36'), C('#221a4e')
    img = np.where(t < 0.6, top * (1 - t / 0.6) + mid * (t / 0.6), mid * (1 - (t - 0.6) / 0.4) + bot * ((t - 0.6) / 0.4))
    img = img.astype(np.float32)
    r = _rng(3)
    n = 260
    sx, sy = r.integers(0, WW, n), r.integers(0, int(HW * 0.6), n)
    br = r.uniform(0.3, 1.0, n)
    for x, y, b in zip(sx, sy, br):
        img[y, x] = C('#d8e4ff') * b
        if b > 0.93 and 0 < x < WW - 1 and 0 < y < HW - 1:
            img[y, x - 1] = img[y, x + 1] = img[y - 1, x] = img[y + 1, x] = C('#6a78b0')
    # 远处的湖面倒映一条淡光带（艾尔姆湖）
    band = (wy > 168) & (wy < 176)
    img[band] = img[band] * 0.7 + C('#2a3070') * 0.3
    return img


def _buildings(seed, n_range, x_range, top_range, base, wmin, wmax):
    r = _rng(seed)
    out, x = [], x_range[0]
    while x < x_range[1]:
        w = int(r.integers(wmin, wmax))
        top = int(r.integers(*top_range))
        out.append((x, min(x + w, WW - X0 - 2), top, base, int(r.integers(0, 4))))
        x += w + int(r.integers(n_range[0], n_range[1]))
    return out


def _block_of(xc):
    return int(np.clip((xc + 60) // 50, 0, 12))


@lru_cache(None)
def far_layer():
    """远处的楼群：低矮、偏蓝，零星暗窗。返回 (RGBA, 窗户块号)。"""
    img = np.zeros((HW, WW, 4), np.float32)
    blk = np.full((HW, WW), -1, np.int16)
    r = _rng(11)
    for x0, x1, top, base, kind in _buildings(11, (-2, 2), (-60, 600), (118, 166), 212, 8, 22):
        ys, xs = slice(top + Y0, base + Y0), slice(x0 + X0, x1 + X0)
        img[ys, xs, :3] = C('#161c40') if kind else C('#12183a')
        img[ys, xs, 3] = 1
        img[top + Y0, xs, :3] = C('#252c58')
        b = _block_of((x0 + x1) / 2)
        for y in range(top + 3, base - 4, 4):
            for x in range(x0 + 2, x1 - 1, 3):
                if r.random() < 0.35:
                    img[y + Y0, x + X0, :3] = C('#7a88c0') if r.random() < 0.7 else C('#c0a870')
                    blk[y + Y0, x + X0] = b
    return img, blk


ORCHIS = (352, 30)        # 奥基斯大厦：中心 x、楼顶 y（世界，中景层）


@lru_cache(None)
def mid_layer():
    """中景：密集的楼、亮着的窗（暖色 / 冷色）、屋顶水箱与天线、奥基斯大厦、街面与路灯。
    返回 (RGBA, 窗户块号, 窗色)。"""
    img = np.zeros((HW, WW, 4), np.float32)
    blk = np.full((HW, WW), -1, np.int16)
    wcol = np.zeros((HW, WW, 3), np.float32)
    r = _rng(21)
    warm = [C('#ffd27a'), C('#ffb050'), C('#fff0c0'), C('#bfe0ff')]
    bl = _buildings(21, (0, 3), (-60, 600), (138, 196), 232, 14, 38)
    # 大树脚下留出广场，奥基斯大厦的位置留空
    bl = [b for b in bl if not (b[0] < 262 and b[1] > 218) and not (b[0] < ORCHIS[0] + 22 and b[1] > ORCHIS[0] - 22)]
    for x0, x1, top, base, kind in bl:
        ys, xs = slice(top + Y0, base + Y0), slice(x0 + X0, x1 + X0)
        img[ys, xs, :3] = C('#0c1028')
        img[ys, xs, 3] = 1
        img[ys, x0 + X0, :3] = C('#1a2244')                  # 左侧受光边
        img[top + Y0, xs, :3] = C('#2a3260')
        if kind == 0:                                          # 水箱
            cx = (x0 + x1) // 2
            img[top - 6 + Y0:top + Y0, cx - 3 + X0:cx + 3 + X0, :3] = C('#141a34')
            img[top - 6 + Y0:top + Y0, cx - 3 + X0:cx + 3 + X0, 3] = 1
        elif kind == 1:                                        # 天线
            cx = x0 + 3
            img[top - 10 + Y0:top + Y0, cx + X0, :3] = C('#2a3260')
            img[top - 10 + Y0:top + Y0, cx + X0, 3] = 1
        b = _block_of((x0 + x1) / 2)
        pw = 2 if kind % 2 == 0 else 1
        for y in range(top + 4, base - 3, 5):
            lit_row = r.random() < 0.85
            for x in range(x0 + 2, x1 - 2, 4):
                if lit_row and r.random() < 0.55:
                    c = warm[int(r.integers(0, 4))]
                    img[y + Y0:y + 2 + Y0, x + X0:x + pw + X0, :3] = c
                    wcol[y + Y0:y + 2 + Y0, x + X0:x + pw + X0] = c
                    blk[y + Y0:y + 2 + Y0, x + X0:x + pw + X0] = b
    # 奥基斯大厦：细高的玻璃塔身，上部收分，尖顶天线
    cx, top = ORCHIS
    for y in range(top, 232):
        t = (y - top) / (232 - top)
        hw = 9 + 10 * min(1, t * 2.2)
        xs = slice(int(cx - hw) + X0, int(cx + hw) + X0)
        img[y + Y0, xs, :3] = C('#101830')
        img[y + Y0, xs, 3] = 1
        img[y + Y0, int(cx - hw) + X0, :3] = C('#3a4a80')
        img[y + Y0, int(cx) + X0, :3] = C('#1c2850')
        if y % 4 == 0 and y > top + 6:
            for x in range(int(cx - hw) + 2, int(cx + hw) - 1, 3):
                c = C('#cfe8ff') if (x + y) % 7 else C('#fff0c0')
                img[y + Y0, x + X0, :3] = c
                wcol[y + Y0, x + X0] = c
                blk[y + Y0, x + X0] = 99                       # 大厦最后熄灯
    for y in range(top - 22, top):                            # 尖顶
        w = max(0, int((y - (top - 22)) / 22 * 8))
        img[y + Y0, cx - w + X0:cx + w + 1 + X0, :3] = C('#1c2850')
        img[y + Y0, cx - w + X0:cx + w + 1 + X0, 3] = 1
    img[top - 34 + Y0:top - 22 + Y0, cx + X0, :3] = C('#6a7aa8')
    img[top - 34 + Y0:top - 22 + Y0, cx + X0, 3] = 1
    # 街面（广场与大道）+ 路灯串
    g = slice(222 + Y0, HW)
    img[g, :, :3] = C('#080a18')
    img[g, :, 3] = 1
    for x in range(-60, 600, 9):
        img[226 + Y0, x + X0, :3] = C('#ffcf70')
        wcol[226 + Y0, x + X0] = C('#ffcf70')
        blk[226 + Y0, x + X0] = _block_of(x)
    return img, blk, wcol


@lru_cache(None)
def near_layer():
    """近景屋顶剪影（只在第 1 小节镜头偏低时入画）。"""
    img = np.zeros((HW, WW, 4), np.float32)
    r = _rng(31)
    for x0, x1, top, base, kind in _buildings(31, (0, 1), (-60, 600), (236, 262), 320, 30, 70):
        img[top + Y0:base + Y0, x0 + X0:x1 + X0, :3] = C('#05060d')
        img[top + Y0:base + Y0, x0 + X0:x1 + X0, 3] = 1
        img[top + Y0, x0 + X0:x1 + X0, :3] = C('#161a30')
        for y in range(top + 5, 300, 7):
            for x in range(x0 + 4, x1 - 3, 6):
                if r.random() < 0.18:
                    img[y + Y0:y + 3 + Y0, x + X0:x + 2 + X0, :3] = C('#e8b060')
    return img


# ================================================================ 碧之大树（按"生长时刻"预先栅格化）
def _seg(tb, nd, fac, p0, p1, w0, w1, t0, t1):
    x0, y0 = p0
    x1, y1 = p1
    pad = max(w0, w1) / 2 + 1
    ix0, ix1 = int(min(x0, x1) - pad) + X0, int(max(x0, x1) + pad) + X0 + 1
    iy0, iy1 = int(min(y0, y1) - pad) + Y0, int(max(y0, y1) + pad) + Y0 + 1
    ix0, iy0 = max(ix0, 0), max(iy0, 0)
    ix1, iy1 = min(ix1, WW), min(iy1, HW)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    yy, xx = np.mgrid[iy0:iy1, ix0:ix1].astype(np.float32)
    px, py = xx - X0 + .5, yy - Y0 + .5
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy + 1e-6
    u = np.clip(((px - x0) * dx + (py - y0) * dy) / L2, 0, 1)
    cx, cy = x0 + u * dx, y0 + u * dy
    side = (px - cx) * (-dy) + (py - cy) * dx
    d = np.hypot(px - cx, py - cy)
    w = (w0 + (w1 - w0) * u) / 2
    m = d < np.maximum(w, 0.6)
    t = t0 + (t1 - t0) * u
    n = d / np.maximum(w, 0.6)
    L = np.sqrt(L2)
    f = ((np.floor(u * L / 5) % 2) * 2 + (side > 0)).astype(np.int8)     # 晶面：沿枝每 5px 换一面，左右分明暗
    reg_t, reg_n, reg_f = tb[iy0:iy1, ix0:ix1], nd[iy0:iy1, ix0:ix1], fac[iy0:iy1, ix0:ix1]
    upd = m & ((t < reg_t) | ((np.abs(t - reg_t) < 0.05) & (n < reg_n)))
    reg_t[upd], reg_n[upd], reg_f[upd] = t[upd], n[upd], f[upd]


def _branch(segs, p, ang, length, w0, t0, speed, depth, r, tips):
    """ang: 与竖直向上的夹角（弧度，右正）。枝条分 4 段、带随机折角地逐渐向上弯；沿途分出 2–3 根子枝。"""
    pts = [p]
    a = ang
    for k in range(4):
        a = a * 0.86 + r.uniform(-0.16, 0.16)
        q = (pts[-1][0] + np.sin(a) * length / 4, pts[-1][1] - np.cos(a) * length / 4)
        pts.append(q)
    ws = np.linspace(w0, max(1.0, w0 * 0.3), 5)
    ts = [t0 + length * k / 4 / speed for k in range(5)]
    for k in range(4):
        segs.append((pts[k], pts[k + 1], ws[k], ws[k + 1], ts[k], ts[k + 1]))
    if depth > 0:
        n = 3 if depth > 1 else 2
        for i in range(n):
            k = int(r.integers(1, 4))
            s = 1 if i % 2 == 0 else -1
            sub_a = float(np.clip(ang + s * r.uniform(0.3, 0.8), -1.5, 1.5))
            _branch(segs, pts[k], sub_a, length * r.uniform(0.35, 0.55), max(1.0, ws[k] * 0.6),
                    ts[k] + 0.08, speed * 0.85, depth - 1, r, tips)
    else:
        tips.append((pts[-1], ts[-1]))


@lru_cache(None)
def tree_arrays():
    """返回 (颜色 RGB, 出现时刻 tb[拍], 光晕出现时刻, 根须 RGB, 根须 tb, 晶簇列表)。"""
    r = _rng(7)
    segs, tips = [], []
    bx, by = TREE_BASE
    top = -46
    # 树干：1.1 破土，约 0.6 拍冲到顶
    tr = [(bx + 7 * np.sin((by - y) / 55), y) for y in np.linspace(by, top, 9)]
    tw = np.linspace(30, 7, 9)
    tt = [1.0 + 0.6 * (by - y) / (by - top) for _, y in tr]
    for k in range(8):
        segs.append((tr[k], tr[k + 1], tw[k], tw[k + 1], tt[k], tt[k + 1]))
    # 枝层：(树干高度, 拍, 长度, 夹角)
    tiers = [(132, 2.0, 180, 1.2), (112, 2.0, 160, 1.05), (88, 2.5, 150, 0.95), (62, 3.0, 135, 0.8),
             (40, 3.0, 115, 0.7), (14, 3.5, 95, 0.55), (-12, 3.5, 70, 0.4), (-36, 3.75, 45, 0.25)]
    for j, (y, beat, ln, ang) in enumerate(tiers):
        px = bx + 7 * np.sin((by - y) / 55)
        for s in (-1, 1):
            if j == 2 and s == 1 or j == 4 and s == -1:     # 左右错开，不那么对称
                continue
            a = s * ang * r.uniform(0.8, 1.2)
            _branch(segs, (px, y + r.uniform(-6, 6)), a, ln * r.uniform(0.85, 1.15), 7 * (ln / 180) + 2.5, beat,
                    260, 2, r, tips)
    # 树干根部的板根
    for s in (-1, 1):
        segs.append(((bx, by - 30), (bx + s * 26, by + 4), 12, 5, 1.0, 1.15))
    tb = np.full((HW, WW), np.inf, np.float32)
    nd = np.ones((HW, WW), np.float32)
    fac = np.zeros((HW, WW), np.int8)
    for p0, p1, w0, w1, t0, t1 in segs:
        _seg(tb, nd, fac, p0, p1, w0, w1, t0, t1)
    # 根须：1.0 起在街面下蔓延（先是裂缝里透出的光），1.1 随树干一起顶破地面
    rtb = np.full((HW, WW), np.inf, np.float32)
    rnd = np.ones((HW, WW), np.float32)
    rfac = np.zeros((HW, WW), np.int8)
    for k in range(9):
        tx = -40 + k * 70 + r.uniform(-20, 20)
        ty = 226 + r.uniform(0, 10)
        pts = [(bx, by + 8)]
        for j in range(1, 6):
            f = j / 5
            pts.append((bx + (tx - bx) * f + r.uniform(-8, 8), by + 8 + (ty - by - 8) * f + np.sin(f * 5 + k) * 4))
        for j in range(5):
            w0 = 5 * (1 - j / 5) + 1
            _seg(rtb, rnd, rfac, pts[j], pts[j + 1], w0, w0 * 0.8, 0.15 + j * 0.3, 0.15 + (j + 1) * 0.3)
    def colorize(tb_, nd_, fac_):
        col = np.zeros((HW, WW, 3), np.float32)
        pal = [[C('#eafffb'), C('#c8fff6')], [C('#8ff5e6'), C('#5ce6d8')], [C('#38c8c8'), C('#2aa8b8')],
               [C('#1a7fa0'), C('#125e84')]]
        lvl = np.digitize(nd_, [0.3, 0.6, 0.85])
        light = (fac_ % 2 == 1)
        for i in range(4):
            for j in range(2):
                m = (lvl == i) & (light == (j == 0))
                col[m] = pal[i][j]
        ring = np.isfinite(tb_)
        edge = ring & ~(np.roll(ring, 1, 0) & np.roll(ring, -1, 0) & np.roll(ring, 1, 1) & np.roll(ring, -1, 1))
        col[edge] = C('#0a3a5a')
        return col
    col = colorize(tb, nd, fac)
    rcol = colorize(rtb, rnd, rfac)
    # 光晕：出现时刻向外扩 3px
    g = tb.copy()
    for _ in range(3):
        g = np.minimum.reduce([g, np.roll(g, 1, 0), np.roll(g, -1, 0), np.roll(g, 1, 1), np.roll(g, -1, 1)])
    g += 0.05
    # 树冠的光：每个枝梢一团青白色的光晕（抖动铺点），出现时刻 = 枝梢长成 + 0.2 拍
    can = np.full((HW, WW), np.inf, np.float32)
    cand = np.ones((HW, WW), np.float32)
    yy, xx = np.mgrid[0:HW, 0:WW]
    for (px, py), tt in tips:
        R = r.uniform(9, 16)
        ix0, ix1 = int(px - R) + X0, int(px + R) + X0 + 1
        iy0, iy1 = int(py - R) + Y0, int(py + R) + Y0 + 1
        if ix1 <= 0 or iy1 <= 0 or ix0 >= WW or iy0 >= HW:
            continue
        ix0, iy0, ix1, iy1 = max(ix0, 0), max(iy0, 0), min(ix1, WW), min(iy1, HW)
        d = np.hypot(xx[iy0:iy1, ix0:ix1] - X0 + .5 - px, yy[iy0:iy1, ix0:ix1] - Y0 + .5 - py) / R
        m = d < 1
        reg, regd = can[iy0:iy1, ix0:ix1], cand[iy0:iy1, ix0:ix1]
        reg[m] = np.minimum(reg[m], tt + 0.2)
        regd[m] = np.minimum(regd[m], d[m])
    return col, tb, g, rcol, rtb, tips, can, cand, rnd


CRYSTAL = ["..W..", ".WCW.", "WCCCW", ".CAC.", "..A.."]


@lru_cache(None)
def crystal_img():
    pal = {'W': '#ffffff', 'C': '#9ff8ec', 'A': '#2ab8c8'}
    img = np.zeros((5, 5, 4), np.float32)
    for y, row in enumerate(CRYSTAL):
        for x, ch in enumerate(row):
            if ch != '.':
                img[y, x, :3] = C(pal[ch])
                img[y, x, 3] = 1
    return img


# ================================================================ 小工具
def _crop(arr, cx, cy, f):
    ix, iy = int(round(X0 + cx * f)), int(round(Y0 + cy * f))
    return arr[iy:iy + H, ix:ix + W]


def _over(dst, rgba):
    a = rgba[..., 3:4]
    dst[:] = dst * (1 - a) + rgba[..., :3] * a


def _sprite(pose, scale=1):
    img = CAMPA.body(pose)
    return np.repeat(np.repeat(img, scale, 0), scale, 1) if scale > 1 else img


def _silhouette(spr, amt, rim, rim_dirs=((0, -1), (-1, 0)), eyes=(), eye_col=None, scale=1):
    """暗色剪影 + 朝光一侧的轮廓光 + 反光的眼睛。"""
    out = spr.copy()
    a = out[..., 3] > 0
    out[a, :3] = out[a, :3] * (1 - amt) + C('#070812') * amt
    edge = np.zeros_like(a)
    for dx, dy in rim_dirs:
        edge |= a & ~np.roll(np.roll(a, -dy, 0), -dx, 1)
    out[edge, :3] = C(rim) if isinstance(rim, str) else rim
    if eye_col is not None:
        for (y, x) in eyes:
            out[y * scale:(y + 1) * scale, x * scale:(x + 1) * scale, :3] = C(eye_col)
    return out


def _zoom(img, z, cx, cy):
    if abs(z - 1) < 1e-3:
        return img
    ys = np.clip(((np.arange(H) + .5 - H / 2) / z + cy).astype(int), 0, H - 1)
    xs = np.clip(((np.arange(W) + .5 - W / 2) / z + cx).astype(int), 0, W - 1)
    return img[ys][:, xs]


def _card(sx):
    """小扑克牌（6x8）：紫色牌背、金边；sx 为水平缩放（翻面）。"""
    base = np.zeros((8, 6, 4), np.float32)
    base[..., 3] = 1
    base[..., :3] = C('#3a1250')
    base[[0, -1], :, :3] = C('#e0b050')
    base[:, [0, -1], :3] = C('#e0b050')
    base[3:5, 2:4, :3] = C('#ffd870')
    if sx < 0:
        base[1:-1, 1:-1, :3] = C('#f6f0e0')
        base[3:5, 2:4, :3] = C('#c02040')
    w = max(1, int(round(abs(sx) * 6)))
    xs = (np.arange(w) * 6 / w).astype(int)
    return base[:, xs]


def _ledge(dst, y_top, x0=250, rim=0.0, t=0.0):
    """前景的楼顶栏杆（屏幕坐标）：石质压顶 + 栏柱，朝向巨树的上沿带一点青光。"""
    top = int(y_top)
    dst[top:, x0:] = C('#07070f')
    dst[top:top + 5, x0 - 6:] = C('#141424')
    dst[top, x0 - 6:] = C('#2a3050') * (1 - rim) + C('#7ff0e4') * rim
    for x in range(x0 + 4, W, 16):
        dst[top + 5:top + 30, x:x + 5] = C('#10101c')
        dst[top + 5:top + 30, x] = C('#1c2036')
    dst[top + 30:top + 33, x0:] = C('#141424')


# ================================================================ 渲染
FPS_POSE = 12


def _shot(b):
    if b < 4:
        return 'A'
    if b < 8:
        return 'B'
    if b < 12:
        return 'C'
    if b < 14:
        return 'D'
    return 'E'


def _world(dst, ctx, b, cx, cy, lights_k, ring_p=0.0, eye_open=0.0):
    """画城市全景（天空、蛇眼、远楼、巨树、中景楼、根须、近景）。返回屏幕坐标下的树干中点与火环参数。"""
    t = ctx.t
    dst[:] = _crop(sky_layer(), cx, cy, PAR['sky'])
    grow = smooth(1.0, 4.0, b)
    # 巨树把天空染成青色
    tx, ty = TREE_BASE[0] - cx * PAR['mid'], 60 - cy * PAR['mid']
    if grow > 0:
        d = np.hypot((XX - tx) / 1.3, YY - ty)
        k = np.clip(1 - d / 260, 0, 1) ** 1.6 * 0.55 * grow
        dst += (BAYER < k * 1.6)[..., None] * C('#0e3a48') * 0.8 + k[..., None] * C('#0a2a38')
    # 蛇眼（在树后）
    if eye_open > 0:
        ex, ey, hw, hh = EYE
        glow = np.clip(1 - np.hypot((XX - ex) / (hw * 1.4), (YY - ey) / (hh * 2.2)), 0, 1) * eye_open
        dst += (BAYER < glow * 0.8)[..., None] * C('#1a4a18') * 0.6
        fx.serpent_eye(dst, ex, ey, hw, hh, eye_open, t)
    # 火环后半圈：在天际线后面燃起一道火墙，把楼群逆光勾成剪影
    if ring_p > 0:
        hz = np.clip(1 - np.abs(YY - (RING[1] - RING[3] - cy * PAR['mid'])) / 60, 0, 1) * min(1, ring_p * 1.5)
        dst += (hz ** 2 * 0.45)[..., None] * C('#ff5a10')
        _ring(dst, ctx, cx, cy, ring_p, back=True)
        ctx.P.render(dst)
    # 远楼
    far, fblk = far_layer()
    fc = _crop(far, cx, cy, PAR['far']).copy()
    fb = _crop(fblk, cx, cy, PAR['far'])
    off = (fb >= 0) & ~lights_k(fb)
    fc[off, :3] = C('#161c40')
    _over(dst, fc)
    # 巨树
    col, tb, tg, rcol, rtb, tips, can, cand, rnd = tree_arrays()
    tbc = _crop(tb, cx, cy, PAR['mid'])
    m = tbc <= b
    canc = _crop(can, cx, cy, PAR['mid'])
    cm = canc <= b
    if cm.any():                                           # 树冠光晕（在枝条后面）
        dd = _crop(cand, cx, cy, PAR['mid'])
        lvl = np.clip((1 - dd) * 0.9 * np.clip((b - canc) / 0.6, 0, 1), 0, 1)
        halo = cm & (BAYER < lvl * 0.8)
        dst[halo] = dst[halo] * 0.4 + np.where((lvl[halo] > 0.55)[:, None], C('#bff8f0'), C('#2ab8c0'))
    if m.any():
        tgc = _crop(tg, cx, cy, PAR['mid'])
        halo = (tgc <= b) & ~m & (BAYER < 0.55)
        dst[halo] = dst[halo] * 0.5 + C('#1aa0b0') * 0.6
        cc = _crop(col, cx, cy, PAR['mid'])
        dst[m] = cc[m]
        front = m & (tbc > b - 0.18)
        dst[front] = C('#ffffff')
        # 晶面的缓慢流光
        shine = m & (np.sin((XX + YY) * 0.08 - t * 3) > 0.93)
        dst[shine] = np.minimum(dst[shine] + 0.25, 1)
        for (px, py), tt in tips:
            if tt + 0.2 <= b:
                sx, sy = px - cx * PAR['mid'], py - cy * PAR['mid']
                if 0 <= sx < W and 0 <= sy < H and (int(t * 8 + px) % 5):
                    blit(dst, crystal_img(), sx, sy, anchor='center')
    # 中景楼 + 窗
    mid, mblk, _ = mid_layer()
    mc = _crop(mid, cx, cy, PAR['mid']).copy()
    mb = _crop(mblk, cx, cy, PAR['mid'])
    off = (mb >= 0) & ~lights_k(mb)
    mc[off, :3] = C('#0c1028')
    _over(dst, mc)
    # 奥基斯大厦的红色航标灯
    ox, oy = ORCHIS[0] - cx * PAR['mid'], ORCHIS[1] - 34 - cy * PAR['mid']
    if int(t * 1.6) % 2 == 0 and 0 <= ox < W and 0 <= oy < H:
        dst[int(oy) - 1:int(oy) + 1, int(ox) - 1:int(ox) + 2] = C('#ff3030')
    # 根须：裂缝里透出的光 -> 顶破街面
    rtbc = _crop(rtb, cx, cy, PAR['mid'])
    rm = rtbc <= b
    if rm.any():
        rc = _crop(rcol, cx, cy, PAR['mid'])
        crack = rm & (b < 1.0) & (_crop(rnd, cx, cy, PAR['mid']) < 0.4)
        dst[crack] = C('#3ae0d0') * (0.6 + 0.4 * np.sin(t * 20))
        solid = rm & (b >= 1.0)
        dst[solid] = rc[solid]
    # 火环（前半圈）；后半圈的火舌画在楼后、但高过屋顶的部分可见 -> 这里一起画
    if ring_p > 0:
        _ring(dst, ctx, cx, cy, ring_p, back=False)
        ctx.P2.render(dst)
    # 近景屋顶
    _over(dst, _crop(near_layer(), cx, cy, PAR['near']))


def _ring(dst, ctx, cx, cy, p, back):
    """幻焰之环：从画面右前方（他脚下一侧）起，两路火头沿椭圆狂奔，p=1 时在左后方相撞闭合。
    一道亮线 + 向上 30px 的抖动火光 + 大量火焰粒子（后半圈的粒子在中景楼之后才画，火舌高过屋顶）。"""
    ex, ey, rx, ry = RING
    ex, ey = ex - cx * PAR['mid'], ey - cy * PAR['mid']
    a0 = 0.12 * np.pi                                      # 起点：右侧（他脚下一侧）
    span = np.pi * min(p, 1.0)
    P = ctx.P if back else ctx.P2
    line = np.zeros((H, W), bool)
    for sgn in (1, -1):
        angs = a0 + sgn * np.linspace(0, span, 240)
        xs, ys = ex + rx * np.cos(angs), ey + ry * np.sin(angs)
        sel = (ys < ey) if back else (ys >= ey)
        xs, ys = xs[sel], ys[sel]
        if not len(xs):
            continue
        xi, yi = xs.astype(int), ys.astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        line[yi[ok], xi[ok]] = True
        if ctx.dt > 0:
            k = min(len(xs), 110)
            idx = ctx.rng.integers(0, len(xs), k)
            P.emit_arrays(xs[idx] + ctx.rng.uniform(-2, 2, k), ys[idx], ctx.rng.uniform(-8, 8, k),
                          ctx.rng.uniform(-150, -40, k), (0.25, 0.6), 'fire',
                          size=np.where(ctx.rng.random(k) < 0.5, 2, 1), drag=0.9, wob=0.2)
            hx, hy = ex + rx * np.cos(angs[-1]), ey + ry * np.sin(angs[-1])
            if (hy < ey) == back:                          # 火头：更大、更亮，甩出蓝色火星
                P.emit(18, (hx - 4, hx + 4), (hy - 6, hy + 1), (-40, 40), (-170, -60), (0.25, 0.6), 'fire', size=3,
                       drag=1.0)
                P.emit(8, hx, hy - 4, (-90, 90), (-120, -10), (0.4, 0.9), 'blue', size=(1, 2), drag=1.4, bright=1.3)
    if not line.any():
        return
    fl = 0.8 + 0.2 * np.sin(ctx.t * 23)
    for k in range(2, 32, 2):                              # 线上方的火光（越高越稀）
        sh = np.zeros_like(line)
        sh[:-k] = line[k:]
        m = sh & (BAYER < (1 - k / 32) * 0.7 * fl)
        dst[m] = dst[m] * 0.4 + C('#ff7a1a') * 0.8
    thick = line | np.roll(line, -1, 0) | np.roll(line, 1, 1)
    dst[thick] = C('#ff9a30')
    dst[line] = C('#fff0a0')


def _lights(b, frame):
    """窗户块号 -> 是否亮着。3.0.5 起从大树附近一块块闪烁熄灭，奥基斯大厦最后。"""
    order = sorted(range(13), key=lambda k: abs(k * 50 - 60 + 25 - 240))
    t_off = {k: 8.5 + i * 0.22 for i, k in enumerate(order)}
    t_off[99] = 11.2
    lut = np.ones(128, bool)
    for k, to in t_off.items():
        if b >= to:
            lut[k] = False
        elif b >= to - 0.45:
            lut[k] = ((frame * 7 + k * 13) % 5) not in (0, 2)
    return lambda arr: lut[np.clip(arr, 0, 127)]


def render(dst, ctx):
    lt, beat = ctx.lt, ctx.sec.beat
    b = lt / beat
    t = ctx.t
    shot = _shot(b)
    frame = int(lt * 30)
    lights = _lights(b, frame)
    eye_open = smooth(8.0, 9.0, b) ** 0.7 if b >= 8 else 0.0
    st = ctx.state

    # ---- 一次性事件（震屏 / 闪光 / 粒子）
    if ctx.crossed(1, 0.0):
        ctx.shake(3, 1.2)
    if ctx.crossed(1, 1.0):
        ctx.shake(7, 0.6)
        ctx.flash('#bff8f0', 0.12, 0.55)
    for bb, kk in ((1, 2.0), (1, 3.0), (1, 3.5)):
        if ctx.crossed(bb, kk):
            ctx.shake(3, 0.25)
    if ctx.crossed(3, 0.0):
        ctx.shake(5, 0.5)
    if ctx.crossed(4, 3.5):
        ctx.shake(6, 0.4)

    if shot == 'A':
        # 低机位全景：1.0 城市 -> 1.1 树干冲天，镜头跟着上摇
        cy = 30 - 86 * smooth(0.9, 3.8, b) + np.sin(t * 40) * (1.5 if 0.1 < b < 1.0 else 0)
        _world(dst, ctx, b, 0, cy, lights)
        if ctx.crossed(1, 1.0):
            x, y = TREE_BASE[0], TREE_BASE[1] - cy * PAR['mid']
            ctx.P.emit(160, (x - 30, x + 30), (y - 20, y + 5), (-90, 90), (-160, -30), (0.5, 1.3), 'mint',
                       size=(1, 2), grav=120, drag=0.6)
        ctx.P.step(ctx.dt)
        ctx.P.render(dst)

    elif shot in ('B', 'D'):
        # 高楼边缘的 2x 中景：巨树在画面左侧的远处
        cx, cy = (90 - 6 * smooth(4, 8, b), -30) if shot == 'B' else (70, -38)
        _world(dst, ctx, b, cx, cy, lights, eye_open=eye_open)
        dst *= 0.72
        ledge_y = 196
        _ledge(dst, ledge_y, rim=0.8)
        k = b - (4 if shot == 'B' else 12)
        if shot == 'B':
            f = int(lt * FPS_POSE)
            seq = CAMPA.meta.get('seq', {}).get('sit', ['sit'])
            if k < 1.0:
                pose = seq[(f // 3) % len(seq)]
            elif k < 1.5:
                pose = 'sit_toss'
            elif k < 2.5:
                pose = 'sit_0'
            elif k < 3.0:
                pose = 'sit_1'
            else:
                pose = 'sit_flick' if k < 3.6 else 'sit_0'
            spr = _sprite(pose, 2)
            seat = CAMPA.meta.get('seat_y', 51) * 2
            x0, y0 = 352, ledge_y - seat + 2
        else:
            seq = CAMPA.meta.get('seq', {}).get('snap', ['idle', 'snap'])
            if k < 0.4:
                pose = 'idle'
            elif k < 1.0:
                pose = seq[min(len(seq) - 1, 1 + int((k - 0.4) / 0.6 * (len(seq) - 1)))]
            else:
                pose = 'snap' if k < 1.7 else 'snap_2'
            spr = _sprite(pose, 2)
            x0, y0 = 340, ledge_y - spr.shape[0] + 2
        eyes = CAMPA.meta.get('eyes_body', {}).get(pose, [])
        sil = _silhouette(spr, 0.8, '#7ff0e4', eyes=eyes, eye_col='#ffd870', scale=2)
        blit(dst, sil, x0, y0)
        hands = CAMPA.meta.get('hands', {}).get(pose, {})
        if shot == 'B':
            _card_business(dst, ctx, k, x0, y0, hands)
        else:
            _snap_business(dst, ctx, k, x0, y0, hands)
        ctx.P.step(ctx.dt)
        ctx.P.render(dst)

    elif shot == 'C':
        # 全景：巨树背后睁开蛇眼，城市的灯一片片熄灭；他的剪影坐在右下角
        cy = -52 - 6 * smooth(8, 12, b)
        _world(dst, ctx, b, 0, cy, lights, eye_open=eye_open)
        _ledge(dst, 236, x0=404, rim=0.5)
        spr = _silhouette(_sprite('sit_0'), 0.9, '#5ad8d0', eyes=CAMPA.meta.get('eyes_body', {}).get('sit_0', []),
                          eye_col='#ffd870')
        blit(dst, spr, 420, 236 - CAMPA.meta.get('seat_y', 51) + 1)
        if ctx.crossed(3, 0.0):
            ex, ey = EYE[:2]
            ctx.P.emit_radial(120, ex, ey, (40, 200), (0.4, 1.2), 'green', size=1, drag=1.5)
        ctx.P.step(ctx.dt)
        ctx.P.render(dst)

    else:  # 'E' 火环
        p = smooth(14.0, 15.5, b) if b < 15.5 else 1.0
        p = 0.15 + 0.85 * p
        cy = -6
        ctx.P.step(ctx.dt)
        ctx.P2.step(ctx.dt)
        _world(dst, ctx, b, 0, cy, lights, ring_p=p, eye_open=1.0)
        # 火光把城市下半部染橙
        k = 0.12 + 0.25 * p + (0.3 * smooth(15.5, 16, b))
        dst += (np.clip((YY - 120) / 150, 0, 1) * k)[..., None] * C('#ff6a18')
        _ledge(dst, 240, x0=410, rim=0.2)
        spr = _silhouette(_sprite('snap_2'), 0.85, '#ffa040')
        blit(dst, spr, 424, 240 - spr.shape[0] + 1)
        if ctx.crossed(4, 3.5):
            ex, ey = RING[0] - 180, RING[1] - cy * PAR['mid'] - 20
            ctx.P2.emit_radial(200, ex, ey, (60, 260), (0.4, 1.0), 'fire', size=(1, 2), drag=1.4)
            ctx.P2.emit_radial(60, ex, ey, (40, 200), (0.4, 1.0), 'blue', size=1, drag=1.2)
        # 最后半拍：急推向巨树与蛇眼，画面烧亮
        if b >= 15.5:
            q = smooth(15.5, 16.0, b)
            dst += C('#ffd8a0') * 0.5 * q ** 2
            np.clip(dst, 0, 1, out=dst)
            dst = _zoom(dst, 1 + 0.9 * q ** 2, 240, 120)
    np.clip(dst, 0, 1, out=dst)
    return dst


def _card_business(dst, ctx, k, x0, y0, hands):
    """2.1 抛牌 -> 空中翻转 -> 2.2.5 接住 -> 2.3 反手甩向巨树。k 为本镜头内的拍数。"""
    hy, hx = hands.get('l', (27, 28))
    hx, hy = x0 + hx * 2, y0 + hy * 2
    if 1.0 <= k < 2.5:
        u = (k - 1.0) / 1.5
        cxp, cyp = hx - 6 * u, hy - 70 * 4 * u * (1 - u)
        sx = np.cos(u * TAU * 2.5)
        img = _card(sx)
        blit(dst, img, cxp, cyp, scale=2, anchor='center')
    elif k >= 3.0:
        u = (k - 3.0) / 0.9
        if u < 1:
            sx0, sy0 = hx, hy
            ex, ey = 120, 60
            cxp, cyp = sx0 + (ex - sx0) * ease_out(u), sy0 + (ey - sy0) * ease_out(u)
            for j in range(1, 6):                         # 残影
                v = max(0, u - j * 0.03)
                px, py = sx0 + (ex - sx0) * ease_out(v), sy0 + (ey - sy0) * ease_out(v)
                dst[int(py) - 1:int(py) + 1, int(px) - 1:int(px) + 1] = C('#e0b050') * (1 - j / 6)
            blit(dst, _card(np.cos(u * 30)), cxp, cyp, scale=2 if u < 0.5 else 1, anchor='center')
    else:
        blit(dst, _card(1.0), hx, hy, scale=2, anchor='center')


def _snap_business(dst, ctx, k, x0, y0, hands):
    """4.1 响指：冲击帧（反相两帧）-> 指尖炸开橙色火花 + 蓝火星，一圈小火环向外扩散。"""
    hy, hx = hands.get('l', (10, 35))
    hx, hy = x0 + hx * 2, y0 + hy * 2
    if ctx.crossed(4, 1.0):
        ctx.P.emit_radial(90, hx, hy, (40, 220), (0.3, 0.9), 'fire', size=(1, 2), drag=2.5)
        ctx.P.emit_radial(24, hx, hy, (30, 160), (0.4, 1.0), 'blue', size=1, drag=1.8, bright=1.2)
        ctx.shake(4, 0.3)
    s = ctx.since(4, 1.0)
    if 0 <= s < 0.062:                                    # 冲击帧
        lum = dst.mean(-1)
        dst[:] = np.where((lum > 0.3)[..., None], C('#1a0804'), C('#ffe8c0'))
    elif s >= 0:
        r = 4 + s * 260
        if r < 300:
            fx.shockwave(dst, hx, hy, r, C('#ff9a30'), w=2)
        g = np.hypot(XX - hx, YY - hy) < 3 + 2 * np.sin(ctx.t * 30)
        dst[g] = C('#fff0b0')


# ================================================================ 专属音效
def _sfx_crack():
    """地底裂开：低频碎裂 + 稀疏的石块崩裂声。"""
    n = int(1.4 * S.SR)
    tt = S.tt(n)
    x = S.lp(S.noise(n), 300, 2) * np.clip(tt / 0.3, 0, 1) * np.exp(-tt / 0.9) * 1.5
    r = np.random.default_rng(5)
    for i in r.integers(0, n - 2000, 22):
        x[i:i + 2000] += S.bp(S.noise(2000), 400, 3000) * np.exp(-np.arange(2000) / 180) * r.uniform(0.2, 0.7)
    return x * 0.8


def _sfx_erupt():
    """巨树破土：低沉的轰鸣上扬 + 晶体的高频泛音群。"""
    n = int(2.2 * S.SR)
    tt = S.tt(n)
    f = 40 + 90 * (1 - np.exp(-tt / 0.5))
    rumble = np.tanh(1.8 * np.sin(2 * np.pi * np.cumsum(f) / S.SR)) * np.exp(-tt / 1.0)
    whoosh = S.sweep_lp(S.noise(n), 200, 5000, 0.6) * np.exp(-tt / 0.7)
    shim = sum(np.sin(2 * np.pi * fr * tt) * np.exp(-tt / 1.2) for fr in (1318.5, 1975.5, 2637.0, 3951.1)) * 0.07
    return (0.8 * rumble + 0.4 * whoosh + shim * np.clip((tt - 0.1) / 0.2, 0, 1)) * 0.9


def _sfx_crystal():
    """枝桠展开：清脆的晶体生长声（快速的高音琶音 + 玻璃般的泛音）。"""
    n = int(1.5 * S.SR)
    x = np.zeros(n)
    for i, m in enumerate((88, 95, 100, 104, 107)):
        o = int(i * 0.035 * S.SR)
        x[o:] += S.inst_bell(m, 1.5)[:n - o] * (0.5 - i * 0.05)
    tt = S.tt(n)
    x += S.hp(S.noise(n), 6000) * np.exp(-tt / 0.05) * 0.2
    return x * 0.55


def _sfx_eye():
    """蛇眼睁开：反向的低音膨胀 + 湿黏的拉扯声 + 一声低沉的"嗡"。"""
    n = int(2.0 * S.SR)
    tt = S.tt(n)
    swell = S.lp(S.noise(n), 500, 2) * (tt / 0.5).clip(0, 1) ** 2 * np.exp(-np.clip(tt - 0.5, 0, None) / 0.4) * 1.6
    hum = sum(np.sin(2 * np.pi * f * tt) for f in (41.2, 82.4, 123.5)) * np.clip(tt / 0.4, 0, 1) * np.exp(-tt / 1.1)
    return (0.5 * swell + 0.5 * hum) * 0.9


def _sfx_off():
    """一片灯熄灭：电流"嗡"地断掉 + 继电器的咔哒。"""
    n = int(0.45 * S.SR)
    tt = S.tt(n)
    f = 120 * np.exp(-tt / 0.15)
    hum = np.sign(np.sin(2 * np.pi * np.cumsum(f) / S.SR)) * np.exp(-tt / 0.12) * 0.25
    click = S.hp(S.noise(n), 3000) * np.exp(-tt / 0.004) * 0.6
    return S.lp(hum, 1500) + click


SFX = {'ph_crack': _sfx_crack, 'ph_erupt': _sfx_erupt, 'ph_crystal': _sfx_crystal, 'ph_eye': _sfx_eye,
       'ph_off': _sfx_off}
