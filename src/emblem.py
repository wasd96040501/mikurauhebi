"""结社《身喰らう蛇》(Ouroboros) 纹章的像素版。

照着官方纹章画（空之轨迹 SC 起使用的社徽，Falcom 官方周边「軌跡シリーズ 紋章ピンズ」C 款同款）：
紫灰色波浪边徽牌 + 银白内场，内场上方七颗绿宝珠排成拱形，正中一颗倒水滴形蓝宝石，
下半部一条黑蛇绕过徽牌打结，身上套三个金环，蛇头在正中朝下、绿眼。

    emblem(size=96, progress=1.0, palette='gold') -> float32 (size, size, 4) RGBA 0..1

几何全部定义在参考图坐标系（486x610 的官方图，徽牌包围盒 x 9..477, y 11..602）里，
按 size 采样成像素；每个 size 的几何缓存一次，之后换 progress / palette 只做掩码拼色。
"""
from functools import lru_cache

import numpy as np
from scipy import ndimage as nd
from scipy.spatial import cKDTree

from pixelkit import hexc

EMBLEM_NOTES = (
    "Ouroboros (身喰らう蛇) society crest as used in-game since Trails in the Sky SC and on "
    "Falcom's official emblem pin: an upright scalloped slate-purple plaque (pointed top, "
    "three cusped lobes per side, small flared foot) framing a pale silver field. In the "
    "field: seven lime-green orbs in dark-navy bezels arranged in a horseshoe arch (one on "
    "top, three down each side), dark leaf/horn filigree, and a large blue teardrop gem "
    "(point up) with a dark crescent under it. Across the lower half a single black snake "
    "wraps around the plaque: its body curls round the left and right edges (thin gold ring "
    "on each), passes behind, and knots in the centre - the neck arches through a large gold "
    "cuff and the head points down at the centre with a green eye, while the other end runs "
    "down in an S-curve so the thin tail hangs into the plaque's foot."
)

# ---------------------------------------------------------------- 参考几何（参考图坐标）

CX, CY = 243.0, 306.5          # 包围盒中心
REF_H = 594.0                  # 包围盒高度（含 1px 余量）

# 徽牌外轮廓：以 (243, 306) 为圆心的极坐标半径表，120 个角度桶，从 -pi 开始（左侧）逆时针。
# 从官方图 alpha 通道提取，左右取 min 对称化（去掉蛇身盘出去的部分）。
_SIL = np.array([
    212, 213, 212, 210, 206, 207, 219, 240, 261, 264, 253, 242, 235, 241, 252, 256, 256, 255,
    255, 261, 276, 274, 267, 268, 269, 270, 273, 278, 284, 295, 295, 284, 278, 273, 270, 269,
    268, 267, 274, 276, 261, 255, 255, 256, 256, 252, 241, 235, 242, 253, 264, 261, 240, 219,
    207, 206, 210, 212, 213, 212, 209, 202, 194, 186, 190, 199, 204, 209, 213, 217, 218, 219,
    219, 219, 228, 235, 237, 237, 235, 232, 240, 244, 246, 246, 244, 240, 246, 291, 295, 296,
    296, 295, 291, 246, 240, 244, 246, 246, 244, 240, 232, 235, 237, 237, 235, 228, 219, 219,
    219, 218, 217, 213, 209, 204, 199, 190, 186, 194, 202, 209], np.float32)
_SIL_C = (243.0, 306.0)

# 七颗绿宝珠（左右对称，从左下沿拱形到右下 = 点亮顺序）
ORBS = [(160, 245), (157, 190), (185, 140), (243, 112), (301, 140), (329, 190), (326, 245)]
ORB_R, ORB_G = 20.0, 12.5      # 镶座外半径 / 绿珠半径

GEM_C, GEM_R, GEM_TIP = (243.0, 234.0), 29.0, 163.0

# 蛇：A 段 = 尾尖 -> S 形上行 -> 从蛇头下穿过 -> 右侧身体 -> 右边缘绕到徽牌后面
#     B 段 = 从徽牌左边缘后面绕出来 -> 左侧身体 -> 大金箍 -> 拱起的脖子 -> 蛇头（尖朝下）
SNAKE_A = [  # (x, y, 半径)
    (249, 586, 2.0), (249, 568, 4.5), (246, 546, 7), (251, 520, 10), (258, 496, 12),
    (252, 470, 14), (236, 448, 15), (224, 422, 16), (222, 398, 17), (232, 377, 17),
    (256, 362, 18), (284, 370, 20), (306, 384, 21), (336, 402, 22), (366, 418, 22),
    (396, 428, 22), (424, 418, 21), (442, 392, 20), (440, 364, 19), (424, 346, 19),
    (402, 338, 19),
]
SNAKE_B = [
    (86, 336, 21), (60, 342, 22), (40, 362, 24), (34, 390, 25), (52, 414, 26), (82, 422, 27),
    (112, 414, 27), (142, 400, 26), (166, 380, 25), (186, 354, 24), (202, 332, 23),
    (222, 316, 22), (244, 312, 23), (259, 322, 24), (264, 338, 24),
]
# 蛇头：从头顶 top 指向吻端 tip 的圆头楔形，最宽半宽 W
HEAD = dict(top=(250, 308), tip=(281, 404), W=30.0)
# 金环：(段, 环心附近的参考点, 沿身体的半长, 比身体多出的半径)
RINGS = [('A', (364, 419), 7.0, 3.5), ('B', (48, 396), 6.0, 3.5), ('B', (178, 364), 17.0, 6.0)]
EYE = (244, 356)
# 徽牌边缘带里的涡卷 (cx, cy, 半径, 起始角, 圈数)，只写左半边
CURLS = [(58, 212, 15, 0.5, 0.9), (72, 292, 13, -2.2, 0.9), (122, 468, 14, 3.4, 0.9),
         (178, 530, 12, 2.6, 0.9), (148, 78, 12, 1.2, 0.8), (214, 44, 9, 0.2, 0.8)]

LIGHT = np.array([-0.55, -0.75, 0.95], np.float32)
LIGHT /= np.linalg.norm(LIGHT)

# ---------------------------------------------------------------- 调色板

_PAL = {
    'gold': dict(  # 官方配色（取自官方图）：石板紫徽牌、银白内场、黑蛇、金环、绿珠、蓝宝石
        line='#0c0a16',
        plq=['#2b2946', '#46446a', '#5e5c86', '#8482b0'],
        plq_eng='#5c5a86',
        fld=['#9796b6', '#b8b8d2', '#d6d7ea', '#f2f2fa'],
        fld_lo='#65638e',
        orn='#2c2a54',
        bez=['#1f1b4a', '#453e82'],
        grn=['#6c9a1a', '#b7e84f', '#e4ff8e', '#ffffff'],
        gem=['#0b1f7a', '#2150dc', '#4c9cff', '#dff6ff'],
        snk_line='#050408',
        snk=['#0d0c15', '#1e1c2c', '#35324c', '#5f5b84'],
        dot='#3a3752',
        au=['#6e5410', '#b89a26', '#ecd84c', '#fffbb0'],
        eye=['#46e07a', '#c8ffd8'],
    ),
    'glow': dict(  # 闪光用高亮版
        line='#7a3cd8',
        plq=['#9a8cf0', '#b8aefc', '#d4ccff', '#f0ecff'],
        plq_eng='#e4dcff',
        fld=['#e6e2ff', '#f4f2ff', '#ffffff', '#ffffff'],
        orn='#a890ff',
        fld_lo='#ece8ff',
        bez=['#8a78f0', '#b0a0ff'],
        grn=['#d8ff70', '#f0ffa8', '#ffffe0', '#ffffff'],
        gem=['#70b0ff', '#a8d8ff', '#e0f4ff', '#ffffff'],
        snk_line='#5a28b8',
        snk=['#8068d8', '#9c88ec', '#bcaeff', '#e6e0ff'],
        dot='#9c88ec',
        au=['#ffd060', '#ffe490', '#fff4c8', '#ffffff'],
        eye=['#b0ffc8', '#ffffff'],
    ),
}


_INK, _CLR = '#140c1c', 'clear'
_PAL['mono'] = dict(  # 单色图章：墨色 + 挖空（透明）
    line=_INK, plq=[_INK, _INK, _INK, _CLR], plq_eng=_INK, fld=[_CLR] * 4, fld_lo=_CLR,
    orn=_INK, bez=[_INK, _INK], grn=[_CLR] * 4, gem=[_CLR, _CLR, _CLR, _CLR],
    snk_line=_CLR, snk=[_INK] * 4, dot=_INK, au=[_CLR] * 4, eye=[_CLR, _CLR],
)


def _c4(c):
    if c == _CLR:
        return np.zeros(4, np.float32)
    return np.r_[hexc(c), 1].astype(np.float32)


def _hx(v):
    return [_c4(c) for c in v] if isinstance(v, list) else _c4(v)


_PALC = {k: {n: _hx(v) for n, v in p.items()} for k, p in _PAL.items()}


# ---------------------------------------------------------------- 几何工具

def _catmull(pts, step=1.0):
    """Catmull-Rom 过控制点，按约 step 参考单位重采样；返回 (xy, 半径, 弧长)。"""
    p = np.asarray(pts, np.float64)
    P = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        n = max(2, int(np.hypot(*(p2[:2] - p1[:2])) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        seg = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                     + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        out.append(seg)
    out.append(p[-1:])
    q = np.vstack(out)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(q[:, :2], axis=0).T))]
    return q[:, :2], q[:, 2], d


def _dil(m):
    g = m.copy()
    g[1:] |= m[:-1]
    g[:-1] |= m[1:]
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    return g


def _shift(m, dy, dx):
    o = np.zeros_like(m)
    H, W = m.shape
    o[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        m[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return o


def _polar(X, Y, table, c):
    n = len(table)
    a = np.arctan2(Y - c[1], X - c[0])
    f = (a + np.pi) / (2 * np.pi) * n - 0.5
    i0 = np.floor(f).astype(int)
    w = f - i0
    R = table[i0 % n] * (1 - w) + table[(i0 + 1) % n] * w
    return np.hypot(X - c[0], Y - c[1]) / R


def _leaf(X, Y, cx, cy, ang, ln, wd):
    """尖头叶片（两端尖的透镜形）。"""
    ca, sa = np.cos(ang), np.sin(ang)
    u = ((X - cx) * ca + (Y - cy) * sa) / ln
    v = (-(X - cx) * sa + (Y - cy) * ca) / wd
    return (np.abs(u) <= 1) & (np.abs(v) <= (1 - u * u))


def _tube(X, Y, pts, k):
    """管状蛇身：返回每像素最近点的弧长 t、归一化横截面 s、受光值、掩码。"""
    xy, rad, arc = _catmull(pts, 0.7)
    tree = cKDTree(xy)
    q = np.stack([X.ravel(), Y.ravel()], 1)
    d, idx = tree.query(q, distance_upper_bound=60)
    ok = np.isfinite(d)
    idx = np.where(ok, idx, 0)
    d = np.where(ok, d, 1e9).reshape(X.shape)
    idx = idx.reshape(X.shape)
    r = rad[idx]
    rp = np.maximum(r, 0.9 / k)              # 至少约 1 像素半宽
    m = d <= rp
    s = np.clip(d / rp, 0, 1)
    nx = (X - xy[idx, 0]) / np.maximum(d, 1e-6)
    ny = (Y - xy[idx, 1]) / np.maximum(d, 1e-6)
    nz = np.sqrt(1 - s * s)
    lit = (nx * s * LIGHT[0] + ny * s * LIGHT[1] + nz * LIGHT[2])
    return dict(t=arc[idx], L=arc[-1], s=s, lit=lit, m=m, d=d, r=rp, xy=xy, arc=arc)


def _curls(S, k, specs):
    """1px 涡卷线（阿基米德螺线的一段），左半边定义、右半边镜像。"""
    m = np.zeros((S, S), bool)
    for cx, cy, r0, a0, turns in specs:
        th = np.linspace(0, turns * 2 * np.pi, 400)
        r = r0 * (1 - th / (turns * 2 * np.pi) * 0.8)
        x = cx + r * np.cos(a0 + th)
        y = cy + r * np.sin(a0 + th)
        j = np.floor((x - CX) * k + S / 2).astype(int)
        i = np.floor((y - CY) * k + S / 2).astype(int)
        ok = (i >= 0) & (i < S) & (j >= 0) & (j < S)
        m[i[ok], j[ok]] = True
    return m | m[:, ::-1]


def _arc_at(tube, p):
    i = np.argmin(np.hypot(tube['xy'][:, 0] - p[0], tube['xy'][:, 1] - p[1]))
    return tube['arc'][i]


# ---------------------------------------------------------------- 按尺寸缓存的几何

@lru_cache(maxsize=16)
def _geom(S):
    k = (S - 2) / REF_H
    jj, ii = np.meshgrid(np.arange(S) + 0.5, np.arange(S) + 0.5)
    X = CX + (jj - S / 2) / k
    Y = CY + (ii - S / 2) / k
    Xa = CX - np.abs(X - CX)                 # 左右对称用：折到左半边
    g = dict(k=k, S=S)

    # 徽牌
    rn = _polar(X, Y, _SIL, _SIL_C)
    plq = rn <= 1
    g['plq_rn'] = rn
    g['plq'] = plq
    px = 1 / k                                # 1 像素 = 多少参考单位
    # 斜面：左上边缘亮、右下边缘暗；内侧一圈刻线
    g['plq_hi'] = plq & ~_shift(plq, 1, 1)
    g['plq_sh'] = plq & ~_shift(plq, -1, -1)
    dist = nd.distance_transform_edt(plq)
    g['plq_dist'] = dist
    ring_in = max(2.6, 17 * k)
    g['plq_groove'] = plq & (dist > ring_in) & (dist <= ring_in + 1)
    g['plq_groove_hi'] = _shift(g['plq_groove'], 1, 1) & plq & ~g['plq_groove'] & (dist > ring_in + 1)
    # 刻花：边缘带里的几个涡卷（小尺寸时省略）
    eng = _curls(S, k, CURLS) if S >= 80 else np.zeros((S, S), bool)
    g['plq_eng'] = eng & plq & (dist > ring_in + 1)

    # 内场：上方大拱 + 下方莲瓣（被蛇压住）
    up = ((X - CX) / 156) ** 2 + ((Y - 214) / 160) ** 2
    lo = ((X - CX) / 96) ** 2 + ((Y - 420) / 64) ** 2
    fld = ((up <= 1) & (Y < 330)) | (lo <= 1)
    g['fld_lo'] = (lo <= 1) & ~((up <= 1) & (Y < 330))
    fld &= plq
    g['fld'] = fld
    g['fld_sh'] = fld & ~_shift(fld, 2, 2)    # 凹进去：左上内沿有阴影
    g['fld_hi'] = fld & ~_shift(fld, -1, -1)
    # 石纹斑点
    hsh = (ii.astype(int) * 73 + jj.astype(int) * 151 + (ii.astype(int) * jj.astype(int)) % 7) % 13
    g['fld_spk'] = fld & (hsh == 0)

    # 内场纹饰：两层弯角、叶片、下方刀叶、宝石下的新月
    rho = np.sqrt(up)
    psi = np.degrees(np.arctan2(np.abs(X - CX), -(Y - 214)))
    orn = np.zeros_like(fld)
    for r0, wmax, a0, a1 in ((0.80, 0.12, 12, 124), (0.665, 0.07, 34, 120)):
        tt = np.clip((psi - a0) / (a1 - a0), 0, 1)
        w = wmax * np.sin(np.pi * tt) ** 0.6
        w = np.maximum(w, np.where((tt > 0) & (tt < 1), 0.6 * px / 156, 0))
        orn |= (psi > a0) & (psi < a1) & (rho >= r0) & (rho <= r0 + w)
    # 顶端小尖饰
    orn |= _leaf(X, Y, CX, 76, np.pi / 2, 14, 9)
    # 宝珠之间朝向宝石的叶片
    for (x0, y0), (x1, y1) in zip(ORBS[:3], ORBS[1:4]):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        vx, vy = GEM_C[0] - mx, GEM_C[1] - 10 - my
        n = np.hypot(vx, vy)
        cx_, cy_ = mx + vx / n * 20, my + vy / n * 20
        orn |= _leaf(Xa, Y, cx_, cy_, np.arctan2(vy, vx), 15, 8)
    # 下方刀叶
    for cx_, cy_, a, ln in ((176, 284, 158, 34), (196, 306, 146, 30), (146, 266, 170, 24)):
        orn |= _leaf(Xa, Y, cx_, cy_, np.radians(a), ln, max(7.0, 1.1 * px))
    # 新月
    dg = np.hypot(X - CX, Y - 238)
    orn |= (dg >= 38) & (dg <= 38 + max(6, 1.0 * px)) & (Y > 258) & (np.abs(X - CX) < 30)
    # 下方莲瓣里的涡纹
    orn |= (lo <= 0.55) & (lo >= 0.55 - max(0.12, 1.2 * px / 64)) & (Y > 440)
    g['orn'] = orn & fld

    # 宝珠（像素对齐，左右严格对称）
    # 直径取偶数像素、圆心落在像素角上，小圆也是圆的，且能严格左右对称
    Db = max(4, 2 * int(round(ORB_R * k)))      # 保持偶数，中轴上那颗才能左右对称
    Dg = Db - 2
    o_bez, o_g, o_hi, o_sh, o_sp = [], [], [], [], []
    for (ox, oy) in ORBS:
        # 左半边算好，右半边镜像；中间那颗落在中轴上（S 为偶数时中轴在像素边界）
        cxp = (ox - CX) * k + S / 2
        if abs(ox - CX) < 1:
            cxp = S / 2
        else:
            side = np.sign(ox - CX)
            cxp = S / 2 + side * np.round(abs(cxp - S / 2))
        cyp = np.round((oy - CY) * k + S / 2)
        dx, dy = jj - cxp, ii - cyp
        dd = np.hypot(dx, dy)
        o_bez.append(dd <= Db / 2 + 0.05)
        rg = Dg / 2 + 0.05
        gm = dd <= rg
        o_g.append(gm)
        o_hi.append(gm & (dx + dy < -0.3) & (Dg >= 3) | gm & (dx < 0) & (dy < 0))
        o_sh.append(gm & (dx + dy > Dg * 0.45) & (Dg >= 3))
        o_sp.append(gm & (dx < -Dg / 2 + 1.2) & (dy < -Dg / 2 + 1.2) & (dx > -Dg / 2 + 0.3)
                    & (dy > -Dg / 2 + 0.3) & (Dg >= 4))
    g['orbs'] = (o_bez, o_g, o_hi, o_sh, o_sp)

    # 宝石：倒水滴
    dx, dy = X - GEM_C[0], Y - GEM_C[1]
    def drop(grow):
        R = GEM_R + grow
        tip = GEM_TIP - grow * 1.3
        body = np.hypot(dx, dy) <= R
        tt = np.clip((Y - tip) / (GEM_C[1] - tip), 0, 1)
        cone = (Y >= tip) & (Y <= GEM_C[1]) & (np.abs(dx) <= R * np.sin(tt * np.pi / 2) ** 1.25 + 0.5 * px)
        return body | cone
    gem = drop(0)
    bez = drop(max(6.5, 1.0 * px))
    g['gem'] = gem
    g['gem_bez'] = bez
    lg = (-dx * 0.6 - dy * 0.35) / GEM_R          # 左上亮
    g['gem_lvl'] = np.digitize(lg, [-0.2, 0.3, 0.75])
    # 刻面：一道斜的亮刻面 + 一个高光点
    g['gem_facet'] = gem & (np.abs(dx + 0.55 * (dy - 6)) < 0.7 * px) & (dy > -30) & (dy < 18)
    g['gem_spec'] = gem & (np.hypot(dx + 11, dy + 5) < max(4, 0.7 * px))
    g['gem_dot'] = np.hypot(X - CX, Y - (GEM_TIP - 10)) <= max(4.5, 0.6 * px)

    # 蛇
    A = _tube(X, Y, SNAKE_A, k)
    B = _tube(X, Y, SNAKE_B, k)
    g['A'], g['B'] = A, B
    g['gap'] = 90.0                           # 在徽牌后面绕过去的那一段（不可见）
    edge_ring = _dil(plq) & ~plq
    for T, zone in ((A, A['t'] > A['L'] - 60), (B, B['t'] < 60)):
        # 两端绕过徽牌边缘钻到背后：端部落在徽牌内的像素藏起来，与徽牌轮廓相交处补一道描边
        T['hide'] = T['m'] & zone & plq
        T['cross'] = T['m'] & zone & edge_ring
        T['m'] = T['m'] & ~T['hide']
        T['band'] = np.digitize(T['lit'], [0.36, 0.7, 0.92])
        # 腹侧鳞点：沿身体每隔一段的小亮点，落在背光的一侧
        per = max(14.0, 2.6 * px)
        T['dots'] = T['m'] & (T['s'] > 0.35) & (T['s'] < 0.85) & (T['lit'] < 0.62) & \
            ((T['t'] % per) < 0.9 * px) & (T['band'] == 1) & (S >= 80)
    rings = []
    for seg, p, hl, extra in RINGS:
        T = A if seg == 'A' else B
        t0 = _arc_at(T, p)
        hlp = max(hl, 0.8 * px)
        rm = (np.abs(T['t'] - t0) <= hlp) & (T['d'] <= T['r'] + max(extra, 0.7 * px))
        # 金环自己的受光：沿用管子的法线，但半径放大
        s = np.clip(T['d'] / (T['r'] + extra), 0, 1)
        nz = np.sqrt(1 - s * s)
        lit = T['lit'] + (nz - np.sqrt(1 - T['s'] ** 2)) * LIGHT[2]   # 换成放大后的 z 分量
        band = np.digitize(lit, [0.35, 0.7, 0.92])
        # 环的两端边缘压暗一格，做出箍的厚度
        edge = np.abs(np.abs(T['t'] - t0) - hlp) < 0.55 * px
        band = np.where(edge, np.maximum(band - 1, 0), band)
        rings.append(dict(seg=seg, t1=t0 + hlp, m=rm, band=band))
    g['rings'] = rings
    # 蛇头
    (tx, ty), (sx, sy) = HEAD['top'], HEAD['tip']
    ax, ay = sx - tx, sy - ty
    Lh = np.hypot(ax, ay)
    ax, ay = ax / Lh, ay / Lh
    v = ((X - tx) * ax + (Y - ty) * ay) / Lh           # 0 头顶 .. 1 吻端
    u = (X - tx) * (-ay) + (Y - ty) * ax               # 横向（正 = 右侧）
    vc = np.clip(v, 0, 1)
    top = np.sqrt(np.clip(1 - ((0.28 - vc) / 0.28) ** 2, 0, 1))
    taper = 1 - 0.55 * np.clip((vc - 0.28) / 0.72, 0, 1) ** 1.6
    snout = np.sqrt(np.clip(1 - ((vc - 0.8) / 0.2) ** 2, 0, 1))   # 吻端收圆
    w = HEAD['W'] * np.where(vc < 0.28, top, taper * np.where(vc > 0.8, snout, 1))
    w = np.maximum(w, 0.6 / k)
    hm = (v >= 0) & (v <= 1) & (np.abs(u) <= w)
    su = np.clip(u / w, -1, 1)
    nzh = np.sqrt(1 - su * su)
    tilt = np.where(v < 0.3, -(0.3 - v) / 0.3, 0.25)   # 头顶朝上，吻部略朝下
    nxh, nyh = su * (-ay) + tilt * ax * 0.8, su * ax + tilt * ay * 0.8
    nn = np.sqrt(nxh ** 2 + nyh ** 2 + nzh ** 2)
    hlit = (nxh * LIGHT[0] + nyh * LIGHT[1] + nzh * LIGHT[2]) / nn
    g['head'] = hm
    g['head_band'] = np.digitize(hlit, [0.36, 0.7, 0.92])
    # 下颌线：吻部右侧一道暗线
    g['head_jaw'] = hm & (v > 0.62) & (v < 0.95) & (su > 0.35) & (su < 0.35 + 1.2 / (k * np.maximum(w, 1))) if S >= 80 \
        else np.zeros_like(hm)
    # 眼睛
    ex = (EYE[0] - CX) * k + S / 2
    ey = (EYE[1] - CY) * k + S / 2
    ey, ex = int(ey), int(ex)
    eyem = np.zeros((S, S), bool)
    hi = np.zeros((S, S), bool)
    eyem[ey, ex] = True
    if S >= 88:
        eyem[ey - 1, ex] = True           # 竖长的眼
    if S >= 120:
        eyem[ey - 1:ey + 1, ex + 1] = True
        hi[ey - 1, ex] = True
    g['eye'], g['eye_hi'] = eyem, hi
    return g


# ---------------------------------------------------------------- 合成

def _ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _put(img, m, col):
    img[m] = col


def _put_lvl(img, m, lvl, cols):
    for i, c in enumerate(cols):
        img[m & (lvl == i)] = c


def _outlined(img, m, col):
    _put(img, _dil(m) & ~m, col)


def _render(S, progress, pal):
    g = _geom(S)
    C = _PALC[pal]
    p = float(np.clip(progress, 0, 1))
    img = np.zeros((S, S, 4), np.float32)

    # 1) 徽牌：从中心向外撑开 (0 .. 0.24)
    reveal = _ease(p / 0.24) * 1.06
    if reveal <= 0:
        return img
    vis = g['plq'] & (g['plq_rn'] <= reveal)
    _outlined(img, vis, C['line'])
    _put(img, vis, C['plq'][1])
    if p >= 0.24:
        _put(img, g['plq_eng'], C['plq_eng'])
        _put(img, g['plq_groove_hi'], C['plq'][2])
        _put(img, g['plq_groove'], C['plq'][0])
        _put(img, g['plq_hi'], C['plq'][3])
        _put(img, g['plq_sh'], C['plq'][0])
    fld = g['fld'] & vis & (g['plq_rn'] <= reveal - 0.12)
    _outlined(img, fld, C['line'])
    _put(img, fld, C['fld'][2])
    _put(img, fld & g['fld_spk'], C['fld'][1])
    _put(img, fld & g['fld_hi'], C['fld'][3])
    _put(img, fld & g['fld_sh'], C['fld'][0])
    lo = fld & g['fld_lo']                     # 蛇下方的莲瓣：比内场暗一档
    _put(img, lo, C['fld_lo'])
    _put(img, lo & g['fld_sh'], C['plq'][0])

    # 2) 内场纹饰 (0.16 .. 0.34)
    oreveal = _ease((p - 0.16) / 0.18) * 1.1
    if oreveal > 0:
        _put(img, g['orn'] & (g['plq_rn'] <= oreveal), C['orn'])

    # 3) 宝珠镶座先出现，绿珠在最后按拱形顺序亮起
    o_bez, o_g, o_hi, o_sh, o_sp = g['orbs']
    for i in range(7):
        if p >= 0.22 + 0.015 * i:
            _put(img, o_bez[i], C['bez'][0])
            _put(img, o_bez[i] & _shift(o_g[i], 1, 1) & ~o_g[i], C['bez'][1])
        if p >= 0.80 + 0.018 * i:
            _put(img, o_g[i], C['grn'][1])
            _put(img, o_hi[i], C['grn'][2])
            _put(img, o_sh[i], C['grn'][0])
            _put(img, o_sp[i], C['grn'][3])
    if p >= 0.30:
        _put(img, g['gem_bez'] | g['gem_dot'], C['bez'][0])
    if p >= 0.93:
        _put_lvl(img, g['gem'], g['gem_lvl'], C['gem'])
        _put(img, g['gem_facet'], C['gem'][2])
        _put(img, g['gem_spec'], C['gem'][3])
        _put(img, g['gem_dot'], C['gem'][2])

    # 4) 蛇：尾尖 -> 右侧身体 -> (徽牌后) -> 左侧身体 -> 脖子 -> 蛇头 (0.30 .. 0.78)
    A, B = g['A'], g['B']
    total = A['L'] + g['gap'] + B['L']
    cut = _ease((p - 0.30) / 0.48) * total if p < 0.78 else total + 1
    if cut > 0:
        for T, off in ((A, 0.0), (B, A['L'] + g['gap'])):
            m = T['m'] & (T['t'] + off <= cut)
            if not m.any():
                continue
            _outlined(img, m, C['snk_line'])
            _put_lvl(img, m, T['band'], C['snk'])
            _put(img, m & T['dots'], C['dot'])
            _put(img, m & T['cross'], C['line'])
            if T is B and cut >= off + B['L'] - 1:
                _outlined(img, g['head'], C['snk_line'])
                _put_lvl(img, g['head'], g['head_band'], C['snk'])
                _put(img, g['head_jaw'], C['snk'][0])
            for R in g['rings']:
                if (R['seg'] == 'A') == (T is A) and cut >= R['t1'] + off:
                    rm = R['m']
                    _outlined(img, rm, C['snk_line'])
                    _put_lvl(img, rm, R['band'], C['au'])
    if p >= 0.78:
        _put(img, g['eye'], C['eye'][0])
        _put(img, g['eye_hi'], C['eye'][1])
    return img


def emblem(size=96, progress=1.0, palette='gold'):
    """Ouroboros 纹章。size: 画布边长（像素，建议 64/96/128）；progress 0..1 为绘制动画进度；
    palette: 'gold'（官方配色，默认）/ 'glow'（闪光高亮）/ 'mono'（单色图章）。"""
    size = int(size)
    if palette in ('official', 'color'):
        palette = 'gold'
    return _render(size, progress, palette)


if __name__ == '__main__':
    import time
    for s in (64, 96, 128):
        emblem(s)
        t = time.perf_counter()
        for _ in range(10):
            emblem(s, 0.9)
        print(s, f'{(time.perf_counter() - t) / 10 * 1000:.1f} ms')
