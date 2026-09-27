"""第三幕 · 钢之圣女：古战场的枪林。

原作：雅里安洛德 = 250 年前「狮子战役」的传说「枪之圣女」莉安娜·桑德罗特，如今是结社第七使徒；
麾下是铁机队（杜巴莉、恩奈娅、艾奈斯）。

场景：血红夕阳下的古战场山丘，漫山插着无数旧枪与破旗（像一片枪的墓地），地平线上是倾颓的要塞城墙，
风把草与灰烬吹向画面左方。

分镜（6 小节 @144，小节从 1 数、拍从 0 数）
  1.0  低机位远景：夕阳前的枪林剪影，她立在丘顶，枪插在土里，闭目，披风向左飘。镜头缓缓横移。
  2.0  硬切特写（画像 3x）：闭着眼 —— 2.1 睁眼，瞳中一点青光。名牌滑入。
  2.2  硬切中景：她扶枪而立；身后三名铁机队员单膝跪地（剪影）。
  3.0  台词。3.0 双手握枪 → 3.2 发力、地面裂开 → 3.2.5 拔出骑枪，土块崩飞。
  4.0  挥枪：枪尖划出一道弧光扫到水平 → 4.0.5 端平（踏步扬尘）→ 4.2 压低重心蓄力，枪尖卷起螺旋光。
  4.3.5 蹬地：化作一道银光向左冲出（残影、金色螺旋尾迹、枪尖前的激波锥）。
  5.0  硬切远景，命中：反相冲击帧 + 顿帧 → 光之十字从枪尖处拔地而起直冲天际，整个战场被照成金白色。
  5.1  地面冲击波，枪与旗被连根掀飞。5.2 十字再次暴涨（第二冲击）。
  6.0  硬切中景：她落地（蹲 → 起身 → 回到立绘站姿），披风慢慢落定；背景的光柱逐渐消散，
       6.1 铁机队起身举武器致敬。最后两拍是定格画面（扑克牌飞入）。
"""
from functools import lru_cache

import numpy as np

import sfx as S
import ui
from gfx import BAYER, H, W, XX, YY, blit, ease_out, hexc, scale_img, smooth

LETTERBOX = 0
SAYS = [('arianrhod', 3, 0.0, 'dialog')]
CUES = [
    (1, 0.0, 'wind', -6), (1, 2.0, 'wind', -9, 0.4),
    (2, 1.0, 'shimmer', -8),                       # 睁眼
    (2, 2.0, 'armor_clank', -4),                   # 切中景：铁机队跪地的甲胄声
    (3, 0.0, 'step', -8),                          # 握枪
    (3, 2.0, 'crumble', -3), (3, 2.0, 'earth_pull', -2),   # 发力，地面裂开
    (3, 2.5, 'sword_draw', -3),                    # 拔出
    (4, 0.0, 'swish', -2),                         # 挥枪弧光
    (4, 0.5, 'step', -2),                          # 踏步端平
    (4, 1.12, 'charge', -3),                       # 蓄力（1.2 秒，正好收在 5.0）
    (4, 3.5, 'dash', 0),                           # 蹬地冲出
    (5, 0.0, 'big_impact', 0), (5, 0.0, 'lance', -2), (5, 0.0, 'holy_cross', 0), (5, 0.0, 'crunch', -4),
    (5, 1.0, 'rumble', -3),                        # 冲击波掀飞枪林
    (5, 2.0, 'boom_low', -1), (5, 2.0, 'shimmer', -4),     # 十字再次暴涨
    (6, 0.0, 'step', 0), (6, 0.0, 'puff', -6),     # 落地
    (6, 1.0, 'armor_clank', -5),                   # 铁机队起身致敬
    (6, 1.0, 'wind', -8, -0.3),
]
ACCENTS = [(5, 0.0, 'riser'), (5, 0.0, 'hit'), (6, 0.0, 'drop'), (7, 0.0, 'riser')]


# ================================================================ 合成音效
def _st(x, pan=0.0):
    x = np.asarray(x, np.float64)
    return np.stack([x * np.sqrt(0.5 * (1 - pan)), x * np.sqrt(0.5 * (1 + pan))], 1).astype(np.float32)


def _holy_cross():
    """光之十字：低频轰鸣 + D 大调钟鸣簇 + 上扬的空气声，约 2.4 秒。"""
    n = int(2.4 * S.SR)
    t = S.tt(n)
    boom = np.tanh(2.5 * np.sin(2 * np.pi * np.cumsum(36 + 120 * np.exp(-t / 0.06)) / S.SR)) * np.exp(-t / 0.7)
    bell = np.zeros(n)
    for m, a in ((74, 0.5), (78, 0.35), (81, 0.35), (86, 0.3), (90, 0.2), (93, 0.15)):
        f = S.mf(m)
        bell += a * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.3))
    bell *= np.clip(t / 0.02, 0, 1) * np.exp(-t / 0.9)
    air = S.sweep_lp(S.noise(n), 600, 9000, 0.7) * np.exp(-t / 0.5) * np.clip(t / 0.03, 0, 1)
    L = 0.75 * boom + 0.35 * bell + 0.4 * air
    R = 0.75 * boom + 0.35 * np.roll(bell, 220) + 0.4 * S.sweep_lp(S.noise(n), 600, 9000, 0.7) * np.exp(-t / 0.5)
    return np.stack([L, R], 1).astype(np.float32)


def _earth_pull():
    """枪尖从土里拔出：沉闷的土石摩擦 + 金属刮擦上扬。"""
    n = int(0.8 * S.SR)
    t = S.tt(n)
    dirt = S.lp(S.noise(n), 900) * (np.clip(t / 0.25, 0, 1) * np.exp(-np.clip(t - 0.3, 0, None) / 0.12))
    f = 900 + 2600 * np.clip(t / 0.5, 0, 1) ** 2
    scrape = S.bp(S.noise(n), 1500, 6000) * (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(f / 40) / S.SR))
    scrape *= np.clip((t - 0.15) / 0.2, 0, 1) * np.exp(-np.clip(t - 0.45, 0, None) / 0.08)
    return _st(1.3 * dirt + 0.4 * scrape)


def _armor_clank():
    """几副铠甲同时动：短促的金属碰撞，错开几毫秒。"""
    n = int(0.5 * S.SR)
    out = np.zeros((n, 2))
    for k, (d, pan) in enumerate(((0.0, -0.4), (0.035, 0.1), (0.07, 0.5))):
        i = int(d * S.SR)
        m = n - i
        t = S.tt(m)
        x = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((1900 + 300 * k, 0.4), (3100 + 200 * k, 0.3),
                                                             (5200, 0.15))) * np.exp(-t / 0.06)
        x += S.hp(S.noise(m), 2500) * np.exp(-t / 0.015) * 0.6
        out[i:] += _st(x, pan)
    return out.astype(np.float32)


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


SFX = {'holy_cross': _holy_cross, 'earth_pull': _earth_pull, 'armor_clank': _armor_clank}


# ================================================================ 调色板
def _c(h):
    return hexc(h)


SKY = ['#1a0512', '#2c0717', '#44091c', '#660f1f', '#8c1a1c', '#b42c19', '#d8491b', '#f07426', '#ffa446']
HORIZON = 176
C_SUN = ('#fff4cc', '#ffd680', '#ff9c3c')
C_CLOUD, C_CLOUD_LIT = _c('#3a0816'), _c('#e0602a')
C_FORT, C_FORT_LIT, C_FORT_WIN = _c('#4c1020'), _c('#86281f'), _c('#d0501e')
C_MID, C_MID_SPEAR = _c('#2a0812'), _c('#1c050d')
C_NEAR, C_NEAR_RIM = _c('#12030a'), _c('#5a1418')
C_GRASS, C_GRASS_LIT = _c('#1c050c'), _c('#7a2018')
C_BANNER, C_BANNER_RIM = _c('#4e0c1a'), _c('#b0302a')
GOLD, TEAL, WHITE = _c('#ffd870'), _c('#6ee6d6'), _c('#ffffff')


# ================================================================ 静态图层（缓存）
def _rgba(h, w):
    return np.zeros((h, w, 4), np.float32)


def _put(img, m, col, a=1.0):
    img[m, :3] = col
    img[m, 3] = a


def _line_px(x0, y0, x1, y1):
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 2
    t = np.linspace(0, 1, n)
    return np.floor(x0 + (x1 - x0) * t).astype(int), np.floor(y0 + (y1 - y0) * t).astype(int)


def _stamp_line(img, x0, y0, x1, y1, col, w=1):
    xs, ys = _line_px(x0, y0, x1, y1)
    h, ww = img.shape[:2]
    for dx in range(w):
        ok = (xs + dx >= 0) & (xs + dx < ww) & (ys >= 0) & (ys < h)
        img[ys[ok], xs[ok] + dx, :3] = col
        if img.shape[2] == 4:
            img[ys[ok], xs[ok] + dx, 3] = 1


@lru_cache(None)
def sky_layer(wide=True):
    """血红夕阳天空：9 阶色带 + Bayer 抖动过渡 + 横向云带（下缘被夕阳照亮）。宽 W+160。"""
    w = W + 160
    yy, xx = np.mgrid[0:H, 0:w].astype(np.float32)
    bay = np.tile(BAYER[:4, :4], (H // 4 + 1, w // 4 + 1))[:H, :w]
    v = np.clip(yy / HORIZON, 0, 1) ** 0.8 * (len(SKY) - 1)
    i0 = np.floor(v).astype(int)
    fr = v - i0
    idx = np.clip(i0 + (bay < fr), 0, len(SKY) - 1)
    pal = np.stack([_c(h) for h in SKY])
    img = pal[idx]
    # 云带（长条，边缘锯齿），太阳一侧下缘亮
    rng = np.random.default_rng(3 if wide else 4)
    for cy, x0, x1, th in ((40, -20, 260, 5), (58, 180, 560, 4), (86, 40, 330, 3), (104, 300, 620, 4),
                           (120, -30, 200, 3), (136, 250, 420, 2), (28, 360, 640, 3)):
        ln = x1 - x0
        xs = np.arange(max(0, x0), min(w, x1))
        prof = th * np.sin(np.pi * (xs - x0) / ln) ** 0.6
        prof += rng.normal(0, 0.5, len(xs)).cumsum() * 0.15
        for x, p in zip(xs, prof):
            p = max(0.0, p)
            y0, y1 = int(cy - p * 0.6), int(cy + p * 0.4) + 1
            img[y0:y1, x] = C_CLOUD
            if p > 0.6:
                img[y1 - 1, x] = C_CLOUD_LIT if (x + y1) % 5 else _c('#ff9a3a')
    return img.astype(np.float32)


@lru_cache(None)
def sun_sprite(r):
    """复古夕阳：亮核 + 色环 + 抖动光晕，下半部被横纹切开。"""
    R = int(r * 1.35) + 2
    img = _rgba(2 * R, 2 * R)
    yy, xx = np.mgrid[0:2 * R, 0:2 * R].astype(np.float32)
    d = np.hypot(xx - R + .5, yy - R + .5)
    bay = np.tile(BAYER[:4, :4], (R // 2 + 1, R // 2 + 1))[:2 * R, :2 * R]
    halo = (d < r * 1.35) & (bay < clamp01((r * 1.35 - d) / (r * 0.4)) * 0.5)
    _put(img, halo, _c(C_SUN[2]), 0.55)
    _put(img, d < r, _c(C_SUN[1]))
    _put(img, d < r * 0.78, _c(C_SUN[0]))
    dy = yy - R
    cut = (dy > r * 0.15) & (((dy - r * 0.15) // 3 + (dy > r * 0.55)) % 2 == 1) & (d < r + 1)
    thick = (dy - r * 0.15) / (r * 0.85)
    cut &= ((dy - r * 0.15) % 6) < 1 + 2.5 * thick
    img[cut, 3] = 0
    return img


@lru_cache(None)
def fortress_layer():
    """地平线上倾颓的要塞：城墙 + 雉堞 + 断塔 + 塌掉的拱门，远景大气色。宽 W+160。"""
    w = W + 160
    img = _rgba(H, w)
    base = HORIZON - 2
    rng = np.random.default_rng(11)
    top = np.full(w, base, float)
    x = 0
    while x < w:
        seg = int(rng.integers(20, 50))
        kind = rng.random()
        if kind < 0.28:                                    # 塔
            tw = int(rng.integers(12, 20))
            th = int(rng.integers(24, 40))
            broken = rng.random() < 0.6
            for i in range(tw):
                jag = int(rng.integers(0, 7)) if broken else (0 if (i // 3) % 2 else 3)
                if x + i < w:
                    top[x + i] = base - th + jag
            seg = tw
        elif kind < 0.40:                                  # 塌口
            for i in range(seg):
                if x + i < w:
                    top[x + i] = base - 2 - 3 * abs(np.sin(i * 0.4))
        else:                                              # 城墙 + 雉堞
            hgt = int(rng.integers(9, 15))
            for i in range(seg):
                if x + i < w:
                    top[x + i] = base - hgt - (2 if (i // 3) % 2 == 0 else 0)
        x += seg
    yy = np.arange(H)[:, None]
    m = yy >= top[None, :]
    m &= yy < HORIZON + 30
    _put(img, m, C_FORT)
    edge = m & ~np.roll(m, 1, 0)
    _put(img, edge, C_FORT_LIT)
    # 塔上的射击孔透出天光；一座塌拱
    for xi in range(4, w - 4, 7):
        if top[xi] < base - 20 and rng.random() < 0.7:
            y0 = int(top[xi]) + 6
            img[y0:y0 + 3, xi, :3] = C_FORT_WIN
    ax, ay = 300, base - 12
    arch = (np.hypot(np.arange(w)[None, :] - ax, (yy - ay) * 1.2) < 9) & (yy > ay - 8) & (yy < base)
    img[arch, 3] = 0
    return img


@lru_cache(None)
def mid_layer():
    """中景：起伏的远丘 + 密密麻麻插着的旧枪剪影，偶尔几面残旗。宽 W+160。"""
    w = W + 160
    img = _rgba(H, w)
    xs = np.arange(w)
    ground = HORIZON + 8 + 5 * np.sin(xs * 0.013) + 3 * np.sin(xs * 0.041 + 1.3) + 2 * np.sin(xs * 0.09)
    yy = np.arange(H)[:, None]
    m = yy >= ground[None, :]
    _put(img, m, C_MID)
    _put(img, m & ~np.roll(m, 1, 0), _c('#4a1018'))
    rng = np.random.default_rng(5)
    for x in rng.uniform(0, w, 150):
        gy = ground[int(x)]
        ln = rng.uniform(8, 22)
        ang = np.radians(90 + rng.normal(0, 14))
        x1, y1 = x + np.cos(ang) * ln, gy - np.sin(ang) * ln
        _stamp_line(img, x, gy + 1, x1, y1, C_MID_SPEAR)
        if rng.random() < 0.12:        # 残旗
            fw = int(rng.integers(3, 6))
            for k in range(fw):
                for j in range(int(rng.integers(2, 4))):
                    xx_, yy_ = int(x1) - k, int(y1) + 1 + j + (k % 2)
                    if 0 <= xx_ < w and 0 <= yy_ < H:
                        img[yy_, xx_, :3] = _c('#3a0a14')
                        img[yy_, xx_, 3] = 1
    return img


def _mound_y(x, crest_x=330, crest_y=196):
    return crest_y + ((x - crest_x) / 150.0) ** 2 * 44 + 2 * np.sin(x * 0.07)


@lru_cache(None)
def near_wide_layer():
    """远景用近处山丘（丘顶在 x=330, y=196）。宽 W+160，屏幕坐标 = 图层坐标 - 80 + 平移。"""
    w = W + 160
    img = _rgba(H, w)
    xs = np.arange(w) - 80
    top = _mound_y(xs)
    yy = np.arange(H)[:, None]
    m = yy >= top[None, :]
    _put(img, m, C_NEAR)
    _put(img, m & ~np.roll(m, 1, 0), C_NEAR_RIM)
    # 近处的碎石 / 残甲
    rng = np.random.default_rng(8)
    for x in rng.uniform(0, w, 40):
        y = int(_mound_y(x - 80)) + int(rng.integers(3, 30))
        if y < H - 1:
            img[y:y + 2, int(x):int(x) + 3, :3] = _c('#2a0a12')
    return img


@lru_cache(None)
def near_crest_layer():
    """中景用：脚下的丘顶（地平线 y≈196）+ 前景插着的高大旧枪与残旗。宽 W+160。"""
    w = W + 160
    img = _rgba(H, w)
    xs = np.arange(w)
    top = 196 + 1.5 * np.sin(xs * 0.05) + 1.0 * np.sin(xs * 0.13 + 2)
    yy = np.arange(H)[:, None]
    m = yy >= top[None, :]
    _put(img, m, C_NEAR)
    _put(img, m & ~np.roll(m, 1, 0), C_NEAR_RIM)
    # 地面纹理：几道暗纹
    for y0 in (206, 214, 226, 240):
        for x in range(0, w, 3):
            if (x * 7 + y0) % 11 < 5:
                img[y0 + (x // 17) % 2, x, :3] = _c('#1c0610')
    # 前景的高大旧枪（画面两侧），部分出画
    for x0, ln, ang, flag in ((70, 180, 98, True), (118, 120, 84, False), (560, 200, 80, True),
                              (610, 140, 95, False)):
        gy = 200
        a = np.radians(ang)
        x1, y1 = x0 + np.cos(a) * ln, gy - np.sin(a) * ln
        _stamp_line(img, x0, gy + 4, x1, y1, _c('#0c0206'), w=3)
        _stamp_line(img, x0 - 1, gy + 4, x1 - 1, y1, _c('#3a0c12'), w=1)
        if flag:                    # 破旗：长条布，下缘撕成齿状
            fx0, fy0 = x1 + np.cos(a) * -6, y1 + 6
            for k in range(26):
                hgt = 22 - k * 0.3 + (5 if k % 5 == 2 else 0) - (4 if k % 7 == 3 else 0)
                xx_ = int(fx0 - k)
                y0_ = int(fy0 + k * 0.25)
                if 0 <= xx_ < w:
                    img[y0_:y0_ + int(hgt), xx_, :3] = C_BANNER
                    img[y0_:y0_ + int(hgt), xx_, 3] = 1
                    img[y0_ + int(hgt) - 1, xx_, :3] = C_BANNER_RIM
    return img


def _layer(dst, img, x, y=0):
    blit(dst, img, x, y)


# ================================================================ 角色图（着色、缩放后缓存）
def _edge(a, dx, dy):
    return a & ~np.roll(np.roll(a, dy, 0), dx, 1)


@lru_cache(None)
def spr(ch, pose, scale=1, look='normal'):
    """look: back = 背光剪影（暗 + 暖色描边）；normal = 夕照（偏暖 + 左侧轮廓光）；
    light = 被圣光照亮（偏白金 + 青色轮廓光）。"""
    img = ch.body(pose).copy()
    a = img[..., 3] > 0
    if look == 'back':
        img[..., :3] = img[..., :3] * 0.32 + _c('#2a0a14') * 0.68
        rim = (_edge(a, 1, 0) | _edge(a, 0, 1)) & a
        img[rim, :3] = img[rim, :3] * 0.3 + _c('#ff8a3a') * 0.7
    elif look == 'normal':
        img[..., :3] *= np.array([1.0, 0.9, 0.86], np.float32)
        rim = _edge(a, 1, 0) & a
        img[rim, :3] = img[rim, :3] * 0.5 + _c('#ffa050') * 0.5
    elif look == 'light':
        img[..., :3] = img[..., :3] * 0.7 + _c('#fff6d8') * 0.3
        rim = (_edge(a, -1, 0) | _edge(a, 0, 1)) & a
        img[rim, :3] = TEAL * 0.6 + WHITE * 0.4
    elif look == 'white':
        img[a, :3] = WHITE
    elif look == 'black':
        img[a, :3] = _c('#08020a')
    return scale_img(img, scale) if scale > 1 else img


def put_char(dst, ch, pose, fx_, fy, scale=1, look='normal', alpha=1.0, tint=None, amt=0.0):
    img = spr(ch, pose, scale, look)
    blit(dst, img, fx_, fy, anchor='bottom', alpha=alpha, tint=tint, tint_amt=amt)
    return img


def tip_screen(ch, pose, fx_, fy, scale):
    """姿势的枪尖在屏幕上的位置。"""
    import chars.arianrhod as A
    x, y = A.tip(pose.split(':')[0])
    return fx_ - A.BW * scale / 2 + x * scale, fy - A.BH * scale + y * scale


# ================================================================ 铁机队（剪影）
@lru_cache(None)
def maiden(kind, pose):
    """铁机队员剪影（约 64px 高的全身比例）：kind 0 杜巴莉（剑）、1 恩奈娅（弓）、2 艾奈斯（斧枪）；
    pose: kneel（单膝跪地，武器立在身侧）/ rise（起身中）/ salute（立正，武器举在面前致敬）。"""
    from chars.arianrhod import Cv
    cv = Cv(80, 44, {'K': '#10040a', 'r': '#d8562a', 'w': '#26101c'})
    X = 20
    if pose == 'kneel':
        y = 80
        cv.poly([(X - 4, y - 30), (X + 4, y - 30), (X + 6, y - 16), (X - 5, y - 16)], 'K')     # 躯干（略前倾）
        cv.poly([(X - 5, y - 16), (X + 6, y - 16), (X + 12, y - 12), (X + 12, y - 8), (X + 3, y - 10)], 'K')
        cv.poly([(X + 8, y - 12), (X + 12, y - 12), (X + 12, y), (X + 8, y)], 'K')              # 前腿小腿
        cv.poly([(X - 5, y - 12), (X + 1, y - 12), (X - 10, y), (X - 14, y), (X - 14, y - 2)], 'K')  # 跪地的腿
        cv.ell(X + 1, y - 34, 3.4, 3.8, 'K')                                                  # 低着的头
        cv.poly([(X - 7, y - 31), (X + 7, y - 31), (X + 5, y - 27), (X - 5, y - 27)], 'K')      # 肩甲
        wx = X - 9
        wtop = {0: y - 44, 1: y - 40, 2: y - 60}[kind]
    elif pose == 'rise':
        y = 80
        cv.poly([(X - 4, y - 42), (X + 4, y - 42), (X + 5, y - 26), (X - 4, y - 26)], 'K')
        cv.poly([(X - 4, y - 26), (X + 5, y - 26), (X + 8, y - 14), (X + 7, y), (X + 3, y), (X + 1, y - 14),
                 (X - 3, y - 12), (X - 7, y), (X - 11, y), (X - 7, y - 16)], 'K')
        cv.ell(X + 0.5, y - 46, 3.4, 3.8, 'K')
        cv.poly([(X - 7, y - 43), (X + 7, y - 43), (X + 5, y - 39), (X - 5, y - 39)], 'K')
        wx = X - 9
        wtop = {0: y - 56, 1: y - 52, 2: y - 72}[kind]
    else:   # salute
        y = 80
        cv.poly([(X - 4, y - 50), (X + 4, y - 50), (X + 4, y - 32), (X - 4, y - 32)], 'K')
        cv.poly([(X - 4, y - 32), (X + 4, y - 32), (X + 4, y), (X + 1, y), (X, y - 16), (X - 1, y),
                 (X - 4, y)], 'K')
        cv.ell(X, y - 54, 3.4, 3.8, 'K')
        cv.poly([(X - 7, y - 51), (X + 7, y - 51), (X + 5, y - 47), (X - 5, y - 47)], 'K')
        cv.stroke([(X - 5, y - 48), (X - 6, y - 42), (X - 2, y - 44)], 'K', 2.4)                 # 抬到胸前的手
        wx = X - 3
        wtop = {0: y - 74, 1: y - 68, 2: y - 79}[kind]
    # 头发：杜巴莉短发、恩奈娅长发、艾奈斯马尾
    hx, hy = (X + 1, y - 34) if pose == 'kneel' else ((X + .5, y - 46) if pose == 'rise' else (X, y - 54))
    if kind == 1:
        cv.poly([(hx - 3, hy - 2), (hx + 4, hy - 2), (hx + 5, hy + 9), (hx + 1, hy + 7)], 'K')
    elif kind == 2:
        cv.stroke([(hx + 2, hy - 3), (hx + 7, hy - 1), (hx + 8, hy + 6)], 'K', 2, 0.8)
    # 武器
    wb = y - (2 if pose != 'salute' else 38)
    if kind == 0:                                          # 剑：竖在身侧 / 举在面前
        cv.stroke([(wx, wb), (wx, wtop)], 'K', 1.6)
        cv.stroke([(wx - 3, wb - (12 if pose != 'salute' else 2)), (wx + 3, wb - (12 if pose != 'salute' else 2))],
                  'K', 1.4)
    elif kind == 1:                                        # 弓
        cv.stroke([(wx, wb), (wx - 4, (wb + wtop) / 2), (wx, wtop)], 'K', 1.4)
        cv.line([(wx, wb), (wx, wtop)], 'w')
    else:                                                  # 斧枪
        cv.stroke([(wx, wb + 2), (wx, wtop)], 'K', 1.4)
        cv.poly([(wx, wtop + 5), (wx - 5, wtop + 7), (wx - 5, wtop + 13), (wx, wtop + 12)], 'K')
        cv.poly([(wx - 1, wtop), (wx + 1, wtop), (wx, wtop - 4)], 'K')
    a = cv.has('Kw')
    # 夕阳在身后偏左：左侧与上侧描一道暖色轮廓光
    rim = a & (~cv.shift(a, -1, 0) | ~cv.shift(a, 0, -1))
    cv.put(rim & a, 'r')
    return cv.rgba('#08020a')


def draw_maidens(dst, lt_pose, xs, fy, scale=1):
    for k, x in enumerate(xs):
        blit(dst, maiden(k, lt_pose[k] if isinstance(lt_pose, (list, tuple)) else lt_pose), x, fy,
             scale=scale, anchor='bottom')


# ================================================================ 动态元素
def grass(dst, t, xs, ybase, lit=1.0, h=(3, 7), gust=0.0):
    """被风吹向左的草：每根草是一条两段的折线，顶端随风摆。xs/ybase: 根部坐标数组。"""
    n = len(xs)
    rng = np.random.default_rng(n)
    hh = rng.uniform(*h, n)
    sway = -1.5 - 2.5 * gust + 1.4 * np.sin(t * 5.0 + xs * 0.31) + 0.8 * np.sin(t * 9.0 + xs * 0.7)
    for k in range(5):
        f = k / 4
        x = np.floor(xs + sway * f * f).astype(int)
        y = np.floor(ybase - hh * f).astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        col = C_GRASS_LIT * lit + C_GRASS * (1 - lit) if k == 4 else C_GRASS
        dst[y[ok], x[ok]] = col


def spear_px(dst, x, y, ang, ln, col=C_NEAR, rim=None, w=1):
    a = np.radians(ang)
    x1, y1 = x + np.cos(a) * ln, y - np.sin(a) * ln
    xs, ys = _line_px(x, y, x1, y1)
    for dx in range(w):
        ok = (xs + dx >= 0) & (xs + dx < W) & (ys >= 0) & (ys < H)
        dst[ys[ok], xs[ok] + dx] = col
    if rim is not None:
        ok = (xs - 1 >= 0) & (xs - 1 < W) & (ys >= 0) & (ys < H)
        dst[ys[ok], xs[ok] - 1] = rim
    # 枪头
    for k in range(4):
        px, py = int(x1 + np.cos(a) * k * 0.5), int(y1 - np.sin(a) * k * 0.5) - 1
        if 0 <= px < W and 0 <= py < H:
            dst[py, px] = col
    return x1, y1


def banner_px(dst, x, y, t, ph, wdt=10, hgt=7, col=C_BANNER, rim=C_BANNER_RIM, wind=1.0):
    """挂在枪头下的破旗：向左飘，每列上下波动，下缘有缺口。"""
    for k in range(wdt):
        off = np.sin(t * 7 * wind + ph + k * 0.7) * (0.4 + k * 0.12) * wind
        top = int(y + off + k * 0.15)
        hh = hgt - (2 if (k + int(ph * 3)) % 4 == 1 else 0) - k * 0.2
        px = int(x - k)
        if 0 <= px < W:
            y0, y1 = max(0, top), min(H, int(top + hh))
            if y1 > y0:
                dst[y0:y1, px] = col
                dst[y1 - 1, px] = rim


def spiral(dst, cx, cy, t, r, n=12, cols=(TEAL, GOLD, WHITE), stretch=0.45, axis=(1, 0)):
    """绕枪轴旋转的光螺旋：一圈 n 个点，按透视压扁（stretch）。"""
    ax, ay = axis
    for i in range(n):
        a = t * 20 + i * 2 * np.pi / n
        rr = r * (0.7 + 0.3 * np.sin(i * 1.7 + t * 6))
        u = np.cos(a) * rr * stretch
        v = np.sin(a) * rr
        x = cx + ax * u - ay * v
        y = cy + ay * u + ax * v
        col = cols[i % len(cols)]
        for dx, dy in ((0, 0), (1, 0)) if i % 3 == 0 else ((0, 0),):
            px, py = int(x) + dx, int(y) + dy
            if 0 <= px < W and 0 <= py < H:
                dst[py, px] = np.minimum(dst[py, px] + col * 0.9, 1.0)


def glow(dst, cx, cy, r, col, k=1.0, flat=1.0):
    """抖动的圆形辉光（加色）。"""
    x0, x1 = max(0, int(cx - r)), min(W, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r * flat)), min(H, int(cy + r * flat) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    d = np.hypot(XX[y0:y1, x0:x1] - cx, (YY[y0:y1, x0:x1] - cy) / flat) / r
    lv = clamp01(1 - d) ** 1.5 * k
    m = BAYER[y0:y1, x0:x1] < lv
    dst[y0:y1, x0:x1][m] = np.minimum(dst[y0:y1, x0:x1][m] + col * 0.6, 1.0)


def light_tint(dst, cx, cy, strength, col=_c('#fff0c0'), r=420):
    """圣光把整个场景染成白金：离光源越近越亮，并往光色偏。"""
    if strength <= 0:
        return
    d = np.hypot(XX - cx, (YY - cy) * 1.4) / r
    k = (clamp01(1 - d) * 0.65 + 0.2) * strength
    dst[:] = dst * (1 - k[..., None] * 0.55) + col * k[..., None] * 0.55 + dst * k[..., None] * 0.35


def cross_of_light(dst, cx, cy_arm, ground, t, grow_v, grow_h, fade=1.0, pulse=0.0):
    """光之十字：从地面 ground 直冲天顶的竖柱 + 在 cy_arm 处横贯的横臂。
    白色核心、金色主体、青色外缘，边缘用 Bayer 抖动做软过渡，并有上升的光纹。"""
    if fade <= 0:
        return
    wv = (14 + 10 * pulse) * grow_h ** 0.3 if grow_v > 0 else 0
    top = ground - (ground + 10) * grow_v
    hw = (200 + 60 * pulse) * grow_h
    hh = (9 + 6 * pulse) * min(1, grow_h * 2)
    # 竖柱
    if grow_v > 0:
        y0, y1 = max(0, int(top)), int(ground) + 1
        xs = slice(max(0, int(cx - wv - 10)), min(W, int(cx + wv + 11)))
        dx = np.abs(XX[y0:y1, xs] - cx)
        shimmer = 0.85 + 0.15 * np.sin(YY[y0:y1, xs] * 0.35 + t * 40)
        lv = clamp01(1 - dx / (wv + 10)) * shimmer * fade
        reg = dst[y0:y1, xs]
        bay = BAYER[y0:y1, xs]
        reg[(bay < lv * 1.4) & (dx > wv)] = np.minimum(reg[(bay < lv * 1.4) & (dx > wv)] * 0.5 + TEAL * 0.8, 1)
        reg[dx <= wv] = GOLD * 0.35 + WHITE * 0.65
        reg[dx <= wv * 0.55] = WHITE
        # 竖柱里的上升光纹
        streak = ((YY[y0:y1, xs] + t * 520) % 23 < 2) & (dx <= wv * 0.9) & (dx > wv * 0.3)
        reg[streak] = _c('#fff3b0')
    # 横臂
    if grow_h > 0:
        y0, y1 = max(0, int(cy_arm - hh - 10)), min(H, int(cy_arm + hh + 11))
        xs = slice(max(0, int(cx - hw)), min(W, int(cx + hw) + 1))
        dy = np.abs(YY[y0:y1, xs] - cy_arm)
        dxn = np.abs(XX[y0:y1, xs] - cx) / max(hw, 1)
        taper = clamp01(1 - dxn ** 3)
        lv = clamp01(1 - dy / (hh + 10)) * taper * fade
        reg = dst[y0:y1, xs]
        bay = BAYER[y0:y1, xs]
        core = dy <= hh * taper
        m = (bay < lv * 1.4) & ~core
        reg[m] = np.minimum(reg[m] * 0.5 + TEAL * 0.8, 1)
        reg[core] = GOLD * 0.35 + WHITE * 0.65
        reg[dy <= hh * 0.5 * taper] = WHITE
        streak = ((XX[y0:y1, xs] + t * 600 * np.sign(XX[y0:y1, xs] - cx)) % 29 < 2) & core & (dy > hh * 0.3 * taper)
        reg[streak] = _c('#fff3b0')
    # 交叉点的星芒
    if grow_h > 0.3:
        glow(dst, cx, cy_arm, 40 + 20 * pulse, WHITE, 0.9 * fade)


def shock_cone(dst, x, y, t, k=1.0):
    """冲锋时枪尖前的激波锥：几条向后张开的弧线（白 / 青）。"""
    for i, (off, col) in enumerate(((0, WHITE), (7, TEAL), (14, GOLD))):
        ax = x - 6 + off
        for s in (-1, 1):
            for j in range(26):
                u = j / 25
                px = int(ax + u * 44)
                py = int(y + s * (3 + u ** 0.7 * 26 * k))
                if 0 <= px < W and 0 <= py < H and (j + i) % (1 + i) == 0:
                    dst[py, px] = np.minimum(dst[py, px] * 0.3 + col, 1)


# ================================================================ 场景状态：动态的枪与旗（远景）
def _init_spears(st):
    rng = np.random.default_rng(21)
    sp = []
    for x in np.concatenate([rng.uniform(20, 280, 16), rng.uniform(370, 470, 6)]):
        base = _mound_y(x)
        sp.append(dict(x=x, y=base + 2, ang=90 + rng.normal(0, 11), ln=rng.uniform(22, 44),
                       flag=rng.random() < 0.45, ph=rng.uniform(0, 6), vx=0.0, vy=0.0, va=0.0, free=False))
    st['spears'] = sp


def _draw_spears(dst, st, t, lit=0.0, wind=1.0):
    col = C_NEAR * (1 - lit) + _c('#3a1a18') * lit
    rim = C_NEAR_RIM * (1 - lit) + _c('#ffe0a0') * lit
    for s in st['spears']:
        x1, y1 = spear_px(dst, s['x'], s['y'], s['ang'], s['ln'], col, rim)
        if s['flag']:
            a = np.radians(s['ang'])
            banner_px(dst, x1 - np.cos(a) * 3, y1 + np.sin(a) * 3, t, s['ph'], wind=wind,
                      col=C_BANNER * (1 - lit) + _c('#a04030') * lit, rim=C_BANNER_RIM * (1 - lit) + GOLD * lit)


def _blow_spears(st, cx, cy, rng, power=1.0):
    for s in st['spears']:
        d = s['x'] - cx
        if abs(d) < 230 * power and not s['free']:
            s['free'] = True
            sgn = np.sign(d) or 1
            s['vx'] = sgn * rng.uniform(140, 320) * power * (1.2 - abs(d) / 300)
            s['vy'] = -rng.uniform(120, 280) * power
            s['va'] = -sgn * rng.uniform(200, 600)


def _step_spears(st, dt):
    for s in st['spears']:
        if s['free']:
            s['vy'] += 420 * dt
            s['x'] += s['vx'] * dt
            s['y'] += s['vy'] * dt
            s['ang'] += s['va'] * dt


# ================================================================ 背景组合
def backdrop(dst, cam, sun_xy, sun_r, near='wide', t=0.0):
    """天空 → 太阳 → 要塞 → 中景枪林 → 近景地面。cam = 水平平移（像素，正 = 镜头右移）。"""
    sky = sky_layer(True)
    ox = int(80 + cam * 0.08)
    dst[:] = sky[:, ox:ox + W]
    sp = sun_sprite(sun_r)
    blit(dst, sp, sun_xy[0] - cam * 0.12 - sp.shape[1] / 2, sun_xy[1] - sp.shape[0] / 2)
    # 太阳前面再压一道云
    for cy, x0, x1 in ((sun_xy[1] + sun_r * 0.35, sun_xy[0] - sun_r * 1.6, sun_xy[0] + sun_r * 0.9),
                       (sun_xy[1] - sun_r * 0.45, sun_xy[0] - sun_r * 0.4, sun_xy[0] + sun_r * 1.8)):
        xs = slice(int(max(0, x0 - cam * 0.12)), int(min(W, x1 - cam * 0.12)))
        y = int(cy)
        if 0 <= y < H - 2:
            dst[y:y + 2, xs] = C_CLOUD
            dst[y + 2, xs] = C_CLOUD_LIT
    _layer(dst, fortress_layer(), -80 - cam * 0.25)
    _layer(dst, mid_layer(), -80 - cam * 0.5)
    if near == 'wide':
        _layer(dst, near_wide_layer(), -80 - cam)
    elif near == 'crest':
        _layer(dst, near_crest_layer(), -80 - cam)


def ash(ctx, n=3, gust=0.0):
    """随风向左飘的灰烬与火星。"""
    P = ctx.P
    P.emit(n, W + 4, (20, H - 20), (-110 - 80 * gust, -50), (-12, 8), (3, 5), 'fire', size=1, bright=0.35,
           wob=0.15)
    if ctx.rng.random() < 0.35:
        P.emit(1, W + 4, (60, 200), (-160, -90), (-20, 0), (2, 3), 'gold', size=1, bright=0.8, wob=0.2)


# ================================================================ 渲染
X_MED, Y_MED = 330, 196          # 中景：脚底位置（2x）
X_WIDE, Y_WIDE = 330, 196        # 远景：丘顶
MAIDENS_X = (398, 430, 462)


def _flut(lt, amp=1.0, fps=12):
    return f':{int(lt * fps) % 4}:{amp:g}'


def render(dst, ctx):
    lt, t = ctx.lt, ctx.t
    ch = ctx.C['arianrhod']
    st = ctx.state
    if 'spears' not in st:
        _init_spears(st)
    beat = ctx.sec.beat
    at = ctx.at
    P, P2 = ctx.P, ctx.P2

    # ---------------------------------------------------------------- 1：远景
    if lt < at(2, 0):
        k = ctx.prog(1, 0, 2, 0)
        cam = 18 * (1 - ease_out(k)) - 6
        backdrop(dst, cam, (X_WIDE - 10, 164), 50, 'wide', t)
        _draw_spears(dst, st, t)
        put_char(dst, ch, 'planted' + _flut(lt, 1.5), X_WIDE - cam, Y_WIDE + 1, 1, 'back')
        xs = np.arange(0, W, 2.0)
        grass(dst, t, xs, _mound_y(xs + cam) + 1, lit=0.6, h=(2, 6), gust=0.5 * smooth(0.3, 1.0, k))
        ash(ctx, 3)
        P.step(ctx.dt)
        P.render(dst)
        return dst

    # ---------------------------------------------------------------- 2.0–2.2：特写，睁眼
    if lt < at(2, 2):
        k = ctx.prog(2, 0, 2, 2)
        cam = -30 + 10 * k
        backdrop(dst, cam, (170, 150), 70, None, t)
        dst[:] = dst * 0.8
        so = ctx.since(2, 1)
        expr = 'closed' if so < 0 else ('stern' if so < 0.07 else None)
        por = ch.portrait(expr)
        img = scale_img(por, 3)
        px, py = 190 - 4 * k, 30
        blit(dst, img, px, py)
        if 0 <= so < 0.5:            # 瞳中青光
            ex, ey = px + 47 * 3, py + 34 * 3
            glow(dst, ex, ey, 10 * (1 - so / 0.5) + 2, TEAL, 1.0)
            if ctx.crossed(2, 1):
                P2.emit_radial(18, ex, ey, (30, 90), (0.2, 0.5), 'mint', size=1, drag=3)
        ash(ctx, 4, gust=0.5)
        for _ in range(2):           # 近景的发丝 / 灰烬从镜头前掠过
            P.emit(1, W + 2, (0, H), (-260, -180), (-10, 10), (1.5, 2), 'gold', size=2, bright=0.25)
        P.step(ctx.dt)
        P2.step(ctx.dt)
        P.render(dst)
        P2.render(dst)
        ui.name_tag(dst, ch, ctx.since(2, 0))
        return dst

    # ---------------------------------------------------------------- 2.2–4.3.5：中景（拔枪、端平、蓄力）
    if lt < at(4, 3.5):
        k = ctx.prog(2, 2, 4, 3.5)
        cam = 10 * k - 4
        charge = ctx.prog(4, 1.12, 5, 0)
        backdrop(dst, cam, (230, 150), 58, 'crest', t)
        if charge > 0:               # 蓄力的光开始照亮场景
            light_tint(dst, 170, 150, 0.35 * charge ** 2, TEAL * 0.4 + WHITE * 0.6)
        # 铁机队跪在她身后
        draw_maidens(dst, 'kneel', [x - cam * 0.9 for x in MAIDENS_X], Y_MED + 1)
        # 姿势时间表
        sched = [((2, 2), 'planted_open'), ((3, 0), 'grip'), ((3, 2), 'pull1'), ((3, 2.5), 'pull2'),
                 ((4, 0), 'swing'), ((4, 0.5), 'guard'), ((4, 2), 'windup')]
        pose = sched[0][1]
        for (b, bt), p in sched:
            if lt >= at(b, bt):
                pose = p
        amp = {'planted_open': 1.2, 'grip': 1.2, 'pull1': 1.5, 'pull2': 2.0, 'swing': 3.0, 'guard': 2.0,
               'windup': 2.5}[pose]
        full = pose + _flut(lt, amp)
        fx_, fy = X_MED - cam, Y_MED
        bob = 0
        if pose == 'pull2':          # 举着枪时的呼吸起伏
            bob = int(np.sin(ctx.since(3, 2.5) * 6) > 0.6)
        if pose == 'windup':         # 蓄力时微微颤动
            bob = int(ctx.rng.random() < 0.3 * charge)
        img = put_char(dst, ch, full, fx_, fy + bob, 2, 'normal')
        tx, ty = tip_screen(ch, pose, fx_, fy + bob, 2)
        # 拔枪：地面裂开、土块崩飞
        gx = fx_ - 160 + 63.6 * 2
        if ctx.crossed(3, 2):
            ctx.shake(3, 0.3)
            P2.emit(40, (gx - 10, gx + 10), Y_MED, (-60, 60), (-160, -40), (0.4, 0.9), 'fire', size=(1, 2),
                    grav=500, bright=0.3)
        if ctx.crossed(3, 2.5):
            ctx.shake(4, 0.25)
            P2.emit(60, (gx - 8, gx + 8), Y_MED - 2, (-120, 120), (-260, -80), (0.5, 1.0), 'fire', size=(1, 3),
                    grav=600, bright=0.35)
            P2.emit(20, gx, Y_MED - 4, (-30, 30), (-60, -10), (0.6, 1.2), 'gold', size=1, bright=0.6, drag=2)
        if at(3, 2) <= lt < at(3, 2.5):   # 裂缝
            c = ctx.prog(3, 2, 3, 2.5)
            for s in (-1, 1):
                for j in range(int(4 + 14 * c)):
                    px, py = int(gx + s * j), int(Y_MED + 1 + (j % 3 == 0) + j * 0.1)
                    if 0 <= px < W and py < H:
                        dst[py, px] = _c('#ff7a30') * (1 - j / 20)
        # 挥枪弧光：沿 pull2 → swing → guard 的枪尖轨迹
        sw = ctx.since(4, 0)
        if -0.02 < sw < 0.3:
            pts = [tip_screen(ch, p, fx_, fy, 2) for p in ('pull2', 'swing', 'guard')]
            prog = clamp01(sw / 0.2)
            fade = 1 - clamp01((sw - 0.12) / 0.18)
            hx, hy = fx_ - 160 + 66 * 2, fy - 224 + (28 + 36) * 2
            for i in range(60):
                u = i / 59 * prog
                bx = (1 - u) ** 2 * pts[0][0] + 2 * u * (1 - u) * (pts[1][0] - 30) + u * u * pts[2][0]
                by = (1 - u) ** 2 * pts[0][1] + 2 * u * (1 - u) * (pts[1][1] - 20) + u * u * pts[2][1]
                for j, col in ((0, WHITE), (4, WHITE), (8, TEAL), (13, TEAL * 0.6)):
                    kk = j / max(1, np.hypot(bx - hx, by - hy))
                    px, py = int(bx + (hx - bx) * kk), int(by + (hy - by) * kk)
                    if 0 <= px < W and 0 <= py < H and BAYER[py, px] < fade * (1 - j / 16) * (0.3 + u):
                        dst[py, px] = col
        if ctx.crossed(4, 0.5):      # 踏步扬尘
            P2.emit(30, (fx_ - 40, fx_ - 10), Y_MED, (-80, 20), (-50, -10), (0.4, 0.8), 'fire', size=(1, 2),
                    bright=0.3, drag=3)
            ctx.shake(2, 0.2)
        # 蓄力：枪尖的光螺旋 + 光核
        if charge > 0:
            spiral(dst, tx + 6, ty, t, 6 + 16 * (1 - charge) + 4 * charge, n=10 + int(10 * charge))
            spiral(dst, tx + 20, ty, t * 0.8 + 1, 4 + 10 * (1 - charge), n=8, cols=(GOLD, WHITE))
            glow(dst, tx, ty, 6 + 14 * charge ** 2, WHITE, 0.6 + 0.4 * charge)
            if ctx.rng.random() < 0.6 + charge:
                a = ctx.rng.uniform(0, 2 * np.pi)
                r = ctx.rng.uniform(30, 60)
                P2.emit_arrays(np.array([tx + np.cos(a) * r]), np.array([ty + np.sin(a) * r]),
                               np.array([-np.cos(a) * r * 3]), np.array([-np.sin(a) * r * 3]), 0.3, 'mint',
                               size=1, bright=0.9)
        xs = np.arange(0, W, 2.0)
        grass(dst, t, xs, 197 + 1.5 * np.sin((xs + cam) * 0.05), lit=0.5, h=(3, 8), gust=0.5 * charge)
        ash(ctx, 3, gust=charge)
        P.step(ctx.dt)
        P2.step(ctx.dt)
        P.render(dst)
        P2.render(dst)
        return dst

    # ---------------------------------------------------------------- 4.3.5–5.0：冲锋
    if lt < at(5, 0):
        k = ctx.prog(4, 3.5, 5, 0)
        cam = 6
        backdrop(dst, cam + 40 * k, (230, 150), 58, 'crest', t)
        light_tint(dst, 200, 150, 0.35 + 0.3 * k, TEAL * 0.4 + WHITE * 0.6)
        dst[:] = dst * (1 - 0.35 * k) + dst.mean(-1, keepdims=True) * 0.35 * k    # 去饱和，突出光迹
        draw_maidens(dst, 'kneel', [x - (cam + 40 * k) * 0.9 for x in MAIDENS_X], Y_MED + 1)
        # 高速横向速度线
        for i in range(30):
            y = int((i * 37 + 11) % 190) + 4
            x0 = int((i * 97 - lt * 2400) % (W + 200)) - 100
            ln = 40 + (i * 13) % 60
            dst[y, max(0, x0):max(0, min(W, x0 + ln))] = dst[y, max(0, x0):max(0, min(W, x0 + ln))] * 0.3 + 0.7
        x_now = X_MED - cam - (X_MED + 260) * k ** 1.3
        # 残影（旧位置、青色半透明）+ 金色螺旋尾迹
        for i in range(1, 6):
            kk = max(0.0, k - i * 0.07)
            xo = X_MED - cam - (X_MED + 260) * kk ** 1.3
            put_char(dst, ch, 'skill:1:3', xo, Y_MED, 2, 'light', alpha=0.5 - i * 0.08, tint=(0.4, 0.95, 0.9),
                     amt=0.8)
        put_char(dst, ch, f'skill:{int(lt * 24) % 4}:3', x_now, Y_MED, 2, 'light')
        tx, ty = tip_screen(ch, 'skill', x_now, Y_MED, 2)
        x_start = X_MED - cam - 160
        for j in range(0, int(max(0, x_start - tx)), 2):
            px = int(tx + j)
            a = j * 0.25 - t * 30
            for r, col in ((10, GOLD), (6, WHITE)):
                py = int(ty + np.sin(a + (r == 6) * np.pi) * r * min(1, j / 40))
                if 0 <= px < W and 0 <= py < H:
                    dst[py, px] = col
        dst[int(ty) - 1:int(ty) + 2, int(max(0, tx)):int(min(W, x_start))] = WHITE
        shock_cone(dst, tx, ty, t, 0.6 + 0.6 * k)
        glow(dst, tx, ty, 18, WHITE, 1.0)
        P2.step(ctx.dt)
        P2.render(dst)
        return dst

    # ---------------------------------------------------------------- 5：命中，光之十字（远景）
    if lt < at(6, 0):
        s5 = ctx.since(5, 0)
        fps = 1 / ctx.dt if ctx.dt else 30
        hitstop = 3 / fps
        fxw, fyw = 262, int(_mound_y(262)) + 1           # 她的落点（左坡），枪尖在十字脚下
        cx, cy = fxw - 80, fyw - 112 + 66                # 枪尖
        ground = cy + 26
        if ctx.crossed(5, 0):
            ctx.shake(8, 0.6)
            for s in st['spears']:
                s['free'] = False
        if ctx.crossed(5, 1):
            _blow_spears(st, cx, ground, np.random.default_rng(5), 1.0)
            ctx.shake(5, 0.4)
        if ctx.crossed(5, 2):
            _blow_spears(st, cx, ground, np.random.default_rng(6), 1.6)
            ctx.shake(6, 0.5)
            ctx.flash('#fff6d0', 0.15, 0.6)
            P2.emit_radial(160, cx, 95, (60, 300), (0.5, 1.2), 'gold', size=(1, 2), drag=1.5)
        # 冲击帧：两帧反相剪影
        if s5 < 2 / fps:
            dst[:] = WHITE
            blit(dst, spr(ch, 'skill:0:3', 1, 'black'), fxw, fyw, anchor='bottom')
            m = near_wide_layer()[..., 3] > 0
            dst[m[:, 80:80 + W]] = _c('#08020a')
            _draw_spears(dst, st, t)
            dst[(dst.sum(-1) < 1.2)] = _c('#08020a')
            glow(dst, cx, cy, 40, _c('#000000'), 1.0)
            return dst
        cam = 0
        backdrop(dst, cam, (340, 160), 50, 'wide', t)
        e = s5 - hitstop
        grow_v = clamp01(e / 0.12) if e > 0 else 0
        grow_h = clamp01((e - 0.08) / 0.16) if e > 0 else 0
        pulse = np.exp(-max(0, ctx.since(5, 2)) / 0.25) if ctx.since(5, 2) >= 0 else 0
        light = 0.25 + 0.75 * max(grow_v, pulse, 1.0 if e <= 0 else 0.0)
        light_tint(dst, cx, 110, light, _c('#fff0c8'))
        cross_of_light(dst, cx, 96, ground, t, grow_v, grow_h, 1.0, pulse)
        # 地面冲击波（压扁的环）
        for b0, pw in ((1, 1.0), (2, 1.5)):
            sb = ctx.since(5, b0)
            if 0 <= sb < 0.6:
                r = 20 + sb * 700 * pw
                from fx import shockwave
                shockwave(dst, cx, ground, r, _c('#fff0c0'), w=3, flat=0.18)
                shockwave(dst, cx, ground, r * 0.8, TEAL, w=2, flat=0.18)
        if e > 0:
            _step_spears(st, ctx.dt)
        _draw_spears(dst, st, t, lit=0.6 * light, wind=2.0)
        put_char(dst, ch, 'skill:1:2', fxw, fyw, 1, 'light')
        # 撞击点的光核 + 火花
        glow(dst, cx, cy, 16 + 10 * pulse + (30 if e <= 0 else 0), WHITE, 1.0)
        if e > 0 and ctx.rng.random() < 0.9:
            P2.emit(6, cx, cy, (-60, 60), (-240, -60), (0.4, 0.9), 'gold', size=1, grav=200)
            P2.emit(3, (cx - 16, cx + 16), ground, (-10, 10), (-300, -160), (0.6, 1.0), 'white', size=1)
        if ctx.crossed(5, 0.25):
            P2.emit_radial(200, cx, cy, (80, 360), (0.4, 1.0), 'gold', size=(1, 2), drag=1.8)
            P2.emit_radial(80, cx, cy, (60, 240), (0.4, 0.8), 'mint', size=1, drag=2)
        xs = np.arange(0, W, 2.0)
        grass(dst, t, xs, _mound_y(xs) + 1, lit=0.8, h=(2, 6), gust=1.5)
        if e > 0:
            P2.step(ctx.dt)
            ash(ctx, 6, gust=2)
            P.step(ctx.dt)
        P.render(dst)
        P2.render(dst)
        return dst

    # ---------------------------------------------------------------- 6：落地、余韵、定格
    s6 = ctx.since(6, 0)
    cam = -6 + 4 * ctx.prog(6, 0, 7, 0)
    backdrop(dst, cam, (230, 150), 58, 'crest', t)
    fade_l = 1 - smooth(0.0, 1.3, s6)
    light_tint(dst, 110, 120, 0.6 * fade_l + 0.08, _c('#fff0c8'))
    # 远处的光柱（十字残光）慢慢消散
    cross_of_light(dst, 96 - cam * 0.5, 70, 190, t, 1.0, 1.0 - 0.6 * smooth(0, 1.2, s6), fade_l ** 0.7, 0)
    if ctx.rng.random() < 0.8:
        P2.emit(2, (70, 130), (40, 190), (-6, 6), (-60, -25), (1.0, 2.0), 'gold', size=1, bright=0.8)
    # 铁机队：6.1 起身致敬
    sm = ctx.since(6, 1)
    mp = 'kneel' if sm < 0 else ('rise' if sm < 0.12 else 'salute')
    draw_maidens(dst, mp, [x - cam * 0.9 for x in MAIDENS_X], Y_MED + 1)
    # 她：落地蹲 → 起身 → 立绘站姿，披风抖动慢慢收小
    if s6 < 0.1:
        pose = 'windup:%d:4' % (int(lt * 12) % 4)
    elif s6 < 0.22:
        pose = 'guard:%d:3' % (int(lt * 12) % 4)
    else:
        amp = 1.0 + 3.0 * np.exp(-(s6 - 0.22) / 0.5)
        pose = 'idle' + _flut(lt, round(amp * 2) / 2)
    fx_ = X_MED - cam
    put_char(dst, ch, pose, fx_, Y_MED, 2, 'normal')
    if ctx.crossed(6, 0):
        P2.emit(50, (fx_ - 60, fx_ + 40), Y_MED, (-120, 120), (-70, -10), (0.4, 0.9), 'fire', size=(1, 2),
                bright=0.35, drag=2.5)
        ctx.shake(3, 0.25)
    xs = np.arange(0, W, 2.0)
    grass(dst, t, xs, 197 + 1.5 * np.sin((xs + cam) * 0.05), lit=0.5, h=(3, 8), gust=0.3)
    ash(ctx, 3)
    P.step(ctx.dt)
    P2.step(ctx.dt)
    P.render(dst)
    P2.render(dst)
    return dst
