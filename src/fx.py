"""特效库：背景、光束、魔法阵、刀光、蛇眼、气焰、卡牌/玫瑰/音符等精灵粒子、转场。"""
import numpy as np
from PIL import Image, ImageDraw

from gfx import BAYER, H, W, XX, YY, blit, clamp01, dilate, dither_mask, hexc, rect, scale_img

TAU = 2 * np.pi
_rng0 = np.random.default_rng(7)
STARS = np.stack([_rng0.uniform(0, W, 260), _rng0.uniform(0, H, 260), _rng0.uniform(0, TAU, 260),
                  _rng0.uniform(1.5, 5, 260), _rng0.integers(0, 3, 260)], 1)
SPEED = np.stack([_rng0.uniform(0, H, 60), _rng0.uniform(30, 140, 60), _rng0.uniform(700, 1500, 60),
                  _rng0.uniform(0, W + 300, 60)], 1)


# ---------------------------------------------------------------- 形状
def polygon_mask(pts):
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).polygon([(float(x), float(y)) for x, y in pts], fill=255)
    return np.array(im) > 0


def circle_mask(cx, cy, r):
    return (XX - cx) ** 2 + (YY - cy) ** 2 <= r * r


def ring_mask(cx, cy, r, w=1.0):
    return np.abs(np.hypot(XX - cx, YY - cy) - r) <= w / 2


def put(dst, mask, color, alpha=1.0):
    c = np.asarray(color, np.float32)
    if alpha >= 1:
        dst[mask] = c
    else:
        dst[mask] = dst[mask] * (1 - alpha) + c * alpha


def add(dst, mask, color, k=1.0):
    dst[mask] += np.asarray(color, np.float32) * k


# ---------------------------------------------------------------- 背景元素
def stars(dst, t, alpha=1.0):
    cols = (np.array([1, 1, 1]), np.array([0.8, 0.6, 1]), np.array([1, 0.85, 0.6]))
    for x, y, ph, sp, kind in STARS:
        b = (0.3 + 0.7 * abs(np.sin(t * sp + ph))) * alpha
        yi, xi = int(y), int(x)
        dst[yi, xi] = np.maximum(dst[yi, xi], cols[int(kind)] * b)


def fog(dst, t, color, strength=0.3, scale=0.02, speed=6.0, seed=0):
    """便宜的像素雾：几层正弦叠加后用抖动阈值化。"""
    ph = seed * 1.7
    v = (np.sin(XX * scale + t * speed * scale + ph) + np.sin(YY * scale * 1.7 - t * 0.3 + ph * 2)
         + np.sin((XX + YY) * scale * 0.6 + t * 0.2 + ph)) / 3
    lvl = np.clip((v + 0.2) * strength, 0, 1)
    add(dst, BAYER < lvl, color, 0.5)


def rays(dst, cx, cy, t, color, n=10, strength=0.35, speed=0.5, r0=20, reach=330):
    a = np.arctan2(YY - cy, XX - cx)
    d = np.hypot(XX - cx, YY - cy)
    m = (np.sin(a * n + t * speed) > 0.55) & (d > r0)
    fall = np.clip(1 - d / reach, 0, 1)
    add(dst, dither_mask(fall * strength * 2) & m, color, 0.5)


def god_rays_down(dst, t, color, x0=240, spread=1.0, strength=0.5):
    """自上而下的神圣光束（盟主）。"""
    a = np.arctan2(XX - x0, YY + 60)
    m = np.sin(a * 22 * spread + t * 0.7) + 0.6 * np.sin(a * 9 - t * 0.4) > 0.8
    fall = np.clip(1 - YY / H, 0, 1) * strength
    add(dst, m & dither_mask(fall), color, 0.6)


def speed_lines(dst, t, color, direction=-1, alpha=0.55):
    for y, ln, sp, off in SPEED:
        x = (off + direction * sp * t) % (W + 300) - 150
        rect(dst, x, y, x + ln, y + 1, color, alpha)


def radial_lines(dst, cx, cy, t, color, n=48, seed=0):
    """漫画集中线。"""
    rng = np.random.default_rng(seed + int(t * 20))
    for _ in range(n):
        ang = rng.uniform(0, TAU)
        r0 = rng.uniform(90, 150)
        r1 = r0 + rng.uniform(80, 220)
        for r in np.arange(r0, r1, 1.0):
            x, y = int(cx + np.cos(ang) * r), int(cy + np.sin(ang) * r)
            if 0 <= x < W and 0 <= y < H:
                dst[y, x] = color


def diamond_wall(dst, c1, c2, size=16, y1=H):
    diag = ((XX + YY) // size + (XX - YY) // size) % 2 == 0
    put(dst, diag & (YY < y1), hexc(c2))


def gears(dst, t, color, alpha=0.25):
    """背景里缓慢转动的齿轮剪影（命运的齿轮）。"""
    for (cx, cy, r, sp, teeth) in ((70, 70, 60, 0.15, 14), (400, 200, 80, -0.1, 18), (430, 40, 36, 0.25, 10),
                                   (40, 230, 44, -0.2, 11)):
        d = np.hypot(XX - cx, YY - cy)
        a = np.arctan2(YY - cy, XX - cx) + t * sp
        tooth = (np.sin(a * teeth) > 0.2)
        m = ((d < r) & (d > r * 0.72)) | ((d >= r) & (d < r + 5) & tooth) | ((d < r * 0.25) & (d > r * 0.15))
        spokes = (d < r * 0.75) & (np.abs(np.sin(a * 3)) < 0.08 * r / np.maximum(d, 1))
        put(dst, (m | spokes) & dither_mask(0.8), color, alpha)


def moon_scene(dst, oy, t):
    from gfx import vgrad
    dst[:] = vgrad('#12020c', '#6a1420', -30 + oy, 230 + oy)
    cx, cy, R = 240, 100 + oy, 78
    d = np.hypot(XX - cx, YY - cy)
    moon = d <= R
    shade = np.clip(0.75 + 0.25 * (-(XX - cx) - (YY - cy)) / R, 0, 1)
    col = hexc('#ff5a3a')[None, None, :] * shade[..., None]
    dst[moon] = col[moon]
    for (mx, my, mr) in ((-26, -18, 13), (20, 12, 17), (-6, 32, 9), (36, -30, 7), (-40, 20, 7)):
        cr = np.hypot(XX - cx - mx, YY - cy - my) < mr
        dst[moon & cr & dither_mask(0.5)] = hexc('#c02a2a')
    halo = (d > R) & (d < R + 20)
    dst[halo & dither_mask(0.25) & (d < R + 10)] += hexc('#7a1a1a')
    dst[halo & dither_mask(0.1)] += hexc('#4a0a14')
    for i, (by, sp) in enumerate(((84, 14), (136, -9), (58, 20))):
        yb = by + oy
        band = (np.abs(YY - yb - 4 * np.sin(XX * 0.035 + i)) < 4) & (((XX + t * sp + i * 70) % 230) < 160)
        dst[band] = hexc('#2a0814')
    ridge = 196 + oy + 6 * np.sin(XX * 0.06) + 4 * np.sin(XX * 0.17 + 1) + 3 * ((XX // 9) % 3)
    cliff = YY >= ridge
    dst[cliff] = hexc('#0a0208')
    dst[cliff & (YY < ridge + 1)] = hexc('#3a0a14')


# ---------------------------------------------------------------- 角色效果
def aura(dst, spr, x, y, scale, color, rng, r=2, anchor='bottom'):
    """角色周围跳动的像素气焰。(x, y) 与 blit 的 anchor 相同。"""
    m = scale_img(spr[..., 3] > 0, scale)
    h, w = m.shape
    pad = r + 3
    m = np.pad(m, pad)
    ring = dilate(m, r) & ~m
    up = np.roll(ring, -2, 0) & (rng.random(ring.shape) < 0.5)
    outer = (ring | up) & ~m
    if anchor == 'bottom':
        y0, x0 = int(round(y - h)) - pad, int(round(x - w / 2)) - pad
    else:
        y0, x0 = int(round(y - h / 2)) - pad, int(round(x - w / 2)) - pad
    ys, xs = np.nonzero(outer)
    ys, xs = ys + y0, xs + x0
    ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    flick = rng.random(ok.sum()) < 0.8
    dst[ys[ok][flick], xs[ok][flick]] = np.asarray(color, np.float32)


def silhouette(img, color=(0.04, 0.02, 0.08), rim=None, eyes=(), eye_color=None):
    out = img.copy()
    a = out[..., 3] > 0
    out[a, :3] = color
    if rim is not None:
        edge = a & ~np.roll(a, -1, axis=1) | a & ~np.roll(a, 1, axis=0)
        out[edge, :3] = rim
    if eye_color is not None:
        for (y, x) in eyes:
            if 0 <= y < out.shape[0] and 0 <= x < out.shape[1]:
                out[y, x, :3] = eye_color
                out[y, x, 3] = 1
    return out


def serpent_eye(dst, cx, cy, hw, hh, open_, t, lid='#40ff90', iris=('#ffd84a', '#e08a1a')):
    if open_ <= 0.01:
        rect(dst, cx - hw, cy, cx + hw, cy + 1, hexc(lid), 0.6)
        return
    x = (XX - cx) / hw
    inside = (np.abs(YY - cy) < hh * open_ * np.clip(1 - x * x, 0, 1)) & (np.abs(x) < 1)
    ri = hh * 0.95
    d = np.hypot(XX - cx, YY - cy)
    a = np.arctan2(YY - cy, XX - cx)
    irm = inside & (d < ri)
    base = np.where((d < ri * 0.55)[..., None], hexc(iris[0]), hexc(iris[1]))
    base = np.where((np.sin(a * 22 + d * 0.3) > 0.4)[..., None], base * 0.78, base)
    dst[inside] = hexc('#1a0610')
    dst[irm] = base[irm]
    dst[inside & (np.abs(d - ri) < 1.0)] = hexc('#6a1a8a')
    pw = 3.5 + 2 * np.sin(t * 3)
    slit = irm & (np.abs(XX - cx) < pw * np.clip(1 - ((YY - cy) / ri) ** 2, 0, 1))
    dst[slit] = hexc('#050208')
    dst[(dilate(slit, 1) & ~slit & irm)] = hexc(lid)
    dst[inside & (np.hypot(XX - cx + ri * 0.35, YY - cy + ri * 0.4) < 4)] = hexc('#fff8d0')
    lidm = dilate(inside, 1) & ~inside
    dst[lidm] = hexc(lid)
    dst[dilate(lidm, 1) & ~lidm & ~inside & dither_mask(0.5)] = hexc(lid) * 0.35


def magic_circle(dst, cx, cy, R, t, color, flat=1.0, dots=12, star=6):
    dx, dy = XX - cx, (YY - cy) / flat
    d = np.hypot(dx, dy)
    a = np.arctan2(dy, dx)
    m = (np.abs(d - R) < 0.8) | (np.abs(d - R * 0.8) < 0.6)
    m |= (np.abs(d - R * 0.9) < 1.6) & (((a + t * 1.2) % (TAU / dots)) < 0.12)
    for k in range(star):
        ang = t * -0.6 + k * TAU / star
        p = np.array([np.cos(ang), np.sin(ang)]) * R * 0.8
        q = np.array([np.cos(ang + TAU / 3), np.sin(ang + TAU / 3)]) * R * 0.8
        v = q - p
        tt = np.clip(((dx - p[0]) * v[0] + (dy - p[1]) * v[1]) / (v @ v), 0, 1)
        m |= np.hypot(dx - p[0] - tt * v[0], dy - p[1] - tt * v[1]) < 0.6
    dst[m] = np.minimum(dst[m] + np.asarray(color, np.float32), 1.3)


def slash_arc(dst, cx, cy, rx, ry, a0, a1, prog, fadev, thick=10, cols=('#ff5a1a', '#c01a1a', '#fff0a0')):
    dx, dy = (XX - cx) / rx, (YY - cy) / ry
    d = np.hypot(dx, dy)
    a = np.arctan2(dy, dx) % TAU
    amax = a0 + (a1 - a0) * prog
    within = (a >= min(a0, amax)) & (a <= max(a0, amax))
    along = np.clip((a - a0) / (a1 - a0 + 1e-6), 0, 1)
    w = thick * np.sin(along * np.pi) / rx
    band = within & (np.abs(d - 1) < w)
    core = within & (np.abs(d - 1) < w * 0.4)
    if fadev < 1:
        keep = dither_mask(fadev)
        band &= keep
        core &= keep
    dst[band & (d > 1)] = hexc(cols[0])
    dst[band & (d <= 1)] = hexc(cols[1])
    dst[core] = hexc(cols[2])


def beam(dst, x0, y0, x1, y1, width, prog, cols=('#ffffff', '#ffe8a0', '#e0a040'), t=0):
    """直线能量束（长枪突刺 / 光柱）。prog 控制长度，宽度带抖动。"""
    vx, vy = x1 - x0, y1 - y0
    L = np.hypot(vx, vy)
    ux, uy = vx / L, vy / L
    along = (XX - x0) * ux + (YY - y0) * uy
    perp = np.abs(-(XX - x0) * uy + (YY - y0) * ux)
    wv = width * (1 + 0.15 * np.sin(along * 0.3 - t * 40))
    inb = (along > 0) & (along < L * prog)
    for k, c in enumerate(reversed(cols)):
        f = 1 - k * 0.33
        dst[inb & (perp < wv * f)] = hexc(c)
    tip = inb & (along > L * prog - 6)
    return tip


def rose(dst, cx, cy, R, t, cols=('#e0b8ff', '#9b5bd6', '#6a2aa8', '#3a1060')):
    """像素大玫瑰：外深内浅的多层花瓣，缓慢旋转。"""
    if R < 2:
        return
    d = np.hypot(XX - cx, YY - cy)
    a = np.arctan2(YY - cy, XX - cx)
    for k in range(5, -1, -1):
        rk = R * (0.25 + 0.15 * k)
        petal = d < rk * (0.78 + 0.22 * np.abs(np.cos(2.5 * a + k * 1.3 + t * (0.5 if k % 2 else -0.4))))
        c = hexc(cols[min(3, (5 - k) // 2 + (k % 2))]) if k > 0 else hexc(cols[0])
        put(dst, petal, c)
        put(dst, dilate(petal, 1) & ~petal & (d < rk * 1.05), hexc('#1a0a28'))


def shockwave(dst, cx, cy, r, color, w=3, flat=1.0):
    d = np.hypot(XX - cx, (YY - cy) / flat)
    put(dst, np.abs(d - r) < w / 2, hexc(color) if isinstance(color, str) else color)
    put(dst, np.abs(d - r * 0.86) < 0.6, (hexc(color) if isinstance(color, str) else color) * 0.6)


# ---------------------------------------------------------------- 精灵粒子（卡牌、玫瑰、音符、镜片）
def _grid(rows, pal):
    h, w = len(rows), max(len(r) for r in rows)
    img = np.zeros((h, w, 4), np.float32)
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            if c != '.':
                img[y, x, :3] = hexc(pal[c])
                img[y, x, 3] = 1
    return img


CARD_PAL = {'K': '#20182c', 'W': '#fbf8ff', 'R': '#e02848', 'G': '#d8b040', 'P': '#9a1838'}
CARD_FRONT = _grid(["KKKKKK", "KWWWWK", "KWRWWK", "KRRRWK", "KWRWWK", "KWWWWK", "KWWWRK", "KKKKKK"], CARD_PAL)
CARD_BACK = _grid(["KKKKKK", "KPGPGK", "KGPGPK", "KPGPGK", "KGPGPK", "KPGPGK", "KGPGPK", "KKKKKK"], CARD_PAL)
CARD_EDGE = _grid(["K", "W", "W", "W", "W", "W", "W", "K"], CARD_PAL)
NOTE = _grid(["..KK", "..KB", "..K.", "..K.", "KKK.", "KBK.", "KK.."], {'K': '#bff0ff', 'B': '#6cc8ff'})
ROSE = _grid([".RR.", "RrRR", "RRrR", ".RR.", "..G.", ".GG.", "..G."], {'R': '#e8285a', 'r': '#8a0f30', 'G': '#40a060'})
PETAL = _grid(["RR", "Rr"], {'R': '#ff3a6a', 'r': '#a01040'})
VPETAL = _grid(["VV", "Vv"], {'V': '#b57ae8', 'v': '#5a2a90'})
SHARD = _grid(["..W", ".WA", "WAa", "Aa."], {'W': '#ffffff', 'A': '#c8c0f0', 'a': '#7a70b0'})


class SpriteBits:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.items = []

    def emit(self, n, x, y, vx, vy, life, kind='card', scale=(1, 2), grav=0.0, drag=0.0):
        r = self.rng
        u = lambda v: r.uniform(*v) if isinstance(v, tuple) else v
        for _ in range(n):
            self.items.append(dict(x=u(x), y=u(y), vx=u(vx), vy=u(vy), life=u(life), ph=r.uniform(0, TAU),
                                   spin=r.uniform(6, 16) * r.choice([-1, 1]), kind=kind,
                                   s=int(r.integers(scale[0], scale[1] + 1)), grav=grav, drag=drag,
                                   sway=r.uniform(0, TAU)))

    def step(self, dt):
        for it in self.items:
            it['vy'] += it['grav'] * dt
            k = np.exp(-it['drag'] * dt)
            it['vx'] *= k
            it['vy'] *= k
            it['x'] += it['vx'] * dt
            it['y'] += it['vy'] * dt
            it['ph'] += it['spin'] * dt
            it['life'] -= dt
        self.items = [i for i in self.items if i['life'] > 0 and -40 < i['x'] < W + 40 and -60 < i['y'] < H + 40]

    def render(self, dst, t):
        for it in self.items:
            k = it['kind']
            if k == 'card':
                c = np.cos(it['ph'])
                img = CARD_FRONT if c > 0.35 else CARD_BACK if c < -0.35 else CARD_EDGE
            elif k == 'note':
                img = NOTE
                it['x'] += np.sin(t * 5 + it['sway']) * 0.4
            elif k == 'petal':
                img = PETAL if np.cos(it['ph']) > 0 else PETAL[::-1]
                it['x'] += np.sin(t * 4 + it['sway']) * 0.6
            elif k == 'vpetal':
                img = VPETAL if np.cos(it['ph']) > 0 else VPETAL[::-1]
                it['x'] += np.sin(t * 4 + it['sway']) * 0.6
            elif k == 'shard':
                img = SHARD if np.cos(it['ph']) > 0 else SHARD[:, ::-1]
            else:
                img = ROSE
            blit(dst, img, it['x'], it['y'], scale=it['s'], anchor='center', alpha=clamp01(it['life'] / 0.3))

    def clear(self):
        self.items = []


# ---------------------------------------------------------------- 转场
def slash_wipe(dst, prev, lt, dur=0.16, color='#ffffff'):
    if prev is None or lt >= dur:
        return
    pos = -120 + (W + 300) * (lt / dur)
    k = XX + (H - YY) * 0.5
    old = k > pos
    dst[old] = prev[old]
    dst[(k <= pos) & (k > pos - 14)] = hexc(color)


def iris_wipe(dst, cx, cy, r, color=(0, 0, 0)):
    """圆形收缩/展开转场：圆外填色。"""
    put(dst, ~circle_mask(cx, cy, r), color)
