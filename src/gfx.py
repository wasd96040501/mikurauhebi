"""低分辨率画布（320x180）上的绘图工具：贴图、像素字、粒子、DOOM 火焰、抖动量化。"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 480, 270
ROOT = Path(__file__).resolve().parent.parent
FONT = str(ROOT / 'assets/fonts/fusion-pixel-12px-proportional-zh_hans.ttf')
FONT_JA = str(ROOT / 'assets/fonts/fusion-pixel-12px-proportional-ja.ttf')


def hexc(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp01(x):
    return max(0.0, min(1.0, x))


def smooth(e0, e1, x):
    t = clamp01((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_back(t):
    t = clamp01(t)
    c = 1.9
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16
BAYER = np.tile(BAYER4, (H // 4 + 1, W // 4 + 1))[:H, :W]


def canvas(color=(0, 0, 0)):
    c = np.empty((H, W, 3), np.float32)
    c[:] = color
    return c


def vgrad(top, bottom, y0=0, y1=H):
    t = np.clip((YY - y0) / max(1, (y1 - y0)), 0, 1)[..., None]
    return (hexc(top) * (1 - t) + hexc(bottom) * t).astype(np.float32)


def scale_img(img, s):
    if s == 1:
        return img
    return np.repeat(np.repeat(img, s, 0), s, 1)


def blit(dst, img, x, y, scale=1, flip=False, alpha=1.0, tint=None, tint_amt=0.0, anchor='topleft'):
    """把 RGBA 图贴到画布上。anchor='bottom' 时 (x, y) 是脚底中心点。"""
    if flip:
        img = img[:, ::-1]
    img = scale_img(img, scale)
    h, w = img.shape[:2]
    if anchor == 'bottom':
        x, y = x - w / 2, y - h
    elif anchor == 'center':
        x, y = x - w / 2, y - h / 2
    x0, y0 = int(round(x)), int(round(y))
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(dst.shape[1], x0 + w), min(dst.shape[0], y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    part = img[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
    rgb, a = part[..., :3], part[..., 3:4] * alpha
    if tint is not None:
        rgb = rgb * (1 - tint_amt) + np.asarray(tint, np.float32) * tint_amt
    reg = dst[dy0:dy1, dx0:dx1]
    reg[:] = reg * (1 - a) + rgb * a


@lru_cache(None)
def bayer(h, w):
    return np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]


def dither_mask(level, shape=(H, W)):
    """0..1 的覆盖率 -> Bayer 抖动布尔遮罩，做像素风的淡入淡出。"""
    return bayer(*shape) < level


def fade(dst, color, level):
    if level <= 0:
        return
    m = dither_mask(level, dst.shape[:2]) if level < 1 else np.ones(dst.shape[:2], bool)
    dst[m] = color


def rect(dst, x0, y0, x1, y1, color, alpha=1.0):
    x0, y0, x1, y1 = [int(round(v)) for v in (x0, y0, x1, y1)]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(dst.shape[1], x1), min(dst.shape[0], y1)
    if x1 > x0 and y1 > y0:
        dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - alpha) + np.asarray(color, np.float32) * alpha


def add_glow(dst, mask, color, strength=1.0):
    dst += mask[..., None].astype(np.float32) * np.asarray(color, np.float32) * strength


# ------------------------------------------------------------------ 像素字
@lru_cache(None)
def text_mask(s, font='zh'):
    f = ImageFont.truetype(FONT_JA if font == 'ja' else FONT, 12)
    w = int(f.getlength(s)) + 2
    im = Image.new('L', (w, 16), 0)
    ImageDraw.Draw(im).text((1, 0), s, font=f, fill=255)
    m = np.array(im) > 110
    rows = np.where(m.any(1))[0]
    return m[1:15] if len(rows) else m


def dilate(m, r=1):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx or dy:
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def text(dst, s, x, y, color, scale=1, outline=(0.02, 0.0, 0.05), anchor='center',
         reveal=None, spacing=0, glow=None, jitter=None, rng=None, font='zh'):
    """画像素字。reveal 为显示的字数（打字机效果）。"""
    if reveal is not None:
        s = s[:max(0, int(reveal))]
        if not s:
            return
    if spacing:
        parts = [text_mask(ch, font) for ch in s]
        hmax = max(p.shape[0] for p in parts)
        m = np.zeros((hmax, sum(p.shape[1] for p in parts) + spacing * (len(parts) - 1)), bool)
        cx = 0
        for p in parts:
            m[:p.shape[0], cx:cx + p.shape[1]] = p
            cx += p.shape[1] + spacing
    else:
        m = text_mask(s, font)
    m = scale_img(m, scale)
    pad = 2
    m = np.pad(m, pad)
    h, w = m.shape
    if anchor == 'center':
        x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
    elif anchor == 'left':
        x0, y0 = int(round(x)) - pad, int(round(y - h / 2))
    else:
        x0, y0 = int(round(x - w)) + pad, int(round(y - h / 2))
    if jitter is not None and rng is not None:
        # 故障风：逐行水平错位
        rows = np.arange(h)
        shift = (rng.random(h) < jitter) * rng.integers(-6, 7, h)
        m = np.stack([np.roll(m[r], shift[r]) for r in rows])
    layers = []
    if glow is not None:
        layers.append((dilate(m, 2) & ~dilate(m, 1), glow, 0.5))
    if outline is not None:
        layers.append((dilate(m, 1) & ~m, outline, 1.0))
    layers.append((m, color, 1.0))
    for mm, col, a in layers:
        ys, xs = np.nonzero(mm)
        ys, xs = ys + y0, xs + x0
        ok = (ys >= 0) & (ys < dst.shape[0]) & (xs >= 0) & (xs < dst.shape[1])
        dst[ys[ok], xs[ok]] = dst[ys[ok], xs[ok]] * (1 - a) + np.asarray(col, np.float32) * a
    return w


# ------------------------------------------------------------------ 色带
def ramp(*hexes):
    return np.stack([hexc(h) for h in hexes])


RAMPS = {
    'fire': ramp('#ffffff', '#fff3a0', '#ffc93c', '#ff7a1a', '#d8301c', '#7a1020', '#2a0810'),
    'green': ramp('#ffffff', '#d8ffe8', '#7dffb0', '#2ee88a', '#12a870', '#0a5a48', '#062a24'),
    'gold': ramp('#ffffff', '#ffe9a0', '#ffc23a', '#e07a10', '#8a2a10', '#3a0c10', '#120408'),
    'blue': ramp('#ffffff', '#bff0ff', '#6cc8ff', '#3a7aff', '#2a3ad0', '#1a1470', '#0a0630'),
    'rose': ramp('#ffffff', '#ffd0e0', '#ff6a9a', '#e0205a', '#9a1040', '#4a0620', '#1a0210'),
    'purple': ramp('#ffffff', '#f0d0ff', '#c080ff', '#8a40e0', '#5a1ab0', '#2a0a60', '#10041e'),
    'violet': ramp('#ffffff', '#f0dcff', '#c89af0', '#9b5bd6', '#6a2aa8', '#3a1060', '#140428'),
    'mint': ramp('#ffffff', '#e0fff8', '#8fe3d4', '#4ab8a8', '#2a7a70', '#10403a', '#041a18'),
    'white': ramp('#ffffff', '#ffffff', '#e8e0ff', '#b0a0e0', '#6050a0', '#302050', '#100818'),
}
RAMP_IDS = {k: i for i, k in enumerate(RAMPS)}
RAMP_ARR = np.stack(list(RAMPS.values()))  # (n, 7, 3)


class Particles:
    """向量化粒子：位置/速度/寿命/色带/尺寸/重力/阻力。"""

    FIELDS = ('x', 'y', 'vx', 'vy', 'life', 'maxlife', 'ramp', 'size', 'grav', 'drag', 'bright', 'wob')

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        for f in self.FIELDS:
            setattr(self, f, np.zeros(0, np.float32))

    def emit(self, n, x, y, vx, vy, life, ramp='fire', size=1, grav=0.0, drag=0.0, bright=1.0, wob=0.0):
        r = self.rng

        def v(val):
            if isinstance(val, tuple):
                return r.uniform(val[0], val[1], n).astype(np.float32)
            return np.full(n, val, np.float32)

        vals = dict(x=v(x), y=v(y), vx=v(vx), vy=v(vy), life=v(life), ramp=np.full(n, RAMP_IDS[ramp], np.float32),
                    size=v(size), grav=v(grav), drag=v(drag), bright=v(bright), wob=v(wob))
        vals['maxlife'] = vals['life'].copy()
        for f in self.FIELDS:
            setattr(self, f, np.concatenate([getattr(self, f), vals[f]]))

    def emit_radial(self, n, x, y, speed, life, ramp, **kw):
        ang = self.rng.uniform(0, 2 * np.pi, n)
        sp = self.rng.uniform(speed[0], speed[1], n) if isinstance(speed, tuple) else np.full(n, speed)
        self.emit_arrays(x + np.zeros(n), y + np.zeros(n), np.cos(ang) * sp, np.sin(ang) * sp, life, ramp, **kw)

    def emit_arrays(self, x, y, vx, vy, life, ramp, size=1, grav=0.0, drag=0.0, bright=1.0, wob=0.0):
        n = len(x)
        r = self.rng

        def v(val):
            if isinstance(val, tuple):
                return r.uniform(val[0], val[1], n).astype(np.float32)
            if np.ndim(val):
                return np.asarray(val, np.float32)
            return np.full(n, val, np.float32)

        vals = dict(x=v(x), y=v(y), vx=v(vx), vy=v(vy), life=v(life), ramp=np.full(n, RAMP_IDS[ramp], np.float32),
                    size=v(size), grav=v(grav), drag=v(drag), bright=v(bright), wob=v(wob))
        vals['maxlife'] = vals['life'].copy()
        for f in self.FIELDS:
            setattr(self, f, np.concatenate([getattr(self, f), vals[f]]))

    def step(self, dt):
        if not len(self.x):
            return
        self.vy += self.grav * dt
        k = np.exp(-self.drag * dt)
        self.vx *= k
        self.vy *= k
        if self.wob.any():
            self.vx += self.wob * self.rng.normal(0, 1, len(self.x)).astype(np.float32) * dt * 60
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt
        keep = (self.life > 0) & (self.x > -40) & (self.x < W + 40) & (self.y > -60) & (self.y < H + 60)
        for f in self.FIELDS:
            setattr(self, f, getattr(self, f)[keep])

    def clear(self):
        for f in self.FIELDS:
            setattr(self, f, np.zeros(0, np.float32))

    def render(self, dst, mode='add'):
        if not len(self.x):
            return
        frac = 1 - self.life / self.maxlife  # 0 新生 -> 1 消亡
        idx = np.clip((frac * 6.99).astype(int), 0, 6)
        col = RAMP_ARR[self.ramp.astype(int), idx] * self.bright[:, None]
        buf = np.zeros_like(dst)
        xi = np.floor(self.x).astype(int)
        yi = np.floor(self.y).astype(int)
        for s in (1, 2, 3):
            sel = np.round(self.size) == s
            if not sel.any():
                continue
            for dy in range(s):
                for dx in range(s):
                    xx, yy = xi[sel] + dx - s // 2, yi[sel] + dy - s // 2
                    ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
                    np.add.at(buf, (yy[ok], xx[ok]), col[sel][ok])
        if mode == 'add':
            dst += buf
        else:
            m = buf.max(-1) > 0
            dst[m] = np.minimum(buf[m], 1.0)


class DoomFire:
    """经典 DOOM 像素火焰：底部点火，一行行向上传播并随机衰减。"""

    def __init__(self, w, h, colors, seed=0):
        self.w, self.h = w, h
        self.f = np.zeros((h, w), np.int32)
        self.rng = np.random.default_rng(seed)
        stops = [hexc(c) for c in colors]
        pal = []
        for i in range(37):
            t = i / 36 * (len(stops) - 1)
            j = min(int(t), len(stops) - 2)
            pal.append(stops[j] * (1 - (t - j)) + stops[j + 1] * (t - j))
        self.pal = np.stack(pal).astype(np.float32)

    def step(self, source, decay=1.0, wind=0, sub=1):
        for _ in range(sub):
            self._step(source, decay, wind)

    def _step(self, source, decay, wind):
        h, w = self.h, self.w
        f = self.f
        f[-1] = source
        r = self.rng.integers(0, 4, (h - 1, w))
        dec = (self.rng.random((h - 1, w)) < decay * 0.5).astype(np.int32) * (1 + (r == 3))
        val = np.maximum(f[1:] - dec, 0)
        xs = np.clip(np.arange(w)[None, :] - r + 1 + wind, 0, w - 1)
        rows = np.broadcast_to(np.arange(h - 1)[:, None], (h - 1, w))
        new = f.copy()
        new[rows, xs] = val
        new[-1] = source
        self.f = new

    def rgba(self):
        img = np.zeros((self.h, self.w, 4), np.float32)
        img[..., :3] = self.pal[self.f]
        img[..., 3] = np.clip(self.f / 6.0, 0, 1)
        return img

    def add_to(self, dst, x=0, y=0, gain=1.0):
        img = self.pal[self.f] * gain
        h, w = img.shape[:2]
        y0, x0 = int(y), int(x)
        ys, ye = max(0, y0), min(H, y0 + h)
        xs, xe = max(0, x0), min(W, x0 + w)
        if ye > ys and xe > xs:
            dst[ys:ye, xs:xe] += img[ys - y0:ye - y0, xs - x0:xe - x0]


# ------------------------------------------------------------------ 后处理
LEVELS = 8


def finalize(frame, scale=4, scan=True):
    """抖动量化到有限色阶，再最近邻放大到 1080p，加一点扫描线。"""
    f = np.clip(frame, 0, 1)
    q = np.floor(f * (LEVELS - 1) + bayer(*f.shape[:2])[..., None] * 0.999) / (LEVELS - 1)
    q = np.clip(q, 0, 1)
    big = np.repeat(np.repeat(q, scale, 0), scale, 1)
    if scan:
        big[scale - 1::scale] *= 0.86
    return (big * 255 + 0.5).astype(np.uint8)
