"""第一幕 · 劫炎（执行者 No.I 马克邦）—— 煌魔城的王座大厅。6 小节 @144（10 s）。

原作（闪之轨迹 II / III）：帝都上空升起的深红之城「煌魔城」。结社最强的执行者马克邦永远一副懒得动的样子，
可一旦「燃起来了」（燃えてきたぜ），就会显出真身「劫炎魔人」，从火里抽出魔剑 Angbar。

分镜（小节.拍，本段内）
  1.0–2.1  全景：深红石砌的哥特大厅，尖拱长窗外是血红的月，王座空着；火盆、吊灯、飘着火星。
           马克邦双手插兜、从画面右侧慢悠悠走进来，风衣下摆一摆一摆，每拍一步（6 步，脚步声落在拍上）
  2.0      名牌滑入；2.1.5 停步
  2.2–2.3  打个大哈欠、抬手挠头，一脸无聊
  3.0      台词（对话框）；地上的熔岩裂缝一点点亮起来，火舌沿着裂缝朝他脚下舔过去，大厅渐渐变暖
  4.0      硬切特写：他咧嘴笑了（grin），眼镜反光一闪，背后火往上蹿（「燃起来了」）
  4.2      切回全景：远侧手向前伸出、张开五指，火从裂缝里旋着卷进掌心
  4.3      一把抓住，从火里抽出魔剑 Angbar（拔剑声落在拍上），剑举过头
  4.3.5    魔人化：头发转为暗红黑、发梢喷火，摘下眼镜、金瞳；剑横到胸前蓄力（预备动作）
  5.0      命中：反色冲击帧 2 帧 + 顿帧 3 帧，然后魔剑贴着地面横扫一整圈，剑尖轨迹留下火焰拖尾，
           拖尾脱手化成一道新月形的火墙，横扫过整个大厅；火墙经过的柱子依次起火，DOOM 火焰吞没画面下半部
  5.2      柱子接连爆燃（第二次爆炸）
  6.0      身后一整面火墙轰起（第三次爆炸）；他把剑扛上肩，逆着火光成为剪影，火星往上飘 —— 定格
  6.x      最后约 0.55 s 让给扑克牌转场

镜头：1–2 小节固定机位看他走进来；3–4 小节镜头往左横移（四层视差：远墙 0.35 / 柱廊 0.7 / 地面 1.0 /
前景巨柱 1.35），他被推到画面右侧三分线；4.0 硬切特写；5.0 顿帧 + 震屏、镜头下沉成低角度；6 小节缓慢拉远。
光：火势（heat）从 0.15 升到 1，给整个大厅和他的轮廓染上暖色（边缘光朝向火的一侧）。
"""
from functools import lru_cache

import numpy as np

import fx
from gfx import H, W, XX, YY, blit, clamp01, dilate, dither_mask, ease_out, hexc, smooth
import sfx as S
import ui

import chars.mcburn as MC

LETTERBOX = 0


# ---------------------------------------------------------------- 柱子上的火
class _Flame:
    """对称扩散的小块 DOOM 火（gfx.DoomFire 会整体往左飘，柱子上的火要往正上方烧），不透明地画。"""
    PAL = None

    def __init__(self, w, h, seed):
        self.f = np.zeros((h, w), np.int32)
        self.rng = np.random.default_rng(seed)
        if _Flame.PAL is None:
            stops = [hexc(c) for c in ('#000000', '#4a0808', '#a01c10', '#e8501c', '#ff9a2a', '#ffd870', '#fff4d0')]
            pal = []
            for i in range(37):
                u = i / 36 * (len(stops) - 1)
                j = min(int(u), len(stops) - 2)
                pal.append(stops[j] * (1 - (u - j)) + stops[j + 1] * (u - j))
            _Flame.PAL = np.stack(pal).astype(np.float32)

    def step(self, src, decay, sub=1):
        for _ in range(sub):
            f = self.f
            h, w = f.shape
            f[-1] = src
            r = self.rng.integers(-1, 2, (h - 1, w))
            dec = (self.rng.random((h - 1, w)) < decay * 0.5).astype(np.int32) * \
                (1 + (self.rng.random((h - 1, w)) < 0.2))
            val = np.maximum(f[1:] - dec, 0)
            xs = np.clip(np.arange(w)[None, :] + r, 0, w - 1)
            rows = np.broadcast_to(np.arange(h - 1)[:, None], (h - 1, w))
            new = f.copy()
            new[rows, xs] = val
            new[-1] = src
            self.f = new

    def draw(self, dst, x, y):
        h, w = self.f.shape
        x0, y0 = int(round(x)), int(round(y))
        xs, xe = max(0, x0), min(W, x0 + w)
        ys, ye = max(0, y0), min(H, y0 + h)
        if xe <= xs or ye <= ys:
            return
        f = self.f[ys - y0:ye - y0, xs - x0:xe - x0]
        m = f > 5
        reg = dst[ys:ye, xs:xe]
        reg[m] = _Flame.PAL[f[m]]


SAYS = [('mcburn', 3, 0.0, 'dialog', (4, 1.5))]      # 台词 3.0 出现，4.1.5 开始滑出，切回全景（4.2）时已收好

CUES = [
    (1, 0.0, 'crackle', -16.0), (1, 0.0, 'step', -8.0, 0.8), (1, 1.0, 'step', -7.0, 0.7),
    (1, 2.0, 'step', -6.0, 0.6), (1, 3.0, 'step', -6.0, 0.5), (2, 0.0, 'step', -5.0, 0.4), (2, 1.0, 'step', -5.0, 0.3),
    (2, 2.0, 'mc_yawn', -5.0, 0.3),
    (3, 0.5, 'crackle', -12.0), (3, 2.0, 'mc_lava', -9.0),
    (4, 0.0, 'fire_whoosh', -7.0), (4, 1.0, 'mc_glint', -6.0),
    (4, 2.0, 'fire_roar', -6.0, 0.2), (4, 3.0, 'sword_draw', -1.0, 0.2), (4, 3.0, 'fire_burst', -6.0),
    (4, 3.5, 'mc_ignite', -3.0), (4, 3.5, 'charge', -9.0),
    (5, 0.0, 'big_impact', 0.0), (5, 0.0, 'slash', -2.0, -0.2), (5, 0.0, 'crunch', -3.0),
    (5, 0.25, 'mc_wall', -2.0, -0.5), (5, 2.0, 'fire_burst', -3.0, -0.4), (5, 2.0, 'crumble', -9.0),
    (6, 0.0, 'fire_burst', -1.0), (6, 0.0, 'boom_low', -3.0), (6, 0.5, 'fire_roar', -6.0), (6, 1.0, 'crackle', -8.0),
]
ACCENTS = [(5, 0.0, 'hit'), (5, 2.0, 'drop'), (6, 0.0, 'hit'), (7, 0.0, 'riser')]


# ---------------------------------------------------------------- 自带音效（44.1 kHz）
def _yawn():
    n = int(1.1 * S.SR)
    t = S.tt(n)
    f = 260 - 120 * t / 1.1
    breath = S.bp(S.noise(n), 300, 1800) * np.sin(np.pi * np.clip(t / 1.1, 0, 1)) ** 1.5
    voice = np.sin(2 * np.pi * np.cumsum(f) / S.SR) * 0.25 * np.sin(np.pi * np.clip(t / 1.0, 0, 1)) ** 2
    return breath * 0.8 + S.lp(voice, 900)


def _glint():
    n = int(0.5 * S.SR)
    t = S.tt(n)
    return (np.sin(2 * np.pi * 3520 * t) + 0.6 * np.sin(2 * np.pi * 5280 * t)) * np.exp(-t / 0.09) * 0.5 + \
        S.hp(S.noise(n), 6000) * np.exp(-t / 0.02) * 0.4


def _lava():
    n = int(1.8 * S.SR)
    t = S.tt(n)
    rumble = S.lp(S.noise(n), 120, 4) * 2.5
    bubbles = np.zeros(n)
    rng = np.random.default_rng(3)
    for i in rng.integers(0, n - 3000, 14):
        m = 2500
        bubbles[i:i + m] += np.sin(2 * np.pi * rng.uniform(90, 200) * S.tt(m) * (1 + S.tt(m) * 4)) * np.exp(-S.tt(m) / 0.02)
    return (rumble + bubbles * 0.4) * np.clip(t / 1.2, 0, 1)


def _ignite():
    """魔人化：低沉的「呼——」吸入 + 火焰炸开。"""
    n = int(1.0 * S.SR)
    t = S.tt(n)
    suck = S.sweep_lp(S.noise(n), 200, 5000, 2.0) * np.clip(t / 0.2, 0, 1) * (t < 0.2)
    burst = S.lp(S.noise(n), 2500) * np.exp(-np.maximum(t - 0.2, 0) / 0.25) * (t >= 0.2)
    sub = np.sin(2 * np.pi * np.cumsum(50 + 60 * np.exp(-np.maximum(t - 0.2, 0) / 0.1)) / S.SR) * \
        np.exp(-np.maximum(t - 0.2, 0) / 0.4) * (t >= 0.2)
    return suck * 0.9 + burst * 1.2 + np.tanh(2 * sub) * 0.8


def _wall():
    """火墙横扫大厅：长长的由近到远的咆哮（声像由右往左）。"""
    n = int(1.6 * S.SR)
    t = S.tt(n)
    roar = S.sweep_lp(S.noise(n), 3500, 400, 0.7) * np.exp(-t / 0.7) * 1.6
    x = roar + S.sfx('fire_roar')[:n] * 0.5
    pan = np.clip(t / 0.7, 0, 1)
    return np.stack([x * (0.5 + 0.5 * pan), x * (1 - 0.6 * pan)], 1) if x.ndim == 1 else x


SFX = {'mc_yawn': _yawn, 'mc_glint': _glint, 'mc_lava': _lava, 'mc_ignite': _ignite, 'mc_wall': _wall}


# ---------------------------------------------------------------- 调色板
def _c(h):
    return hexc(h)


STONE = [_c(h) for h in ('#0e0307', '#1c060c', '#2c0a12', '#3e1018', '#56181e', '#74262a')]
GOLD = _c('#b87434')
LAVA = [_c(h) for h in ('#5a0a0a', '#c0220e', '#ff6a1a', '#ffb440', '#fff0b0')]

PAR = {'far': 0.35, 'mid': 0.7, 'floor': 1.0, 'fg': 1.35}
PAN = 70                     # 镜头横移范围（地面层像素）
HORIZON = 150                # 远墙墙脚 / 地面起点
VP = (240, 62)               # 地面透视消失点
MARK_X, FEET_Y = 338, 238    # 他停下的位置（屏幕，镜头归零时）与脚底
PILLARS_MID = [18, 150, 290, 430, 560]      # 柱廊层坐标里柱子的 x（左缘）
PILLAR_W = 30


def _cam(ctx):
    """镜头横移量（地面层像素，0..PAN）：1–2 小节固定在 40 看他走进来，3–4 小节缓缓横移到 0（他移到画面右侧三分线），
    6 小节慢慢拉回 14。"""
    b = 40 * (1 - smooth(0, 1, ctx.prog(3, 0, 4, 2)))
    c = 14 * smooth(0, 1, ctx.prog(5, 1, 6, 3))
    return b + c


def _poly(pts, h, w):
    return fx.polygon_mask(pts)[:h, :w] if (h, w) == (H, W) else _pmask(pts, h, w)


def _pmask(pts, h, w):
    from PIL import Image, ImageDraw
    im = Image.new('L', (w, h), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=1)
    return np.array(im) > 0


def _arch(cx, top, bot, half, h, w):
    """尖拱窗的遮罩：两段圆弧在顶端相交。"""
    yy, xx = np.mgrid[0:h, 0:w]
    r = half * 1.6
    spring = top + np.sqrt(r * r - (r - half) ** 2)       # 拱脚高度
    left = (xx - (cx + half - r)) ** 2 + (yy - spring) ** 2 < r * r
    right = (xx - (cx - half + r)) ** 2 + (yy - spring) ** 2 < r * r
    body = (np.abs(xx - cx) < half) & (yy >= spring) & (yy < bot)
    return ((left & right & (yy < spring)) | body) & (yy >= top)


def _bricks(h, w, bh=7, bw=18, seed=0, base=2):
    """深红石砖：随机明暗的砖块 + 暗色灰缝。返回 (h, w, 3)。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    row = yy // bh
    col = (xx + (row % 2) * (bw // 2)) // bw
    tone = rng.integers(0, 3, (row.max() + 2, col.max() + 2))
    img = np.zeros((h, w, 3), np.float32)
    idx = np.clip(base - 1 + tone[row, col], 0, len(STONE) - 1)
    img[:] = np.stack(STONE)[idx]
    mortar = (yy % bh == 0) | ((xx + (row % 2) * (bw // 2)) % bw == 0)
    img[mortar] = STONE[0]
    # 砖块上缘一道受光
    img[(yy % bh == 1) & ~mortar] = img[(yy % bh == 1) & ~mortar] * 1.25
    return img


# ---------------------------------------------------------------- 静态层（缓存）
@lru_cache(None)
def far_layer():
    """远墙：血月、尖拱长窗（窗棂 + 玫瑰花窗）、王座与台阶、垂挂的红旗。宽 W + PAN*0.35 + 8。"""
    w = W + int(PAN * PAR['far']) + 24
    h = HORIZON + 4
    img = _bricks(h, w, seed=1, base=2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # 越往上越暗
    img *= (0.45 + 0.55 * np.clip(yy / h, 0, 1))[..., None]
    cx = w // 2
    # 窗外的天空 + 血月
    sky = np.zeros((h, w, 3), np.float32)
    t = np.clip(yy / 120, 0, 1)[..., None]
    sky[:] = _c('#16020a') * (1 - t) + _c('#6a0c16') * t
    moon_c, moon_r = (cx + 8, 50), 22
    d = np.hypot(xx - moon_c[0], yy - moon_c[1])
    sky[d < moon_r + 6] = sky[d < moon_r + 6] * 0.6 + _c('#8a1420') * 0.4
    sky[d < moon_r] = _c('#e8302c')
    sky[(d < moon_r) & (np.hypot(xx - moon_c[0] + 6, yy - moon_c[1] + 5) < moon_r * 0.75)] = _c('#ff5a44')
    sky[(d < moon_r) & (((xx * 3 + yy * 7) % 23) < 3) & (d > 6)] = _c('#b81c24')     # 月面暗斑
    # 云带
    cloud = (np.sin(xx * 0.05 + np.sin(yy * 0.3) * 2) > 0.6) & (np.abs(yy - 70) < 3)
    sky[cloud] = _c('#3a0610')
    # 主窗
    win = _arch(cx, 14, 128, 42, h, w)
    img[win] = sky[win]
    frame = dilate(win, 3) & ~win
    img[frame] = STONE[4]
    img[dilate(win, 1) & ~win] = STONE[5]
    # 窗棂：两根竖棂 + 一道横档 + 顶部玫瑰窗圆
    for mx in (cx - 14, cx + 14):
        m = win & (np.abs(xx - mx) < 1.5) & (yy > 40)
        img[m] = STONE[3]
    img[win & (np.abs(yy - 96) < 1.2)] = STONE[3]
    rr = np.hypot(xx - cx, yy - 34)
    img[win & (np.abs(rr - 12) < 1.3)] = STONE[3]
    img[win & (np.abs(rr - 5) < 1.0)] = STONE[3]
    for k in range(8):
        a = k * np.pi / 4
        m = win & (rr < 12) & (rr > 5) & (np.abs((xx - cx) * np.sin(a) - (yy - 34) * np.cos(a)) < 0.8)
        img[m] = STONE[3]
    # 两侧小窗
    for sx in (cx - 150, cx + 150):
        sw = _arch(sx, 38, 118, 16, h, w)
        img[sw] = sky[sw] * 0.8
        img[dilate(sw, 2) & ~sw] = STONE[4]
        img[sw & (np.abs(xx - sx) < 1)] = STONE[3]
    # 垂挂的红旗（金边）
    for bx in (cx - 88, cx + 76):
        m = (xx >= bx) & (xx < bx + 14) & (yy >= 10) & (yy < 104 - 6 * (np.abs(xx - bx - 7) < 3))
        img[m] = _c('#6a0c18')
        img[m & ((xx == bx) | (xx == bx + 13))] = GOLD * 0.7
        img[m & (xx == bx + 1)] = _c('#8a1a24')
        # 旗上的衔尾蛇圈
        r2 = np.hypot(xx - bx - 7, yy - 48)
        img[m & (np.abs(r2 - 4.5) < 0.8)] = GOLD * 0.8
    # 王座 + 三级台阶
    for k, (y0, half) in enumerate([(138, 64), (131, 50), (124, 38)]):
        m = (np.abs(xx - cx) < half) & (yy >= y0) & (yy < y0 + 7)
        img[m] = STONE[3 + (k % 2)]
        img[m & (yy == y0)] = STONE[5]
    back = _pmask([(cx - 13, 124), (cx - 13, 92), (cx - 9, 80), (cx - 5, 88), (cx, 70), (cx + 5, 88), (cx + 9, 80),
                   (cx + 13, 92), (cx + 13, 124)], h, w)
    img[back] = _c('#1a0408')
    img[dilate(back, 1) & ~back] = GOLD * 0.55
    seat = (np.abs(xx - cx) < 17) & (yy >= 108) & (yy < 124)
    img[seat] = _c('#240610')
    img[seat & (yy == 108)] = GOLD * 0.6
    img[(np.abs(xx - cx) < 3) & (yy >= 96) & (yy < 104)] = _c('#c01c28')        # 王座上的红宝石
    # 墙脚
    img[yy >= HORIZON - 2] = STONE[1]
    return img


@lru_cache(None)
def mid_layer():
    """柱廊：粗壮的哥特柱（柱头、柱础、竖向凹槽）+ 顶部拱肋。返回 RGBA。"""
    w = W + int(PAN * PAR['mid']) + 60
    h = HORIZON + 14
    img = np.zeros((h, w, 4), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for px in PILLARS_MID:
        m = (xx >= px) & (xx < px + PILLAR_W) & (yy < h)
        u = (xx - px) / PILLAR_W
        col = np.where(u < 0.18, 4, np.where(u < 0.55, 3, np.where(u < 0.85, 2, 1)))
        for k in range(1, 5):
            img[m & (col == k), :3] = STONE[k]
        img[m & (((xx - px) % 7) == 3) & (u > 0.1) & (u < 0.9), :3] = STONE[1]       # 凹槽
        img[m, 3] = 1
        # 柱头 / 柱础
        for y0, y1 in ((22, 30), (h - 14, h - 6)):
            cap = (xx >= px - 5) & (xx < px + PILLAR_W + 5) & (yy >= y0) & (yy < y1)
            img[cap, :3] = STONE[3]
            img[cap & (yy == y0), :3] = STONE[5]
            img[cap & (yy == y1 - 1), :3] = STONE[1]
            img[cap, 3] = 1
        base = (xx >= px - 7) & (xx < px + PILLAR_W + 7) & (yy >= h - 6)
        img[base, :3] = STONE[2]
        img[base & (yy == h - 6), :3] = STONE[4]
        img[base, 3] = 1
    # 顶部拱肋：相邻两柱之间的尖拱
    for a, b in zip(PILLARS_MID[:-1], PILLARS_MID[1:]):
        cx = (a + b + PILLAR_W) / 2
        half = (b - a - PILLAR_W) / 2 + 4
        outer = ~_arch(cx, -40, 60, half, h, w) & (yy < 24) & (xx > a + PILLAR_W // 2) & (xx < b + PILLAR_W // 2)
        img[outer, :3] = STONE[2]
        img[outer, 3] = 1
        edge = dilate(outer, 1) & ~outer & (yy < 26)
        img[edge, :3] = STONE[4]
        img[edge, 3] = 1
    return img


@lru_cache(None)
def floor_layer():
    """透视石地板 + 熔岩裂缝。返回 (rgb, crack 像素列表 (y, x, 离他多远 0..1), 裂缝辉光遮罩)。"""
    w = W + PAN + 24
    h = H - HORIZON
    rgb = np.zeros((h, w, 3), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ys = yy + HORIZON
    k = np.clip((ys - HORIZON) / (H - HORIZON), 0, 1)
    rgb[:] = (STONE[1] * (1 - k[..., None]) + STONE[2] * k[..., None])
    # 窗户在抛光地面上的倒影（中间一道暗红竖带）
    cx = VP[0] + PAN * 0.5
    refl = (np.abs(xx - cx) < 26 - 14 * k) & dither_mask_local(h, w, 0.45)
    rgb[refl] = rgb[refl] * 0.6 + _c('#5a0e16') * 0.4
    # 地砖缝：横线按透视间距，竖线汇向消失点
    for yl in (152, 157, 164, 173, 185, 200, 219, 243):
        rgb[yl - HORIZON] = STONE[0]
    depth = (ys - VP[1]) / (H - VP[1])
    for xb in range(-400, w + 400, 64):
        xl = cx + (xb - cx) * depth
        rgb[(np.abs(xx - xl) < 0.5)] = STONE[0]
    # 熔岩裂缝：几条从大厅深处 / 两侧爬向他脚下的锯齿线
    rng = np.random.default_rng(5)
    crack = np.zeros((h, w), bool)
    pts_all = []
    target = (MARK_X + 6, FEET_Y - HORIZON - 2)
    starts = [(40, 4), (150, 2), (250, 6), (-10, 60), (70, 110), (200, 118), (560, 30), (470, 118)]
    for sx, sy in starts:
        x, y = float(sx), float(sy)
        path = []
        for _ in range(400):
            dx, dy = target[0] - x, target[1] - y
            d = np.hypot(dx, dy)
            if d < 4:
                break
            ang = np.arctan2(dy, dx) + rng.normal(0, 0.9)
            x += np.cos(ang) * 1.6
            y += np.sin(ang) * 0.8
            path.append((x, y))
        for i, (x, y) in enumerate(path):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < h and 0 <= xi < w:
                crack[yi, xi] = True
                pts_all.append((yi, xi, 1 - i / max(1, len(path) - 1)))
        # 分叉
        for j in range(0, len(path) - 10, 40):
            bx, by = path[j]
            ang = rng.uniform(0, 2 * np.pi)
            for s in range(rng.integers(6, 16)):
                bx += np.cos(ang) * 1.3 + rng.normal(0, 0.4)
                by += np.sin(ang) * 0.6 + rng.normal(0, 0.3)
                xi, yi = int(round(bx)), int(round(by))
                if 0 <= yi < h and 0 <= xi < w:
                    crack[yi, xi] = True
    rgb[crack] = _c('#1a0204')
    glow = dilate(crack, 1) & ~crack
    pts = np.array(pts_all, np.float32)
    return rgb, crack, glow, pts


def dither_mask_local(h, w, level):
    B = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16
    return np.tile(B, (h // 4 + 1, w // 4 + 1))[:h, :w] < level


@lru_cache(None)
def fg_layer():
    """前景：画面左缘一根巨大的柱子（几乎是剪影）+ 右下角一个铁火盆的剪影。RGBA，宽 W + PAN*1.35。"""
    w = W + int(PAN * PAR['fg']) + 8
    img = np.zeros((H, w, 4), np.float32)
    yy, xx = np.mgrid[0:H, 0:w]
    m = (xx < 30 + (yy > 240) * 6)
    img[m, :3] = _c('#0a0205')
    img[m & (xx >= 27 + (yy > 240) * 6), :3] = _c('#3a0c12')       # 朝火的一侧轮廓光
    img[m, 3] = 1
    return img


# ---------------------------------------------------------------- 角色
@lru_cache(None)
def body_img(pose):
    return MC.body(pose)


@lru_cache(None)
def sword_img(angle, reveal, length):
    return MC.sword(angle, reveal, length)


def lit(img, heat, rim_col, silhouette=0.0):
    """火光照明：整体偏暖、朝下 / 朝左（火的方向）的轮廓染上火色；silhouette>0 时压暗成剪影。"""
    out = img.copy()
    a = out[..., 3] > 0
    warm = np.array([1 + 0.25 * heat, 1 - 0.05 * heat, 1 - 0.25 * heat], np.float32)
    out[..., :3] = np.clip(out[..., :3] * warm, 0, 1)
    if silhouette > 0:
        out[a, :3] = out[a, :3] * (1 - silhouette) + _c('#12020a') * silhouette
    if heat > 0.05:
        edge = a & (~np.roll(a, 1, 1) | ~np.roll(a, -1, 0) | ~np.roll(a, -1, 1))
        k = min(1.0, heat) * (0.55 + 0.45 * silhouette)
        out[edge, :3] = out[edge, :3] * (1 - k) + rim_col * k
    return out


# 姿势时间表（本段内拍数 B = lt / beat）
WALK = ['walk0', 'walk1', 'walk2', 'walk3']
WALK_END, WALK_SPEED = 5.5, 33.0          # 走 5.5 拍（6 步，接地落在 1.0 … 2.1），每拍 33px
T_REACH, T_PULL, T_RAISE, T_WIND, T_SW1, T_HIT = 14.0, 15.0, 15.3, 15.5, 15.82, 16.0
HITSTOP = 3 / 30.0
T_FOLLOW = 16.0          # + HITSTOP
T_SHOULDER = 17.6


def pose_at(B, beat):
    """-> (姿势, 屏幕 x（镜头归零时）, 上下起伏, 剑抽出比例)"""
    if B < WALK_END:
        i = int(B * 2) % 4
        x = MARK_X + WALK_SPEED * (WALK_END - B)
        return WALK[i], x, (-2 if i % 2 else 0), 0.0
    if B < 6.0:
        return 'idle', MARK_X, 0, 0.0
    if B < 7.8:
        return 'yawn', MARK_X, (-1 if B < 7.2 else 0), 0.0
    if B < T_REACH:
        return 'idle', MARK_X, 0, 0.0
    if B < T_PULL:
        return 'reach', MARK_X, 0, 0.0
    if B < T_RAISE:
        return 'pull0', MARK_X, 0, float(clamp01((B - T_PULL) / (T_RAISE - T_PULL - 0.05)))
    if B < T_WIND:
        return 'pull1', MARK_X, -1, 1.0
    if B < T_SW1:
        return 'dwind', MARK_X, 1, 1.0
    if B < T_HIT:
        return 'dswing1', MARK_X, 1, 1.0
    if B < T_HIT + HITSTOP / beat:
        return 'dswing2', MARK_X, 1, 1.0
    if B < T_SHOULDER:
        return 'dfollow', MARK_X, 0, 1.0
    return 'dshoulder', MARK_X, 0, 1.0


def sprite_origin(x, bob):
    """全身图（2x）左上角在屏幕上的位置。"""
    return x - 48, FEET_Y - 144 + bob


def blade_screen(pose, x, bob, reveal=1.0):
    """-> (握点, 剑尖, 剑图, 剑图左上角, 在身后?) 屏幕坐标。"""
    b = MC.BLADE.get(pose)
    if b is None:
        return None
    gx, gy, ang, ln, behind = b
    ox, oy = sprite_origin(x, bob)
    grip = np.array([ox + gx * 2 + 1, oy + gy * 2 + 1], np.float32)
    img, (sx, sy) = sword_img(float(ang), round(float(reveal), 1), float(ln))
    L = (3 + (46 * ln - 3) * reveal) * 2
    th = np.radians(ang)
    tip = grip + L * np.array([np.cos(th), np.sin(th)], np.float32)
    return grip, tip, img, (grip[0] - sx * 2 - 1, grip[1] - sy * 2 - 1), behind


# 挥剑的连续轨迹（插值关键帧的握点与角度），火焰拖尾沿着真实剑尖路径
ARC_KEYS = [(T_WIND, 'dwind'), (T_SW1, 'dswing1'), (T_HIT, 'dswing2'), (T_HIT + 0.35, 'dfollow')]


def arc_sample(B):
    """B 时刻剑的 (握点, 剑尖)（屏幕，镜头归零），在关键帧之间插值；角度走最短的顺时针方向。"""
    ks = ARC_KEYS
    if B <= ks[0][0]:
        k0 = k1 = ks[0]
        f = 0.0
    elif B >= ks[-1][0]:
        k0 = k1 = ks[-1]
        f = 0.0
    else:
        for a, b in zip(ks[:-1], ks[1:]):
            if a[0] <= B < b[0]:
                k0, k1 = a, b
                f = (B - a[0]) / (b[0] - a[0])
                break
    b0, b1 = MC.BLADE[k0[1]], MC.BLADE[k1[1]]
    gx = b0[0] + (b1[0] - b0[0]) * f
    gy = b0[1] + (b1[1] - b0[1]) * f
    a0, a1 = b0[2], b1[2]
    if a1 < a0:
        a1 += 360
    ang = a0 + (a1 - a0) * f
    ln = b0[3] + (b1[3] - b0[3]) * f
    ox, oy = sprite_origin(MARK_X, 1)
    grip = np.array([ox + gx * 2 + 1, oy + gy * 2 + 1])
    th = np.radians(ang)
    tip = grip + 92 * ln * np.array([np.cos(th), np.sin(th)])
    return grip, tip


# ---------------------------------------------------------------- 动态效果
def draw_flames(dst, ys, xs, heights, rng):
    """一排竖直的小火舌：底部金黄 -> 橙 -> 红，高度逐像素给。"""
    if not len(xs):
        return
    hmax = int(heights.max()) + 1
    for j in range(hmax):
        sel = heights > j
        yy, xx = ys[sel] - j, xs[sel]
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        fr = j / np.maximum(heights[sel][ok], 1)
        col = np.where(fr[:, None] < 0.3, LAVA[4], np.where(fr[:, None] < 0.6, LAVA[3], np.where(fr[:, None] < 0.85, LAVA[2], LAVA[1])))
        dst[yy[ok], xx[ok]] = col


def brazier(dst, x, y, t, scale=1.0, seed=0):
    """立式火盆：黑铁盆 + 跳动的火（屏幕坐标，x,y 为盆口中心）。"""
    x, y = int(round(x)), int(round(y))
    w = int(7 * scale)
    y0 = max(0, y)
    if -20 < x < W + 20:
        xs = np.arange(max(0, x - w), min(W, x + w + 1))
        if len(xs):
            dst[y0:min(H, y + 3), xs] = _c('#140408')
            dst[y + 3:min(H, y + int(18 * scale)), max(0, x - 1):min(W, x + 2)] = _c('#1c060c')
            rng = np.random.default_rng(int(t * 12) + seed)
            hs = (rng.integers(3, 9, len(xs)) * scale * (1 - np.abs(xs - x) / (w + 1))).astype(int)
            draw_flames(dst, np.full(len(xs), y - 1), xs, hs, rng)


def chandelier(dst, x, y, t, lit_k=1.0):
    x, y = int(round(x)), int(round(y))
    if not (-40 < x < W + 40):
        return
    dst[0:y, max(0, x):min(W, x + 1)] = _c('#1a060a')
    for k in range(-16, 17):
        yy = y + int(3 * (1 - (k / 16) ** 2))
        if 0 <= x + k < W:
            dst[yy:yy + 2, x + k] = _c('#2a0a10')
    rng = np.random.default_rng(int(t * 10))
    for k in (-16, -8, 0, 8, 16):
        cx = x + k
        cy = y + int(3 * (1 - (k / 16) ** 2)) - 1
        if 0 <= cx < W and cy - 4 >= 0:
            dst[cy - 2:cy, cx] = _c('#d8c8a0')
            if rng.random() < 0.9:
                dst[cy - 4:cy - 2, cx] = LAVA[3] * lit_k
                dst[cy - 5, cx] = LAVA[4] * lit_k


def disc(dst, x, y, r, col):
    x0, x1 = int(max(0, x - r - 1)), int(min(W, x + r + 2))
    y0, y1 = int(max(0, y - r - 1)), int(min(H, y + r + 2))
    if x1 <= x0 or y1 <= y0:
        return
    m = (XX[y0:y1, x0:x1] - x) ** 2 + (YY[y0:y1, x0:x1] - y) ** 2 <= r * r
    dst[y0:y1, x0:x1][m] = col


def crescent(dst, cx, base_y, height, thick, alpha_k, t):
    """新月形火墙：竖立在地上、凹面朝右（朝着挥剑的他），外红内白。"""
    ry = height / 2
    rx = ry * 0.55
    cy = base_y - ry
    x0, x1 = int(max(0, cx - rx - 4)), int(min(W, cx + rx * 0.6 + thick))
    y0, y1 = int(max(0, cy - ry - 6)), int(min(H, base_y + 2))
    if x1 <= x0 or y1 <= y0:
        return None
    xx, yy = XX[y0:y1, x0:x1], YY[y0:y1, x0:x1]
    wob = 2.5 * np.sin(yy * 0.35 + t * 30)
    d_out = ((xx - cx + wob) / rx) ** 2 + ((yy - cy) / ry) ** 2
    d_in = ((xx - cx - thick + wob) / (rx * 0.92)) ** 2 + ((yy - cy) / (ry * 0.96)) ** 2
    band = (d_out < 1) & (d_in > 1)
    core = (d_out < 1) & (d_in > 1) & (((xx - cx - thick * 0.45 + wob) / (rx * 0.97)) ** 2 + ((yy - cy) / ry) ** 2 > 1)
    outer = (d_out < 1.25) & ~band & (d_in > 1) & dither_mask_local(y1 - y0, x1 - x0, 0.5)
    reg = dst[y0:y1, x0:x1]
    if alpha_k < 1:
        keep = dither_mask_local(y1 - y0, x1 - x0, alpha_k)
        band &= keep
        core &= keep
        outer &= keep
    reg[outer] = LAVA[1]
    reg[band] = LAVA[2]
    reg[core & (d_out < 0.93)] = LAVA[3]
    reg[core & (d_out < 0.8)] = LAVA[4]
    return (y0, y1, x0, x1)


def warm_grade(dst, heat, t):
    """整个大厅按火势偏暖：压蓝、提红，下半部额外加一层火光。"""
    if heat <= 0:
        return dst
    g = np.array([1 + 0.3 * heat, 1 + 0.06 * heat, 1 - 0.3 * heat], np.float32)
    dst *= g
    floor_glow = np.clip((YY - 110) / 160, 0, 1)[..., None] * np.array([0.14, 0.04, 0.0], np.float32)
    dst += floor_glow * heat * (0.85 + 0.15 * np.sin(t * 17))
    return dst


# ---------------------------------------------------------------- 主渲染
def heat_at(B):
    h = 0.12
    h += 0.25 * smooth(8, 12, B) + 0.2 * smooth(12, 15.5, B)
    h += 0.45 * smooth(15.9, 16.2, B)
    return min(1.0, h)


def render(dst, ctx):
    lt, beat = ctx.lt, ctx.sec.beat
    B = lt / beat
    t = ctx.t
    st = ctx.state
    P, P2 = ctx.P, ctx.P2
    hitstop = T_HIT <= B < T_HIT + HITSTOP / beat
    dt = 0.0 if hitstop else ctx.dt

    # ---- 4.0–4.2：特写（咧嘴笑、镜片反光、火往上蹿）
    if 12.0 <= B < 14.0:
        return closeup(dst, ctx, B)

    cam = _cam(ctx)
    low = 8 * smooth(15.95, 16.05, B) * (1 - smooth(18, 22, B))       # 命中时镜头下沉（低角度）
    heat = heat_at(B)

    # ---- 背景三层
    far = far_layer()
    ox = int(round(cam * PAR['far']))
    dst[:HORIZON + 4] = far[:, ox:ox + W]
    mid = mid_layer()
    ox = int(round(cam * PAR['mid']))
    m = mid[:, ox:ox + W]
    a = m[..., 3:4]
    dst[:m.shape[0]] = dst[:m.shape[0]] * (1 - a) + m[..., :3] * a
    # 火盆与吊灯（柱廊层）
    for bx in (118, 262, 400):
        brazier(dst, bx - cam * PAR['mid'], HORIZON - 12, t, 1.0, bx)
    chandelier(dst, 240 - cam * PAR['mid'], 40, t)
    chandelier(dst, 470 - cam * PAR['mid'], 30, t + 0.3)
    # 柱子被点燃（火墙经过之后）
    ign = st.setdefault('ign', {})
    for px in PILLARS_MID:
        sx = px - cam * PAR['mid'] + PILLAR_W / 2
        if px in ign:
            age = lt - ign[px]
            k = smooth(0, 0.3, age)
            # 柱身被火照亮：底部最亮、往上渐暗
            x0, x1 = max(0, int(sx - 24)), min(W, int(sx + 24))
            if x1 > x0:
                prof = np.clip(1 - np.abs(np.arange(x0, x1) - sx) / 24, 0, 1)[None, :, None]
                vert = np.clip((YY[:HORIZON, :1] - 10) / (HORIZON - 10), 0, 1)[..., None] ** 1.5
                dst[:HORIZON, x0:x1] += (np.array([0.3, 0.08, 0.0], np.float32) * prof * vert * k *
                                         (0.85 + 0.15 * np.sin(t * 23 + px)))
            # 顺着柱子往上爬的火（每根柱子一小块 DOOM 火）
            pf = st.setdefault(f'pil{px}', _Flame(56, 136, px))
            if not hitstop:
                srcp = np.zeros(56, np.int32)
                srcp[12:44] = int(36 * min(1.0, age / 0.2))
                pf.step(srcp, 0.5 + 0.08 * np.sin(px), sub=2)
            pf.draw(dst, sx - 28, HORIZON - 132)
            if not hitstop and ctx.frame % 3 == 0:
                P.emit(1, (sx - 12, sx + 12), (HORIZON - 110, HORIZON - 30), (-8, 8), (-60, -20),
                       (0.4, 0.9), 'fire', size=1)
    # 地面
    fl, crack, glow, pts = floor_layer()
    ox = int(round(cam * PAR['floor']))
    dst[HORIZON:] = fl[:, ox:ox + W]
    cr = crack[:, ox:ox + W]
    gl = glow[:, ox:ox + W]
    pulse = 0.5 + 0.5 * np.sin(t * 5)
    lava_k = 0.35 + 0.65 * smooth(8, 13, B)
    dst[HORIZON:][cr] = LAVA[1] * (0.4 + 0.6 * lava_k) + LAVA[2] * 0.3 * pulse * lava_k
    dst[HORIZON:][gl] += LAVA[0] * 0.8 * lava_k
    # 火舌沿裂缝舔向他（3.0 起火头前进，4.2 起卷进掌心）
    front = smooth(8.5, 14.0, B)
    if front > 0 and B < 16.0:
        sel = pts[:, 2] < front
        sp = pts[sel][::3]
        if len(sp):
            rng = np.random.default_rng(int(t * 12))
            ys = sp[:, 0].astype(int) + HORIZON
            xs = sp[:, 1].astype(int) - ox
            near = 1 - np.clip((front - sp[:, 2]) * 3, 0, 1)            # 火头处最高
            hs = (rng.integers(1, 5, len(sp)) * (0.5 + 1.2 * near) * (0.6 + 0.8 * smooth(12, 15, B))).astype(int)
            draw_flames(dst, ys, xs, hs, rng)
            if not hitstop and ctx.frame % 2 == 0:
                j = rng.integers(0, len(sp), 3)
                P.emit_arrays(xs[j].astype(np.float32), ys[j].astype(np.float32), rng.uniform(-8, 8, 3),
                              rng.uniform(-50, -20, 3), (0.4, 0.9), 'fire', size=1)

    # ---- DOOM 火：命中后从已被火墙扫过的地面烧起，吞没下半个画面
    fire = ctx.fire('mcburn', ['#000000', '#2a0408', '#7a0e10', '#d8301c', '#ff7a1a', '#ffc24a', '#fff0b0'],
                    W, 150, 3)
    wall_x = st.get('wall_x', W + 100)
    src = np.zeros(W, np.int32)
    if B >= T_HIT:
        passed = np.arange(W) > wall_x
        src[passed] = 36
        if B >= 20.0:                           # 6.0 第三次爆炸：整排火墙轰起
            src[:] = 36
        rng = np.random.default_rng(ctx.frame)
        src = np.where(rng.random(W) < 0.12, src // 2, src)
    if not hitstop:
        decay = 0.95 - 0.4 * smooth(19.8, 20.3, B) + 0.3 * smooth(22.5, 23.8, B)
        fire.step(src, decay=decay, sub=2, wind=0)
    fire.add_to(dst, 0, H - 150 + low, gain=0.8)

    # ---- 新月火墙（命中 + 顿帧之后从剑尖处脱出，向左扫过大厅）
    t_launch = ctx.at(5, 0) + HITSTOP
    if lt >= t_launch:
        age = lt - t_launch
        _, tip0 = arc_sample(T_HIT)
        x_c = tip0[0] - cam - 10 - 560 * age
        height = 110 + 170 * smooth(0, 0.45, age)
        st['wall_x'] = x_c
        k = 1 - smooth(0.6, 1.1, age)
        if k > 0:
            crescent(dst, x_c, FEET_Y + 10 + low, height, 30 + 10 * smooth(0, 0.4, age), k, t)
            if not hitstop:
                P2.emit(int(24 * k), (x_c - 12, x_c + 10), (FEET_Y - height, FEET_Y), (40, 160), (-120, 20),
                        (0.2, 0.6), 'fire', size=(1, 3), drag=2.0)
        for px in PILLARS_MID:
            sx = px - cam * PAR['mid'] + PILLAR_W / 2
            if px not in ign and x_c < sx:
                ign[px] = lt
                ctx.shake(3, 0.2)
                P2.emit_radial(60, sx, HORIZON - 30, (40, 200), (0.3, 0.9), 'fire', size=(1, 3), drag=2.5)
    if ctx.crossed(5, 2.0):                     # 5.2 柱子接连爆燃
        for px in PILLARS_MID:
            sx = px - cam * PAR['mid'] + PILLAR_W / 2
            P2.emit_radial(90, sx, HORIZON - 60, (60, 260), (0.4, 1.1), 'fire', size=(1, 3), drag=2.0)
        ctx.shake(5, 0.35)
        ctx.flash('#ff9040', 0.12, 0.25)
    if ctx.crossed(6, 0.0):                     # 6.0 身后火墙轰起
        P2.emit(260, (0, W), (H - 30, H), (-30, 30), (-320, -120), (0.5, 1.4), 'fire', size=(1, 3), drag=1.2)
        ctx.shake(6, 0.5)
        ctx.flash('#ffb060', 0.14, 0.3)

    # ---- 角色
    pose, x, bob, reveal = pose_at(B, beat)
    x = x - cam
    silh = 0.8 * smooth(19.6, 20.4, B)
    rim = LAVA[3] if B < 16 else LAVA[4]
    spr = lit(body_img(pose), heat, rim, silh)
    ox_, oy_ = sprite_origin(x, bob + low)
    sw = blade_screen(pose, x, bob + low, reveal) if pose in MC.BLADE else None
    if sw is not None:
        grip, tip, simg, (sx0, sy0), behind = sw
        simg = lit(simg, heat * 0.4, LAVA[4], 0.0)
    if sw is not None and behind:
        blit(dst, simg, sx0, sy0, scale=2)
    # 魔人化的火焰气（身后）
    if pose.startswith('d') or pose == 'skill':
        if B < 19.6 and ctx.frame % 2 == 0:
            fx.aura(dst, body_img(pose), x, FEET_Y + bob + low, 2, LAVA[1], ctx.rng, 1)
    blit(dst, spr, ox_, oy_, scale=2)
    if sw is not None and not behind:
        blit(dst, simg, sx0, sy0, scale=2)
    # 剑身上的火（从剑上喷出来）
    if sw is not None and not hitstop:
        n = 6 if reveal >= 1 else 3
        f = ctx.rng.random(n)
        pxs = grip[0] + (tip[0] - grip[0]) * f
        pys = grip[1] + (tip[1] - grip[1]) * f
        P.emit_arrays(pxs, pys, ctx.rng.uniform(-10, 10, n), ctx.rng.uniform(-70, -30, n), (0.2, 0.5), 'fire',
                      size=(1, 2))
    # 魔人化：发梢喷火
    if MC.POSES.get(pose, {}).get('demon') and not hitstop:
        for hx, hy in MC.META['hair_fire']:
            P.emit(1, ox_ + hx * 2 + ctx.rng.uniform(-2, 2), oy_ + hy * 2, (-12, 12), (-90, -50), (0.15, 0.45),
                   'fire', size=(1, 2))
    # 4.2–4.3：火从裂缝旋着卷进伸出的掌心
    if T_REACH <= B < T_PULL + 0.2:
        hx, hy = MC.HAND_FAR['reach']
        cx_, cy_ = ox_ + hx * 2 + 1, oy_ + hy * 2 + 1
        k = smooth(T_REACH, T_PULL, B)
        # 三股火流从脚下的裂缝螺旋着卷上来、汇进掌心
        for arm in range(3):
            for i in range(26):
                u = ((i / 26) + t * 1.8 + arm / 3) % 1.0          # 0 = 地面，1 = 掌心
                ang = arm * 2.094 + u * 7.0 + t * 6
                r = 46 * (1 - u) + 2
                fxp = cx_ + r * np.cos(ang)
                fyp = cy_ + (1 - u) * (FEET_Y - cy_) * 0.9 + r * np.sin(ang) * 0.35
                col = LAVA[2] if u < 0.5 else (LAVA[3] if u < 0.85 else LAVA[4])
                disc(dst, fxp, fyp, 1.2 + 1.8 * (1 - u) * (0.4 + 0.6 * k), col)
        P.emit(4, (cx_ - 50, cx_ + 50), (FEET_Y - 10, FEET_Y), 0, 0, (0.25, 0.4), 'fire', size=(1, 2))
        # 粒子往掌心吸
        if len(P.x):
            sel = (P.life > 0) & (np.abs(P.x - cx_) < 90)
            P.vx[sel] += (cx_ - P.x[sel]) * 0.25
            P.vy[sel] += (cy_ - P.y[sel]) * 0.25
        rr = 4 + 5 * k + np.sin(t * 40)
        d = np.hypot(XX - cx_, YY - cy_)
        fx.put(dst, d < rr * 1.3, LAVA[1])
        fx.put(dst, d < rr, LAVA[2])
        fx.put(dst, d < rr * 0.55, LAVA[4])
    # 4.3.5 魔人化的一瞬：人物周围火焰炸开
    if ctx.crossed(4, 3.5):
        P2.emit_radial(160, x, FEET_Y - 80, (80, 260), (0.3, 0.8), 'fire', size=(1, 3), drag=2.0)
        ctx.flash('#ff5a1e', 0.1, 0.25)
        ctx.shake(3, 0.25)
    if ctx.crossed(4, 3.0):
        P2.emit_radial(80, *(MC.HAND_FAR['pull0'][0] * 2 + ox_, MC.HAND_FAR['pull0'][1] * 2 + oy_), (40, 160),
                       (0.2, 0.5), 'gold', size=(1, 2), drag=2.5)

    # ---- 挥剑的火焰拖尾：沿剑尖真实路径（插值关键帧）画一道由细到粗的弧形火痕
    if T_WIND + 0.15 <= B < T_HIT + 1.0:
        Ba = B if B < T_HIT else max(T_HIT, B - HITSTOP / beat)       # 顿帧期间拖尾也停住
        head = min(Ba, T_HIT + 0.35)
        tail = max(T_WIND + 0.1, head - 0.5)
        fade_k = 1 - smooth(T_HIT + 0.25, T_HIT + 1.0, B)
        tips = np.array([arc_sample(sb)[1] for sb in np.linspace(tail, head, 40)]) + np.array([-cam, low])
        seg = np.hypot(*np.diff(tips, axis=0).T)
        pts = [tips[0]]
        for i in range(1, len(tips)):
            m = max(1, int(seg[i - 1] / 2))
            pts += [tips[i - 1] + (tips[i] - tips[i - 1]) * (j + 1) / m for j in range(m)]
        pts = np.array(pts)
        n = len(pts)
        for layer, (col, kr) in enumerate(((LAVA[1], 1.0), (LAVA[2], 0.7), (LAVA[3], 0.45), (LAVA[4], 0.22))):
            for i in range(0, n, 1):
                u = i / max(1, n - 1)
                r = (1.5 + 11 * u ** 1.3) * kr * (0.4 + 0.6 * fade_k)
                if r < 0.6 or (layer == 0 and fade_k < 1 and ((i * 7 + ctx.frame) % 5) >= 5 * fade_k):
                    continue
                disc(dst, pts[i][0], pts[i][1], r, col)
    # ---- 粒子（火星、火花）
    if not hitstop:
        P.emit(int(1 + 2 * heat) if ctx.frame % 2 else 0, (0, W), H + 2, (-8, 8), (-60, -20), (1.5, 3.5), 'fire', bright=0.8, wob=0.4, size=1)
    P.step(dt)
    P2.step(dt)
    P.render(dst)
    P2.render(dst)

    # ---- 前景巨柱
    fg = fg_layer()
    ox = int(round(cam * PAR['fg']))
    f = fg[:, ox:ox + W]
    a = f[..., 3:4]
    dst[:] = dst * (1 - a) + f[..., :3] * a

    dst = warm_grade(dst, heat, t)

    # ---- 命中：反色冲击帧（2 帧），纯黑底 + 白色人物 / 剑 / 拖尾
    if T_HIT <= B < T_HIT + 2 / 30.0 / beat:
        lum = dst.mean(-1, keepdims=True)
        dst[:] = np.where(lum > 0.55, 0.0, 1.0) * np.array([1.0, 0.95, 0.85], np.float32)
        silo = body_img('dswing2')[..., 3] > 0
        blit(dst, np.dstack([np.zeros(silo.shape + (3,), np.float32), silo.astype(np.float32)]), ox_, oy_, scale=2)
    if ctx.crossed(5, 0.0):
        ctx.shake(9, 0.7)
        P2.emit_radial(200, *arc_sample(T_HIT)[1] + np.array([-cam, 0]), (100, 420), (0.3, 1.0), 'fire',
                       size=(1, 3), drag=2.2)

    # ---- 名牌（第 2 小节）
    if 4.0 <= B < 8.0:
        ui.name_tag(dst, ctx.C['mcburn'], lt - ctx.at(2, 0))
    return dst


def closeup(dst, ctx, B):
    """4.0–4.2 特写：暗红底 + 往上蹿的火，头像 2x；1 拍后镜片反光、嘴角咧开。"""
    t = ctx.t
    dst[:] = _c('#1a0306')
    dst[:] += (YY / H)[..., None] * np.array([0.35, 0.05, 0.02], np.float32)
    fire = ctx.fire('mc_close', ['#000000', '#2a0408', '#8a1410', '#e0401c', '#ff9a2a', '#ffe08a', '#ffffff'],
                    W, 140, 9)
    k = smooth(12.0, 13.0, B)
    fire.step(np.full(W, int(20 + 16 * k), np.int32), decay=1.3 - 0.4 * k, sub=2)
    fire.add_to(dst, 0, H - 140, gain=0.9)
    dst = warm_grade(dst, 0.4, t)
    fx.speed_lines(dst, t, np.array([0.5, 0.12, 0.05], np.float32), direction=1, alpha=0.35)
    expr = 'smirk' if B < 12.9 else 'grin'
    p = MC.portrait(expr)
    p = lit(p, 0.3, LAVA[3])
    push = int(4 * smooth(12, 14, B))
    blit(dst, p, 160 - push, 22 - push, scale=2)
    ctx.P.emit(3, (80, 400), H + 2, (-10, 10), (-120, -60), (0.8, 1.6), 'fire', size=(1, 2))
    ctx.P.step(ctx.dt)
    ctx.P.render(dst)
    # 眼镜反光：4.1 拍一道十字星
    g = ctx.since(4, 1.0)
    if 0 <= g < 0.3:
        gx, gy = 160 - push + 47 * 2, 22 - push + 33 * 2
        L = int(14 * np.sin(np.pi * g / 0.3))
        dst[gy, max(0, gx - L):gx + L + 1] = 1.0
        dst[max(0, gy - L):gy + L + 1, gx] = 1.0
        dst[gy - 1:gy + 2, gx - 1:gx + 2] = 1.0
    return dst
