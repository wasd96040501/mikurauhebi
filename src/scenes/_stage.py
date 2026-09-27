"""肯帕雷拉的剧场（序章 / 标题 / 谢幕共用）：空荡的旧剧场、深红帷幕、脚灯、追光、前景座椅剪影。

世界坐标：x 0..480，y -90..270（WORLD_H = 360，数组下标 = y + TOP）。镜头 cam_y = 0 时正好是舞台全景；
cam_y < 0 往上看到穹顶和吊灯。zoom=2 时取 240x135 的区域整数放大（像素密度保持一致）。
所有静态层都按"亮版"绘制，再乘一个暗版；每帧用光照遮罩（追光、脚灯）在两版之间插值。
"""
from functools import lru_cache

import numpy as np

import fx
from gfx import BAYER, H, W, blit, clamp01, dilate, hexc
import ui

TOP = 90
WORLD_H = H + TOP
YW, XW = np.mgrid[-TOP:H, 0:W].astype(np.float32)          # 世界坐标网格
BAYER_W = np.tile(BAYER[:4, :4], (WORLD_H // 4 + 1, W // 4 + 1))[:WORLD_H, :W]

# 舞台几何
OPEN_X0, OPEN_X1 = 64, 416          # 台口
VAL_Y0, VAL_Y1 = 48, 66             # 顶部垂幔
FLOOR_Y = 204                       # 台面（帷幕下沿 / 角色脚底）
LIP_Y = 220                         # 台唇（脚灯）
CUR_Y0 = VAL_Y1                     # 帷幕顶
CUR_H = FLOOR_Y - CUR_Y0
EMBLEM_XY = (240, 128)


def C(h):
    return hexc(h)


def _wy(y):
    return int(y) + TOP


# ================================================================ 静态层
@lru_cache(None)
def layers():
    lit = np.zeros((WORLD_H, W, 3), np.float32)
    X, Y = XW, YW
    # --- 墙：暗酒红，竖向墙板 + 金色腰线
    lit[:] = C('#3a1420')
    panel = ((X.astype(int) % 60) < 3) & (Y > -40)
    lit[panel] = C('#2a0c16')
    lit[(np.abs(Y - 150) < 2)] = C('#7a5424')
    lit[(np.abs(Y - 152) < 1)] = C('#b88a3a')
    # --- 穹顶：深色，带放射彩绘肋条，吊灯剪影
    dome = Y < 16
    lit[dome] = C('#24101c')
    ang = np.arctan2(Y + 140, X - 240)
    rib = dome & (np.abs(np.sin(ang * 14)) < 0.08)
    lit[rib] = C('#4a2a2a')
    ring = np.abs(np.hypot(X - 240, (Y + 140) * 1.6) - 170) < 1.5
    lit[ring & dome] = C('#8a6428')
    _chandelier(lit)
    # --- 包厢（左右两层）
    for x0, x1 in ((0, 34), (446, 480)):
        for y0 in (34, 96):
            box = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y0 + 44)
            lit[box] = C('#12060a')
            drape = box & (Y < y0 + 10 + 4 * np.sin((X - x0) * 0.4))
            lit[drape] = C('#7a1024')
            rail = (X >= x0) & (X < x1) & (Y >= y0 + 30) & (Y < y0 + 44)
            lit[rail] = C('#5a3a18')
            lit[rail & (((X.astype(int)) % 6) < 2)] = C('#2a1a0c')
            lit[(X >= x0) & (X < x1) & (np.abs(Y - (y0 + 30)) < 1)] = C('#d8a848')
    # --- 台框：金色立柱 + 横楣 + 中央涡卷饰
    for x0, x1 in ((36, OPEN_X0), (OPEN_X1, 444)):
        col = (X >= x0) & (X < x1) & (Y >= 18) & (Y < LIP_Y)
        lit[col] = C('#b08030')
        flute = col & ((((X - x0).astype(int)) % 5) == 0)
        lit[flute] = C('#6a4818')
        lit[col & ((X - x0) < 2)] = C('#f0d080')
        lit[col & ((x1 - X) <= 2)] = C('#5a3a14')
        cap = (X >= x0 - 4) & (X < x1 + 4) & (Y >= 18) & (Y < 30)
        lit[cap] = C('#d8a848')
        lit[cap & (Y > 26)] = C('#7a5420')
        base = (X >= x0 - 3) & (X < x1 + 3) & (Y >= 196) & (Y < LIP_Y)
        lit[base] = C('#8a6020')
        lit[base & (Y < 198)] = C('#e0b860')
    lintel = (X >= 36) & (X < 444) & (Y >= 18) & (Y < VAL_Y0)
    lit[lintel] = C('#9a7028')
    lit[lintel & (Y < 21)] = C('#f0d080')
    lit[lintel & (Y > 44)] = C('#5a3a14')
    lit[lintel & (np.abs(Y - 33) < 6) & (((X.astype(int) + 3) % 12) < 6)] = C('#c89840')
    cart = ((X - 240) / 34) ** 2 + ((Y - 30) / 16) ** 2 < 1
    lit[cart] = C('#e0b050')
    lit[cart & (((X - 240) / 30) ** 2 + ((Y - 30) / 12) ** 2 < 1)] = C('#3a1420')
    lit[(((X - 240) / 34) ** 2 + ((Y - 30) / 16) ** 2 > 0.85) & cart] = C('#f8e098')
    # 台口内：深处（帷幕打开后另画舞台深处）
    opening = (X >= OPEN_X0) & (X < OPEN_X1) & (Y >= VAL_Y0) & (Y < FLOOR_Y)
    lit[opening] = C('#0a0408')
    # --- 台面：木地板，透视木纹
    floor = (Y >= FLOOR_Y) & (Y < LIP_Y) & (X >= 20 + (LIP_Y - Y) * 1.0) & (X < 460 - (LIP_Y - Y) * 1.0)
    lit[floor] = C('#6a3e22')
    k = (X - 240) / (1 + (Y - FLOOR_Y) * 0.06)
    plank = floor & ((np.floor(k / 14).astype(int) % 2) == 0)
    lit[plank] = C('#5a3220')
    lit[floor & (np.abs(np.mod(k, 14)) < 0.7)] = C('#3a1e10')
    lit[floor & (Y < FLOOR_Y + 1)] = C('#2a140a')
    lip = (Y >= LIP_Y) & (Y < LIP_Y + 8) & (X >= 20) & (X < 460)
    lit[lip] = C('#2a160c')
    lit[lip & (Y < LIP_Y + 1)] = C('#8a6030')
    # 乐池 / 观众席深处
    lit[(Y >= LIP_Y + 8)] = C('#080406')
    # --- 垂幔（帷幕顶的短幔）
    val = (X >= OPEN_X0) & (X < OPEN_X1) & (Y >= VAL_Y0) & (Y < VAL_Y1 + 5 * np.abs(np.sin((X - OPEN_X0) * np.pi / 44)))
    lit[val] = C('#9a1830')
    lit[val & (np.sin((X - OPEN_X0) * np.pi / 11) > 0.6)] = C('#c02a40')
    lit[val & (np.sin((X - OPEN_X0) * np.pi / 11) < -0.5)] = C('#5a0818')
    vb = VAL_Y1 + 5 * np.abs(np.sin((X - OPEN_X0) * np.pi / 44))
    fringe = (X >= OPEN_X0) & (X < OPEN_X1) & (Y >= vb - 2) & (Y < vb + 1)
    lit[fringe] = C('#e0b050')
    lit[fringe & (X.astype(int) % 3 == 0) & (Y > vb - 1)] = C('#7a5420')
    dark = lit * DARK_K
    # --- 前景：座椅椅背剪影（单独一层，视差更快）
    fg = np.zeros((WORLD_H + 40, W, 4), np.float32)
    YF, XF = np.mgrid[-TOP:H + 40, 0:W].astype(np.float32)
    for row, (y0, pitch, hh, off) in enumerate(((244, 26, 14, 0), (258, 32, 18, 11), (276, 40, 24, 5))):
        u = np.mod(XF + off, pitch)
        top = y0 - hh * np.sqrt(np.clip(1 - ((u - pitch / 2) / (pitch / 2 - 2)) ** 2, 0, 1))
        seat = (YF >= top) & (u > 1) & (u < pitch - 1)
        seat |= (YF >= y0 + 2)
        fg[seat, :3] = C('#0e0608') * (1 + 0.3 * (2 - row))
        fg[seat, 3] = 1
        edge = seat & ~np.roll(seat, 1, 0)
        fg[edge, :3] = C('#3a1a1e')
    return dict(lit=lit, dark=dark, fg=fg)


def _chandelier(lit):
    X, Y = XW, YW
    stem = (np.abs(X - 240) < 1) & (Y < -40)
    lit[stem] = C('#5a4020')
    body = ((X - 240) / 40) ** 2 + ((Y + 30) / 10) ** 2 < 1
    body &= Y > -34
    lit[body] = C('#6a4a20')
    for i in range(-4, 5):
        cx = 240 + i * 9
        cy = -28 + abs(i) * 1.5
        arm = (np.abs(X - cx) < 1) & (Y > cy - 6) & (Y < cy)
        lit[arm] = C('#b08030')
        lit[(np.abs(X - cx) < 1) & (np.abs(Y - (cy - 7)) < 1)] = C('#ffe0a0')
    drops = (np.abs(Y + 18 - 3 * np.cos((X - 240) * 0.3)) < 1) & (np.abs(X - 240) < 34) & ((X.astype(int) % 4) == 0)
    lit[drops] = C('#d8c8a0')


# ================================================================ 帷幕
@lru_cache(None)
def _curtain_half():
    """闭合时的半幅帷幕（左半），RGBA (CUR_H, 176)。右半镜像。"""
    w = (OPEN_X1 - OPEN_X0) // 2
    img = np.zeros((CUR_H, w, 4), np.float32)
    yy, xx = np.mgrid[0:CUR_H, 0:w].astype(np.float32)
    ph = np.sin(xx * 2 * np.pi / 22 + 0.6 * np.sin(yy * 0.03 + xx * 0.05))
    cols = [C('#300410'), C('#5a0818'), C('#8e1428'), C('#b01e34'), C('#d23a4c')]
    idx = np.clip(((ph + 1) / 2 * 4.99).astype(int), 0, 4)
    img[..., :3] = np.stack(cols)[idx]
    img[..., :3] *= (0.75 + 0.25 * np.clip(yy / 30, 0, 1))[..., None]
    img[..., 3] = 1
    fr = yy >= CUR_H - 4
    img[fr, :3] = C('#d8a848')
    img[fr & (yy >= CUR_H - 2) & (xx.astype(int) % 3 == 1), :3] = C('#7a5420')
    img[fr & (yy == CUR_H - 4), :3] = C('#f0d080')
    img[:, -1, :3] = C('#200208')
    return img


@lru_cache(512)
def curtain(open_q=0, drop_q=0):
    """整幅帷幕 RGBA（世界坐标 x OPEN_X0.., y CUR_Y0..）。
    open_q: 0..50 打开程度（两半向两侧收拢、下摆斜斜提起）；drop_q: 0..CUR_H 从上往下落下时露出的高度（0 = 落到底）。"""
    o = open_q / 50
    half = _curtain_half()
    h, w = half.shape[:2]
    out = np.zeros((h, w * 2, 4), np.float32)
    ww = max(2, int(round(w * (1 - 0.86 * o))))
    xs = (np.arange(ww) * w / ww).astype(int)
    squashed = half[:, xs].copy()
    if o > 0:   # 收拢后下摆向外侧斜提（系带）
        u = np.arange(ww) / max(1, ww - 1)
        lift = (o * 70 * u ** 1.6).astype(int)
        rows = np.arange(h)[:, None]
        squashed[rows >= (h - lift)[None, :]] = 0
        # 越收拢越暗（褶子挤在一起）
        squashed[..., :3] *= 1 - 0.25 * o
    out[:, :ww] = squashed
    out[:, 2 * w - ww:] = squashed[:, ::-1]
    if drop_q:
        out[:h - drop_q] = out[drop_q:]
        out[h - drop_q:] = 0
        out = np.roll(out, 0, 0)
    return out


DARK_K = np.array([0.2, 0.17, 0.26], np.float32)


def draw_curtain(lit, dark, open_=0.0, drop=0.0):
    """把帷幕画进亮 / 暗两版。open_: 0 闭合 .. 1 收到两侧；drop: 0 落到底 .. 1 完全升起（从上方落下时用）。"""
    oq = int(round(clamp01(open_) * 50))
    img = curtain(oq, 0)
    dy = int(round(-drop * (CUR_H + 24)))
    if dy <= -(CUR_H + 24):
        return
    y = CUR_Y0 + TOP + dy
    blit(lit, img, OPEN_X0, y)
    d = img.copy()
    d[..., :3] *= DARK_K
    blit(dark, d, OPEN_X0, y)


# ================================================================ 光
def spot_mask(cx, cy=FLOOR_Y, rx=46, src=(240, -150), wash=1.0):
    """追光：地面光斑 + 帷幕上的椭圆光晕，返回世界坐标 (WORLD_H, W) 0..1。"""
    X, Y = XW, YW
    pool = np.clip(1 - np.hypot((X - cx) / rx, (Y - cy - 3) / (rx * 0.22)), 0, 1)
    wall = np.clip(1 - np.hypot((X - cx) / (rx * 1.15), (Y - cy + 50) / 78), 0, 1) * wash
    m = np.maximum(pool * 1.2, wall)
    return np.clip(m * 1.6, 0, 1)


def beam_mask(cx, cy=FLOOR_Y, rx=46, src=(240, -150)):
    """空气中的光柱（丁达尔），0..1。"""
    X, Y = XW, YW
    t = np.clip((Y - src[1]) / (cy - src[1]), 0, 1)
    ax = src[0] + (cx - src[0]) * t
    half = 5 + (rx - 5) * t
    d = np.abs(X - ax) / half
    m = np.clip(1 - d, 0, 1) ** 0.7 * (Y < cy + 2) * (0.35 + 0.65 * t)
    return m


@lru_cache(8)
def _cached_spot(cx, rx, wash10):
    return spot_mask(cx, rx=rx, wash=wash10 / 10), beam_mask(cx, rx=rx)


def foot_mask():
    return _foot()


@lru_cache(None)
def _foot():
    X, Y = XW, YW
    m = np.clip(1 - (FLOOR_Y + 6 - Y) / 70, 0, 1) * (Y < LIP_Y) * (X > 60) * (X < 420)
    return m ** 1.5


def light(world_lit, world_dark, L):
    """在暗版 / 亮版之间按光照 L 插值（L 可 > 1 过曝一点）。"""
    return world_dark + (world_lit - world_dark) * L[..., None]


def lamps(world, on):
    """台唇上的一排脚灯。"""
    if on <= 0:
        return
    for x in range(44, 440, 28):
        y = LIP_Y + TOP - 1
        world[y:y + 2, x - 2:x + 3] = C('#3a2a1a')
        world[y - 1:y + 1, x - 1:x + 2] = C('#ffe0a0') * on + world[y - 1:y + 1, x - 1:x + 2] * (1 - on)
        g = (np.abs(XW[y - 4:y + 1, x - 6:x + 7] - x) < 6)
        world[y - 4:y + 1, x - 6:x + 7][g & (BAYER_W[y - 4:y + 1, x - 6:x + 7] < 0.25 * on)] += C('#ffb060') * 0.3


def haze(world, beam, strength, color='#ffe8c0'):
    """把光柱加到世界图上（抖动，像素风）。"""
    if strength <= 0:
        return
    m = BAYER_W < beam * strength
    world[m] += C(color) * 0.16


# ================================================================ 镜头
def camera(world, cam_y=0.0, zoom=1, cx=240, cy=135):
    """world (WORLD_H, W, 3) -> 屏幕。zoom=2 时以 (cx, cy)（屏幕 / 世界坐标，cam_y=0 时）为中心取 240x135 放大。"""
    y0 = int(round(cam_y)) + TOP
    if zoom == 1:
        return world[y0:y0 + H].copy()
    w, h = W // zoom, H // zoom
    x0 = int(np.clip(round(cx - w / 2), 0, W - w))
    yy = int(np.clip(round(cy - h / 2) + y0, 0, WORLD_H - h))
    crop = world[yy:yy + h, x0:x0 + w]
    return np.repeat(np.repeat(crop, zoom, 0), zoom, 1)


# ================================================================ 角色小工具
def rim(spr, color, side='left', amt=1.0):
    """给精灵加一道边缘光（面向光源一侧的轮廓像素染色）。"""
    out = spr.copy()
    a = out[..., 3] > 0
    if side == 'left':
        e = a & ~np.roll(a, 1, 1)
    elif side == 'right':
        e = a & ~np.roll(a, -1, 1)
    elif side == 'top':
        e = a & ~np.roll(a, 1, 0)
    else:
        e = a & (~np.roll(a, 1, 1) | ~np.roll(a, -1, 1) | ~np.roll(a, 1, 0))
    out[e, :3] = out[e, :3] * (1 - amt) + np.asarray(color, np.float32) * amt
    return out


def shade(spr, k, tint=(0.35, 0.3, 0.5)):
    """整体压暗（在暗处 / 逆光时）：k=1 原色，k=0 全剪影色。"""
    out = spr.copy()
    out[..., :3] = out[..., :3] * k + np.asarray(tint, np.float32) * 0.12 * (1 - k)
    return out


def hand_screen(ch, pose, which, x, y, scale=1):
    """body(pose) 以 anchor='bottom' 贴在 (x, y) 时，某只手指尖的屏幕坐标。"""
    hy, hx = ch.meta['hands'][pose if pose in ch.meta['hands'] else 'snap_2'][which]
    spr = ch.body(pose)
    h, w = spr.shape[:2]
    return x - w * scale / 2 + (hx + 0.5) * scale, y - h * scale + (hy + 0.5) * scale


@lru_cache(None)
def gold_emblem(size=96):
    """纹章的金线刺绣版：按官方配色的明度映射到 5 阶金色。"""
    img = ui.emblem_img(size, 1.0, 'gold').copy()
    lum = img[..., :3] @ np.array([0.3, 0.55, 0.15], np.float32)
    ramp = np.stack([C('#2a1606'), C('#6a4410'), C('#b07a20'), C('#e8b440'), C('#fff0a8')])
    idx = np.clip((lum / max(lum.max(), 1e-6) * 4.99).astype(int), 0, 4)
    img[..., :3] = ramp[idx]
    return img


def emblem_scan(world, prog, glow=0.0, cx=240, cy=128, size=96, alpha=1.0):
    """帷幕上的纹章：先是暗色刺绣（mono），一道金光自上而下逐行扫过，扫过的行亮成金色。"""
    if alpha <= 0:
        return
    mono = ui.emblem_img(size, 1.0, 'mono').copy()
    gold = gold_emblem(size) if glow < 0.95 else ui.emblem_img(size, 1.0, 'glow').copy()
    h = mono.shape[0]
    line = int(prog * (h + 4)) - 2
    img = mono.copy()
    img[..., :3] = img[..., :3] * 0.5 + C('#300410') * 0.5
    if line > 0:
        img[:max(0, line)] = gold[:max(0, line)]
    if 0 <= line < h:
        a = img[line, :, 3] > 0
        img[line, a, :3] = C('#fff4c0')
        if line + 1 < h:
            a2 = img[line + 1, :, 3] > 0
            img[line + 1, a2, :3] = C('#ffc860')
    img[..., 3] *= alpha
    if glow > 0:
        m = gold[..., 3] > 0
        g = dilate(dilate(m, 1), 1) & ~m
        gi = np.zeros_like(img)
        gi[g] = (*C('#ffc860'), min(1.0, glow) * 0.8 * alpha)
        blit(world, gi, cx, cy + TOP, anchor='center')
    blit(world, img, cx, cy + TOP, anchor='center')


# ================================================================ 合成
def compose(ctx, cam=0.0, ambient=0.1, spot=0.0, spot_x=240, spot_rx=46, foot=0.0, curtain_open=0.0,
            curtain_drop=0.0, behind=None, emblem=None, beam=None, lamps_on=0.0, glow=None, actors=None,
            fg_par=0.35):
    """画一帧剧场（世界坐标），返回 (world, fgdy)。
    behind(lit, dark)：帷幕打开时画舞台深处；emblem = (扫光进度, 发光, 透明度)；
    glow = [(x, y, r, color, k)] 点光源（火焰等），加在光照上并染色；actors(world) 画角色 / 粒子。"""
    Ls = layers()
    lit, dark = Ls['lit'].copy(), Ls['dark'].copy()
    if behind is not None:
        behind(lit, dark)
    if curtain_drop < 1 or curtain_open < 1:
        draw_curtain(lit, dark, curtain_open, curtain_drop)
    L = np.full((WORLD_H, W), ambient, np.float32)
    bm = None
    if spot > 0:
        sp, bm = _cached_spot(int(spot_x), int(spot_rx), 10)
        L += sp * spot
    if foot > 0:
        L += _foot() * foot
    tint = None
    if glow:
        tint = np.zeros((WORLD_H, W, 3), np.float32)
        for gx, gy, r, col, k in glow:
            if k <= 0:
                continue
            y0, y1 = max(0, int(gy - r) + TOP), min(WORLD_H, int(gy + r) + TOP)
            x0, x1 = max(0, int(gx - r)), min(W, int(gx + r))
            if y1 <= y0 or x1 <= x0:
                continue
            d = np.clip(1 - np.hypot(XW[y0:y1, x0:x1] - gx, YW[y0:y1, x0:x1] - gy) / r, 0, 1) ** 1.5 * k
            L[y0:y1, x0:x1] += d
            tint[y0:y1, x0:x1] += d[..., None] * np.asarray(col, np.float32)
    world = light(lit, dark, np.minimum(L, 1.25))
    if tint is not None:
        world += tint * 0.35 * lit
    if emblem is not None:
        prog, eg, ea = emblem
        emblem_scan(world, prog, eg, alpha=ea * (1 - clamp01(curtain_open * 3)) * (1 - clamp01(curtain_drop * 2)))
    if bm is not None and spot > 0:
        haze(world, bm, 0.55 * spot if beam is None else beam)
    lamps(world, lamps_on)
    if actors is not None:
        actors(world)
    fgdy = int(round(-cam * fg_par))
    blit(world, Ls['fg'], 0, fgdy)
    return world


def view(world):
    """world 里 y 0..H 那一段（粒子 / 精灵按世界坐标画在这里）。"""
    return world[TOP:TOP + H]


def to_screen(x, y, cam_y=0.0, zoom=1, cx=240, cy=135):
    """世界坐标 -> 屏幕坐标（与 camera() 的取景一致）。"""
    if zoom == 1:
        return x, y - int(round(cam_y))
    w, h = W // zoom, H // zoom
    x0 = int(np.clip(round(cx - w / 2), 0, W - w))
    yy = int(np.clip(round(cy - h / 2) + int(round(cam_y)) + TOP, 0, WORLD_H - h)) - TOP
    return (x - x0) * zoom, (y - yy) * zoom
