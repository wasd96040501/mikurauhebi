"""总攻：噬身之蛇全员同时出手（4 小节 @144，E 小调，双踩 + 半音爬升低音，全片最快的一段）。

分镜（小节.拍）
--------------
  1.0  硬切进来（盟主之后）。每 2 拍一个人，全在拍点上硬切，切入的头两帧是反相冲击帧（角色黑剪影压在主色上）：
       1.0 马克邦 —— 挥剑，脚下窜起的魔焰墙，剑上喷火；1.1 第二下（火舌爆开）
  1.2  莱恩哈特 —— 斩击帧 + 按真实剑尖轨迹画的火焰新月；1.3 收势，剑上的鬼火拖尾
  2.0  雅里安洛德 —— 长枪突刺，枪尖射出一道光束；2.1 枪尖炸开巨大的光之十字
  2.2  布卢布兰 —— 身后绽开巨大的紫玫瑰，扑克牌与花瓣从手上爆出；2.3 第二波
  3.0  薇塔 —— 法杖顶端展开魔法阵；3.1 苍蓝的鸟从阵中掠过画面
  3.2  肯帕雷拉 —— 响指：冲击环 + 集中线 + 火焰与扑克牌爆开；3.3 第二道冲击环
  4.0  全景：血红的月亮下，六人剪影站在山脊上（轮廓红光）；盟主巨大而淡的身影浮在月亮里。
       镜头从山脊缓缓仰到月亮；4.2 起眼睛一双双亮起（随鼓的 riser），4.3.5 画面被白光吞掉 -> Logo 砸出。

每个镜头都按角色模块的 POSES / META 在运行时取帧和挂点，缺了就退回 'skill' / 'idle'，
挂点缺了就用精灵包围盒估一个，保证其他人改名字也不会崩。
"""
from functools import lru_cache

import numpy as np

import cast
import fx
from gfx import BAYER, H, W, blit, clamp01, ease_out, hexc, smooth

# (小节, 拍, 音效, dB)
CUES = [
    (1, 0.0, 'fire_burst', -2), (1, 0.0, 'crunch', -6), (1, 1.0, 'fire_roar', -8),
    (1, 2.0, 'slash', -1), (1, 2.0, 'crunch', -7), (1, 3.0, 'fire_whoosh', -8),
    (2, 0.0, 'lance', -1), (2, 0.0, 'crunch', -7), (2, 1.0, 'shimmer', -6),
    (2, 2.0, 'cards', -2), (2, 2.0, 'crunch', -7), (2, 3.0, 'glass', -9),
    (3, 0.0, 'magic_beam', -3), (3, 0.0, 'crunch', -7), (3, 1.0, 'wing', -5),
    (3, 2.0, 'snap', 0), (3, 2.0, 'fire_whoosh', -4), (3, 3.0, 'impact', -8),
    (4, 0.0, 'big_impact', -3), (4, 0.0, 'boom_low', -4),
    (4, 2.0, 'heartbeat', -4), (4, 3.0, 'charge', -5),
]
# 音乐本身在 1.0 / 2.0 / 3.0 / 4.0 有镲、4.3 前有 riser、下一段 1.0 是 slam；这里只要求群像落下时的全奏
ACCENTS = [(4, 0.0, 'hit')]

BEAT = 60 / 144
IMPACT_FRAMES = 2 / 30 + 1e-3
YY, XX = np.mgrid[0:H, 0:W]


# ---------------------------------------------------------------- 取帧与挂点（对别人的模块保持宽容）
def _char(key):
    try:
        return cast.get(key)
    except Exception:
        return None


def _poses(ch):
    m = getattr(ch, 'mod', None)
    for attr in ('POSES',):
        p = getattr(m, attr, None)
        if isinstance(p, dict) and p:
            return set(p)
    meta_p = ch.meta.get('poses') if ch is not None else None
    return set(meta_p) if meta_p else None


@lru_cache(None)
def sprite(key, wants):
    """按顺序尝试 wants 里的姿势，返回 (姿势名, RGBA)。"""
    ch = _char(key)
    if ch is None or ch.mod is None:
        return 'idle', ch.body('idle') if ch is not None else np.zeros((72, 48, 4), np.float32)
    known = _poses(ch)
    for p in tuple(wants) + ('skill', 'idle'):
        if known is not None and p not in known and p not in ('skill', 'idle'):
            continue
        try:
            img = ch.mod.body(p)
            if img is not None and img.ndim == 3 and img.shape[2] == 4 and (img[..., 3] > 0).any():
                return p, img
        except Exception:
            continue
    return 'idle', ch.body('idle')


def _root(key, pose, img):
    """脚底根点（精灵像素坐标）。莱恩哈特的大画布有显式根点，其余按「底边中点」。"""
    if key == 'leonhardt':
        try:
            return ch_mod('leonhardt').pose_info(pose)['root']
        except Exception:
            pass
    return img.shape[1] / 2, img.shape[0]


def ch_mod(key):
    ch = _char(key)
    return getattr(ch, 'mod', None)


def _pt(img, v, order='auto'):
    """META 里的点有的写 (y, x) 有的写 (x, y)：取落在不透明像素上的那种解释。"""
    if v is None:
        return None
    a, b = float(v[0]), float(v[1])
    h, w = img.shape[:2]
    alpha = img[..., 3] > 0

    def hit(x, y):
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(0, yi - 2), min(h, yi + 3), max(0, xi - 2), min(w, xi + 3)
        return 0 <= xi < w and 0 <= yi < h and alpha[y0:y1, x0:x1].any()
    if order == 'yx' or (order == 'auto' and hit(b, a) and not hit(a, b)):
        return b, a
    if order == 'xy' or hit(a, b):
        return a, b
    return b, a


def _meta_pt(key, pose, img, *names):
    """按名字顺序找挂点：per-pose 字典 / 单点 / anchors；都找不到返回 None。"""
    ch = _char(key)
    if ch is None:
        return None
    meta = ch.meta
    anchors = meta.get('anchors', {})
    for n in names:
        v = anchors.get(pose, {}).get(n) if isinstance(anchors.get(pose), dict) else None
        if v is not None:
            return float(v[0]), float(v[1])            # vita.anchors 是 (x, y)
        v = meta.get(n)
        if isinstance(v, dict):
            v = v.get(pose)
            if isinstance(v, dict):
                v = v.get('r') or v.get('l')
        if v is not None and len(v) >= 2 and np.isscalar(v[0]):
            return _pt(img, v[:2])
    return None


def _bbox_pt(img, fx_=0.0, fy=0.0):
    ys, xs = np.nonzero(img[..., 3] > 0)
    return xs.min() + (xs.max() - xs.min()) * fx_, ys.min() + (ys.max() - ys.min()) * fy


class Placed:
    """一个贴到屏幕上的精灵：记录缩放 / 翻转 / 左上角，负责精灵坐标 -> 屏幕坐标。"""

    def __init__(self, key, pose, img, x, y, s, flip=False):
        self.key, self.pose, self.img, self.s, self.flip = key, pose, img, s, flip
        rx, ry = _root(key, pose, img)
        if flip:
            rx = img.shape[1] - rx
        self.x0, self.y0 = x - s * rx, y - s * ry

    def to(self, p):
        if p is None:
            return None
        px = self.img.shape[1] - p[0] if self.flip else p[0]
        return self.x0 + self.s * px, self.y0 + self.s * p[1]

    def draw(self, dst, sil=None, rim=None, alpha=1.0):
        img = self.img
        if sil is not None:
            img = img.copy()
            img[img[..., 3] > 0, :3] = sil
        elif rim is not None:
            img = _rim(img, rim)
        blit(dst, img, self.x0, self.y0, scale=self.s, flip=self.flip, alpha=alpha)


def _rim(img, col, k=0.75):
    a = img[..., 3] > 0
    inner = a & np.roll(a, 1, 1) & np.roll(a, -1, 1) & np.roll(a, 1, 0)
    e = inner & ~(np.roll(inner, 1, 1) & np.roll(inner, -1, 1) & np.roll(inner, 1, 0))
    out = img.copy()
    out[e, :3] = out[e, :3] * (1 - k) + np.asarray(col, np.float32) * k
    return out


# ---------------------------------------------------------------- 镜头背景：斜切色块 + 速度线
def _split_bg(dst, c_dark, c_light, slope, t, lines, direction):
    d = (YY - H / 2) - slope * (XX - W / 2)
    dst[:] = hexc(c_dark)
    lit = d > 0
    dst[lit] = hexc(c_light)
    band = (d > -3) & (d < 0)
    dst[band] = dst[band] * 0.4 + hexc(lines) * 0.6
    # 暗半边里的抖动过渡，让色块不至于死平
    grad = lit & (d < 40) & (BAYER < (1 - d / 40) * 0.5)
    dst[grad] = hexc(c_dark) * 0.4 + hexc(c_light) * 0.6
    fx.speed_lines(dst, t * 4, hexc(lines), direction=direction, alpha=0.45)


def _flash(dst, col):
    dst[:] = hexc(col)


# ---------------------------------------------------------------- 各人的镜头
def _shot_mcburn(dst, ctx, u, first):
    _split_bg(dst, '#140404', '#5a1206', 0.45, ctx.t, '#ff8a3a', -1)
    fire = ctx.fire('as_mc', ['#000000', '#1a0608', '#6a1408', '#ff5a1e', '#ffb04a', '#fff0b0', '#ffffff'], W, 120, 9)
    src = np.full(W, 36 if u < 1.4 else 20, np.int32)
    src[(np.arange(W) // 11) % 4 == 0] = 28
    fire.step(src, decay=1.2, wind=1, sub=2)
    fire.add_to(dst, 0, H - fire.h, gain=0.9)
    pose, img = sprite('mcburn', ('dswing2', 'skill') if u < 1 else ('dfollow', 'dswing2', 'skill'))
    pl = Placed('mcburn', pose, img, 300 - 6 * u, 300, 3)
    if first:
        _flash(dst, '#ffb070')
        pl.draw(dst, sil=hexc('#140404'))
        return
    fx.aura(dst, pl.img if not pl.flip else pl.img[:, ::-1], pl.x0 + img.shape[1] * 1.5, pl.y0 + img.shape[0] * 3, 3,
            hexc('#ff7a2a'), ctx.rng, 2)
    pl.draw(dst, rim=hexc('#ffb060'))
    h = _meta_pt('mcburn', pose, img, 'hand_far', 'hand_near') or _bbox_pt(img, 0.1, 0.25)
    hx, hy = pl.to(h)
    ang = None
    bl = ctx.C['mcburn'].meta.get('blade', {}).get(pose) if 'mcburn' in ctx.C else None
    if bl:
        ang = np.radians(bl[2])
    ang = ang if ang is not None else np.pi
    n = 10
    r = ctx.rng.uniform(0, 110, n)
    ctx.P.emit_arrays(hx + np.cos(ang) * r, hy + np.sin(ang) * r, ctx.rng.normal(0, 30, n),
                      -ctx.rng.uniform(40, 140, n), (0.2, 0.5), 'fire', size=(1, 3), drag=1.5)
    if abs(u - 1.0) < 0.04:
        ctx.P.emit_radial(90, hx, hy, (60, 260), (0.3, 0.7), 'fire', size=(1, 3), drag=2)


def _shot_leonhardt(dst, ctx, u, first):
    _split_bg(dst, '#0a0816', '#4a0c10', -0.5, ctx.t, '#ffb070', -1)
    pose, img = sprite('leonhardt', ('slash2',) if u < 0.9 else ('slash3', 'slash2'))
    x, y = 330 - 8 * min(u, 1.2), 262
    pl = Placed('leonhardt', pose, img, x, y, 3)
    if first:
        _flash(dst, '#ffd8a0')
        pl.draw(dst, sil=hexc('#100606'))
        return
    try:
        from scenes.leonhardt import _crescent, _slash_path
        path = _slash_path(x, x, x - 6, y, s=3)
        fv = 1 - smooth(0.4, 1.6, u)
        _crescent(dst, path, min(1.0, 0.6 + u * 1.5), fadev=fv, thick=0.5)
    except Exception:
        fx.slash_arc(dst, x - 60, y - 150, 180, 120, np.pi * 1.1, np.pi * 1.9, min(1, u * 3), 1 - smooth(0.4, 1.6, u),
                     thick=16)
    pl.draw(dst, rim=hexc('#ff8a4a'))
    info = {}
    try:
        info = ch_mod('leonhardt').pose_info(pose)
    except Exception:
        pass
    if info.get('hand'):
        hx, hy = pl.to(info['hand'])
        tx, ty = pl.to(info['tip'])
        n = 8
        k = ctx.rng.uniform(0.1, 1, n)
        ctx.P.emit_arrays(hx + (tx - hx) * k, hy + (ty - hy) * k, ctx.rng.normal(40, 20, n), -ctx.rng.uniform(40, 120, n),
                          (0.2, 0.45), 'fire', size=(1, 2), drag=1.2)


def _shot_arianrhod(dst, ctx, u, first):
    _split_bg(dst, '#0c1620', '#b8c8d8', 0.35, ctx.t, '#ffffff', -1)
    pose, img = sprite('arianrhod', ('skill', 'swing'))
    pl = Placed('arianrhod', pose, img, 330 - 5 * u, 300, 2)
    tipm = ctx.C['arianrhod'].meta.get('weapon_tips', {}).get(pose) if 'arianrhod' in ctx.C else None
    tip = _pt(img, tipm, 'yx') if tipm else (_meta_pt('arianrhod', pose, img, 'weapon_tip') or _bbox_pt(img, 0.0, 0.3))
    tx, ty = pl.to(tip)
    if first:
        _flash(dst, '#ffffff')
        pl.draw(dst, sil=hexc('#0c1620'))
        return
    fx.beam(dst, tx, ty, tx - 480, ty, 7, min(1.0, u * 4), cols=('#ffffff', '#c8fff4', '#6ee6d6'), t=ctx.t)
    if u >= 1.0:
        a = u - 1.0
        k = 1 - smooth(0.2, 0.9, a)
        L = 30 + 260 * ease_out(a / 0.25)
        w = 7 * k + 1
        for m in ((np.abs(XX - tx) < w) & (np.abs(YY - ty) < L), (np.abs(YY - ty) < w * 0.8) & (np.abs(XX - tx) < L * 0.7)):
            m &= BAYER < k
            dst[m] = hexc('#ffffff')
        if abs(a) < 0.04:
            ctx.P.emit_radial(80, tx, ty, (80, 300), (0.3, 0.7), 'white', size=(1, 2), drag=2)
            ctx.shake(4, 0.25)
    pl.draw(dst, rim=hexc('#ffffff'))


def _shot_bleublanc(dst, ctx, u, first):
    _split_bg(dst, '#12081e', '#46186a', -0.4, ctx.t, '#d8b0ff', 1)
    fx.rose(dst, 150, 120, 70 + 12 * ease_out(min(1.0, u * 2)), ctx.t)
    pose, img = sprite('bleublanc', ('skill', 'cast'))
    pl = Placed('bleublanc', pose, img, 300 + 5 * u, 300, 3)
    if first:
        _flash(dst, '#e8c8ff')
        pl.draw(dst, sil=hexc('#12081e'))
        return
    pl.draw(dst, rim=hexc('#e0b8ff'))
    h = _meta_pt('bleublanc', pose, img, 'hand_skill', 'hand') or _bbox_pt(img, 0.15, 0.2)
    hx, hy = pl.to(h)
    for when in (0.05, 1.0):
        if abs(u - when) < 0.04 and not ctx.state.get(('bb', when)):
            ctx.state[('bb', when)] = True
            ctx.bits.emit(14, hx, hy, (-260, 60), (-220, 80), (0.6, 1.1), 'card', scale=(2, 3), drag=1.5)
            ctx.bits.emit(12, hx, hy, (-200, 80), (-160, 60), (0.6, 1.2), 'petal', scale=(2, 3), drag=1.2, grav=60)
    ctx.P.emit(3, (hx - 10, hx + 10), (hy - 10, hy + 10), (-60, 60), (-60, 20), (0.3, 0.6), 'violet', size=(1, 2))


_BIRD = [  # 苍蓝的鸟（像素剪影），两帧扇翅
    ["....B........B....", "...BbB......BbB...", "..BbbbB....BbbbB..", ".BbbccbBBBBbccbbB.", "Bbb..ccWccWcc..bbB",
     ".......ccccc......", "........cc........"],
    ["..................", "........BB........", "..BBBbbbWWbbbBBB..", "BbbbbccWccWccbbbbB", ".Bbb..ccccc..bbB..",
     "..B.....cc.....B..", "..................."[:18]],
]
_BIRD_PAL = {'B': '#1a3a9a', 'b': '#3a7aff', 'c': '#8fd8ff', 'W': '#ffffff'}


@lru_cache(None)
def _bird(k):
    rows = _BIRD[k]
    img = np.zeros((len(rows), max(len(r) for r in rows), 4), np.float32)
    for y, r in enumerate(rows):
        for x, ch in enumerate(r):
            if ch in _BIRD_PAL:
                img[y, x, :3] = hexc(_BIRD_PAL[ch])
                img[y, x, 3] = 1
    return img


def _shot_vita(dst, ctx, u, first):
    _split_bg(dst, '#050a20', '#18348a', 0.5, ctx.t, '#8fd8ff', 1)
    pose, img = sprite('vita', ('cast', 'skill') if u < 1 else ('cast_b', 'cast', 'skill'))
    pl = Placed('vita', pose, img, 250, 300, 3)
    st = _meta_pt('vita', pose, img, 'staff_tip', 'staff_orb') or _pt(img, ctx.C['vita'].meta.get('weapon_tip'), 'yx') \
        if 'vita' in ctx.C else None
    st = st or _bbox_pt(img, 0.9, 0.0)
    sx, sy = pl.to(st)
    if first:
        _flash(dst, '#a8e0ff')
        pl.draw(dst, sil=hexc('#050a20'))
        return
    R = 30 + 40 * ease_out(min(1.0, u * 2))
    fx.magic_circle(dst, sx, sy, R, ctx.t * 3, hexc('#6cc8ff'), flat=1.0)
    fx.magic_circle(dst, sx, sy, R * 0.55, -ctx.t * 4, hexc('#3a7aff'), flat=1.0, star=5)
    pl.draw(dst, rim=hexc('#8fd8ff'))
    ctx.P.emit(3, (sx - 4, sx + 4), (sy - 4, sy + 4), (-80, 80), (-80, 40), (0.3, 0.6), 'blue', size=(1, 2), drag=1)
    if u >= 1.0:                                   # 苍鸟从阵中掠过
        a = u - 1.0
        bx = sx - 40 - a * 420
        by = sy - 10 + 30 * np.sin(a * 3)
        blit(dst, _bird(int(ctx.t * 12) % 2), bx, by, scale=4, anchor='center')
        ctx.P.emit(4, (bx + 20, bx + 36), (by - 6, by + 6), (40, 120), (-20, 20), (0.3, 0.6), 'blue', size=(1, 2))


def _shot_campanella(dst, ctx, u, first):
    _split_bg(dst, '#16040c', '#7a220e', -0.45, ctx.t, '#ffd070', -1)
    pose, img = sprite('campanella', ('snap', 'snap_2') if u < 1 else ('snap_2', 'snap'))
    pl = Placed('campanella', pose, img, 240, 300, 3)
    ch = ctx.C.get('campanella') if hasattr(ctx.C, 'get') else ctx.C['campanella']
    hands = ch.meta.get('hands', {}).get(pose, {}) if ch is not None else {}
    h = hands.get('l') if isinstance(hands, dict) else None
    h = _pt(img, h) if h is not None else _bbox_pt(img, 0.9, 0.1)
    hx, hy = pl.to(h)
    if first:
        _flash(dst, '#ffd8a0')
        pl.draw(dst, sil=hexc('#16040c'))
        return
    for when in (0.0, 1.0):
        a = u - when
        if 0 <= a < 0.9:
            fx.shockwave(dst, hx, hy, 10 + 420 * ease_out(a / 0.9), hexc('#ffd070'), w=3)
    if u < 0.5:
        fx.radial_lines(dst, hx, hy, ctx.t, hexc('#fff0c0'), n=30, seed=7)
    pl.draw(dst, rim=hexc('#ffb060'))
    for when in (0.05, 1.0):
        if abs(u - when) < 0.04 and not ctx.state.get(('cp', when)):
            ctx.state[('cp', when)] = True
            ctx.P.emit_radial(110, hx, hy, (60, 300), (0.3, 0.8), 'fire', size=(1, 3), drag=2)
            ctx.bits.emit(10, hx, hy, (-240, 240), (-260, 40), (0.6, 1.2), 'card', scale=(2, 3), drag=1.2, grav=200)


SHOTS = [_shot_mcburn, _shot_leonhardt, _shot_arianrhod, _shot_bleublanc, _shot_vita, _shot_campanella]


# ---------------------------------------------------------------- 群像：血月下的剪影
# 群像站位：(角色, 姿势偏好, 屏幕 x, 排, 翻转)。两排错开：后排站在山脊上（更远、偏暗），前排在近处的岩石上。
# 雅里安洛德站在最左边缘，长枪和披风的尖角伸出画面，不再压住左半边；肯帕雷拉在正中、最靠前。
LINEUP = [
    ('vita', ('skill', 'raise', 'idle'), 118, 'back', False),
    ('bleublanc', ('skill', 'idle'), 420, 'back', False),
    ('arianrhod', ('idle', 'grip'), 34, 'front', False),
    ('leonhardt', ('skill', 'idle'), 176, 'front', False),
    ('mcburn', ('skill', 'idle'), 336, 'front', True),
    ('campanella', ('snap', 'idle'), 252, 'fore', False),
]
ROW_Y = {'back': (200, 1.0), 'front': (238, 0.6), 'fore': (252, 0.6)}   # 脚底 y = y0 + oy * k
SIL = {'back': (0.10, 0.03, 0.06), 'front': (0.03, 0.01, 0.03), 'fore': (0.02, 0.0, 0.02)}


def _eyes(key, pose):
    ch = _char(key)
    if ch is None:
        return []
    e = ch.meta.get('eyes_body', {})
    return list(e.get(pose, e.get('idle', [])))


def _grandmaster_in_moon(dst, a, mcx, mcy, R):
    """月亮里巨大而淡的盟主：暗红的剪影，只画在月盘内；月晕描一圈金边。"""
    k = smooth(0, 0.5, a)
    if k <= 0:
        return
    _, gm = sprite('grandmaster', ('raise_hi', 'raise', 'idle'))
    h, w = gm.shape[:2]
    x0, y0 = int(mcx - w / 2), int(mcy - R + 6)
    m = np.zeros((H, W), bool)
    ys, xs = np.nonzero(gm[..., 3] > 0)
    ys, xs = ys + y0, xs + x0
    ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    m[ys[ok], xs[ok]] = True
    disc = np.hypot(XX - mcx, YY - mcy) < R - 1
    body = m & disc & (BAYER < 0.35 + 0.65 * k)
    dst[body] = dst[body] * (1 - 0.55 * k) + hexc('#5a0610') * 0.55 * k
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)) & disc
    dst[edge] = dst[edge] * (1 - 0.5 * k) + hexc('#ffb070') * 0.5 * k
    d = np.hypot(XX - mcx, YY - mcy)
    halo = (d >= R) & (d < R + 2) & (BAYER < 0.6 * k)
    dst[halo] = hexc('#ffc860')


def _group(dst, ctx, lt):
    a = lt - 12 * BEAT                       # 本镜头内的时间
    oy = 36 + 30 * (1 - ease_out(min(1.0, a / (3 * BEAT))))
    fx.moon_scene(dst, oy, ctx.t)
    _grandmaster_in_moon(dst, a, 240, 100 + oy, 78)
    n_lit = int(max(0, (a - 2 * BEAT) / (0.25 * BEAT)))      # 4.2 起每 1/4 拍点亮一双眼睛
    for i, (key, wants, x, row, flip) in enumerate(LINEUP):
        pose, img = sprite(key, wants)
        eyes = _eyes(key, pose)
        lit = i < n_lit
        ch = ctx.C.get(key)
        prim = ch.fx('primary') if ch is not None else hexc('#ff4040')
        rimc = np.minimum(prim * 0.75 + 0.25, 1)                # 每人自己的主色做轮廓光
        ec = np.minimum(prim * 1.4 + 0.15, 1) if key != 'campanella' else hexc('#ff4040')
        sil = fx.silhouette(img, color=SIL[row], rim=rimc if row != 'back' else rimc * 0.7,
                            eyes=eyes if lit else (), eye_color=ec)
        y0, ky = ROW_Y[row]
        pl = Placed(key, pose, img, x, y0 + oy * ky, 2, flip=flip)
        blit(dst, sil, pl.x0, pl.y0, scale=2, flip=flip)
        if lit and eyes:                                     # 眼睛亮起时的一点闪光
            age = a - 2 * BEAT - i * 0.25 * BEAT
            if age < 0.2:
                for (ey, ex) in eyes[:1]:
                    gx, gy = pl.to((ex + 0.5, ey + 0.5))
                    L = int(8 * (1 - age / 0.2)) + 2
                    col = np.minimum(ec * 1.2 + 0.3, 1)
                    for dd in range(-L, L + 1):
                        for (px, py) in ((gx + dd, gy), (gx, gy + dd // 2)):
                            if 0 <= int(px) < W and 0 <= int(py) < H:
                                dst[int(py), int(px)] = col
    # 前排脚下的近景岩石
    rock = YY > 246 + oy * 0.6 + 5 * np.sin(XX * 0.05) + 3 * np.sin(XX * 0.19)
    dst[rock] = hexc('#060104')
    # 月下飘散的火星
    ctx.P.emit(3, (0, W), H + 2, (-20, 20), (-80, -30), (1.5, 2.5), 'fire', size=(1, 2), wob=0.3, bright=0.8)
    # 最后半拍：白光吞没画面，交给 Logo
    wf = smooth(15.3 * BEAT, 16 * BEAT, lt)
    if wf > 0:
        m = BAYER < wf
        dst[m] = hexc('#fff0f0')


# ---------------------------------------------------------------- 入口
def render(dst, ctx):
    lt = ctx.lt
    beat = lt / BEAT
    if beat < 12:
        i = min(int(beat // 2), len(SHOTS) - 1)
        t0 = i * 2 * BEAT
        u = (lt - t0) / BEAT                  # 本镜头内的拍数
        first = (lt - t0) < IMPACT_FRAMES
        if ctx.crossed(1 + (2 * i) // 4, (2 * i) % 4):
            ctx.P.clear()
            ctx.bits.clear()
            ctx.shake(5, 0.2)
        SHOTS[i](dst, ctx, u, first)
    else:
        if ctx.crossed(4, 0.0):
            ctx.P.clear()
            ctx.bits.clear()
            ctx.shake(7, 0.4)
        if lt - 12 * BEAT < IMPACT_FRAMES:
            dst[:] = hexc('#ff3a2a')
        else:
            _group(dst, ctx, lt)
    ctx.P.step(ctx.dt)
    ctx.P.render(dst)
    ctx.bits.step(ctx.dt)
    ctx.bits.render(dst, ctx.t)
    np.clip(dst, 0, 1, out=dst)
    return dst
