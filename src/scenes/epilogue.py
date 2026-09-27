"""谢幕：回到剧场。执行者与使徒们的剪影在台上站成一排，他在最前面鞠躬、道别；
响指一打，他在幻焰里化为火星，一大蓬扑克牌炸开四散；众人剪影化作蓝火散去，帷幕落下，纹章发光。

3 小节 @90（1 拍 = 0.667 s）；配乐在 2.2 给响指留空，之后钢片琴 / 竖琴洒牌下行，第 3 小节 D 大三和弦。
分镜（小节.拍）：
  1.0  全景：台上灯全亮，众人逆光剪影列队；他张开双臂亮相（掌声）-> 1.1 - 1.2 深鞠躬；1.2 字幕「告辞了」
  2.0  切中景 2x：起身（2.0 - 2.0.6）-> 抬手捏指（2.1 - 2.1.8，预备）
  2.2  响指！顿帧 3 帧 + 白闪 -> 切回全景：幻焰火柱吞没他（溶解 6 帧），扑克牌爆开四散
  2.3  众人剪影由外向内化作蓝火星散去；牌雨持续
  3.0  帷幕从上方落下（回弹），纹章金光一扫、发光；3.2 起全场只剩纹章与飘落的牌（导演最后 0.8 s 淡出）
"""
from functools import lru_cache

import numpy as np

import fx
from gfx import H, W, blit, clamp01, ease_out, fade, hexc, smooth
import ui
from scenes import _stage as st

LETTERBOX = 0
SAYS = [('campa_exit', 1, 2.0, 'sub')]
CUES = [
    (1, 0.0, 'applause', -12), (1, 1.0, 'swish', -10),
    (2, 1.0, 'swish', -12),
    (2, 2.0, 'snap', -3), (2, 2.02, 'fire_burst', -6), (2, 2.05, 'fire_whoosh', -7), (2, 2.08, 'cards', -4),
    (2, 2.5, 'cards', -8, 0.5), (2, 3.0, 'sparkle', -8, -0.4), (2, 3.3, 'sparkle', -10, 0.4),
    (3, 0.0, 'curtain_fall', -6), (3, 0.6, 'thud', -9), (3, 1.0, 'shimmer', -8),
]
ACCENTS = []      # 2.2 的留空和第 3 小节的 D 大三和弦已写在配乐里


def _curtain_fall():
    """厚重帷幕落下：布料的长「刷——」+ 滑轮的摩擦。"""
    import sfx as S
    n = int(1.0 * S.SR)
    x = S.sweep_lp(S.noise(n), 3000, 400, 0.8) * S.env(n, 0.05, 0.3, 0.6, 0.3) * 0.6
    squeak = S.sine(S.np.linspace(900, 700, n), n) * S.env(n, 0.1, 0.2, 0.2, 0.2) * 0.05
    return (x + squeak).astype(np.float32)


def _thud():
    import sfx as S
    n = int(0.6 * S.SR)
    return (S.lp(S.noise(n), 180) * S.expd(n, 0.08) * 1.4 + S.sine(55, n) * S.expd(n, 0.12)).astype(np.float32)


SFX = {'curtain_fall': _curtain_fall, 'thud': _thud}

CH = 'campanella'
XC, FOOT = 240, st.FLOOR_Y + 3          # 他站在台前
TROUPE = [('mcburn', 118), ('leonhardt', 172), ('bleublanc', 310), ('vita', 362), ('arianrhod', 420)]
TROUPE_Y = st.FLOOR_Y - 4                # 众人略靠后


@lru_cache(None)
def _sil(key, ch):
    img = ch.body('idle')
    a = img[..., 3] > 0
    ys, xs = np.nonzero(a)
    img = img[:, xs.min():xs.max() + 1]
    s = fx.silhouette(img, color=(0.06, 0.03, 0.07), rim=hexc('#ffd8a0'))
    # 逆光：顶部 / 两侧描一圈暖色
    e = (img[..., 3] > 0) & ~np.roll(img[..., 3] > 0, 1, 1)
    s[e, :3] = hexc('#ffb870')
    rng = np.random.default_rng(len(key))
    noise = rng.random(img.shape[:2])
    return s, noise


def _pose(ctx):
    a, lt = ctx.at, ctx.lt
    frames = [(a(1, 0), 'appear_2'), (a(1, 0.8), 'bowin_t0'), (a(1, 1.1), 'bowin_t1'), (a(1, 1.4), 'bow'),
              (a(2, 0), 'bowout_t0'), (a(2, 0.25), 'bowout_t1'), (a(2, 0.5), 'idle'), (a(2, 0.9), 'raise_t0'),
              (a(2, 1.15), 'snap_0'), (a(2, 1.45), 'lift_t0'), (a(2, 1.7), 'snap'), (a(2, 2.0), 'snap_2')]
    pose = 'appear_2'
    for t0, p in frames:
        if lt >= t0:
            pose = p
    k = ctx.since(2, 2.0) - 0.1               # 顿帧 0.1 s 之后开始烧
    if k >= 0:
        i = int(k / 0.09)
        pose = f'vanish_{i}' if i < 6 else None
    return pose


def render(dst, ctx):
    lt, t = ctx.lt, ctx.t
    ch = ctx.C[CH]
    P, P2, bits, fg = ctx.P, ctx.P2, ctx.bits, ctx.fg
    snap = ctx.at(2, 2.0)
    pose = _pose(ctx)
    hx, hy = st.hand_screen(ch, 'snap_2', 'l', XC, FOOT)
    if ctx.crossed(2, 2.0):
        ctx.flash('#ffffff', 0.18, 0.9)
        ctx.shake(6, 0.5)
        P.emit_radial(160, hx, hy, (60, 300), (0.4, 1.0), 'fire', size=(1, 3), drag=2.5)
        P2.emit_radial(120, XC, FOOT - 36, (60, 260), (0.5, 1.3), 'blue', size=(1, 2), drag=2)
        bits.emit(70, (XC - 10, XC + 10), (FOOT - 60, FOOT - 20), (-320, 320), (-380, -60), (2.0, 3.6),
                  kind='card', scale=(1, 2), grav=210, drag=0.8)
    # 牌雨（响指之后一直下）
    if lt > snap + 0.3:
        if ctx.rng.random() < 0.55:
            fg.emit(1, (0, W), -12, (-20, 20), (30, 60), (6, 8), kind='card', scale=(1, 2), grav=10, drag=0.4)
    fire_k = clamp01(ctx.since(2, 2.0) / 0.1) * (1 - smooth(snap + 0.4, snap + 1.6, lt)) if lt >= snap else 0.0
    if fire_k > 0.05:
        P.emit(int(4 * fire_k), (XC - 12, XC + 12), FOOT - 4, (-20, 20), (-160, -70), (0.4, 0.9), 'fire', size=1, drag=1)
        P2.emit(int(3 * fire_k), (XC - 14, XC + 14), FOOT - 10, (-30, 30), (-150, -60), (0.5, 1.1), 'blue', size=1)
    # 众人化作蓝火星：由外向内
    fade_t = {}
    for i, (k, x) in enumerate(TROUPE):
        order = [0, 4, 1, 3, 2][i]
        t0 = ctx.at(2, 3.0) + order * 0.08
        fade_t[k] = clamp01((lt - t0) / 0.35)
    P.step(ctx.dt)
    P2.step(ctx.dt)
    bits.step(ctx.dt)
    fg.step(ctx.dt)

    def behind(lit, dark):
        y0, y1 = st.VAL_Y0 + st.TOP, st.FLOOR_Y + st.TOP
        x0, x1 = st.OPEN_X0, st.OPEN_X1
        yy = np.linspace(0, 1, y1 - y0)[:, None, None]
        grad = hexc('#120a24') * (1 - yy) + hexc('#5a2a3a') * yy ** 1.5
        lit[y0:y1, x0:x1] = grad
        dark[y0:y1, x0:x1] = grad * 0.6
        # 天幕上的逆光（众人身后的一片暖光）
        Y, X = np.mgrid[y0 - st.TOP:y1 - st.TOP, x0:x1].astype(np.float32)
        glow = np.clip(1 - np.hypot((X - 240) / 200, (Y - 196) / 70), 0, 1)
        m = fx.BAYER[:y1 - y0, :x1 - x0] < glow * 0.8
        lit[y0:y1, x0:x1][m] += hexc('#ff9a50') * 0.25
        dark[y0:y1, x0:x1][m] += hexc('#ff9a50') * 0.2

    def actors(world):
        v = st.view(world)
        for k, x in TROUPE:
            f = fade_t[k]
            if f >= 1:
                continue
            s, noise = _sil(k, ctx.C[k])
            img = s.copy()
            if f > 0:
                gone = noise < f * 1.2
                edge = (noise < f * 1.2 + 0.12) & ~gone & (img[..., 3] > 0)
                img[gone, 3] = 0
                img[edge, :3] = hexc('#6ab8ff')
                if ctx.rng.random() < 0.9:
                    ys, xs = np.nonzero(edge)
                    if len(ys):
                        j = ctx.rng.integers(0, len(ys), min(6, len(ys)))
                        h_, w_ = img.shape[:2]
                        P2.emit_arrays(x - w_ / 2 + xs[j], TROUPE_Y - h_ + ys[j], ctx.rng.normal(0, 10, len(j)),
                                       ctx.rng.uniform(-60, -20, len(j)), (0.5, 1.0), 'blue', size=1)
            blit(v, img, x, TROUPE_Y, anchor='bottom')
        if fire_k > 0.01:
            fire = ctx.fire('epi', ui.fire_palette('#ff8a20'), 64, 150, 9)
            src = np.zeros(64, np.int32)
            wd = int(8 + 14 * fire_k)
            src[32 - wd:32 + wd] = int(36 * min(1, fire_k * 1.3))
            fire.step(src, decay=0.3 + 0.8 * (1 - fire_k), wind=ctx.frame % 2, sub=2)
            fire.add_to(v, XC - 32, FOOT - 150, gain=0.75)
        if pose is not None:
            spr = ch.body(pose)
            spr = st.rim(spr, hexc('#ffe0b0'), 'top', 0.5)
            spr = st.rim(spr, hexc('#ffb070'), 'left', 0.5)
            blit(v, spr, XC, FOOT, anchor='bottom')
        if snap - 0.05 <= lt < snap + 0.12:      # 响指那一下的星芒
            r = int(3 + 12 * (1 - abs(lt - snap) / 0.12))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for i in range(r):
                    x, y = int(hx + dx * i), int(hy + dy * i)
                    if 0 <= x < W and 0 <= y < H:
                        v[y, x] = hexc('#fff8d0')
        P.render(v)
        P2.render(v)
        bits.render(v, t)

    drop = 1 - _fall(ctx.since(3, 0)) if lt >= ctx.at(3, 0) else 1.0
    eglow = 0.0
    scan = 1.0
    if lt >= ctx.at(3, 0):
        k = ctx.since(3, 0)
        scan = clamp01((k - 0.5) / 0.8)
        eglow = 0.4 + 0.3 * np.sin(k * 4) + 0.5 * np.exp(-max(0, k - 1.3) * 3) * (k > 1.3)
    glow = [(XC, FOOT - 40, 170, hexc('#ff8a30'), 1.0 * fire_k)] if fire_k > 0 else []
    spot_x = XC
    world = st.compose(ctx, cam=0, ambient=0.34 - 0.2 * smooth(ctx.at(3, 0), ctx.at(3, 2), lt), spot=1.0,
                       spot_x=spot_x, spot_rx=40, foot=0.7 * (1 - 0.6 * smooth(ctx.at(3, 0), ctx.at(3, 1), lt)),
                       curtain_open=1.0 if drop > 0.999 else 0.0, curtain_drop=drop, behind=behind,
                       emblem=(scan, eglow, 1.0) if lt >= ctx.at(3, 0) and drop < 0.08 else None, lamps_on=1.0, glow=glow,
                       actors=actors)
    if ctx.at(2, 0) <= lt < snap + 0.1:          # 中景：起身、抬手；响指后顿帧 0.1 s 再切回全景
        k = ctx.since(2, 0)
        out = st.camera(world, 0, 2, cx=XC + 6, cy=168 - 6 * ease_out(k / 1.3))
    else:
        out = st.camera(world, 0, 1)
    fg.render(out, t)
    dst[:] = out
    return dst


def _fall(k):
    """帷幕落下的进度 0..1（0.45 s 落到底，带一次小回弹）。"""
    if k < 0.45:
        return (k / 0.45) ** 2
    b = k - 0.45
    return 1 - 0.06 * np.sin(b * 14) * np.exp(-b * 6)
