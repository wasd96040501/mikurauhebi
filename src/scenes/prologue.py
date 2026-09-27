"""序章：空荡的旧剧场。追光、灰尘、帷幕上的衔尾蛇纹章逐行亮起；一张牌飘落、化作幻焰，
肯帕雷拉从火里现身、鞠躬，抬手，对镜头意味深长地一眨眼——响指落在「标题」第一拍。

6 小节 @90（1 拍 = 0.667 s）。分镜（小节.拍）：
  1.0  黑场，镜头从穹顶吊灯缓缓下摇到舞台；1.2 追光「咔」地亮起（闪两下），灰尘在光柱里飘
  1.2 - 3.0  纹章被一道金光自上而下逐行点亮；3.0 亮完一闪
  3.0 - 3.3  一张扑克牌从光柱顶端翻飞着飘下，3.3 落在舞台上
  4.0  牌燃成幻焰（橙焰 + 蓝火星），火柱升起，照亮帷幕
  5.0  切中景（2x）：他从火里蹲伏着现身、起身展臂（5.0 - 5.2），鞠躬（5.2 - 5.3.5）；台词
  6.0  起身、抬手（6.0 - 6.1.5）；6.2 切特写：立绘 2x + 抬起的手，6.2.5 眨眼；6.3 音乐静默，屏息
"""
import numpy as np

import cast
import fx
from gfx import H, W, blit, clamp01, ease_out, fade, hexc, smooth
import ui
from scenes import _stage as st

LETTERBOX = 0
SAYS = [('campa_enter', 5, 0.0, 'dialog')]
CUES = [
    (1, 2.0, 'spot_on', -2), (1, 2.35, 'spot_on', -8),
    (3, 0.0, 'chime', -10),                    # 纹章亮完
    (3, 0.0, 'card_flip', -8), (3, 1.5, 'card_flip', -12), (3, 3.0, 'card_land', -4),
    (4, 0.0, 'fire_burst', -3), (4, 0.03, 'fire_whoosh', -4), (4, 0.1, 'sparkle', -8), (4, 0.5, 'fire_roar', -9),
    (5, 0.0, 'fire_whoosh', -2), (5, 0.0, 'puff', -6), (5, 1.0, 'swish', -9), (5, 2.0, 'swish', -8),
    (6, 0.0, 'swish', -10), (6, 2.0, 'shimmer', -10), (6, 2.5, 'glint', -6),
]
ACCENTS = []      # 序章无鼓：第 5 小节的膨胀和 6.3 的静默已写在配乐里


def _spot_on():
    """老剧场追光：继电器「咔哒」+ 灯丝嗡的一声。"""
    import sfx as S
    n = int(0.5 * S.SR)
    click = S.hp(S.noise(n), 1800) * S.expd(n, 0.006) * 1.2
    thump = S.sine(70, n) * S.expd(n, 0.05) * 0.8
    hum = (S.sine(120, n) * 0.3 + S.sine(240, n) * 0.15) * S.env(n, 0.02, 0.2, 0.3, 0.2) * 0.4
    return (click + thump + hum).astype(np.float32)


def _card_land():
    import sfx as S
    n = int(0.25 * S.SR)
    tap = S.bp(S.noise(n), 900, 4000) * S.expd(n, 0.012)
    return (tap * 0.9 + S.sine(1400, n) * S.expd(n, 0.03) * 0.15).astype(np.float32)


def _glint():
    """眨眼的一声「叮」：高音铃 + 微光。"""
    import sfx as S
    n = int(0.9 * S.SR)
    x = sum(S.sine(f, n) * a for f, a in ((2637, 0.5), (3951, 0.3), (5274, 0.15))) * S.expd(n, 0.25)
    return (x * 0.6).astype(np.float32)


SFX = {'spot_on': _spot_on, 'card_land': _card_land, 'glint': _glint}

CH = 'campanella'
X0 = 240                    # 他站的位置 / 牌落点


def _campa(ctx):
    return ctx.C[CH]


# ---------------------------------------------------------------- 时间线
def _spot(ctx):
    k = ctx.since(1, 2.0)
    if k < 0:
        return 0.0
    for t0, v in ((0.0, 0.8), (0.07, 0.1), (0.16, 0.9), (0.24, 0.35), (0.3, 1.0)):
        if k >= t0:
            val = v
    return val


def _pose_prologue(ctx):
    """第 5-6 小节他的动作帧（12 fps 取帧）。返回 (姿势名, 现身进度)."""
    lt = ctx.lt
    a = ctx.at
    frames = [(a(5, 0.0), 'appear_0'), (a(5, 0.3), 'appear_t0'), (a(5, 0.6), 'appear_1'), (a(5, 1.0), 'rise_t0'),
              (a(5, 1.3), 'appear_2'), (a(5, 2.0), 'bowin_t0'), (a(5, 2.25), 'bowin_t1'), (a(5, 2.5), 'bow'),
              (a(6, 0.0), 'bowout_t0'), (a(6, 0.3), 'bowout_t1'), (a(6, 0.6), 'idle'), (a(6, 0.9), 'raise_t0'),
              (a(6, 1.15), 'snap_0'), (a(6, 1.4), 'lift_t0'), (a(6, 1.6), 'snap')]
    pose = None
    for t0, p in frames:
        if lt >= t0:
            pose = p
    return pose


def render(dst, ctx):
    lt = ctx.lt
    ch = _campa(ctx)
    P, P2 = ctx.P, ctx.P2
    # ---- 镜头：穹顶 -> 舞台
    tilt = smooth(ctx.at(1, 0.5), ctx.at(2, 3), lt)
    cam = -90 * (1 - tilt)
    spot = _spot(ctx)
    ambient = 0.02 + 0.1 * smooth(0, ctx.at(2, 0), lt)
    # 纹章逐行点亮
    eprog = ctx.prog(1, 2.5, 3, 0)
    eglow = 0.0
    if lt >= ctx.at(3, 0):
        eglow = 0.35 + 0.65 * np.exp(-ctx.since(3, 0) * 3)
    if ctx.crossed(3, 0):
        ctx.flash('#ffd870', 0.2, 0.25)
    # ---- 牌
    card = None
    t_fall0, t_land = ctx.at(3, 0), ctx.at(3, 3)
    if t_fall0 <= lt < t_land:
        p = (lt - t_fall0) / (t_land - t_fall0)
        y = -20 + (st.FLOOR_Y - 3 + 20) * (p ** 1.3)
        x = X0 + 26 * np.sin(p * np.pi * 2.6) * (1 - p) ** 0.8
        card = ('fall', x, y, p)
    elif t_land <= lt < ctx.at(4, 0):
        card = ('flat', X0, st.FLOOR_Y - 1, ctx.since(3, 3))
    # ---- 火
    t_ign = ctx.at(4, 0)
    fire_k = 0.0
    if lt >= t_ign:
        grow = ease_out(ctx.since(4, 0) / 0.5)
        fade_ = 1 - smooth(ctx.at(5, 0.3), ctx.at(5, 2.5), lt)
        fire_k = grow * fade_
    if ctx.crossed(4, 0):
        ctx.flash('#ff9a40', 0.22, 0.4)
        ctx.shake(3, 0.35)
        P.emit_radial(90, X0, st.FLOOR_Y - 4, (40, 190), (0.4, 1.0), 'fire', size=(1, 2), drag=2.5, grav=-60)
        P2.emit_radial(50, X0, st.FLOOR_Y - 4, (60, 220), (0.5, 1.2), 'blue', size=(1, 2), drag=2)
    if ctx.crossed(5, 0):
        ctx.flash('#ffe0b0', 0.2, 0.5)
        P.emit_radial(70, X0, st.FLOOR_Y - 36, (40, 170), (0.4, 0.9), 'fire', size=(1, 2), drag=2.5, grav=-40)
        P2.emit_radial(60, X0, st.FLOOR_Y - 36, (50, 200), (0.4, 1.1), 'blue', size=1, drag=2)
    # ---- 灰尘（只在光柱里）
    if spot > 0:
        n = 2
        yy = ctx.rng.uniform(-60, st.FLOOR_Y, n)
        tt_ = np.clip((yy + 150) / (st.FLOOR_Y + 150), 0, 1)
        half = 5 + 41 * tt_
        xx = 240 + ctx.rng.uniform(-1, 1, n) * half * 0.8
        P2.emit_arrays(xx, yy, ctx.rng.normal(0, 3, n), ctx.rng.uniform(-3, 4, n), (2.5, 4.5), 'gold', size=1,
                       bright=0.35, wob=0.05)
    # 火焰里的蓝火星
    if fire_k > 0.05:
        k = int(3 * fire_k)
        P2.emit(k, (X0 - 12, X0 + 12), st.FLOOR_Y - 6, (-25, 25), (-140, -60), (0.5, 1.1), 'blue', size=1, drag=1)
        P.emit(k, (X0 - 10, X0 + 10), st.FLOOR_Y - 10, (-15, 15), (-120, -50), (0.4, 0.8), 'fire', size=1, drag=1)
    P.step(ctx.dt)
    P2.step(ctx.dt)
    pose = _pose_prologue(ctx)

    # ---- 特写（6.2 起）
    if lt >= ctx.at(6, 2.0):
        return _closeup(dst, ctx, ch)

    def actors(world):
        v = st.view(world)
        # 火柱（DOOM 火，从牌的位置升起）
        if fire_k > 0.01:
            fire = ctx.fire('pro', ui.fire_palette('#ff8a20'), 64, 150, 3)
            src = np.zeros(64, np.int32)
            wd = int(6 + 16 * fire_k)
            src[32 - wd:32 + wd] = int(36 * min(1, fire_k * 1.4))
            fire.step(src, decay=0.32 + 0.6 * (1 - fire_k), wind=ctx.frame % 2, sub=2)
            fire.add_to(v, X0 - 32, st.FLOOR_Y - 148, gain=0.72)
        # 他
        if pose is not None:
            spr = ch.body(pose)
            since = ctx.since(5, 0)
            if since < 0.35:        # 刚从火里出来：先是一团燃烧的剪影
                k = since / 0.35
                sil = fx.silhouette(spr, color=hexc('#ff9a30') * (1 - k) + hexc('#4a1a10') * k,
                                    rim=hexc('#ffe0a0'))
                spr = sil if k < 0.6 else st.rim(spr, hexc('#ffb060'), 'all', 0.8)
            else:
                warm = clamp01(fire_k * 1.5)
                if warm > 0.05:
                    spr = st.rim(spr, hexc('#ffa040'), 'right', warm)
                spr = st.rim(spr, hexc('#fff0d0'), 'top', 0.5)
            blit(v, spr, X0, st.FLOOR_Y + 1, anchor='bottom')
        # 牌
        if card is not None:
            if card[0] == 'fall':
                _, x, y, p = card
                c = np.cos(p * 14)
                img = fx.CARD_FRONT if c > 0.35 else fx.CARD_BACK if c < -0.35 else fx.CARD_EDGE
                blit(v, img, x, y, scale=2, anchor='center')
                if ctx.rng.random() < 0.6:
                    P2.emit(1, x, y, (-4, 4), (-12, -2), (0.4, 0.8), 'gold', size=1, bright=0.7)
                # 牌经过光柱时闪一下
                if abs(x - 240) < 5 + 41 * (y + 150) / (st.FLOOR_Y + 150) and c > 0.9:
                    v[int(y) - 1:int(y) + 1, int(x):int(x) + 2] = hexc('#ffffff')
            else:
                _, x, y, k = card
                flat = np.zeros((3, 12, 4), np.float32)
                flat[..., 3] = 1
                flat[..., :3] = hexc('#fbf8ff')
                flat[2, :, :3] = hexc('#8a8098')
                flat[1, 5:7, :3] = hexc('#e02848')
                blit(v, flat, x, y, anchor='bottom')
                # 落地后隐隐发光，预告要燃起来
                pulse = 0.5 + 0.5 * np.sin(k * 14)
                if k > 0.4 and pulse > 0.7:
                    v[int(y) - 3, int(x) - 1:int(x) + 2] += hexc('#ff8a20') * 0.6
        P.render(v)
        P2.render(v)

    glow = []
    if fire_k > 0:
        flick = 1 + 0.15 * np.sin(lt * 37) + 0.1 * np.sin(lt * 23)
        glow.append((X0, st.FLOOR_Y - 40, 170, hexc('#ff8a30'), 0.9 * fire_k * flick))
    if pose is not None:
        glow.append((X0, st.FLOOR_Y - 40, 90, hexc('#ffd8a0'), 0.25))
    world = st.compose(ctx, cam=cam, ambient=ambient, spot=spot * (1 - 0.5 * fire_k), foot=0.25 * smooth(
        ctx.at(4, 0), ctx.at(5, 0), lt), emblem=(eprog, eglow, 1.0 - 0.45 * smooth(ctx.at(5, 0), ctx.at(5, 1), lt)), lamps_on=smooth(ctx.at(4, 0), ctx.at(5, 0), lt),
        glow=glow, actors=actors)
    # ---- 镜头
    if lt >= ctx.at(5, 0):
        # 中景：从 2x 缓慢拉近（整数放大，靠平移做"推"）
        drift = smooth(ctx.at(5, 0), ctx.at(6, 2), lt)
        out = st.camera(world, 0, 2, cx=240, cy=178 - 8 * drift)
    else:
        out = st.camera(world, cam, 1)
    # 开场黑
    if lt < ctx.at(1, 1.5):
        fade(out, hexc('#000000'), 1 - smooth(ctx.at(1, 0.5), ctx.at(1, 1.5), lt) * 0.8)
    dst[:] = out
    return dst


# ---------------------------------------------------------------- 特写
def _closeup(dst, ctx, ch):
    lt = ctx.lt
    k = ctx.since(6, 2.0)
    from gfx import vgrad
    dst[:] = vgrad('#08040c', '#1a0812')
    # 背后的追光光晕 + 余烬
    d = np.hypot(fx.XX - 240, (fx.YY - 70) * 1.3)
    fx.add(dst, (d < 120) & fx.dither_mask(0.35), hexc('#3a1a24'), 1.0)
    fx.add(dst, (d < 80) & fx.dither_mask(0.35), hexc('#3a1a24'), 1.0)
    ctx.P2.render(dst)
    expr = 'wink' if lt >= ctx.at(6, 2.5) else 'smirk'
    face = ch.portrait(expr)
    face = st.rim(face, hexc('#ffb070'), 'right', 0.7)
    push = int(round(4 * ease_out(k / 0.6)))           # 特写里再轻轻推一点
    blit(dst, face, 240 - 6, 8 - push, scale=2, anchor='topleft') if False else \
        blit(dst, face, 160 - push, 12 - push, scale=2)
    hand = ch.mod.portrait_hand('snap')
    hy = 104 - int(round(10 * ease_out(k / 0.35)))     # 手从下方抬进画面
    blit(dst, st.rim(hand, hexc('#ffb070'), 'right', 0.6), 300 - push, hy, scale=2)
    if lt >= ctx.at(6, 2.5):
        g = ctx.since(6, 2.5)
        if g < 0.4:     # 眨眼那一下的星光（在可见的那只眼旁）
            r = int(2 + 8 * np.sin(np.pi * g / 0.4))
            cx, cy = 160 - push + 2 * 27, 12 - push + 2 * 36
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for i in range(r):
                    x, y = cx + dx * i, cy + dy * i
                    if 0 <= x < W and 0 <= y < H:
                        dst[y, x] = hexc('#fff8e0')
    # 最后一拍：音乐静默，画面只剩呼吸般的暗角
    if lt >= ctx.at(6, 3.0):
        fade(dst, hexc('#000000'), 0.25 * smooth(ctx.at(6, 3.0), ctx.at(7, 0), lt))
    return dst
