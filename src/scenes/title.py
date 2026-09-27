"""标题：响指 -> 帷幕炸开 -> 「噬身之蛇」砸出，Boss 战开始。2 小节 @144（1 拍 = 0.417 s）。

分镜（小节.拍）：
  1.0  响指（配乐 slam + 全局音效 snap / impact / curtain）：白闪，帷幕从中缝向两侧猛地收开，
       缝里炸出一蓬扑克牌；台后是紫色虚空，大纹章 + 旋转光芒；他在台左，打完响指的随势姿势
  1.0  「噬身之蛇」砸下（第一帧放大 + 冲击波），1.1 OUROBOROS 打字，1.1.5 身喰らう蛇
  1.2  两道追光从台顶交叉扫过，火星上升；他收手站定
  2.0  （镲）硬切中景 2x：他从胸前抽出一张牌（2.1 预备）
  2.2  甩牌：牌飞向镜头、越变越大（接导演的扑克牌转场，最后 0.55 s）
"""
import numpy as np

import fx
from gfx import H, W, blit, clamp01, ease_out, fade, hexc, smooth, text
import i18n
import ui
from scenes import _stage as st

LETTERBOX = 0
SAYS = []
# 响指 / 冲击 / 帷幕 / 微光已在 mixdown 的全局音效里（标题第一拍）
CUES = [
    (1, 0.02, 'cards', -4, 0.0), (1, 0.05, 'fire_whoosh', -8),
    (1, 1.0, 'shimmer', -8),
    (1, 2.0, 'swish', -10, -0.5), (1, 2.5, 'swish', -12, 0.5),     # 追光扫过
    (2, 1.0, 'card_flip', -8),
    (2, 2.0, 'whip', -6), (2, 2.05, 'swish', -4),
]
ACCENTS = [(1, 0.0, 'hit')]     # 与内置 slam 重合（score 会跳过重复）

CH = 'campanella'
XC = 96                          # 他在台左的位置


def _behind(lt, t):
    """帷幕后的舞台深处：紫色虚空、旋转光芒、大纹章（自发光，亮 / 暗两版都画）。"""
    def draw(lit, dark):
        y0, y1 = st.VAL_Y0 + st.TOP, st.FLOOR_Y + st.TOP
        x0, x1 = st.OPEN_X0, st.OPEN_X1
        yy = np.linspace(0, 1, y1 - y0)[:, None, None]
        grad = hexc('#1c0830') * (1 - yy) + hexc('#4a1450') * yy
        for L in (lit, dark):
            L[y0:y1, x0:x1] = grad
            reg = L[y0:y1, x0:x1]
            Y, X = np.mgrid[y0 - st.TOP:y1 - st.TOP, x0:x1].astype(np.float32)
            a = np.arctan2(Y - 112, X - 240)
            d = np.hypot(X - 240, Y - 112)
            m = (np.sin(a * 12 + t * 1.3) > 0.55) & (fx.BAYER[:y1 - y0, :x1 - x0] < np.clip(1 - d / 260, 0, 1) * 0.9)
            reg[m] += hexc('#8a3ae0') * 0.35
            m2 = (np.sin(a * 5 - t * 0.8) > 0.8) & (fx.BAYER[:y1 - y0, :x1 - x0] < 0.3)
            reg[m2] += hexc('#ffc860') * 0.2
            r = 30 + lt * 380
            if r < 400:
                ring = np.abs(d - r) < 1.5
                reg[ring] = hexc('#ffe0a0')
            ui.draw_emblem(L, 240, 112 + st.TOP, 96, 1.0, 'glow' if lt < 0.12 else 'gold', scale=1,
                           glow=0.55 + 0.25 * np.sin(t * 6))
    return draw


def _pose(ctx):
    a, lt = ctx.at, ctx.lt
    frames = [(a(1, 0), 'snap_2'), (a(1, 1.5), 'snap'), (a(1, 2.0), 'lift_t0'), (a(1, 2.5), 'raise_t0'),
              (a(1, 3.0), 'idle'), (a(2, 0.5), 'breath'), (a(2, 1.0), 'flick_0'), (a(2, 1.75), 'flick_t0'),
              (a(2, 2.0), 'flick_1'), (a(2, 2.5), 'flick_2')]
    pose = 'snap_2'
    for t0, p in frames:
        if lt >= t0:
            pose = p
    return pose


def render(dst, ctx):
    lt, t = ctx.lt, ctx.t
    ch = ctx.C[CH]
    P, bits = ctx.P, ctx.bits
    opn = ease_out(lt / 0.32)
    if ctx.crossed(1, 0):
        ctx.flash('#ffffff', 0.2, 0.85)
        ctx.shake(7, 0.5)
        bits.emit(46, (232, 248), (70, 200), (-340, 340), (-280, 40), (1.2, 2.4), kind='card', scale=(1, 2),
                  grav=260, drag=0.6)
        P.emit_radial(260, 240, 112, (60, 360), (0.5, 1.4), 'gold', size=(1, 2), drag=2)
        P.emit_radial(120, 240, 112, (40, 200), (0.6, 1.4), 'purple', size=1, drag=1.5)
    if ctx.crossed(2, 0):
        P.emit_radial(120, 240, 112, (60, 260), (0.4, 1.0), 'gold', size=1, drag=2)
    P.emit(3, (60, 420), 210, (-8, 8), (-70, -25), (1.2, 2.2), 'fire', bright=0.7, wob=0.3)
    pose = _pose(ctx)
    # 甩出的牌
    flick_t = ctx.since(2, 2.0)
    P.step(ctx.dt)
    bits.step(ctx.dt)

    def actors(world):
        v = st.view(world)
        spr = ch.body(pose)
        spr = st.rim(spr, hexc('#c89aff'), 'right', 0.8)
        spr = st.rim(spr, hexc('#fff0d0'), 'top', 0.5)
        blit(v, spr, XC, st.FLOOR_Y + 1, anchor='bottom')
        if pose in ('snap_2',) and lt < 0.3:        # 响指瞬间的火花
            hx, hy = st.hand_screen(ch, pose, 'l', XC, st.FLOOR_Y + 1)
            r = int(2 + 7 * (1 - lt / 0.3))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                for i in range(r if dx == 0 or dy == 0 else r // 2):
                    x, y = int(hx + dx * i), int(hy + dy * i)
                    if 0 <= x < W and 0 <= y < H:
                        v[y, x] = hexc('#fff4c0')
        P.render(v)
        bits.render(v, t)

    # 两道交叉扫过的追光
    sweep = smooth(ctx.at(1, 2), ctx.at(2, 2), lt)
    world = st.compose(ctx, cam=0, ambient=0.3, spot=1.0, spot_x=XC, spot_rx=34, foot=0.5, curtain_open=opn,
                       behind=_behind(lt, t), lamps_on=1.0, actors=actors, beam=0.4)
    v = st.view(world)
    for sx, s0, s1 in ((0, 140, 330), (480, 340, 150)):
        if ctx.lt >= ctx.at(1, 2):
            ex = s0 + (s1 - s0) * sweep
            bm = st.beam_mask(ex, rx=26, src=(sx, -40))
            st.haze(world, bm, 0.5, '#e8d0ff')
    if lt < ctx.at(2, 0):
        out = st.camera(world, 0, 1)
        _title_text(out, lt, ctx)
    else:
        k = ctx.since(2, 0)
        out = st.camera(world, 0, 2, cx=XC + 14, cy=160 - 3 * ease_out(k / 1.2))
    # 甩向镜头的牌（屏幕坐标，越飞越大）
    if flick_t >= 0:
        hx, hy = st.hand_screen(ch, 'flick_1', 'r', XC, st.FLOOR_Y + 1)
        sx0, sy0 = st.to_screen(hx, hy, 0, 2, XC + 14, 160 - 3 * ease_out(ctx.since(2, 0) / 1.2))
        p = flick_t / 0.34
        if p < 1:
            x = sx0 + (-40 - sx0) * p ** 1.4
            y = sy0 + (300 - sy0) * p ** 1.8 - 60 * np.sin(p * np.pi)
            c = np.cos(p * 18)
            img = fx.CARD_FRONT if c > 0.3 else fx.CARD_BACK if c < -0.3 else fx.CARD_EDGE
            blit(out, img, x, y, scale=int(2 + 5 * p), anchor='center')
    dst[:] = out
    return dst


def _title_text(out, lt, ctx):
    sc = 5 if lt < 0.07 else 4
    y = 176 + (1 - ease_out(lt / 0.12)) * -30
    if i18n.LANG == 'en':     # 英文版：OUROBOROS 做大标题，第二行留空，日文仍在原位
        text(out, 'OUROBOROS', 240, y, hexc('#ffd870'), scale=sc - 1, glow=hexc('#8a3ae0'), spacing=2)
    else:
        text(out, '噬身之蛇', 240, y, hexc('#ffd870'), scale=sc, glow=hexc('#8a3ae0'), spacing=3)
        text(out, 'OUROBOROS', 240, 208, hexc('#e0c8ff'), spacing=6, reveal=ctx.since(1, 1.0) * 30)
    text(out, '身喰らう蛇', 240, 226, hexc('#b8a0e0'), font='ja', spacing=4, reveal=ctx.since(1, 1.5) * 16)
