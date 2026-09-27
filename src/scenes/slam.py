"""Logo 砸出：纹章 + 「噬身之蛇」。2 小节 @144，之后接谢幕。"""
import numpy as np

import fx
from gfx import H, W, hexc, smooth, text, vgrad
import i18n
import ui

CUES = [(1, 0, 'big_impact', -2), (1, 0.05, 'shimmer', -3), (1, 0.1, 'glitch', -8), (2, 2, 'wind', -8)]
ACCENTS = [(1, 0, 'hit')]


def render(dst, ctx):
    lt = ctx.lt
    dst = vgrad('#06020c', '#1c0830')
    fx.stars(dst, ctx.t, 0.8)
    fx.rays(dst, 240, 86, ctx.t, hexc('#6a2ab0'), n=12, strength=0.45 * smooth(0, 0.3, lt), speed=0.6)
    P = ctx.P
    if ctx.crossed(1, 0):
        P.emit_radial(420, 240, 86, (80, 380), (0.5, 1.6), 'gold', size=(1, 2), drag=2)
        P.emit_radial(160, 240, 86, (40, 180), (0.8, 1.6), 'purple', size=1, drag=1)
        ctx.shake(6, 0.6)
        ctx.flash('#ffffff', 0.25, 0.8)
    P.emit(3, (0, W), H + 2, (-6, 6), (-50, -15), (2, 3.5), 'gold', bright=0.7, wob=0.3)
    P.step(ctx.dt)
    P.render(dst)
    r = 30 + lt * 420
    if r < 520:
        fx.shockwave(dst, 240, 86, r, hexc('#ffe0a0'), w=3)
    ui.draw_emblem(dst, 240, 86, 72, 1.0, 'glow' if lt < 0.1 else 'gold', glow=0.6 + 0.2 * np.sin(lt * 6))
    sc = 5 if lt < 0.07 else 4
    if i18n.LANG == 'en':
        text(dst, 'OUROBOROS', 240, 190, hexc('#ffd870'), scale=sc - 1, glow=hexc('#8a3ae0'), spacing=2)
        text(dst, '身喰らう蛇', 240, 226, hexc('#c8a8ff'), font='ja', spacing=4, reveal=(lt - 0.2) * 16)
    else:
        text(dst, '噬身之蛇', 240, 190, hexc('#ffd870'), scale=sc, glow=hexc('#8a3ae0'), spacing=3)
        text(dst, 'OUROBOROS', 240, 226, hexc('#c8a8ff'), spacing=5, reveal=(lt - 0.2) * 30)
    return dst
