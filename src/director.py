"""v3 导演：按 story.py 的段落逐帧调用各场景模块，统一处理转场、闪光、震屏、遮幅、台词。

场景模块接口（src/scenes/<段落名>.py）
----------------------------------------
    LETTERBOX = 0            # 上下遮幅高度（像素），0 = 不遮
    MUSIC_SHAKE = True       # False = 不要导演按音乐重音自动震屏（场景自己控制）
    SAYS = [('mcburn', 3, 0.0, 'dialog')]   # (lines.LINES 的键, 小节, 拍, 'dialog' | 'sub'[, (收起小节, 拍)])
                                            # 导演自动绘制；默认显示 2 小节后滑出
    CUES = [(5, 0.0, 'slash', 0.0)]        # 音效：(小节, 拍, sfx 名, 增益 dB)，mix 读取
    ACCENTS = [(5, 0.0, 'hit')]            # 希望音乐配合的重音：hit / stop / riser / drop，score 读取
    SFX = {'my_sound': fn}                 # 场景自带的合成音效：fn() -> (n, 2) float32 @ 44.1 kHz

    def render(dst, ctx):    # dst: (270, 480, 3) float32 黑底画布；返回画好的画布（可以是新数组）
        ...

  小节/拍都是本段内的编号：小节从 1 数，拍从 0 数，按本段的 BPM 换算。

ctx 提供
--------
    ctx.t / ctx.lt / ctx.dt / ctx.frame      全局时刻、本段内时刻、帧间隔、帧号
    ctx.sec                                   story.Sec（bpm、bar、beat、bars、dur…）
    ctx.at(bar, beat)                         本段内某拍的时刻（秒）
    ctx.since(bar, beat)                      lt - at(bar, beat)，可为负
    ctx.crossed(bar, beat)                    这一帧刚好越过该拍（一次性触发粒子 / 震屏用）
    ctx.prog(bar0, beat0, bar1, beat1)        两拍之间的 0..1 进度（已 clamp）
    ctx.shake(amp, dur) / ctx.flash(color, dur, strength)
    ctx.P / ctx.P2（粒子）ctx.bits / ctx.fg（扑克牌、花瓣等小精灵）ctx.rng
    ctx.fire(name, colors, w, h, seed)       持久的 DOOM 火焰缓冲
    ctx.state                                 本段私有 dict，跨帧保存状态
    ctx.C[key]                                cast.Char（立绘 / 全身像 / META）

命令
----
    python director.py ../out/video_noaudio.mp4                 整片
    python director.py --only mcburn ../preview/mcburn.mp4      只渲染某几段（逗号分隔），带音乐时间轴
    python director.py --preview 20.5 24.1 ...                  导出静帧到 preview/
    python director.py --sheet mcburn                           该段每半拍一张的联系表 preview/sheet_mcburn.png
"""
import argparse
import importlib
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

import cast
import fx
from fx import SpriteBits
from gfx import H, W, DoomFire, Particles, blit, canvas, clamp01, dither_mask, fade, finalize, hexc, rect, smooth, text
import lines
import story
import ui
import vertical

ROOT = Path(__file__).resolve().parent.parent
C = {k: cast.get(k) for k in cast.ORDER}

# 段与段之间的转场：card = 肯帕雷拉的扑克牌翻面；cut = 硬切（场景自己处理）
TRANSITIONS = {
    ('title', 'mcburn'): 'card', ('mcburn', 'leonhardt'): 'card', ('leonhardt', 'arianrhod'): 'card',
    ('arianrhod', 'bleublanc'): 'card', ('bleublanc', 'vita'): 'card', ('vita', 'phantom'): 'card',
}
CARD_IN = 0.55      # 牌飞入（上一段最后这么多秒）
CARD_OUT = 0.42     # 翻面后放大到全屏


# ---------------------------------------------------------------- 场景加载
def load_scene(name):
    try:
        return importlib.import_module(f'scenes.{name}')
    except ModuleNotFoundError as e:
        if e.name not in (f'scenes.{name}', 'scenes'):
            raise
        return None


class _Placeholder:
    LETTERBOX = 0
    SAYS = []

    def __init__(self, name):
        self.name = name

    def render(self, dst, ctx):
        dst[:] = hexc('#140c20')
        b = int(ctx.lt / ctx.sec.bar) + 1
        k = ctx.lt / ctx.sec.beat % 4
        text(dst, f'[{self.name}]', W / 2, 110, hexc('#c8b8e8'), scale=2)
        text(dst, f'bar {b}  beat {k:.1f}', W / 2, 150, hexc('#ffd870'))
        rect(dst, 0, H - 4, W * ctx.lt / ctx.sec.dur, H, hexc('#6a2ab0'))
        return dst


def scene(name):
    return load_scene(name) or _Placeholder(name)


# ---------------------------------------------------------------- 扑克牌
@lru_cache(None)
def card_back(w=46, h=64):
    """牌背：深紫底、金边、中央衔尾蛇纹章。"""
    img = np.zeros((h, w, 4), np.float32)
    img[..., 3] = 1
    img[..., :3] = hexc('#2a0e3e')
    yy, xx = np.mgrid[0:h, 0:w]
    img[((xx + yy) // 3) % 2 == 0, :3] = hexc('#321248')
    for d, col in ((0, '#140620'), (1, '#e0b050'), (2, '#2a0e3e'), (3, '#8a6020')):
        edge = (xx == d) | (yy == d) | (xx == w - 1 - d) | (yy == h - 1 - d)
        inside = (xx >= d) & (yy >= d) & (xx <= w - 1 - d) & (yy <= h - 1 - d)
        img[edge & inside, :3] = hexc(col)
    for cx, cy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):   # 圆角
        img[(np.abs(xx - cx) + np.abs(yy - cy)) < 2, 3] = 0
    from emblem import emblem
    e = emblem(size=34, progress=1.0, palette='gold')
    blit(img[..., :3], e, (w - 34) // 2, (h - 34) // 2)
    return img


def card_sprite(sx, back=True):
    """sx: 水平缩放 -1..1（翻面），返回 RGBA。"""
    img = card_back()
    h, w = img.shape[:2]
    nw = max(1, int(round(abs(sx) * w)))
    xs = (np.arange(nw) * w / nw).astype(int)
    out = img[:, xs].copy()
    if not back:
        out[..., :3] = hexc('#f6f0e0')
        out[:, [0, -1], :3] = hexc('#8a6020')
        out[[0, -1], :, :3] = hexc('#8a6020')
    return out


# ---------------------------------------------------------------- 上下文
class Ctx:
    def __init__(self, director):
        self.d = director
        self.C = C
        self.rng = np.random.default_rng(7)
        self.reset()

    def reset(self):
        self.P = Particles(seed=1)
        self.P2 = Particles(seed=2)
        self.bits = SpriteBits(3)
        self.fg = SpriteBits(8)
        self.fires = {}
        self.state = {}
        self.prev_lt = -1e9

    def at(self, bar=1, beat=0.0):
        return (bar - 1) * self.sec.bar + beat * self.sec.beat

    def since(self, bar=1, beat=0.0):
        return self.lt - self.at(bar, beat)

    def crossed(self, bar=1, beat=0.0):
        w = self.at(bar, beat)
        return self.prev_lt < w <= self.lt or (self.prev_lt < 0 <= self.lt and w <= 0 and self.lt == 0)

    def prog(self, bar0, beat0, bar1, beat1):
        a, b = self.at(bar0, beat0), self.at(bar1, beat1)
        return float(clamp01((self.lt - a) / max(1e-6, b - a)))

    def shake(self, amp, dur=0.35):
        self.d.shakes.append((self.t, amp, dur))

    def flash(self, color='#ffffff', dur=0.18, strength=0.7):
        self.d.flashes.append((self.t, hexc(color) if isinstance(color, str) else color, dur, strength))

    def fire(self, name, colors, w=W, h=150, seed=0):
        if name not in self.fires:
            self.fires[name] = DoomFire(w, h, colors, seed)
        return self.fires[name]


# ---------------------------------------------------------------- 导演
class Director:
    def __init__(self):
        self.ctx = Ctx(self)
        self.cur = None
        self.last = None          # 上一帧（转场用）
        self.frozen = None        # 上一段最后一帧
        self.flashes = []         # (t, color, dur, strength)
        self.shakes = []          # (t, amp, dur)
        self.rng = np.random.default_rng(11)
        self._plan_music_hits()

    def _plan_music_hits(self):
        """音乐里的全奏重音 -> 轻微震屏（score.hits() 可用时）。"""
        try:
            import score
            for t, kind, strength in score.hits():
                if kind in ('tutti', 'hit', 'slam'):
                    sec = story.section_at(t + 1e-6).name
                    mod = load_scene(sec)
                    if getattr(mod, 'MUSIC_SHAKE', True):     # 场景可以自己管震屏（例如顿帧时要静止）
                        self.shakes.append((t, 1.5 + 3 * strength, 0.25))
        except Exception as e:
            print('[director] no score hits:', e)

    def frame(self, t, dt, frame_no=0):
        sec = story.section_at(t)
        ctx = self.ctx
        if sec.name != self.cur:
            self.frozen = self.last
            self.prev_name, self.cur = self.cur, sec.name
            ctx.reset()
            self.mod = scene(sec.name)
        ctx.sec, ctx.t, ctx.dt, ctx.frame = sec, t, dt, frame_no
        ctx.lt = t - sec.t0
        dst = canvas()
        dst = self.mod.render(dst, ctx)
        if dst is None:
            raise RuntimeError(f'scenes.{sec.name}.render 没有返回画布')
        dst = np.ascontiguousarray(dst, np.float32)
        self.says = []
        lb = getattr(self.mod, 'LETTERBOX', 0)
        if lb:
            dst[:lb] = 0
            dst[H - lb:] = 0
        for say in getattr(self.mod, 'SAYS', []):
            key, bar, beat, style = say[:4]
            end = say[4] if len(say) > 4 else (bar + 2, beat)     # 默认显示 2 小节
            lt = ctx.since(bar, beat)
            left = ctx.at(*end) - ctx.lt                            # 距离收起还有多少秒
            if lt >= 0 and left > -0.15:
                leave = clamp01(-left / 0.15)
                if vertical.VERTICAL:              # 竖屏版：台词挪到画面下方（vertical.compose 画）
                    who = key.split('_')[0]
                    self.says.append((style, C[who] if who in C else C['campanella'], lines.get(key), lt, leave))
                elif style == 'dialog':
                    who = key.split('_')[0]
                    ui.dialog(dst, C[who] if who in C else C['campanella'], lines.get(key), lt, leave=leave)
                else:
                    ui.subtitle(dst, lines.get(key), lt, y=H - max(lb, 16) - 16 if lb else 238, leave=leave)
        self.last = dst.copy()
        dst = self._transition(dst, t, sec)
        for when, col, dur, strength in self.flashes:
            if when <= t < when + dur:
                fade(dst, col, (1 - (t - when) / dur) * strength)
        if t > story.DURATION - 0.8 and not vertical.VERTICAL:
            fade(dst, hexc('#000000'), smooth(story.DURATION - 0.8, story.DURATION - 0.1, t))
        amp = 0.0
        for when, a, dur in self.shakes:
            if when <= t < when + dur:
                amp = max(amp, a * (1 - (t - when) / dur))
        if amp > 0.5:
            dx, dy = self.rng.integers(-int(amp), int(amp) + 1, 2)
            dst = np.roll(np.roll(dst, dy, 0), dx, 1)
        ctx.prev_lt = ctx.lt
        if vertical.VERTICAL:
            dst = vertical.compose(dst, self.says, t)
        return dst

    def _transition(self, dst, t, sec):
        i = story.ORDER.index(sec.name)
        nxt = story.ORDER[i + 1] if i + 1 < len(story.ORDER) else None
        prv = story.ORDER[i - 1] if i > 0 else None
        # 牌飞入：上一段的最后 CARD_IN 秒
        if nxt and TRANSITIONS.get((sec.name, nxt)) == 'card' and t > sec.t1 - CARD_IN:
            p = (t - (sec.t1 - CARD_IN)) / CARD_IN
            e = 1 - (1 - p) ** 3
            x = -30 + (W / 2 + 30) * e
            y = H + 40 - (H / 2 + 40) * e + np.sin(p * np.pi) * -30
            sx = np.cos(p * np.pi * 3.5) if p < 0.86 else np.cos(0.5 * np.pi * (1 - (1 - p) / 0.14))
            sc = 1 + 1.2 * e
            img = card_sprite(sx if abs(sx) > 0.05 else 0.05)
            fade(dst, hexc('#000000'), 0.35 * e)
            blit(dst, img, x, y, scale=2 if p > 0.35 else 1, anchor='center')
        # 翻面后：新场景在牌面里，牌放大到全屏
        if prv and TRANSITIONS.get((prv, sec.name)) == 'card' and t < sec.t0 + CARD_OUT and self.frozen is not None:
            p = (t - sec.t0) / CARD_OUT
            flip = min(1, p / 0.3)
            grow = smooth(0.25, 1.0, p)
            cw = 46 * 2 * (1 - grow) + (W + 40) * grow
            ch_ = 64 * 2 * (1 - grow) + (H + 60) * grow
            cw *= flip
            yy, xx = np.mgrid[0:H, 0:W]
            inside = (np.abs(xx - W / 2) < cw / 2) & (np.abs(yy - H / 2) < ch_ / 2)
            border = inside & ~((np.abs(xx - W / 2) < cw / 2 - 3) & (np.abs(yy - H / 2) < ch_ / 2 - 3))
            bg = self.frozen.copy()
            fade(bg, hexc('#000000'), 0.35 + 0.4 * p)
            out = np.where(inside[..., None], dst, bg)
            out[border] = hexc('#e0b050')
            return out
        return dst


# ---------------------------------------------------------------- 入口
def _frames(sections=None):
    n = story.FRAMES
    for i in range(n):
        t = i / story.FPS
        if sections:
            s = story.section_at(t)
            if s.name not in sections:
                continue
        yield i, t


def run_video(out, sections=None):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fw, fh = (vertical.VW, vertical.VH) if vertical.VERTICAL else (W, H)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{fw * 4}x{fh * 4}',
           '-r', str(story.FPS), '-i', '-']
    wav = ROOT / 'out' / 'soundtrack.wav'
    if sections and wav.exists():   # 分段预览时带上对应的声音
        t0 = story.SECS[sections[0]].t0
        cmd += ['-ss', f'{t0:.3f}', '-i', str(wav), '-shortest', '-c:a', 'aac', '-b:a', '192k']
    # 整片用 veryslow + CRF 20：比 slow + CRF 16 小约 1/3，VMAF 只低 0.3（98.3 vs 98.6，最差 1% 帧 96.2）
    cmd += ['-c:v', 'libx264', '-preset', 'veryslow' if not sections else 'fast', '-crf', '20' if not sections else '16',
            '-tune', 'animation',
            '-pix_fmt', 'yuv420p', str(out)]
    if vertical.VERTICAL:       # 4 倍的 1920x3416 缩到 1080x1920
        cmd[-3:-3] = ['-vf', 'scale=1080:1920:flags=lanczos']
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    d = Director()
    for k, (i, t) in enumerate(_frames(sections)):
        f = d.frame(t, 1 / story.FPS, i)
        proc.stdin.write(finalize(f).tobytes())
        if k % 150 == 0:
            print(f'frame {i}/{story.FRAMES}', flush=True)
    proc.stdin.close()
    proc.wait()
    print('wrote', out)


def run_preview(times, outdir=None):
    outdir = Path(outdir or ROOT / 'preview')
    outdir.mkdir(parents=True, exist_ok=True)
    d = Director()
    want = {int(round(x * story.FPS)) for x in times}
    secs = {story.section_at(i / story.FPS).name for i in want}
    for i, t in _frames(secs):
        if i > max(want):
            break
        f = d.frame(t, 1 / story.FPS, i)
        if i in want:
            p = outdir / f'f_{t:06.2f}.png'
            Image.fromarray(finalize(f, scale=2, scan=False)).save(p)
            print(p)


def run_sheet(name, step_beats=0.5, cols=4):
    """某一段每 step_beats 拍一张的联系表（1x 分辨率），检查动作节奏。"""
    s = story.SECS[name]
    d = Director()
    want = {}
    k = 0.0
    while k < s.bars * 4:
        want[int(round((s.t0 + k * s.beat) * story.FPS))] = k
        k += step_beats
    tiles = []
    for i, t in _frames([name]):
        f = d.frame(t, 1 / story.FPS, i)
        if i in want:
            img = finalize(f, scale=1, scan=False)
            b = want[i]
            tiles.append((img, f'{int(b // 4) + 1}.{b % 4:g}'))
    th, tw = H + 12, W + 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * th, cols * tw, 3), 30, np.uint8)
    for j, (img, lab) in enumerate(tiles):
        r, c = divmod(j, cols)
        sheet[r * th + 12:r * th + 12 + H, c * tw:c * tw + W] = img
    im = Image.fromarray(sheet)
    from PIL import ImageDraw
    dr = ImageDraw.Draw(im)
    for j, (_, lab) in enumerate(tiles):
        r, c = divmod(j, cols)
        dr.text((c * tw + 2, r * th), f'bar.beat {lab}', fill=(255, 220, 120))
    p = ROOT / 'preview' / f'sheet_{name}.png'
    p.parent.mkdir(exist_ok=True)
    im.save(p)
    print(p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?')
    ap.add_argument('--only')
    ap.add_argument('--preview', nargs='+', type=float)
    ap.add_argument('--sheet')
    ap.add_argument('--step', type=float, default=0.5)
    a = ap.parse_args()
    if a.preview:
        run_preview(a.preview)
    elif a.sheet:
        run_sheet(a.sheet, a.step)
    else:
        run_video(a.out, a.only.split(',') if a.only else None)
