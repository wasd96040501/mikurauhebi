"""噬身之蛇 PV v3 配乐（约 100.67 s，带速度变化）。时间全部来自 story.py。

用法（仓库根目录）：
    .venv/bin/python src/score.py out/music.wav [--stems DIR] [--quiet]
产出 out/music.mid（带 set_tempo 的完整总谱）与 out/music.wav（44.1 kHz / 16 bit 立体声，
长度正好 story.DURATION，约 -16 LUFS，采样峰值 <= -3 dBFS）。

给画面 / 音效用（不渲染，只算谱，毫秒级）：
    import score; score.hits()   -> [(秒, kind, strength 0..1), ...]
      kind: 'slam'（标题 / 盟主 / Logo 的大重拍）、'tutti'（全奏重音，含各幕技能命中）、
            'downbeat'（每段第一拍）、'crash'（鼓组镲片重击）、'stop'（停顿起点）、
            'drop'（次低频下潜）、'riser'（渐强起点；终点就是随后的重音 / 段落首拍）

流程：Python 写谱 -> 每个声部单独用 FluidSynth + GeneralUser GS 渲染 -> numpy 混音
（EQ、声像、合成厅堂混响、铜锣、次低频、噪声渐强）-> 总线压缩、指挥推子、限幅、响度归一。

音乐结构（调性 / 主奏）：
  prologue   90 BPM  D 小调  八音盒陈述动机，竖琴琶音、低音弦乐、人声 pad、拨弦 + 木鱼“钟摆”；
                             第 5 小节渐强，第 6 小节属和弦悬置，最后一拍近乎静音（留给响指）
  title     144 BPM  D 小调  管弦击打 + 定音鼓 + 铜锣，铜管奏动机，Boss 战律动开始
  mcburn               失真吉他 riff + 低音铜管（动机扩大），半速重摇滚
  leonhardt            口琴独奏动机（回忆）-> 跳弓弦乐 + 太鼓 + 小号英雄旋律
  arianrhod            圆号 + 小号骑士号角，军鼓
  bleublanc            羽管键琴 + 拨弦 + 钟琴，3+3+2 的圆舞曲式律动
  vita                 女高音（合唱单线 + Solo Vox）+ 竖琴十六分琶音；结尾 G -> B7 为转 E 小调作准备
  phantom             E 小调  半速，八音盒动机回归，暗色宽 pad，逐步推起
  grandmaster          管风琴 + 合唱 + 管钟 + 乐队，动机四倍扩大
  assault              全员，“盘蛇”八分音型 + 低音半音爬升
  slam                 Logo 重击后余响
  epilogue   90 BPM  D 小调  八音盒动机，响指留空，洒牌（钢片琴 / 竖琴下行），D 大三和弦结束

动机（衔尾蛇绕着主音打转）：D(附点四分) Eb(八分) D(四分) C#(四分) | Bb(二分) A(二分)。

每一幕的重音钩子：src/scenes/<段>.py 若定义 ACCENTS = [(小节, 拍, kind), ...]，
kind 为 'hit' / 'stop' / 'riser' / 'drop'，则替换该幕默认的 [(5, 0, 'hit'), (7, 0, 'riser')]
（非“幕”的段落则在内置重音之外追加）。只用 ast 读取字面量，读不到才 import。
"""
import ast
import importlib
import itertools
import math
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import story  # noqa: E402

ROOT = SRC.parent
SF2 = ROOT / 'assets' / 'soundfonts' / 'GeneralUser-GS.sf2'
SR = 44100
DURATION = story.DURATION
N = int(round(DURATION * SR))
TPB = 480
TARGET_LUFS = -16.15     # 自测 BS.1770 与 ffmpeg ebur128 相差约 0.15 LU
CEILING_DB = -3.2        # 采样峰值上限

# ================================================================ 时间轴 ==
# 全局“拍”坐标：各段按自己的 BPM 累加拍数。每段小节数都是整数，所以段首、小节首都落在整数拍上。

SEC = story.SECS
B0 = {}
_b = 0.0
for _n in story.ORDER:
    B0[_n] = _b
    _b += SEC[_n].bars * 4
END = _b


def P(sec, bar=1, beat=0.0):
    """段 sec 第 bar 小节（从 1 数）第 beat 拍（从 0 数）的全局拍位置。"""
    return B0[sec] + (bar - 1) * 4 + beat


def sec_of(b):
    """全局拍所在的段。"""
    cur = story.ORDER[0]
    for n in story.ORDER:
        if b + 1e-9 >= B0[n]:
            cur = n
    return cur


def secs(b):
    """全局拍 -> 秒（分段线性，与 story.Sec.T 一致）。"""
    n = sec_of(b)
    return SEC[n].t0 + (b - B0[n]) * SEC[n].beat


def beat_len(b):
    return SEC[sec_of(b)].beat


# 自检：全局拍与 story 的时刻逐拍一致
for _n in story.ORDER:
    for _bar in range(1, SEC[_n].bars + 1):
        for _k in (0, 1.5, 3.75):
            assert abs(secs(P(_n, _bar, _k)) - SEC[_n].T(_bar, _k)) < 1e-9
assert abs(secs(END) - DURATION) < 1e-9

# ================================================================ 和声 ====

NOTE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6,
        'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
KIND = {'': (0, 4, 7), 'm': (0, 3, 7), '7': (0, 4, 7, 10), 'o7': (0, 3, 6, 9), '5': (0, 7),
        'maj7': (0, 4, 7, 11), 'm7': (0, 3, 7, 10), 'sus4': (0, 5, 7), '7sus4': (0, 5, 7, 10)}


@dataclass
class Chord:
    name: str
    root: int
    pcs: tuple
    kind: str
    bass: int


def chord(sym):
    body, _, bass = sym.partition('/')
    r = body[:2] if len(body) > 1 and body[1] in '#b' else body[:1]
    kind = body[len(r):]
    root = NOTE[r]
    pcs = tuple((root + i) % 12 for i in KIND[kind])
    return Chord(sym, root, pcs, kind, NOTE[bass] if bass else root)


# (小节, 拍, 和弦)，小节是段内编号
HARMONY = {
    'prologue': [(1, 0, 'Dm'), (3, 0, 'Bb'), (3, 2, 'A'), (4, 0, 'Gm'), (5, 0, 'Eb'), (5, 2, 'A'),
                 (6, 0, 'A7sus4'), (6, 1.5, 'A7'), (6, 3, None)],
    'title': [(1, 0, 'Dm'), (1, 1.5, 'Eb'), (1, 2, 'Dm'), (1, 3, 'A'), (2, 0, 'Bb'), (2, 2, 'A')],
    'mcburn': [(1, 0, 'Dm'), (1, 3, 'Eb'), (2, 0, 'Dm'), (2, 2, 'A'), (3, 0, 'Bb'), (4, 0, 'A'),
               (5, 0, 'Dm'), (5, 3, 'A'), (6, 0, 'Bb'), (6, 2, 'A')],
    'leonhardt': [(1, 0, 'Dm'), (1, 3, 'A'), (2, 0, 'Bb'), (2, 2, 'A'), (3, 0, 'Gm'), (4, 0, 'Eb'),
                  (4, 2, 'A'), (5, 0, 'Dm'), (6, 0, 'Gm'), (6, 2, 'A')],
    'arianrhod': [(1, 0, 'Dm'), (2, 0, 'C'), (3, 0, 'Bb'), (4, 0, 'A'), (5, 0, 'Dm'), (5, 2, 'C'),
                  (6, 0, 'Bb'), (6, 2, 'A7')],
    'bleublanc': [(1, 0, 'Dm'), (2, 0, 'Gm'), (3, 0, 'C7'), (4, 0, 'F'), (4, 2, 'A7'), (5, 0, 'Dm'),
                  (6, 0, 'Bb'), (6, 2, 'A')],
    'vita': [(1, 0, 'Dm'), (2, 0, 'Bb'), (3, 0, 'Gm'), (4, 0, 'A'), (5, 0, 'Bb'), (5, 2, 'C'),
             (6, 0, 'G'), (6, 2, 'B7')],
    'phantom': [(1, 0, 'Em'), (2, 0, 'C'), (2, 2, 'B'), (3, 0, 'Am'), (4, 0, 'B7')],
    'grandmaster': [(1, 0, 'Em'), (1, 3, 'F'), (2, 0, 'C'), (2, 2, 'B'), (3, 0, 'Am'), (3, 2, 'F'),
                    (4, 0, 'Bsus4'), (4, 1, 'B'), (4, 2, 'B7')],
    'assault': [(1, 0, 'Em'), (1, 2, 'F'), (2, 0, 'D/F#'), (2, 2, 'G'), (3, 0, 'E7/G#'), (3, 2, 'Am'),
                (4, 0, 'A#o7'), (4, 2, 'B')],
    'slam': [(1, 0, 'Em')],
    'epilogue': [(1, 0, 'Dm'), (2, 0, 'Bb'), (2, 2, 'A'), (3, 0, 'D')],
}

HARM = []
for _n in story.ORDER:
    for bar, beat, sym in HARMONY[_n]:
        HARM.append([P(_n, bar, beat), None, chord(sym) if sym else None])
for _i in range(len(HARM)):
    HARM[_i][1] = HARM[_i + 1][0] if _i + 1 < len(HARM) else END


def harm_at(beat):
    for s, e, ch in HARM:
        if s <= beat + 1e-9 < e:
            return ch
    return None


def segments(b0, b1, min_len=0.0):
    """[b0,b1) 内的和声段 [start, end, chord]，短于 min_len 的并入前一段。"""
    out = []
    for s, e, ch in HARM:
        s2, e2 = max(s, b0), min(e, b1)
        if e2 - s2 <= 1e-9 or ch is None:
            continue
        if out and (e2 - s2 < min_len - 1e-9) and abs(out[-1][1] - s2) < 1e-9:
            out[-1][1] = e2
            continue
        out.append([s2, e2, ch])
    return out


def place(pc, lo):
    """把音级放进 [lo, lo+11]。"""
    return lo + (pc - lo) % 12


def voicing(ch, lo, hi, k, prev=None, top=None):
    """在 [lo,hi] 内为和弦选 k 个音：覆盖和弦音、避免二度与低音区密集音程、尽量靠近上一个排列。"""
    cands = [n for n in range(lo, hi + 1) if n % 12 in ch.pcs]
    if top is not None:
        cands = [n for n in cands if top - 19 <= n < top]
        combos = (c + (top,) for c in itertools.combinations(cands, k - 1))
    else:
        combos = itertools.combinations(cands, k)
    best, best_pen = None, 1e9
    fifth = (ch.root + 7) % 12
    for combo in combos:
        cover = {n % 12 for n in combo}
        pen = 0.0
        for pc in ch.pcs:
            if pc not in cover:
                pen += 3 if (len(ch.pcs) >= 4 and pc == fifth) else 25
        gaps = np.diff(combo)
        pen += 8 * np.sum(gaps < 3) + 2 * max(0, combo[-1] - combo[0] - 19)
        # 低音区（E3 以下）两音间隔小于纯五度会发浑
        for a, b in zip(combo, combo[1:]):
            if a < 52 and b - a < 7:
                pen += 10
        if combo[0] % 12 != ch.bass and len(ch.pcs) > 2:
            pen += 2
        if prev is not None and len(prev) == len(combo):
            pen += 0.35 * sum(abs(a - b) for a, b in zip(combo, prev))
        else:
            pen += 0.15 * abs(np.mean(combo) - (lo + hi) / 2)
        if pen < best_pen:
            best, best_pen = combo, pen
    return list(best)


def below(p, ch, min_int=3):
    """p 之下至少 min_int 半音的最高和弦音（给第二声部配三度 / 六度）。"""
    q = p - min_int
    while q % 12 not in ch.pcs:
        q -= 1
    return q


# ============================================================ 乐谱对象 ====

class Track:
    def __init__(self, name, prog, bank=0, drum=False, expr=110, human=2.5):
        self.name, self.prog, self.bank, self.drum, self.human = name, prog, bank, drum, human
        self.notes = []
        self.ccs = [(0.0, 7, 100), (0.0, 11, expr), (0.0, 91, 0), (0.0, 93, 0)]
        self.progs = [(0.0, bank, prog)]

    def n(self, beat, dur, pitch, vel):
        if dur > 1e-6:
            self.notes.append([float(beat), float(dur), int(pitch), int(np.clip(round(vel), 1, 127))])

    def chord(self, beat, dur, pitches, vel):
        for p in pitches:
            self.n(beat, dur, p, vel)

    def cc(self, beat, num, val):
        self.ccs.append((float(beat), num, int(np.clip(round(val), 0, 127))))

    def ramp(self, b0, b1, v0, v1, num=11, res=0.125, curve=1.0):
        steps = max(1, int(round((b1 - b0) / res)))
        for i in range(steps + 1):
            x = i / steps
            self.cc(b0 + (b1 - b0) * x, num, v0 + (v1 - v0) * x ** curve)

    def program(self, beat, prog, bank=0):
        self.progs.append((float(beat), bank, prog))


TRACKS = {}


def mk(name, prog, **kw):
    TRACKS[name] = Track(name, prog, **kw)
    return TRACKS[name]


drums = mk('drums', 16, drum=True, expr=127, human=0.0)       # GS Power kit
bass = mk('bass', 34, expr=115, human=1.5)
gtr = mk('guitar', 30, expr=110, human=1.5)
lead = mk('lead_guitar', 29, expr=104, human=2.0)
brass = mk('brass', 61, expr=112)
horns = mk('horns', 60, expr=110)
tpt = mk('trumpet', 56, expr=112)
tbn = mk('trombone', 57, expr=112)
tuba = mk('tuba', 58, expr=110)
strf = mk('strings_fast', 48, expr=110, human=1.5)
strs = mk('strings_slow', 49, expr=100)
trem = mk('strings_trem', 44, expr=100)
cb = mk('contrabass', 43, expr=108)
choir = mk('choir', 52, expr=104)
oohs = mk('voice_oohs', 53, expr=100)
sop = mk('soprano', 52, expr=112, human=1.0)
vox = mk('solo_vox', 85, expr=100, human=1.0)
organ = mk('organ', 19, expr=100)
timp = mk('timpani', 47, expr=120, human=0.0)
taiko = mk('taiko', 116, expr=120, human=0.0)
ohit = mk('orch_hit', 55, expr=110, human=0.0)
mbox = mk('music_box', 10, expr=115, human=1.0)
cel = mk('celesta', 8, expr=110, human=1.0)
glock = mk('glockenspiel', 9, expr=110, human=1.0)
bells = mk('tubular_bells', 14, expr=110, human=0.0)
harp = mk('harp', 46, expr=110, human=1.5)
hpsi = mk('harpsichord', 6, expr=112, human=1.5)
pizz = mk('pizzicato', 45, expr=110, human=2.0)
hca = mk('harmonica', 22, expr=112, human=1.0)

# 鼓组音符（GS Power kit）
KICK, SIDE, SNARE, SNARE2 = 36, 37, 38, 40
HAT, PHAT, OHAT, TAMB = 42, 44, 46, 54
CRASH, CRASH2, RIDE, BELL, CHINA, SPLASH = 49, 57, 51, 53, 52, 55
TFL, TL, TM, TM1, TH, TH1 = 41, 43, 45, 47, 48, 50
WOOD_HI, WOOD_LO = 76, 77

# 旋律记谱：'D5:1.5 Eb5:.5 D5:1 C#5:1 | Bb4:2 A4:2'，r = 休止，| 分小节
_NRE = re.compile(r'^([A-G][#b]?)(-?\d)$')


def pitch_of(nm):
    m = _NRE.match(nm)
    return 12 * (int(m.group(2)) + 1) + NOTE[m.group(1)]


def mel(tr, sec, bar, spec, vel, legato=0.95, shift=0, accent=6):
    """把旋律写到 tr 上，返回 [(beat, dur, pitch)]。小节首拍的音 +accent 力度。"""
    out = []
    for i, bs in enumerate(spec.split('|')):
        off = 0.0
        for tok in bs.split():
            nm, d = tok.split(':')
            d = float(d)
            if nm != 'r':
                p = pitch_of(nm) + shift
                b = P(sec, bar + i, off)
                v = vel(off) if callable(vel) else vel + (accent if abs(off) < 1e-9 else 0)
                tr.n(b, d * legato, p, v)
                out.append((b, d, p))
            off += d
        assert off <= 4 + 1e-6, (sec, bar + i, off)
    return out


def mel_notes(tr, notes, vel, legato=0.95, shift=0):
    for b, d, p in notes:
        tr.n(b, d * legato, p + shift, vel)


def second_voice(tr, notes, vel, legato=0.95, shift=0, min_int=3):
    """在旋律下方配和弦内的三度 / 六度。"""
    for b, d, p in notes:
        ch = harm_at(b)
        tr.n(b, d * legato, below(p, ch, min_int) + shift, vel)


def motif(octv, bar=None):
    """动机两小节（D 所在八度 octv）。"""
    return (f'D{octv}:1.5 Eb{octv}:.5 D{octv}:1 C#{octv}:1 | '
            f'Bb{octv - 1}:2 A{octv - 1}:2')


# ============================================================ 通用声部 ====

HITS_OUT = []          # (beat, kind, strength)
FX = []                # (kind, beat, gain)：后期合成的铜锣 / 次低频 / 噪声渐强 / 低频闷击
STOPS = []             # (b_from, b_to)


def pad(tr, b0, b1, lo, hi, k, vel, min_len=1.0, prev=None, extra_low=None, legato=1.0):
    """跟随和声的长音铺底（换和弦才重新起音）。"""
    for s, e, ch in segments(b0, b1, min_len):
        v = voicing(ch, lo, hi, k, prev)
        prev = v
        notes = list(v)
        if extra_low is not None:
            notes.append(place(ch.bass, extra_low))
        tr.chord(s, (e - s) * legato, notes, vel)
    return prev


def roots(tr, b0, b1, lo, vel, legato=0.98, min_len=1.0, octave=False):
    for s, e, ch in segments(b0, b1, min_len):
        r = place(ch.bass, lo)
        tr.n(s, (e - s) * legato, r, vel)
        if octave:
            tr.n(s, (e - s) * legato, r + 12, vel - 8)


def arp(tr, b0, b1, lo, hi, step, vel, shape='updown', dur=None, extra_pcs=(), accent=10):
    """跟随和声的分解和弦（竖琴 / 羽管键琴）。"""
    for s, e, ch in segments(b0, b1, 0.5):
        pcs = set(ch.pcs) | set(extra_pcs)
        tones = [n for n in range(place(ch.bass, lo), hi + 1) if n % 12 in pcs]
        if not tones:
            continue
        if shape == 'up':
            seq = tones
        else:
            seq = tones + tones[-2:0:-1]
        for j, b in enumerate(np.arange(s, e - 1e-9, step)):
            p = seq[j % len(seq)]
            tr.n(b, dur if dur else step * 1.8, p, vel + (accent if j % 4 == 0 else 0))


def bassline(b0, b1, lo=31, vel=112, step=0.5, dur=0.42, approach=True, pattern=None):
    """八分低音（换和弦前半拍 / 一拍加经过音）。pattern 给定时只在这些拍位上弹。"""
    beats = np.arange(b0, b1 - 1e-9, step)
    if pattern is not None:
        beats = [b for b in beats if any(abs(((b - b0) % 4) - k) < 1e-6 for k in pattern)]
    for beat in beats:
        ch = harm_at(beat)
        if ch is None:
            continue
        seg = next(s for s in HARM if s[0] <= beat + 1e-9 < s[1])
        p = place(ch.bass, lo)
        v = vel if (beat % 1) < 1e-9 else vel - 12
        if approach and seg[1] - seg[0] >= 2 and seg[1] < b1 + 1e-9:
            nxt = harm_at(seg[1])
            left = seg[1] - beat
            if nxt is not None and left <= 2 * step + 1e-9:
                q = place(nxt.bass, lo)
                d = q - p
                if abs(d) >= 2:
                    p = q - int(np.sign(d)) * (1 if left <= step + 1e-9 else 2)
                elif left <= step + 1e-9:
                    p = p + 12
        bass.n(beat, dur, p, v)


def ostinato(b0, b1, lo=57, octave_up=False, vel=(100, 76), dur=0.2, tr=None):
    """弦乐十六分“绕主音”音型：根 根 X 根，X 依拍位取五音 / 六音 / 五音 / 八度。"""
    tr = tr or strf
    for beat in np.arange(b0, b1 - 1e-9, 0.25):
        ch = harm_at(beat)
        if ch is None:
            continue
        r = place(ch.root, lo)
        fifth = 6 if ch.kind == 'o7' else 7
        color = {'m': 8, '': 9, '7': 8, 'o7': 9, '5': 8, 'maj7': 9, 'm7': 8, 'sus4': 8,
                 '7sus4': 8}[ch.kind]
        bi = int(np.floor(beat + 1e-9)) % 4
        k = int(round((beat - np.floor(beat + 1e-9)) * 4))
        top = (fifth, color, fifth, 12)[bi]
        p = (r, r, r + top, r)[k]
        v = vel[0] if k == 0 else vel[1]
        tr.n(beat, dur, p, v)
        if octave_up:
            tr.n(beat, dur, p + 12, v - 10)


# 吉他 riff：(拍位, 时值, 类型, 根音偏移)；A = 强音强力和弦，M = 闷音，N = 单音 + 八度
RIFF_C = [(0, .9, 'A'), (1, .2, 'M'), (1.5, .2, 'M'), (2, .45, 'A'), (2.5, .2, 'M'),
          (3, .2, 'M'), (3.25, .2, 'M'), (3.5, .2, 'M'), (3.75, .2, 'M')]
RIFF_G = [(0, .45, 'A'), (.5, .2, 'M'), (.75, .2, 'M'), (1, .2, 'M'), (1.5, .2, 'M'),
          (1.75, .2, 'M'), (2, .45, 'A'), (2.5, .2, 'M'), (2.75, .2, 'M'), (3, .2, 'M'),
          (3.5, .2, 'M'), (3.75, .2, 'M')]
RIFF_16 = [(k, .2, 'A' if k % 2 == 0 else 'M') for k in np.arange(0, 4, 0.25)]
RIFF_MC = [(0, .45, 'A', 0), (.5, .2, 'M', 0), (.75, .2, 'M', 0), (1, .45, 'A', 1),
           (1.5, .2, 'M', 0), (1.75, .2, 'M', 0), (2, .2, 'M', 0), (2.25, .2, 'M', 0),
           (2.5, .45, 'A', 0), (3, .45, 'N', -1), (3.5, .2, 'M', 0), (3.75, .2, 'M', 0)]
RIFF_MUTE = [(k, .2, 'M') for k in np.arange(0, 4, 0.5)]
RIFF_LILT = [(0, .5, 'A'), (1.5, .5, 'A'), (3, .5, 'A')]


def guitar_bar(b, riff, vel=108, tr=None):
    tr = tr or gtr
    for item in riff:
        off, dur, kind = item[:3]
        shift = item[3] if len(item) > 3 else 0
        beat = b + off
        ch = harm_at(beat)
        if ch is None:
            continue
        if any(abs(s - beat) < 1e-9 for s, _, _ in HARM) and kind == 'M':
            kind = 'A'                                  # 换和弦的那一格改成重音
        if ch.root != 2:
            shift = 0                                   # 半音绕行只在主和弦上
        r = place(ch.root, 38) + shift
        fifth = 6 if ch.kind == 'o7' else 7
        if kind == 'A':
            tr.chord(beat, dur, [r, r + fifth, r + 12], vel)
        elif kind == 'N':
            tr.chord(beat, dur, [r, r + 12], vel - 6)
        else:
            tr.chord(beat, dur, [r, r + fifth], vel - 22)


def drum_bar(b, style, crash=False, v=1.0):
    def d(k, note, vel, dur=0.2):
        drums.n(b + k, dur, note, vel * v)
    if crash:
        d(0, CRASH, 118, 1.0)
        d(0, CRASH2, 100, 1.0)
    skip0 = crash
    if style == 'rock':
        for k in (0, .5, 1.5, 2, 2.5):
            d(k, KICK, 116 if k in (0, 2) else 98)
        for k in (1, 3):
            d(k, SNARE, 118)
        for k in np.arange(0, 4, .5):
            if not (skip0 and k == 0):
                d(k, HAT, 86 if k % 1 == 0 else 64)
    elif style == 'ride':
        for k in (0, .5, 1.5, 2, 2.5):
            d(k, KICK, 116 if k in (0, 2) else 98)
        for k in (1, 3):
            d(k, SNARE, 120)
        for k in np.arange(0, 4, .5):
            if not (skip0 and k == 0):
                d(k, BELL if k % 1 == 0 else RIDE, 84 if k % 1 == 0 else 68)
    elif style == 'heavy':            # 半速重摇滚（麦克班）
        for k, vv in ((0, 122), (.75, 100), (1.5, 110), (2.75, 100), (3.25, 96)):
            d(k, KICK, vv)
        d(2, SNARE, 124)
        d(2, SNARE2, 92)
        d(3.75, SNARE, 64)
        for k in np.arange(0, 4, .5):
            if not (skip0 and k == 0):
                d(k, BELL if k % 1 == 0 else RIDE, 88 if k % 1 == 0 else 70)
    elif style == 'gallop':
        for beat in range(4):
            for k, vv in ((0, 116), (.5, 94), (.75, 90)):
                d(beat + k, KICK, vv)
        for k in (1, 3):
            d(k, SNARE, 122)
        for k in np.arange(0, 4, .5):
            if not (skip0 and k == 0):
                d(k, RIDE, 88 if k % 1 == 0 else 72)
    elif style == 'military':         # 军鼓：重音 + 装饰十六分
        for k, vv in ((0, 118), (1.5, 96), (2, 112), (3.5, 92)):
            d(k, KICK, vv)
        for k, vv in ((0, 110), (.5, 70), (.75, 76), (1, 120), (1.5, 72), (1.75, 78),
                      (2, 108), (2.25, 68), (2.5, 72), (2.75, 80), (3, 122), (3.5, 96)):
            d(k, SNARE, vv, 0.12)
        for k in (1, 3):
            d(k, PHAT, 70)
    elif style == 'lilt':             # 3+3+2
        d(0, KICK, 116)
        d(1.5, KICK, 102)
        d(1, SIDE, 80)
        d(3, SNARE, 112)
        for k in np.arange(0, 4, .5):
            if not (skip0 and k == 0):
                d(k, HAT, 84 if k in (0, 1.5, 3) else 56)
        for k in (1.5, 3.5):
            d(k, TAMB, 62)
    elif style == 'light':
        d(0, KICK, 108)
        d(2, KICK, 100)
        for k in (1, 3):
            d(k, SIDE, 86)
        for k in np.arange(0, 4, .5):
            if not (skip0 and k == 0):
                d(k, HAT, 66 if k % 1 == 0 else 48)
    elif style == 'phantom':          # 半速：大鼓 + 军鼓在第 3 拍 + 低通鼓
        d(0, KICK, 118)
        d(1.5, KICK, 100)
        d(2, SNARE, 112)
        d(2, TFL, 96)
        d(3.5, TL, 90)
        for k in (0, 1, 2, 3):
            if not (skip0 and k == 0):
                d(k, BELL, 60)
    elif style == 'half':             # 盟主
        for k in (0, 1.5, 2.5):
            d(k, KICK, 124)
        d(2, SNARE, 127)
        d(2, SNARE2, 110)
        for k in (1, 3):
            d(k, CRASH2, 84, 0.5)
        d(2, CHINA, 92, 0.5)
    elif style == 'double':           # 总攻
        for k in np.arange(0, 4, .25):
            d(k, KICK, 110 if k % 1 == 0 else 86)
        for k in range(4):
            d(k + .5, SNARE, 122)
            if not (skip0 and k == 0):
                d(k, CHINA if k % 2 == 0 else CRASH, 88, 0.3)


def clear(tracks, b0, b1):
    for tr in tracks:
        tr.notes = [n for n in tr.notes if not (b0 - 1e-9 <= n[0] < b1 - 1e-9)]


def fill_toms(b, beats=1.0, v=1.0):
    """b 之前 beats 拍的通鼓下行填充（落点在 b）。"""
    clear([drums], b - beats, b)
    seq = (TH1, TH1, TH, TH, TM, TM, TM1, TM1, TL, TL, TFL, TFL)
    grid = np.arange(b - beats, b - 1e-9, 0.25)
    for i, k in enumerate(grid):
        note = seq[int(i * len(seq) / len(grid))]
        drums.n(k, 0.2, note, (98 + 26 * i / max(1, len(grid) - 1)) * v)
    drums.n(b - 0.5, 0.2, KICK, 110 * v)
    drums.n(b - 0.25, 0.2, KICK, 116 * v)


def cymbal_swell(b0, b1, v0=24, v1=100, note=CRASH2, step=0.125):
    grid = np.arange(b0, b1 - 1e-9, step)
    for i, k in enumerate(grid):
        drums.n(k, step, note, v0 + (v1 - v0) * (i / max(1, len(grid) - 1)) ** 1.5)


def scale_run(b0, b1, end_pitch, step, vel0, vel1, tr=None, pcs=None, dur=None):
    """结束在 end_pitch 的上行音阶（默认用 b1 处的调：D 和声小调 / E 和声小调）。"""
    tr = tr or strf
    if pcs is None:
        key = 'E' if B0['phantom'] <= b1 < B0['slam'] + 1e-9 else 'D'
        pcs = {'D': (2, 4, 5, 7, 9, 10, 1), 'E': (4, 6, 7, 9, 11, 0, 3)}[key]
    grid = list(np.arange(b0, b1 - 1e-9, step))
    notes = [n for n in range(20, end_pitch + 1) if n % 12 in pcs][-len(grid):]
    for i, (b, p) in enumerate(zip(grid, notes)):
        tr.n(b, dur or step * 0.95, p, vel0 + (vel1 - vel0) * i / max(1, len(grid) - 1))


def timp_pitch(pc):
    p = place(pc, 38)
    return p if p <= 45 else p - 12


def tutti(b, strength=1.0, kind='tutti', gong=0.0, sub=0.0, boom=0.35, cym=True):
    """全奏重音：管弦击打 + 低音铜管 + 定音鼓 + 太鼓 + 镲。"""
    ch = harm_at(b)
    s = strength
    r = place(ch.root, 50)
    ohit.chord(b, 0.9, [r, r + 12], 104 + 20 * s)
    timp.n(b, 1.0, timp_pitch(ch.bass), 110 + 17 * s)
    taiko.n(b, 1.0, 36, 108 + 19 * s)
    taiko.n(b, 1.0, 48, 96 + 18 * s)
    tbn.chord(b, 1.2, voicing(ch, 45, 62, 3), 110 + 16 * s)
    tuba.n(b, 1.2, place(ch.bass, 33), 108 + 16 * s)
    horns.chord(b, 1.0, voicing(ch, 55, 70, 3), 104 + 16 * s)
    strf.chord(b, 0.35, voicing(ch, 55, 79, 4), 104 + 18 * s)
    if cym:
        drums.n(b, 1.0, CRASH, 110 + 16 * s)
        drums.n(b, 1.0, CHINA, 90 + 20 * s)
        drums.n(b, 0.3, KICK, 127)
    if gong:
        FX.append(('gong', b, gong))
    if sub:
        FX.append(('sub', b, sub))
    elif boom:
        FX.append(('boom', b, boom * s))
    HITS_OUT.append((b, kind, s))


def riser(b, beats=2.0, strength=1.0, soft=False):
    """在 b 之前 beats 拍内推起：军鼓十六 -> 三十二分、镲渐强、定音鼓滚奏、弦乐上行、噪声渐强。"""
    b0 = b - beats
    HITS_OUT.append((b0, 'riser', 0.5 * strength))
    FX.append(('riser', b, (0.5 if soft else 1.0) * strength, beats))
    if soft:
        cymbal_swell(b0, b, 20, 70)
        return
    # 滚奏在落点前 1/8 拍收住（“吸一口气”），让重音真正冒出来
    breath = b - 0.125
    clear([drums], b0, b)
    cymbal_swell(b0, breath, 20, 88)
    grid = list(np.arange(b0, b0 + beats / 2, 0.25)) + list(np.arange(b0 + beats / 2, breath - 1e-9, 0.125))
    for i, k in enumerate(grid):
        drums.n(k, 0.1, SNARE, 64 + 44 * (i / len(grid)) ** 1.2)
    for k in np.arange(b0, b - 1e-9, 1.0):
        drums.n(k, 0.2, KICK, 108)
    tgrid = np.arange(b0, breath - 1e-9, 0.125)
    for i, k in enumerate(tgrid):
        timp.n(k, 0.1, timp_pitch(harm_at(k).bass) if harm_at(k) else 45, 58 + 40 * i / len(tgrid))
    scale_run(b0, b, 85, 0.25 if beats >= 2 else 0.125, 80, 110)


def drop(b, strength=1.0):
    """次低频下潜 + 大鼓 / 定音鼓低音。"""
    ch = harm_at(b)
    FX.append(('sub', b, strength))
    drums.n(b, 0.4, KICK, 127)
    drums.n(b, 1.0, CRASH, 100 + 20 * strength)
    timp.n(b, 1.5, timp_pitch(ch.bass), 124)
    taiko.n(b, 1.0, 36, 124)
    ohit.chord(b, 0.6, [place(ch.root, 40), place(ch.root, 52)], 100 + 20 * strength)
    cb.n(b, 3.5, place(ch.bass, 28), 118)
    HITS_OUT.append((b, 'drop', strength))


def stop(b):
    """重音后本拍余下部分全体停顿，下一拍再进。"""
    e = math.floor(b + 1e-9) + 1
    tutti(b, 0.8, cym=True)
    STOPS.append((b + 0.3, e))
    HITS_OUT.append((b + 0.3, 'stop', 0.8))


def section_downbeat(sec, strength):
    HITS_OUT.append((P(sec), 'downbeat', strength))


# ============================================================ 各段 ========

def write_prologue():
    s = 'prologue'
    b0, sil = P(s), P(s, 6, 3)
    # 八音盒：第 1 小节末拍弱起，第 2-3 小节动机，第 4-5 小节回应，第 6 小节动机压缩成问句
    mb = mel(mbox, s, 1, 'r:3 A5:1 | D6:1.5 Eb6:.5 D6:1 C#6:1 | Bb5:2 A5:2 | '
             'Bb5:1.5 C6:.5 Bb5:1 A5:1 | G5:1.5 Bb5:.5 A5:1 C#6:1 | D6:.75 Eb6:.25 D6:.5 C#6:1.5',
             lambda off: 84 if off % 1 else 94, legato=1.0)
    mel_notes(cel, [n for n in mb if n[0] >= P(s, 4)], 72, legato=1.0, shift=-12)
    mel_notes(cel, [n for n in mb if P(s, 2) <= n[0] < P(s, 4)], 44, legato=1.0)
    # 竖琴：八分琶音（Dm 加九音），第 5 小节改十六分，第 6 小节属七和弦上行刮奏
    arp(harp, b0, P(s, 2), 38, 76, 0.5, 64, extra_pcs=(4,))
    arp(harp, P(s, 2), P(s, 5), 38, 72, 0.5, 60)
    arp(harp, P(s, 5), P(s, 6), 38, 81, 0.25, 70, dur=0.6)
    gl = [n for n in range(45, 94) if n % 12 in (9, 1, 4, 7)]
    grid = np.arange(P(s, 6, 0.5), sil - 0.13, 0.125)
    for i, b in enumerate(grid):
        harp.n(b, 0.5, gl[int(i * len(gl) / len(grid))], 62 + 30 * i / len(grid))
    # 温暖的低音弦乐：空五度 -> 跟随和声；第 5 小节加高声部渐强
    strs.cc(b0, 11, 70)
    strs.chord(b0, 8 - 0.02, [38, 45], 84)
    pad(strs, P(s, 3), sil, 43, 62, 3, 84)
    strs.ramp(P(s, 4), P(s, 6, 2.5), 70, 118, curve=1.4)
    pad(strs, P(s, 5), sil, 62, 79, 3, 88)
    cb.cc(b0, 11, 90)
    roots(cb, b0, sil, 31, 70)
    # 人声 pad：第 2 小节渐入，第 5 小节膨胀
    oohs.cc(b0, 11, 40)
    oohs.ramp(P(s, 2), P(s, 3), 40, 80)
    pad(oohs, P(s, 2), sil, 57, 74, 3, 72)
    oohs.ramp(P(s, 5), P(s, 6, 2.5), 80, 124, curve=1.3)
    # 第 6 小节：震音弦乐属七和弦渐强，悬在最后一拍前
    trem.cc(P(s, 5, 2), 11, 30)
    trem.chord(P(s, 5, 2), sil - P(s, 5, 2), voicing(chord('A7'), 57, 79, 4), 90)
    trem.ramp(P(s, 5, 2), sil, 30, 120, curve=1.5)
    # 钟摆：拨弦根音 / 五音交替 + 木鱼滴答（第 1-5 小节）
    for b in np.arange(b0, P(s, 6), 1.0):
        ch = harm_at(b)
        k = int(round(b - b0)) % 2
        p = place(ch.bass, 43) if k == 0 else place((ch.bass + 7) % 12, 38)
        pizz.n(b, 0.5, p, 64 if k == 0 else 54)
        drums.n(b, 0.1, WOOD_HI if k == 0 else WOOD_LO, 44 if k == 0 else 36)
    section_downbeat(s, 0.3)
    HITS_OUT.append((P(s, 5), 'downbeat', 0.5))          # 肯帕雷拉在火焰中现身


def write_title():
    s = 'title'
    b0, b1 = P(s), P(s, 3)
    tutti(b0, 1.0, kind='slam', gong=1.0, sub=0.5)
    section_downbeat(s, 1.0)
    # 动机：铜管组 D5、小号 D5、圆号 D4、弦乐 D6
    brass.cc(b0, 11, 118)
    m = mel(brass, s, 1, motif(5), 122)
    mel_notes(tpt, m, 110)
    mel_notes(horns, m, 114, shift=-12)
    mel_notes(strf, m, 104, legato=1.0, shift=12)
    # 铺底：弦乐、合唱
    strs.cc(b0, 11, 112)
    pad(strs, b0, b1, 57, 81, 4, 110)
    choir.cc(b0, 11, 118)
    pad(choir, b0, b1, 55, 76, 4, 112)
    for sgs, e, ch in segments(b0, b1, 1.0):
        tbn.chord(sgs, min(e - sgs, 1.5) * 0.9, voicing(ch, 45, 62, 3), 116)
        cb.n(sgs, (e - sgs) * 0.95, place(ch.bass, 31), 110)
        timp.n(sgs, 0.5, timp_pitch(ch.bass), 112)
    ohit.chord(P(s, 1, 1.5), 0.4, [51, 63], 100)
    # 律动：第 1 小节前两拍让重拍余响，第 3 拍起吉他 / 贝斯 / 鼓进入
    gtr.chord(b0, 1.9, [38, 45, 50], 118)
    for off, dur, kind in RIFF_C:
        if off >= 2:
            guitar_bar(b0, [(off, dur, kind)], 110)
    guitar_bar(P(s, 2), RIFF_C, 110)
    bassline(P(s, 1, 2), b1)
    bass.n(b0, 1.9, 38, 118)
    for k in (2, 2.5):
        drums.n(b0 + k, 0.2, KICK, 116)
    drums.n(b0 + 3, 0.2, SNARE, 118)
    drums.n(b0 + 3.5, 0.2, SNARE, 104)
    drum_bar(P(s, 2), 'rock', crash=True)
    fill_toms(P('mcburn'), 1.0)
    ostinato(P(s, 1, 2), b1, vel=(96, 72))
    taiko.n(P(s, 2), 0.5, 36, 118)


# ---- 各幕 -------------------------------------------------------------------

def write_mcburn():
    s = 'mcburn'
    # 鼓：前四小节半速重摇滚，第 5-6 小节加倍
    for bar in range(1, 7):
        style = 'heavy' if bar <= 4 else 'rock'
        drum_bar(P(s, bar), style, crash=bar in (1,), v=0.85 if bar == 3 else 1.0)
    # 吉他 riff（第 3 小节只闷音，让出台词）
    for bar in range(1, 7):
        riff = RIFF_MC if bar in (1, 2, 4) else (RIFF_MUTE if bar == 3 else RIFF_G)
        guitar_bar(P(s, bar), riff, 112 if bar != 3 else 100)
    for bar in (1, 2, 4):
        guitar_bar(P(s, bar), [x for x in RIFF_MC if x[2] in ('A', 'N')], 96, tr=lead)
    bassline(P(s), P(s, 7), vel=112)
    # 低音铜管：动机四倍扩大（懒洋洋的压倒性力量）
    tbn.cc(P(s), 11, 118)
    m = mel(tbn, s, 1, 'D3:3 Eb3:1 | D3:2 C#3:2 | Bb2:4 | A2:4', 112, legato=0.97)
    mel_notes(tuba, m, 100, legato=0.97, shift=-12)
    mel_notes(horns, m, 100, legato=0.97, shift=12)
    tbn.ramp(P(s, 3), P(s, 3, 3), 96, 118)
    # 铺底
    strs.cc(P(s), 11, 84)
    pad(strs, P(s), P(s, 5), 52, 72, 3, 80)
    # 第 5-6 小节：铜管组 + 小号高八度动机，合唱加入，弦乐八度十六分
    m2 = mel(brass, s, 5, motif(5), 124)
    mel_notes(tpt, m2, 116)
    mel_notes(lead, m2, 100, shift=-12)
    choir.cc(P(s, 5), 11, 116)
    pad(choir, P(s, 5), P(s, 7), 55, 76, 4, 112)
    strs.cc(P(s, 5), 11, 108)
    pad(strs, P(s, 5), P(s, 7), 62, 84, 4, 104)
    ostinato(P(s, 5), P(s, 7), octave_up=True, vel=(104, 80))
    for sg, e, ch in segments(P(s, 5, 0.5), P(s, 7), 1.0):
        tbn.chord(sg, min(e - sg, 1.5) * 0.9, voicing(ch, 45, 62, 3), 112)


def write_leonhardt():
    s = 'leonhardt'
    # 第 1-2 小节：孤独的口琴陈述动机（哈梅尔的回忆），长音弦乐 + 轻竖琴
    hca.cc(P(s), 11, 112)
    mel(hca, s, 1, motif(5), 104, legato=1.0)
    strs.cc(P(s), 11, 76)
    pad(strs, P(s), P(s, 3), 50, 74, 4, 78)
    roots(cb, P(s), P(s, 3), 31, 70)
    arp(harp, P(s), P(s, 3), 50, 77, 0.5, 50, extra_pcs=())
    drums.n(P(s), 1.0, CRASH2, 64)                        # 余响里轻轻的镲
    cymbal_swell(P(s, 2, 1), P(s, 3), 16, 84)
    for k, vv in ((3.25, 70), (3.5, 86), (3.75, 100)):
        drums.n(P(s, 2, k), 0.12, SNARE, vv)
    # 第 3 小节起：跳弓弦乐 + 太鼓 + 小号英雄 / 暗色旋律
    strf.cc(P(s, 3), 11, 116)
    ostinato(P(s, 3), P(s, 5), lo=55, vel=(108, 84), dur=0.14)
    ostinato(P(s, 5), P(s, 7), lo=55, octave_up=True, vel=(112, 88), dur=0.14)
    for bar in range(3, 7):
        b = P(s, bar)
        for k, note, vv in ((0, 36, 122), (.75, 36, 96), (1.5, 36, 108), (2, 48, 110),
                            (2.5, 36, 98), (3, 36, 112), (3.5, 48, 100)):
            taiko.n(b + k, 0.4, note, vv * (0.88 if bar == 3 else 1.0))
        style = 'light' if bar == 3 else 'rock'
        drum_bar(b, style, crash=(bar == 3), v=0.9 if bar == 3 else 1.0)
        if bar >= 4:
            guitar_bar(b, RIFF_C, 106)
    bassline(P(s, 3), P(s, 7), vel=108)
    tpt.cc(P(s, 3), 11, 112)
    t = mel(tpt, s, 3, 'G4:1 Bb4:1 D5:1.5 C5:.5 | Bb4:1.5 G4:.5 A4:1 C#5:1 | '
            'D5:1.5 E5:.5 F5:1 A5:1 | G5:1.5 F5:.5 E5:1 C#5:1',
            lambda off: 104, legato=0.93)
    second_voice(horns, [x for x in t if x[0] >= P(s, 5)], 104)
    mel_notes(brass, [x for x in t if x[0] >= P(s, 5)], 108)
    strs.cc(P(s, 3), 11, 92)
    pad(strs, P(s, 3), P(s, 7), 55, 76, 4, 92)
    for sg, e, ch in segments(P(s, 4), P(s, 7), 2.0):
        tbn.chord(sg, min(e - sg, 1.5) * 0.9, voicing(ch, 45, 62, 3), 104)


def write_arianrhod():
    s = 'arianrhod'
    for bar in range(1, 7):
        style = 'military' if bar <= 4 else 'gallop'
        drum_bar(P(s, bar), style, crash=(bar == 1), v=0.88 if bar == 3 else 1.0)
        riff = RIFF_G if bar != 3 else RIFF_MUTE
        guitar_bar(P(s, bar), riff, 110 if bar != 3 else 96)
        timp.n(P(s, bar), 0.6, timp_pitch(harm_at(P(s, bar)).bass), 108)
    bassline(P(s), P(s, 7), vel=110)
    # 骑士号角：小号主旋律，圆号三 / 六度，铜管组在第 5-6 小节加入
    fan = mel(tpt, s, 1, 'A4:.75 A4:.25 D5:1 D5:.75 E5:.25 F5:1 | E5:1.5 D5:.5 C5:1 G4:1 | '
              'D5:3 F5:1 | E5:.75 E5:.25 E5:.75 F5:.25 E5:1 C#5:1 | '
              'F5:1.5 G5:.5 F5:1 E5:1 | D5:1 F5:1 E5:1 C#5:1',
              lambda off: 116 if off % 1 == 0 else 100, legato=0.9)
    horns.cc(P(s), 11, 118)
    second_voice(horns, fan, 110, legato=0.92)
    mel_notes(horns, [x for x in fan if x[0] < P(s, 3)], 100, legato=0.92, shift=-12)
    mel_notes(brass, [x for x in fan if x[0] >= P(s, 5)], 118, legato=0.92)
    for sg, e, ch in segments(P(s), P(s, 7), 1.0):
        tbn.chord(sg, min(e - sg, 1.5) * 0.9, voicing(ch, 45, 62, 3), 108)
    strf.cc(P(s), 11, 100)
    ostinato(P(s), P(s, 5), vel=(90, 68))
    ostinato(P(s, 5), P(s, 7), octave_up=True, vel=(100, 78))
    strs.cc(P(s), 11, 90)
    pad(strs, P(s), P(s, 7), 55, 76, 4, 88)
    choir.cc(P(s, 5), 11, 112)
    pad(choir, P(s, 5), P(s, 7), 55, 76, 4, 108)


def write_bleublanc():
    s = 'bleublanc'
    for bar in range(1, 7):
        drum_bar(P(s, bar), 'lilt', crash=(bar == 1), v=0.85 if bar == 3 else 1.0)
        guitar_bar(P(s, bar), RIFF_LILT, 96 if bar < 5 else 106)
    bassline(P(s), P(s, 7), vel=108, pattern=(0, 1.5, 3), dur=0.9, approach=False)
    # 拨弦：3+3+2 的“嘭-嚓-嚓”
    for bar in range(1, 7):
        b = P(s, bar)
        for k, role in ((0, 'b'), (.5, 'c'), (1, 'c'), (1.5, 'f'), (2, 'c'), (2.5, 'c'), (3, 'b'),
                        (3.5, 'c')):
            ch = harm_at(b + k)
            if role == 'b':
                pizz.n(b + k, 0.5, place(ch.bass, 43), 100)
            elif role == 'f':
                pizz.n(b + k, 0.5, place((ch.root + 7) % 12, 43), 92)
            else:
                pizz.chord(b + k, 0.4, voicing(ch, 55, 67, 2), 70 if bar != 3 else 60)
    # 羽管键琴：带波音的主旋律（动机的 D-Eb-D 本身就是波音）
    hm = mel(hpsi, s, 1, 'D5:.25 Eb5:.25 D5:.5 A4:.5 F5:.5 E5:.5 D5:.5 C#5:.5 D5:.5 | '
             'G5:.25 A5:.25 G5:.5 D5:.5 Bb5:.5 A5:.5 G5:.5 F#5:.5 G5:.5 | '
             'C6:1 Bb5:.5 G5:.5 E5:1 r:1 | '
             'F5:.25 G5:.25 A5:.5 C6:.5 A5:.5 G5:.5 E5:.5 C#5:.5 A4:.5 | '
             'D6:.25 Eb6:.25 D6:1 C#6:.5 D6:.5 A5:.5 F5:.5 A5:.5 | '
             'Bb5:.5 A5:.5 G5:.5 F5:.5 E5:.5 C#5:.5 A4:1',
             lambda off: 112 if off in (0, 1.5, 3) else 96, legato=0.85)
    mel_notes(hpsi, [x for x in hm if x[0] >= P(s, 5)], 90, legato=0.85, shift=-12)
    arp(hpsi, P(s, 3), P(s, 4), 50, 67, 0.25, 70, shape='up', dur=0.22)
    mel_notes(glock, [x for x in hm if x[0] >= P(s, 5)], 96, legato=0.9)
    # 钟琴装饰：小节末的上行花音
    for bar, top in ((1, 86), (2, 91), (4, 88)):
        b = P(s, bar, 3.5)
        ch = harm_at(b)
        tones = [n for n in range(top - 14, top + 1) if n % 12 in ch.pcs][-4:]
        for i, p in enumerate(tones):
            glock.n(b + 0.125 * i, 0.2, p, 80 + 8 * i)
    strs.cc(P(s), 11, 80)
    pad(strs, P(s), P(s, 7), 57, 76, 3, 76)
    horns.cc(P(s, 5), 11, 100)
    pad(horns, P(s, 5), P(s, 7), 53, 67, 3, 96)


def write_vita():
    s = 'vita'
    for bar in range(1, 7):
        style = 'rock' if bar != 3 else 'ride'
        drum_bar(P(s, bar), style, crash=(bar == 1), v=0.86 if bar == 3 else 1.0)
        riff = RIFF_C if bar in (1, 2, 4) else (None if bar == 3 else RIFF_G)
        if riff:
            guitar_bar(P(s, bar), riff, 106)
    gtr.chord(P(s, 3), 3.8, [43, 50, 55], 92)
    bassline(P(s), P(s, 7), vel=108)
    # 女高音（合唱单线 + Solo Vox 叠加），翱翔
    sop.cc(P(s), 11, 118)
    v = mel(sop, s, 1, 'A4:1 D5:1.5 E5:.5 F5:1 | D5:1 F5:1 Bb5:2 | A5:1.5 G5:.5 D5:2 | '
            'E5:1 A5:1 C#6:2 | D6:2 E6:2 | D6:2 D#6:2', 116, legato=1.0)
    mel_notes(vox, v, 96, legato=1.0)
    # 竖琴十六分琶音
    harp.cc(P(s), 11, 118)
    arp(harp, P(s), P(s, 7), 50, 86, 0.25, 84, dur=0.5)
    choir.cc(P(s), 11, 96)
    pad(choir, P(s), P(s, 7), 52, 72, 4, 92)
    choir.ramp(P(s, 5), P(s, 6), 96, 118)
    strs.cc(P(s), 11, 94)
    pad(strs, P(s), P(s, 7), 57, 79, 4, 92)
    strf.cc(P(s), 11, 96)
    ostinato(P(s, 4), P(s, 7), vel=(88, 66))
    brass.cc(P(s, 5), 11, 110)
    pad(brass, P(s, 5), P(s, 7), 55, 72, 4, 108, min_len=2.0)
    for sg, e, ch in segments(P(s, 5), P(s, 7), 2.0):
        tbn.chord(sg, (e - sg) * 0.9, voicing(ch, 45, 62, 3), 106)
        tuba.n(sg, (e - sg) * 0.9, place(ch.bass, 33), 100)


def write_phantom():
    s = 'phantom'
    b0, b1 = P(s), P(s, 5)
    drop(b0, 1.0)
    section_downbeat(s, 0.9)
    # 半速鼓，第 3 小节起通鼓 / 军鼓推起
    for bar in range(1, 5):
        drum_bar(P(s, bar), 'phantom', crash=False, v=0.9 if bar < 3 else 1.0)
    for i, k in enumerate(np.arange(P(s, 3), P(s, 4), 0.5)):
        drums.n(k + 0.25, 0.15, SNARE, 50 + 30 * i / 8)
    for bar in range(1, 5):
        for k, vv in ((0, 116), (1.5, 96), (2, 108), (3.5, 96)):
            taiko.n(P(s, bar, k), 0.4, 36, vv * (0.9 if bar < 3 else 1.0))
    # 八音盒动机（E 小调），钢片琴在第 3 小节起低八度叠加
    mb = mel(mbox, s, 1, 'E6:1.5 F6:.5 E6:1 D#6:1 | C6:2 B5:2 | C6:1.5 D6:.5 C6:1 B5:1 | '
             'F#5:1 A5:1 B5:1 D#6:1', 104, legato=1.0)
    mel_notes(cel, [n for n in mb if n[0] >= P(s, 3)], 84, legato=1.0, shift=-12)
    mel_notes(horns, [n for n in mb if n[0] < P(s, 3)], 84, legato=0.98, shift=-24)
    # 暗色的宽 pad：弦乐、人声、管风琴（柔）
    strs.cc(b0, 11, 90)
    pad(strs, b0, b1, 47, 71, 4, 96)
    strs.ramp(P(s, 3), b1, 90, 122)
    oohs.cc(b0, 11, 96)
    pad(oohs, b0, b1, 55, 74, 4, 96)
    oohs.ramp(P(s, 3), b1, 96, 124)
    organ.cc(b0, 11, 76)
    pad(organ, b0, b1, 52, 71, 3, 90, extra_low=40)
    organ.ramp(P(s, 3), b1, 76, 110)
    roots(cb, P(s, 1, 3.5), b1, 36, 104)
    # 吉他长音，第 3 小节起震音渐强；贝斯全音符 -> 八分
    gtr.chord(b0, 3.8, [40, 47, 52], 112)
    gtr.cc(b0, 11, 110)
    for i, b in enumerate(np.arange(P(s, 3), b1, 0.25)):
        ch = harm_at(b)
        r = place(ch.root, 38)
        gtr.chord(b, 0.2, [r, r + 7], 60 + 50 * i / 32)
    bass.n(b0, 7.8, 28, 116)
    bassline(P(s, 3), b1, vel=104)
    # 上行弦乐推入盟主
    strf.cc(P(s, 3), 11, 110)
    scale_run(P(s, 3), P(s, 4), 83, 0.5, 80, 100, dur=0.45)


def write_grandmaster():
    s = 'grandmaster'
    b0, b1 = P(s), P(s, 5)
    tutti(b0, 1.0, kind='slam', gong=0.75)
    section_downbeat(s, 1.0)
    for bar in range(1, 5):
        drum_bar(P(s, bar), 'half', crash=True)
        for k, vv in ((0, 127), (2, 124), (2.5, 104), (3.5, 96)):
            taiko.n(P(s, bar, k), 0.4, 36, vv)
        taiko.n(P(s, bar, 2), 0.4, 48, 112)
    # 动机四倍扩大：合唱四部（女高音唱旋律）+ 铜管 + 管风琴高声部
    melody = 'E5:3 F5:1 | E5:2 D#5:2 | C5:4 | B4:4'
    choir.cc(b0, 11, 127)
    top = mel(brass, s, 1, melody, 124, legato=0.97)
    mel_notes(tpt, top, 112, legato=0.97)
    mel_notes(horns, top, 118, legato=0.97, shift=-12)
    prev = None
    for b, d, p in top:
        for sg, e, ch in segments(b, b + d, 0.5):
            v = voicing(ch, 48, p, 4, prev, top=p)
            prev = v
            choir.chord(sg, e - sg, v, 122)
    for b, d, p in top:
        bells.n(b, d, p, 104)
    organ.cc(b0, 11, 124)
    pad(organ, b0, b1, 55, 84, 5, 118, extra_low=40)
    strs.cc(b0, 11, 118)
    pad(strs, b0, b1, 69, 91, 4, 112)
    tbn.cc(b0, 11, 118)
    pad(tbn, b0, b1, 45, 62, 3, 120)
    for sg, e, ch in segments(b0, b1, 1.0):
        tuba.n(sg, e - sg, place(ch.bass, 33), 116)
        cb.n(sg, e - sg, place(ch.bass, 31), 112)
        timp.n(sg, 0.8, timp_pitch(ch.bass), 124)
        r = place(ch.root, 38)
        gtr.chord(sg, (e - sg) * 0.95, [r, r + 7, r + 12], 116)
    strf.cc(b0, 11, 116)
    ostinato(b0, b1, lo=64, octave_up=False, vel=(110, 86))
    bassline(b0, b1, vel=110)


def write_assault():
    s = 'assault'
    b0, b1 = P(s), P(s, 5)
    tutti(b0, 0.9)
    section_downbeat(s, 0.9)
    for bar in range(1, 4):
        drum_bar(P(s, bar), 'double', crash=True)
    for k in np.arange(0, 2, .25):
        drums.n(P(s, 4, k), 0.12, KICK, 104 + 10 * k)
        drums.n(P(s, 4, k), 0.12, SNARE, 90 + 12 * k)
    drums.n(P(s, 4), 0.5, CRASH, 118)
    for b in np.arange(b0, P(s, 3), 0.5):
        taiko.n(b, 0.3, 36, 112 if b % 1 == 0 else 92)
    for i, b in enumerate(np.arange(P(s, 3), P(s, 4, 2), 0.25)):
        taiko.n(b, 0.2, 36 if i % 2 == 0 else 48, 96 + 26 * i / 24)
    for bar in range(1, 5):
        guitar_bar(P(s, bar), RIFF_16 if bar < 4 else RIFF_16[:8], 114)
    bassline(b0, P(s, 4, 2), vel=112)
    bass.n(P(s, 4, 2), 1.9, 35, 116)
    # “盘蛇”：铜管 + 主音吉他 + 圆号，每个和弦一组 根-上邻-根-下导
    UP = {'Em': 1, 'F': 2, 'D/F#': 2, 'G': 2, 'E7/G#': 1, 'Am': 2, 'A#o7': 1, 'B': 1}
    for sg, e, ch in segments(b0, P(s, 4, 2)):
        top = place(ch.root, 72)
        for j, b in enumerate(np.arange(sg, e - 1e-9, 0.5)):
            p = (top, top + UP[ch.name], top, top - 1)[j % 4]
            brass.n(b, 0.46, p, 122 if j % 4 == 0 else 108)
            tpt.n(b, 0.46, p, 108 if j % 4 == 0 else 96)
            lead.n(b, 0.48, p - 12, 100)
            horns.n(b, 0.46, p - 12, 110)
    for tr, p, v in ((brass, 83, 124), (tpt, 83, 116), (horns, 71, 118), (lead, 71, 104)):
        tr.n(P(s, 4, 2), 1.98, p, v)
    brass.ramp(P(s, 4, 2), b1, 96, 127)
    # 弦乐十六分琶音 + 结尾上行
    strf.cc(b0, 11, 118)
    for sg, e, ch in segments(b0, P(s, 4, 2)):
        tones = sorted({place(pc, 64) for pc in ch.pcs})
        tones = tones + [t + 12 for t in tones]
        seq = [0, 1, 2, 3, 4, 3, 2, 1]
        for j, b in enumerate(np.arange(sg, e - 1e-9, 0.25)):
            strf.n(b, 0.2, tones[seq[j % 8] % len(tones)], 104 if j % 4 == 0 else 86)
    organ.cc(b0, 11, 116)
    pad(organ, b0, b1, 57, 79, 4, 112, extra_low=40)
    choir.cc(b0, 11, 118)
    pad(choir, b0, b1, 55, 79, 4, 116)
    choir.ramp(P(s, 4), b1, 110, 127)
    strs.cc(b0, 11, 110)
    pad(strs, b0, b1, 64, 88, 4, 108)
    for sg, e, ch in segments(b0, b1):
        tbn.chord(sg, (e - sg) * 0.9, voicing(ch, 45, 62, 3), 110)
        cb.n(sg, e - sg, place(ch.bass, 31), 112)
        timp.n(sg, 0.5, timp_pitch(ch.bass), 116)


def write_slam():
    s = 'slam'
    b0, b1 = P(s), P(s, 3)
    tutti(b0, 1.0, kind='slam', gong=1.15, sub=0.7)
    section_downbeat(s, 1.0)
    drums.n(b0, 1, CHINA, 112)
    em = chord('Em')
    ring = b1 - b0 - 0.3
    for tr, lo, hi, k, v in ((organ, 52, 83, 5, 118), (choir, 52, 79, 4, 122),
                             (strs, 52, 86, 5, 116), (tbn, 45, 62, 3, 120),
                             (horns, 55, 71, 3, 118), (brass, 62, 79, 4, 120)):
        tr.cc(b0, 11, 124)
        tr.chord(b0, ring, voicing(em, lo, hi, k), v)
        tr.ramp(b0 + 0.5, b1 - 0.3, 122, 6, curve=0.6)
    for tr, p in ((tuba, 40), (cb, 40), (bass, 28)):
        tr.cc(b0, 11, 124)
        tr.n(b0, ring, p, 120)
        tr.ramp(b0 + 0.5, b1 - 0.3, 122, 6, curve=0.6)
    organ.chord(b0, ring, [40], 110)
    gtr.cc(b0, 11, 118)
    gtr.chord(b0, 5, [40, 47, 52], 120)
    gtr.ramp(b0 + 0.5, b0 + 5, 118, 0, curve=0.6)
    roll = np.arange(b0 + 1, b1 - 0.5, 0.125)
    for i, b in enumerate(roll):
        timp.n(b, 0.1, 40, 90 * (1 - i / len(roll)) ** 1.6 + 8)
    bells.n(b0, 4, 76, 110)
    bells.n(b0, 4, 64, 100)


def write_epilogue():
    s = 'epilogue'
    b0 = P(s)
    snap = P(s, 2, 2)
    section_downbeat(s, 0.3)
    # 八音盒独自复述动机
    mel(mbox, s, 1, 'D6:1.5 Eb6:.5 D6:1 C#6:1 | Bb5:1.85', lambda off: 80, legato=1.0)
    oohs.cc(b0, 11, 64)
    pad(oohs, b0, snap - 0.12, 57, 74, 3, 60)
    strs.cc(b0, 11, 62)
    pad(strs, b0, snap - 0.12, 50, 69, 3, 64)
    roots(cb, b0, snap - 0.12, 31, 56)
    for k, p in enumerate((38, 45, 50, 53)):
        harp.n(b0 + k * 0.5, 2.0, p, 50)
    # 响指之后：洒牌（钢片琴 + 竖琴 + 钟琴 A 大三和弦下行瀑布）
    a_tones = [n for n in range(57, 94) if n % 12 in (9, 1, 4)][::-1]
    grid = np.arange(snap + 0.25, P(s, 3) - 0.1, 0.125)
    for i, b in enumerate(grid):
        p = a_tones[int(i * len(a_tones) / len(grid))]
        cel.n(b, 0.4, p, 78 - 10 * i / len(grid))
        if i % 2 == 0:
            harp.n(b, 0.8, p - 12, 64)
        if i < 5:
            glock.n(b, 0.3, p + 12 if p + 12 < 100 else p, 64 - 6 * i)
    # 皮卡第三度：D 大三和弦，响到结束
    b3 = P(s, 3)
    ring = END - b3
    dmaj = [38, 45, 50, 54, 57, 62, 66, 69, 74]
    for i, p in enumerate(dmaj):
        harp.n(b3 + 0.08 * i, ring - 0.08 * i, p, 70)
    mbox.n(b3 + 0.7, ring - 0.7, 90, 78)
    mbox.n(b3 + 0.7, ring - 0.7, 86, 70)
    cel.chord(b3, ring, [74, 78, 81], 60)
    strs.cc(b3, 11, 84)
    strs.chord(b3, ring, [50, 57, 62, 66], 76)
    oohs.cc(b3, 11, 80)
    oohs.chord(b3, ring, [62, 66, 69], 70)
    cb.n(b3, ring, 38, 60)


# ---- 重音钩子 ----------------------------------------------------------------

DEFAULT_ACCENTS = [(5, 0.0, 'hit'), (7, 0.0, 'riser')]


def scene_accents(name):
    """读 src/scenes/<name>.py 的 ACCENTS；优先 ast 读字面量（不执行场景代码）。"""
    path = SRC / 'scenes' / f'{name}.py'
    if not path.exists():
        return None
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) else [])
            if any(isinstance(t, ast.Name) and t.id == 'ACCENTS' for t in targets):
                return [tuple(a) for a in ast.literal_eval(node.value)]
        return None
    except (ValueError, SyntaxError):
        pass
    try:
        mod = importlib.import_module(f'scenes.{name}')
        acc = getattr(mod, 'ACCENTS', None)
        return None if acc is None else [tuple(a) for a in acc]
    except Exception as e:           # 场景模块坏了不能拖垮配乐
        print(f'[score] scenes.{name} ACCENTS unreadable: {e}', file=sys.stderr)
        return None


def apply_accent(sec, bar, beat, kind, strength=1.0):
    b = P(sec, bar, beat)
    if not (B0[sec] - 1e-9 <= b <= B0[sec] + SEC[sec].bars * 4 + 1e-9):
        print(f'[score] accent {(sec, bar, beat, kind)} out of range, ignored', file=sys.stderr)
        return
    nxt_calm = abs(b - B0['leonhardt']) < 1e-6          # 进入口琴独奏：只用镲 / 噪声渐强
    if kind == 'hit':
        if b - 1 >= B0[sec] - 1e-9 and sec_of(b - 1) == sec_of(b):
            fill_toms(b, 1.0)
        tutti(b, strength)
    elif kind == 'stop':
        stop(b)
    elif kind == 'riser':
        riser(b, 2.0, strength, soft=nxt_calm)
    elif kind == 'drop':
        drop(b, strength)
    else:
        print(f'[score] unknown accent kind {kind!r}', file=sys.stderr)


def write_accents():
    for name in story.ORDER:
        acc = scene_accents(name)
        if name in story.ACTS:
            for bar, beat, kind in (acc if acc is not None else DEFAULT_ACCENTS):
                apply_accent(name, bar, beat, kind)
        elif acc:
            for bar, beat, kind in acc:
                b = P(name, bar, beat)
                if any(abs(h[0] - b) < 1e-6 and h[1] in ('slam', 'tutti', 'drop') for h in HITS_OUT):
                    continue                    # 内置重音已经在这里
                apply_accent(name, bar, beat, kind)
    for name in story.ACTS:
        section_downbeat(name, 0.7)
    # 盟主、总攻之前的推起（非幕段落的内置转场）
    riser(P('grandmaster'), 2.0, 1.0)
    riser(P('assault'), 2.0, 0.9)
    riser(P('slam'), 2.0, 1.0)


# ---- 后处理：静音窗、停顿 ----------------------------------------------------------

def cut_window(b0, b1, tail=0.02, keep=()):
    """[b0,b1) 内不起音；跨过 b0 的音截断在 b0 - tail。"""
    for tr in TRACKS.values():
        if tr.name in keep:
            continue
        out = []
        for n in tr.notes:
            s, d = n[0], n[1]
            if b0 - 1e-9 <= s < b1 - 1e-9:
                continue
            if s < b0 and s + d > b0 - tail:
                n[1] = b0 - tail - s
                if n[1] <= 0.01:
                    continue
            out.append(n)
        tr.notes = out


SILENCES = []          # (b0, b1, floor_db) 给母带门限


def post():
    sil0, sil1 = P('prologue', 6, 3), P('title')
    cut_window(sil0, sil1)
    SILENCES.append((sil0, sil1, -60.0, 0.02, 0.003))
    snap = P('epilogue', 2, 2)
    cut_window(snap - 0.12, snap + 0.22)
    SILENCES.append((snap - 0.1, snap + 0.2, -14.0, 0.03, 0.03))
    for b0, b1 in STOPS:
        cut_window(b0, b1, tail=0.0)
        SILENCES.append((b0, b1, -20.0, 0.015, 0.004))


_COMPOSED = False


def compose():
    global _COMPOSED
    if _COMPOSED:
        return
    write_prologue()
    write_title()
    write_mcburn()
    write_leonhardt()
    write_arianrhod()
    write_bleublanc()
    write_vita()
    write_phantom()
    write_grandmaster()
    write_assault()
    write_slam()
    write_epilogue()
    write_accents()
    post()
    _COMPOSED = True


def crash_times():
    out = []
    for b, d, p, v in sorted(drums.notes):
        if p in (CRASH, CHINA) and v >= 100:
            if not out or b - out[-1][0] > 0.3:
                out.append((b, v))
    return out


def hits():
    """[(秒, kind, strength)]，按时间排序。只算谱，不渲染。"""
    compose()
    ev = [(secs(b), kind, round(float(st), 3)) for b, kind, st in HITS_OUT]
    ev += [(secs(b), 'crash', round(v / 127, 3)) for b, v in crash_times()]
    ev = sorted(set((round(t, 4), k, s) for t, k, s in ev))
    return ev


# ============================================================ MIDI ========

def beat2tick(b):
    return int(round(b * TPB))


PROTECT = None


def protected():
    global PROTECT
    if PROTECT is None:
        PROTECT = {beat2tick(b) for b, _, _ in HITS_OUT} | {beat2tick(B0[n]) for n in story.ORDER}
    return PROTECT


def track_events(tr, ch):
    rng = np.random.default_rng(zlib.crc32(tr.name.encode()))
    ev = []
    for b, bank, prog in tr.progs:
        t = beat2tick(b)
        if not tr.drum:
            ev.append((t, 0, mido.Message('control_change', channel=ch, control=0, value=bank)))
        ev.append((t, 0, mido.Message('program_change', channel=ch, program=prog)))
    for b, num, val in tr.ccs:
        ev.append((beat2tick(b), 1, mido.Message('control_change', channel=ch, control=num, value=val)))
    notes = sorted(tr.notes, key=lambda n: (n[2], n[0]))
    prot = protected()
    for i, (b, d, p, v) in enumerate(notes):
        on = beat2tick(b)
        if tr.human > 0 and on not in prot and on % TPB != 0:
            on = max(0, on + int(np.clip(rng.normal(0, tr.human), -6, 6)))
        vel = int(np.clip(v + (int(rng.integers(-3, 4)) if tr.human > 0 else 0), 1, 127))
        off = beat2tick(b + d)
        if i + 1 < len(notes) and notes[i + 1][2] == p:
            off = min(off, beat2tick(notes[i + 1][0]) - 1)
        if off <= on:
            continue
        ev.append((on, 3, mido.Message('note_on', channel=ch, note=p, velocity=vel)))
        ev.append((off, 2, mido.Message('note_off', channel=ch, note=p, velocity=0)))
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev


def meta_track(tail_beats=8):
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage('track_name', name='Ouroboros PV v3', time=0))
    meta.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    ev = []
    for n in story.ORDER:
        ev.append((beat2tick(B0[n]), mido.MetaMessage('set_tempo', tempo=int(round(6e7 / SEC[n].bpm)))))
        ev.append((beat2tick(B0[n]), mido.MetaMessage('marker', text=n)))
    for n, key in (('prologue', 'Dm'), ('phantom', 'Em'), ('epilogue', 'Dm')):
        ev.append((beat2tick(B0[n]), mido.MetaMessage('key_signature', key=key)))
    last = 0
    for t, msg in sorted(ev, key=lambda e: e[0]):
        meta.append(msg.copy(time=t - last))
        last = t
    meta.append(mido.MetaMessage('end_of_track', time=beat2tick(END + tail_beats) - last))
    return meta


def make_midi(tracks, combined=True, tail_beats=8):
    """combined：每个声部一轨，旋律声部占 0-8,10-15 通道，超过 15 个的放到 MIDI 端口 1。"""
    mid = mido.MidiFile(type=1, ticks_per_beat=TPB)
    mid.tracks.append(meta_track(tail_beats))
    mel_i = 0
    for tr in tracks:
        if tr.drum:
            port, ch = 0, 9
        elif combined:
            port, k = divmod(mel_i, 15)
            ch = k if k < 9 else k + 1
            mel_i += 1
        else:
            port, ch = 0, 0
        mt = mido.MidiTrack()
        mt.append(mido.MetaMessage('track_name', name=tr.name, time=0))
        mt.append(mido.MetaMessage('midi_port', port=port, time=0))
        last = 0
        for t, _, msg in track_events(tr, ch):
            mt.append(msg.copy(time=t - last))
            last = t
        mt.append(mido.MetaMessage('end_of_track', time=max(0, beat2tick(END + tail_beats) - last)))
        mid.tracks.append(mt)
    return mid


def check_midi_timing(mid):
    """用 mido 的 tempo 换算核对段首时刻与 story 一致。"""
    tempo, t, tick = 500000, 0.0, 0
    marks = {}
    for msg in mido.merge_tracks([mid.tracks[0]]):
        tick += msg.time
        t += mido.tick2second(msg.time, TPB, tempo)
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        if msg.type == 'marker':
            marks[msg.text] = t
    err = max(abs(marks[n] - SEC[n].t0) for n in story.ORDER)
    assert err < 2e-4, err
    return err


# ====================================================== 渲染与混音 ========

def render_stem(tr, workdir):
    mid_path = workdir / f'{tr.name}.mid'
    wav_path = workdir / f'{tr.name}.wav'
    make_midi([tr], combined=False).save(mid_path)
    subprocess.run(['fluidsynth', '-ni', '-q', '-R', '0', '-C', '0', '-g', '0.6',
                    '-r', str(SR), '-O', 'float', '-T', 'wav',
                    '-o', 'synth.polyphony=1024', '-o', 'synth.cpu-cores=1',
                    '-F', str(wav_path), str(SF2), str(mid_path)],
                   check=True, capture_output=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sr, x = wavfile.read(wav_path)
    assert sr == SR
    x = x.astype(np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    out = np.zeros((N + SR * 4, 2))
    m = min(len(x), len(out))
    out[:m] = x[:m]
    return out


def sos_filter(x, kind, fc, order=2):
    sos = butter(order, fc, kind, fs=SR, output='sos')
    return sosfilt(sos, x, axis=0)


def shelf(x, fc, gain_db, kind='low'):
    """RBJ 搁架 EQ。"""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / SR
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    cw, sa = np.cos(w0), 2 * np.sqrt(A) * alpha
    if kind == 'low':
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    else:
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)


def peaking(x, fc, gain_db, q=1.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / SR
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
    return lfilter(np.array(b) / a[0], np.array(a) / a[0], x, axis=0)


def active_level(x):
    """分轨“演奏时”的电平：50 ms 窗 RMS 的 90 百分位（dBFS）。"""
    mono = x.mean(1)
    w = int(0.05 * SR)
    k = len(mono) // w
    r = np.sqrt((mono[:k * w].reshape(k, w) ** 2).mean(1) + 1e-20)
    db = 20 * np.log10(r)
    act = db[db > db.max() - 40]
    return float(np.percentile(act, 90))


def pan_stereo(x, pan):
    th = (pan + 1) * np.pi / 4
    l, r = np.cos(th) * np.sqrt(2), np.sin(th) * np.sqrt(2)
    return np.stack([x[:, 0] * l, x[:, 1] * r], 1)


def haas_widen(x, ms=12.0):
    d = int(ms / 1000 * SR)
    mono = x.mean(1)
    right = np.concatenate([np.zeros(d), mono[:-d]])
    right = peaking(right, 2500, -2.0, 0.8)
    return np.stack([mono, right], 1)


def hall_ir(seconds=3.2, rt60=2.3, predelay=0.022, seed=7):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = np.zeros((n + int(predelay * SR), 2))
    for c in range(2):
        noise = rng.standard_normal(n)
        env = np.exp(-6.9 * t / rt60) * np.clip(t / 0.012, 0, 1)
        bright = sos_filter(noise, 'low', 7000)
        dark = sos_filter(noise, 'low', 1800)
        mix = np.exp(-t / 0.5)
        ir[int(predelay * SR):, c] = (bright * mix + dark * (1 - mix)) * env
    for dt, g in ((0.011, .5), (0.017, .4), (0.023, .35), (0.031, .3), (0.041, .22), (0.053, .18)):
        i = int(dt * SR)
        ir[i, 0] += g * rng.choice([-1, 1])
        ir[int(i * 1.07), 1] += g * rng.choice([-1, 1])
    ir = sos_filter(ir, 'high', 150)
    return ir / np.sqrt((ir ** 2).sum(0).mean())


def gong(length=7.0, seed=11, f0=73.42):
    """合成大锣（tam-tam）：非谐分音，高频分音延迟绽放。"""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    freqs = np.concatenate([[f0, f0 * 1.52, f0 * 2.03, f0 * 2.61],
                            np.exp(rng.uniform(np.log(180), np.log(4500), 90))])
    for i, f in enumerate(freqs):
        amp = (1.0 if i < 4 else 0.45) * (100 / f) ** 0.45
        bloom = 0.02 + 0.5 * min(1.0, f / 3000) * rng.uniform(0.5, 1.2)
        dec = 5.5 * (200 / f) ** 0.35 + rng.uniform(-0.3, 0.3)
        e = (1 - np.exp(-t / bloom)) * np.exp(-t / max(dec, 0.6))
        wob = 1 + 0.0025 * np.sin(2 * np.pi * rng.uniform(0.3, 2.0) * t + rng.uniform(0, 6.28))
        ph = 2 * np.pi * np.cumsum(f * wob) / SR + rng.uniform(0, 6.28)
        pan = rng.uniform(-0.6, 0.6)
        s = amp * e * np.sin(ph)
        out[:, 0] += s * (1 - pan)
        out[:, 1] += s * (1 + pan)
    thump = np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.18)
    noise = sos_filter(rng.standard_normal(n), 'band', [300, 3000]) * np.exp(-t / 0.05)
    out += (0.8 * thump + 0.25 * noise)[:, None]
    return out / np.abs(out).max()


def sub_drop(length=2.8, f_end=36.71, f_start=95.0, decay=0.9):
    """次低频下潜：D1 正弦 + 少量谐波。"""
    n = int(length * SR)
    t = np.arange(n) / SR
    f = f_end + (f_start - f_end) * np.exp(-t / 0.09)
    ph = 2 * np.pi * np.cumsum(f) / SR
    e = np.clip(t / 0.004, 0, 1) * np.exp(-t / decay)
    x = np.tanh(1.6 * np.sin(ph)) * e
    x = sos_filter(x, 'low', 180, 4)
    return np.stack([x, x], 1) / np.abs(x).max()


def noise_riser(length, seed=5, gap=0.035):
    """噪声渐强（反向镲的感觉），在终点前 gap 秒收掉，给重音留出瞬态。"""
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / n
    out = np.zeros((n, 2))
    for c in range(2):
        w = rng.standard_normal(n)
        lo = sos_filter(w, 'band', [400, 3000])
        hi = sos_filter(w, 'high', 3000)
        out[:, c] = lo * t ** 2.2 + 0.8 * hi * t ** 4
    g0, f = int(gap * SR), int(0.012 * SR)
    out[n - g0:] = 0
    out[n - g0 - f:n - g0] *= np.linspace(1, 0, f)[:, None]
    return out / np.abs(out).max()


def add_at(buf, sig, t0, gain):
    i = int(round(t0 * SR))
    if i < 0:
        sig, i = sig[-i:], 0
    m = min(len(sig), len(buf) - i)
    if m > 0:
        buf[i:i + m] += sig[:m] * gain


# ---- 响度（ITU-R BS.1770-4）----

def _kweight():
    fs = SR
    G, Q, fc = 3.99984385397, 0.7071752369554193, 1681.9744509555319
    A = 10 ** (G / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    cw = np.cos(w0)
    b1 = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * cw),
          A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
    a1 = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * cw),
          (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    Q, fc = 0.5003270373253953, 38.13547087613982
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    cw = np.cos(w0)
    b2 = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    a2 = [1 + alpha, -2 * cw, 1 - alpha]
    return (np.array(b1) / a1[0], np.array(a1) / a1[0]), (np.array(b2) / a2[0], np.array(a2) / a2[0])


def lufs(x):
    (b1, a1), (b2, a2) = _kweight()
    y = lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)
    w, hop = int(0.4 * SR), int(0.1 * SR)
    p2 = np.cumsum(np.concatenate([np.zeros((1, 2)), y ** 2]), axis=0)
    starts = np.arange(0, len(y) - w, hop)
    z = (p2[starts + w] - p2[starts]) / w
    lk = -0.691 + 10 * np.log10(z.sum(1) + 1e-20)
    g = z[lk > -70]
    rel = -0.691 + 10 * np.log10(g.sum(1).mean() + 1e-20) - 10
    g2 = z[(lk > -70) & (lk > rel)]
    return -0.691 + 10 * np.log10(g2.sum(1).mean() + 1e-20)


def compressor(x, thr_db, ratio, attack=0.015, release=0.2, knee=6.0, block=64):
    mono = np.sqrt((x ** 2).mean(1))
    k = len(mono) // block
    lv = np.sqrt((mono[:k * block].reshape(k, block) ** 2).mean(1) + 1e-20)
    db = 20 * np.log10(lv)
    over = db - thr_db
    gr = np.where(over <= -knee / 2, 0,
                  np.where(over >= knee / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    dt = block / SR
    aa, ar = np.exp(-dt / attack), np.exp(-dt / release)
    sm = np.zeros(k)
    g = 0.0
    for i in range(k):
        c = aa if gr[i] > g else ar
        g = c * g + (1 - c) * gr[i]
        sm[i] = g
    gain = 10 ** (-np.interp(np.arange(len(mono)), np.arange(k) * block + block / 2, sm) / 20)
    return x * gain[:, None]


def limiter(x, ceiling_db, lookahead=0.005, release=0.08, block=32):
    """前瞻峰值限幅：4 倍过采样检测。"""
    ceil = 10 ** (ceiling_db / 20)
    up = np.abs(resample_poly(x, 4, 1, axis=0)).max(1)
    pk = up.reshape(-1, 4).max(1)[:len(x)]
    k = int(np.ceil(len(pk) / block))
    pk = np.concatenate([pk, np.zeros(k * block - len(pk))]).reshape(k, block).max(1)
    la = max(1, int(round(lookahead * SR / block)))
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-12))
    need_la = np.array([need[i:i + la + 1].min() for i in range(k)])
    ar = np.exp(-(block / SR) / release)
    g = np.ones(k)
    cur = 1.0
    for i in range(k):
        cur = need_la[i] if need_la[i] < cur else ar * cur + (1 - ar) * need_la[i]
        g[i] = cur
    gain = np.interp(np.arange(len(x)), np.arange(k) * block, g)
    return x * gain[:, None]


# 分轨：目标“演奏电平”（相对 dB）、声像、混响发送、EQ
MIX = {
    'drums':          (0.0,   0.0,  0.10, 'drums'),
    'bass':           (-5.0,  0.0,  0.00, 'bass'),
    'guitar':         (-5.0,  0.0,  0.05, 'guitar'),
    'lead_guitar':    (-9.5,  0.3,  0.18, 'guitar'),
    'brass':          (-3.0, -0.15, 0.28, 'brass'),
    'horns':          (-5.5,  0.35, 0.32, 'brass'),
    'trumpet':        (-4.0,  0.2,  0.28, 'brass'),
    'trombone':       (-6.0, -0.35, 0.26, 'low'),
    'tuba':           (-11.0, 0.05, 0.20, 'low'),
    'strings_fast':   (-5.0, -0.45, 0.25, 'strings'),
    'strings_slow':   (-7.0,  0.45, 0.35, 'strings'),
    'strings_trem':   (-8.0,  0.3,  0.35, 'strings'),
    'contrabass':     (-11.0, 0.25, 0.20, 'low'),
    'choir':          (-5.0,  0.0,  0.40, 'choir'),
    'voice_oohs':     (-6.0,  0.0,  0.45, 'choir'),
    'soprano':        (-2.0,  0.05, 0.38, 'solo'),
    'solo_vox':       (-9.0, -0.05, 0.38, 'solo'),
    'organ':          (-7.0,  0.0,  0.35, 'organ'),
    'timpani':        (-5.0,  0.1,  0.30, 'low'),
    'taiko':          (-6.0, -0.1,  0.22, 'taiko'),
    'orch_hit':       (-6.0,  0.0,  0.30, 'hit'),
    'music_box':      (-3.0, -0.1,  0.45, 'bright'),
    'celesta':        (-8.0,  0.25, 0.45, 'bright'),
    'glockenspiel':   (-9.0,  0.3,  0.40, 'bright'),
    'tubular_bells':  (-8.0, -0.2,  0.45, 'bright'),
    'harp':           (-5.0, -0.3,  0.40, 'harp'),
    'harpsichord':    (-3.5, -0.2,  0.30, 'harpsi'),
    'pizzicato':      (-6.0,  0.2,  0.30, 'pizz'),
    'harmonica':      (-2.5,  0.1,  0.38, 'solo'),
}


def eq(x, kind):
    if kind == 'drums':
        x = sos_filter(x, 'high', 30)
        x = peaking(x, 400, -2.0, 1.0)
        x = shelf(x, 9000, 1.5, 'high')
    elif kind == 'guitar':
        x = sos_filter(x, 'high', 90)
        x = sos_filter(x, 'low', 7500)
        x = peaking(x, 250, -2.0, 1.0)
        x = peaking(x, 2800, 1.5, 1.0)
    elif kind == 'bass':
        x = sos_filter(x, 'high', 32)
        x = sos_filter(x, 'low', 4500)
    elif kind == 'brass':
        x = sos_filter(x, 'high', 80)
        x = peaking(x, 300, -1.5, 1.0)
    elif kind == 'strings':
        x = sos_filter(x, 'high', 55)
        x = peaking(x, 280, -1.5, 1.0)
    elif kind == 'choir':
        x = sos_filter(x, 'high', 110)
        x = shelf(x, 6000, 1.5, 'high')
    elif kind == 'solo':
        x = sos_filter(x, 'high', 160)
        x = peaking(x, 3000, 1.0, 1.0)
    elif kind == 'organ':
        x = sos_filter(x, 'high', 35)
        x = peaking(x, 300, -2.0, 0.9)
    elif kind == 'low':
        x = sos_filter(x, 'high', 30)
        x = peaking(x, 250, -1.5, 1.0)
    elif kind == 'taiko':
        x = sos_filter(x, 'high', 42)
    elif kind == 'hit':
        x = sos_filter(x, 'high', 80)
    elif kind == 'bright':
        x = sos_filter(x, 'high', 250)
    elif kind == 'harp':
        x = sos_filter(x, 'high', 70)
    elif kind in ('harpsi', 'pizz'):
        x = sos_filter(x, 'high', 70)
    return x


def conductor():
    """指挥推子（dB，按全局拍插值）：序章轻而渐强，各幕第 3 小节让出台词、第 5 小节顶上去，谢幕轻。"""
    pts = [(P('prologue'), -7.0), (P('prologue', 2), -6.5), (P('prologue', 4), -5.0),
           (P('prologue', 5), -3.5), (P('prologue', 6, 2.8), -2.0),
           (P('title') - 0.01, -2.0), (P('title'), 1.5), (P('title', 2), 0.0)]
    for a in story.ACTS:
        head = -3.0 if a == 'leonhardt' else 0.0         # 口琴独奏的两小节明显轻下来
        pts += [(P(a), head), (P(a, 3) - 0.01, head), (P(a, 3), -1.5), (P(a, 4), -0.5),
                (P(a, 5) - 0.01, -0.3), (P(a, 5), 1.0), (P(a, 6), 0.0), (P(a, 7) - 0.02, 0.0)]
    pts += [(P('phantom'), -0.5), (P('phantom', 3), -1.0), (P('phantom', 5) - 0.01, 0.5),
            (P('grandmaster'), 1.0), (P('assault') - 0.01, 1.0), (P('assault'), 0.5),
            (P('slam') - 0.01, 1.0), (P('slam'), 2.0), (P('slam', 2), 0.0),
            (P('slam', 3) - 0.01, -4.0), (P('epilogue'), -6.0), (END, -6.0)]
    pts.sort()
    t = np.arange(N) / SR
    xs = [secs(b) for b, _ in pts]
    db = np.interp(t, xs, [v for _, v in pts])
    return 10 ** (db / 20)


def mix_stems(stems, dump=None):
    L = N + SR * 4
    dry = np.zeros((L, 2))
    send = np.zeros((L, 2))
    for name, x in stems.items():
        if np.abs(x).max() < 1e-7:
            continue
        level, pan, snd, kind = MIX[name]
        lv = active_level(x)
        g = 10 ** ((level - 12.0 - lv) / 20)
        y = eq(x, kind) * g
        if name == 'guitar':
            y = haas_widen(y, 13.0)
        elif abs(pan) > 1e-3:
            y = pan_stereo(y, pan)
        dry += y
        send += y * snd
        if dump is not None:
            wavfile.write(dump / f'{name}.wav', SR, y[:N].astype(np.float32))
    gg, sd = gong(), sub_drop()
    boom = sub_drop(1.2, f_end=41.2, f_start=110.0, decay=0.35)
    ref = 10 ** (-14.0 / 20)
    for kind, b, gain, *rest in FX:
        t0 = secs(b)
        if kind == 'gong':
            add_at(send, gg, t0, ref * gain * 0.8)
            add_at(dry, gg, t0, ref * gain)
        elif kind == 'sub':
            add_at(dry, sd, t0, 10 ** (-13.0 / 20) * gain)
        elif kind == 'boom':
            add_at(dry, boom, t0, 10 ** (-13.0 / 20) * gain)
        elif kind == 'riser':
            beats = rest[0]
            length = secs(b) - secs(b - beats)
            nr = noise_riser(length, seed=int(b))
            add_at(dry, nr, t0 - length, 10 ** (-24.0 / 20) * gain)
            add_at(send, nr, t0 - length, 10 ** (-26.0 / 20) * gain)
    ir = hall_ir()
    wet = np.stack([fftconvolve(send[:, 0], ir[:, 0]), fftconvolve(send[:, 1], ir[:, 1])], 1)[:L]
    return dry + 0.55 * wet


def gates(y):
    """静音窗（序章末拍、谢幕响指、停顿）；首尾淡入淡出。"""
    t = np.arange(len(y)) / SR
    g = np.ones(len(y))
    for b0, b1, floor_db, down_s, up_s in SILENCES:
        s0, s1 = secs(b0), secs(b1)
        floor = 10 ** (floor_db / 20)
        down = np.clip((t - s0) / down_s, 0, 1)
        up = np.clip((t - (s1 - up_s)) / up_s, 0, 1)
        gw = 1 - (1 - floor) * down * (1 - up)
        g *= np.where((t > s0 - 0.001) & (t < s1 + 0.001), gw, 1.0)
    g *= np.clip(t / 0.005, 0, 1)
    fade0 = DURATION - 1.4
    g *= np.where(t > fade0, np.cos(np.clip((t - fade0) / (DURATION - fade0), 0, 1) * np.pi / 2) ** 1.5,
                  1.0)
    return y * g[:, None]


def master(x):
    x = sos_filter(x, 'high', 28, 2)
    x = shelf(x, 90, 0.5, 'low')
    x = peaking(x, 220, -1.0, 0.8)
    x = x * 10 ** ((-14.0 - lufs(x[:N])) / 20)
    x = compressor(x, thr_db=-18.0, ratio=2.0, attack=0.02, release=0.25)
    x = x[:N] * conductor()[:, None]
    gain_db = TARGET_LUFS - lufs(x)
    y = pre = x
    for _ in range(6):
        pre = x * 10 ** (gain_db / 20)
        y = gates(limiter(pre, CEILING_DB))
        err = TARGET_LUFS - lufs(y)
        gain_db += err
        if abs(err) < 0.05:
            break
    over = 20 * np.log10(np.abs(pre).max()) - CEILING_DB
    print(f'limiter: pre-limit peak exceeds ceiling by {over:.1f} dB')
    return y


def to_int16(y, seed=3):
    rng = np.random.default_rng(seed)
    d = (rng.random(y.shape) - rng.random(y.shape)) / 32768.0
    z = np.clip(np.round((y + d) * 32767), -32768, 32767).astype(np.int16)
    z[-1] = 0
    return z


def print_harmony():
    for n in story.ORDER:
        rows = []
        for bar in range(1, SEC[n].bars + 1):
            names = []
            for s, e, ch in HARM:
                if P(n, bar) - 1e-9 <= s < P(n, bar + 1) - 1e-9:
                    off = s - P(n, bar)
                    lab = ch.name if ch else '(静)'
                    names.append(lab if off == 0 else f'{lab}@{off:g}')
            rows.append(' '.join(names) or '-')
        print(f'{n:<12}' + ' | '.join(rows))


def stem_report(stems):
    """分轨检查：原始峰值、演奏电平、每段电平（找静音 / 过弱的音色）。"""
    print(f'{"stem":<15}{"peak":>7}{"active":>8}  per-section active RMS dBFS (- = no notes)')
    for name, x in stems.items():
        tr = TRACKS[name]
        pk = 20 * np.log10(np.abs(x).max() + 1e-12)
        lv = active_level(x) if np.abs(x).max() > 0 else -200
        cols = []
        for n in story.ORDER:
            has = any(B0[n] <= b < B0[n] + SEC[n].bars * 4 for b, *_ in tr.notes)
            if not has:
                cols.append('   -')
                continue
            w = int(0.05 * SR)
            mono = x.mean(1)
            wins = set()
            for b, d, *_ in tr.notes:                  # 只看有音符在响的窗口
                if B0[n] <= b < B0[n] + SEC[n].bars * 4:
                    t0, t1 = secs(b), secs(b) + min(max(d * beat_len(b), 0.1), 1.0)
                    wins.update(range(int(t0 * SR) // w, int(t1 * SR) // w + 1))
            r = [20 * np.log10(np.sqrt((mono[i * w:(i + 1) * w] ** 2).mean()) + 1e-10) for i in sorted(wins)]
            cols.append(f'{np.percentile(r, 90):4.0f}')
        flag = '  CLIP?' if pk > 0 else ''
        print(f'{name:<15}{pk:7.1f}{lv:8.1f}  ' + ' '.join(cols) + flag)


def main(argv):
    args = [a for a in argv[1:]]
    out = Path(args[0] if args and not args[0].startswith('--') else 'out/music.wav')
    stems_dir = None
    if '--stems' in args:
        stems_dir = Path(args[args.index('--stems') + 1])
        stems_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which('fluidsynth'):
        sys.exit('需要 fluidsynth：brew install fluid-synth')
    if not SF2.exists():
        sys.exit(f'缺少音色库 {SF2}')
    compose()
    if '--quiet' not in args:
        print_harmony()
    out.parent.mkdir(parents=True, exist_ok=True)
    mid = make_midi(list(TRACKS.values()), tail_beats=0)
    err = check_midi_timing(mid)
    mid.save(out.with_suffix('.mid'))
    print(f'wrote {out.with_suffix(".mid")} (tempo map max error {err * 1000:.3f} ms)')
    with tempfile.TemporaryDirectory() as td, ThreadPoolExecutor(6) as ex:
        live = [tr for tr in TRACKS.values() if tr.notes]
        stems = dict(zip([tr.name for tr in live], ex.map(lambda tr: render_stem(tr, Path(td)), live)))
    if '--quiet' not in args:
        stem_report(stems)
    y = master(mix_stems(stems, stems_dir))
    assert len(y) == N
    wavfile.write(out, SR, to_int16(y))
    pk = 20 * np.log10(np.abs(y).max())
    print(f'wrote {out} ({len(y) / SR:.4f} s = story.DURATION {DURATION:.4f})  '
          f'peak {pk:.2f} dBFS  {lufs(y):.2f} LUFS')


if __name__ == '__main__':
    main(sys.argv)
