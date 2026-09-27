"""v3 最终混音：out/music.wav（score.py）+ 按分镜排好的音效 -> out/soundtrack.wav。

音效来源：
  1. 这里的 GLOBAL_CUES（序章环境声、响指、标题、转场扑克牌、谢幕……）
  2. 各场景模块 scenes/<段>.py 的 CUES = [(小节, 拍, 名, 增益dB)]，名字先查模块自己的 SFX 字典，再查 sfx.py
  3. 各场景 SAYS 里的台词 -> 打字音
重的音效（冲击、斩击、响指）落下时把音乐压低一点（ducking），让它"打进"音乐里而不是叠在上面。
"""
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

import sfx as S
import story
from director import TRANSITIONS, CARD_IN, load_scene
from lines import LINES

SR = S.SR
ROOT = Path(__file__).resolve().parent.parent
HEAVY = ('impact', 'big_impact', 'snap', 'fire_burst', 'slash', 'lance', 'whip', 'crunch', 'bass_drop')
WIDE = ('cards', 'sparkle', 'glass', 'applause', 'crackle', 'wind', 'shimmer')
CPS = 14   # 与 ui.dialog 默认打字速度一致
FX_GAIN = 0.3        # 音效总线：让音乐领跑，音效在重拍上"打进去"
TARGET_LUFS = -14.0  # 成片响度
CEILING_DB = -1.0


def db(g):
    return 10 ** (g / 20)


def global_cues():
    """(秒, 名, 线性增益, 声像)"""
    P, TT, E = story.SECS['prologue'], story.SECS['title'], story.SECS['epilogue']
    ev = [(P.T(1), 'boom_low', 0.35, 0), (P.T(1, 2), 'wind', 0.25, -0.3), (P.T(3, 2), 'wind', 0.2, 0.3),
          (TT.T(1), 'snap', 1.0, 0), (TT.T(1) + 0.03, 'impact', 1.0, 0), (TT.T(1) + 0.05, 'curtain', 0.7, 0),
          (TT.T(1) + 0.1, 'shimmer', 0.5, 0)]
    for (a, b), kind in TRANSITIONS.items():
        if kind == 'card':
            t1 = story.SECS[a].t1
            ev += [(t1 - CARD_IN, 'card_flip', 0.55, -0.4), (t1 - 0.05, 'swish', 0.35, 0.2)]
    return ev


def scene_cues():
    ev, custom = [], {}
    for name in story.ORDER:
        mod = load_scene(name)
        if mod is None:
            continue
        sec = story.SECS[name]
        for fn_name, fn in getattr(mod, 'SFX', {}).items():
            custom[(name, fn_name)] = fn
        for c in getattr(mod, 'CUES', []):
            bar, beat, snd, gdb = (list(c) + [0.0])[:4]
            pan = c[4] if len(c) > 4 else 0.0
            ev.append((sec.T(bar, beat), (name, snd), db(gdb), pan))
        for key, bar, beat, style, *_ in getattr(mod, 'SAYS', []):
            t0 = sec.T(bar, beat)
            cps = CPS if style == 'dialog' else 16
            for i, ch in enumerate(LINES[key][0]):
                if ch not in '「」，。—…“”、！？——':
                    ev.append((t0 + i / cps, 'blip', 0.18, 0))
    return ev, custom


_cache = {}


def sound(name, custom):
    if name in _cache:
        return _cache[name]
    if isinstance(name, tuple):
        sec, snd = name
        x = custom[(sec, snd)]() if (sec, snd) in custom else S.sfx(snd)
    else:
        x = S.sfx(name)
    x = np.asarray(x, np.float64)
    _cache[name] = x
    return x


def load_wav(path):
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, w.getnchannels()) / 32768
    return x.astype(np.float64)


def main(out, stems=None):
    n = int(round(story.DURATION * SR))
    music = np.zeros((n, 2))
    mpath = ROOT / 'out' / 'music.wav'
    if mpath.exists():
        m = load_wav(mpath)[:n]
        music[:len(m)] = m
    else:
        print('[mix] out/music.wav missing, SFX only')
    L = np.zeros(n + SR * 4)
    R = np.zeros(n + SR * 4)
    duck = np.ones(n + SR * 4)
    ev, custom = scene_cues()
    ev = global_cues() + ev
    for when, name, gain, pan in ev:
        x = sound(name, custom)
        base = name[1] if isinstance(name, tuple) else name
        if base in WIDE and x.ndim == 1:
            pan = pan or 0.3
        i = int(round(when * SR))
        if x.ndim == 1:
            x = np.stack([x * np.sqrt(0.5 * (1 - pan)), x * np.sqrt(0.5 * (1 + pan))], 1)
        if i < 0:
            x, i = x[-i:], 0
        k = min(len(x), len(L) - i)
        L[i:i + k] += x[:k, 0] * gain * FX_GAIN
        R[i:i + k] += x[:k, 1] * gain * FX_GAIN
        if base in HEAVY:
            d = int(0.45 * SR)
            duck[i:i + d] = np.minimum(duck[i:i + d], 1 - 0.35 * gain * np.exp(-np.arange(d) / (0.16 * SR))[:len(duck[i:i + d])])
    fx = np.stack([L, R], 1)
    ir_n = int(1.3 * SR)
    ir = S.bp(S.rng.normal(0, 1, ir_n), 300, 7000) * np.exp(-S.tt(ir_n) / 0.28)
    ir /= np.sqrt(np.sum(ir ** 2))
    fx = fx + 0.18 * np.stack([fftconvolve(fx[:, c], ir)[:len(fx)] for c in (0, 1)], 1)
    fx = fx[:n]
    if stems:
        np.save(stems + '_fx.npy', fx.astype(np.float32))
        np.save(stems + '_music.npy', (music * duck[:n, None]).astype(np.float32))
    mix = music * duck[:n, None] + fx
    fade_n = int(0.5 * SR)
    mix[-fade_n:] *= np.linspace(1, 0, fade_n)[:, None] ** 2
    from score import limiter, lufs   # 与配乐同一套响度测量和前瞻限幅
    mix *= 10 ** ((TARGET_LUFS - lufs(mix)) / 20)
    mix = limiter(mix, CEILING_DB)
    print(f'[mix] {lufs(mix):.1f} LUFS, peak {20 * np.log10(np.abs(mix).max()):.1f} dBFS')
    data = (np.clip(mix, -1, 1) * 32767).astype('<i2')
    with wave.open(str(out), 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    print(f'wrote {out}  ({len(ev)} cues, {n / SR:.2f}s)')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--stems=')]
    st = [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--stems=')]
    main(Path(args[0] if args else ROOT / 'out' / 'soundtrack.wav'), st[0] if st else None)
