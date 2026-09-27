"""音效合成：响指、火焰、刀光、魔法、冲击……全部由噪声和振荡器现做。"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt


SR = 44100
N = int(110 * SR)   # 够装下整片（v3 约 101 秒）
rng = np.random.default_rng(2024)


def mf(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tt(n):
    return np.arange(n) / SR


def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR / 2 - 100), 'low', fs=SR, output='sos'), x)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc, 'high', fs=SR, output='sos'), x)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)


def sweep_lp(x, f0, f1, curve=1.0):
    """分块扫频低通，做呼啸和上扬。"""
    out = np.zeros_like(x)
    blocks = 64
    edges = np.linspace(0, len(x), blocks + 1).astype(int)
    zi = None
    for i in range(blocks):
        f = f0 + (f1 - f0) * (i / (blocks - 1)) ** curve
        sos = butter(2, min(max(f, 30), SR / 2 - 200), 'low', fs=SR, output='sos')
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        seg, zi = sosfilt(sos, x[edges[i]:edges[i + 1]], zi=zi)
        out[edges[i]:edges[i + 1]] = seg
    return out


def env(n, a=0.005, d=0.1, s=0.7, r=0.05, hold=None):
    """ADSR；hold 为按住的时长（秒），默认铺满。"""
    t = tt(n)
    hold = hold if hold is not None else n / SR - r
    e = np.where(t < a, t / max(a, 1e-6), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-6)))
    rel = t > hold
    e[rel] *= np.exp(-(t[rel] - hold) / max(r, 1e-6))
    return e


def expd(n, tau):
    return np.exp(-tt(n) / tau)


def phase(freq, n, vib=0.0, vib_rate=5.5, vib_delay=0.12):
    f = np.full(n, freq, float) if np.isscalar(freq) else freq
    if vib:
        t = tt(n)
        f = f * (1 + vib * np.sin(2 * np.pi * vib_rate * t) * np.clip((t - vib_delay) / 0.1, 0, 1))
    return np.cumsum(f) / SR


def pulse(freq, n, duty=0.5, **kw):
    ph = phase(freq, n, **kw) % 1
    return np.where(ph < duty, 1.0, -1.0) - (2 * duty - 1)


def tri(freq, n, **kw):
    ph = phase(freq, n, **kw) % 1
    return 4 * np.abs(ph - 0.5) - 1


def saw(freq, n, **kw):
    ph = phase(freq, n, **kw) % 1
    return 2 * ph - 1


def sine(freq, n, **kw):
    return np.sin(2 * np.pi * phase(freq, n, **kw))


def noise(n):
    return rng.uniform(-1, 1, n)


class Bus:
    def __init__(self):
        self.L = np.zeros(N + SR * 3)
        self.R = np.zeros(N + SR * 3)

    def add(self, t, x, gain=1.0, pan=0.0):
        i = int(round(t * SR))
        if i < 0:
            x = x[-i:]
            i = 0
        n = min(len(x), len(self.L) - i)
        if n <= 0:
            return
        gl = gain * np.sqrt(0.5 * (1 - pan))
        gr = gain * np.sqrt(0.5 * (1 + pan))
        self.L[i:i + n] += x[:n] * gl
        self.R[i:i + n] += x[:n] * gr


# ================================================================ 乐器
def inst_lead(m, dur):
    n = int((dur + 0.25) * SR)
    f = mf(m)
    x = 0.6 * pulse(f, n, 0.5, vib=0.012) + 0.4 * pulse(f * 1.003, n, 0.25, vib=0.012)
    x = lp(x, 5200)
    return x * env(n, 0.004, 0.25, 0.75, 0.12, hold=dur)


def inst_arp(m, dur):
    n = int((dur + 0.05) * SR)
    x = pulse(mf(m), n, 0.125)
    return lp(x, 6000) * env(n, 0.002, 0.06, 0.3, 0.03, hold=dur)


def inst_bass(m, dur):
    n = int((dur + 0.04) * SR)
    f = mf(m)
    x = 0.8 * tri(f, n) + 0.35 * pulse(f, n, 0.5)
    x = lp(x, 1400)
    return x * env(n, 0.002, 0.12, 0.6, 0.03, hold=dur)


def inst_guitar(m, dur, mute=True):
    """失真锯齿力量和弦（根音 + 五度 + 八度）。"""
    n = int((dur + (0.05 if mute else 0.4)) * SR)
    f = mf(m)
    x = sum(saw(f * r * (1 + d), n) for r, d in ((1, 0), (1.5, 0.002), (2, -0.003), (1, 0.004)))
    x = np.tanh(x * 3.0)
    x = lp(x, 2600 if mute else 3800, 2)
    e = env(n, 0.003, 0.09 if mute else 0.6, 0.25 if mute else 0.5, 0.05 if mute else 0.4, hold=dur)
    return x * e


def inst_pad(ms, dur, bright=1800):
    n = int((dur + 0.6) * SR)
    x = np.zeros(n)
    for m in ms:
        for d in (-0.006, 0, 0.007):
            x += saw(mf(m) * (1 + d), n)
    x = lp(x / (3 * len(ms)), bright)
    return x * env(n, 0.25, 0.8, 0.8, 0.5, hold=dur)


def inst_bell(m, dur=1.2):
    """八音盒 / 钟琴：非谐泛音快速衰减。"""
    n = int(dur * SR)
    f = mf(m)
    x = (sine(f, n) * expd(n, 0.5) + 0.5 * sine(f * 2.76, n) * expd(n, 0.18)
         + 0.25 * sine(f * 5.4, n) * expd(n, 0.07))
    return x * np.clip(tt(n) / 0.002, 0, 1)


# ================================================================ 鼓
def kick(big=1.0):
    n = int(0.45 * SR)
    t = tt(n)
    f = 45 + 130 * np.exp(-t / 0.035)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.22 * big))
    click = hp(noise(n), 2000) * np.exp(-t / 0.004) * 0.4
    return np.tanh((x + click) * 1.6)


def snare():
    n = int(0.3 * SR)
    t = tt(n)
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    nz = bp(noise(n), 1200, 8000) * np.exp(-t / 0.11)
    return 0.6 * body + 0.9 * nz


def hat(open_=False):
    n = int((0.25 if open_ else 0.06) * SR)
    return hp(noise(n), 7000) * np.exp(-tt(n) / (0.09 if open_ else 0.018))


def crash(dur=1.8):
    n = int(dur * SR)
    return hp(noise(n), 4000) * np.exp(-tt(n) / (dur * 0.35))


# ================================================================ 音效
def sfx(name):
    if name == 'snap':
        n = int(0.12 * SR)
        t = tt(n)
        click = hp(noise(n), 1500, 4) * np.exp(-t / 0.0025)
        body = bp(noise(n), 1800, 3600, 2) * np.exp(-t / 0.018)
        pop = np.sin(2 * np.pi * 1200 * t) * np.exp(-t / 0.01)
        return 1.4 * click + 1.2 * body + 0.3 * pop
    if name in ('fire_whoosh', 'fire_roar'):
        dur = 0.9 if name == 'fire_whoosh' else 1.8
        n = int(dur * SR)
        t = tt(n)
        x = sweep_lp(noise(n), 250, 3200 if name == 'fire_whoosh' else 1200, 0.6)
        e = np.clip(t / 0.05, 0, 1) * np.exp(-t / (dur * 0.35))
        crack = np.zeros(n)
        for i in rng.integers(0, n - 400, 70 if name == 'fire_roar' else 35):
            crack[i:i + 300] += hp(noise(300), 3000) * np.exp(-np.arange(300) / 40) * rng.uniform(0.3, 1)
        return x * e * 1.4 + crack * 0.35 * e
    if name == 'fire_burst':
        n = int(0.9 * SR)
        t = tt(n)
        x = lp(noise(n), 1800) * np.exp(-t / 0.18)
        sub = np.sin(2 * np.pi * np.cumsum(40 + 80 * np.exp(-t / 0.08)) / SR) * np.exp(-t / 0.3)
        return 1.2 * x + 1.0 * sub
    if name in ('impact', 'big_impact'):
        big = name == 'big_impact'
        dur = 2.4 if big else 1.2
        n = int(dur * SR)
        t = tt(n)
        f = 32 + 90 * np.exp(-t / 0.07)
        sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.7 if big else 0.35))
        hit = lp(noise(n), 3000) * np.exp(-t / 0.05)
        x = np.tanh(1.8 * sub) + 0.8 * hit
        if big:
            x[:len(crash(2.4))] += 0.5 * crash(2.4)[:n]
        return x
    if name == 'shimmer':
        n = int(1.4 * SR)
        x = np.zeros(n)
        for k, m in enumerate((86, 89, 93, 98, 101, 105)):
            i = int(k * 0.05 * SR)
            b = inst_bell(m, 1.4)[:n - i]
            x[i:] += b * 0.35
        return x
    if name == 'glitch':
        n = int(0.3 * SR)
        x = np.zeros(n)
        for i in range(0, n, int(0.025 * SR)):
            seg = pulse(rng.uniform(200, 1800), int(0.02 * SR), 0.5)
            x[i:i + len(seg)] = seg[:n - i] * rng.uniform(0.3, 1)
        return np.round(x * 4) / 4
    if name == 'cards':
        n = int(0.6 * SR)
        x = np.zeros(n)
        for k in range(22):
            i = int((k / 22) ** 0.8 * 0.5 * SR)
            m = int(0.012 * SR)
            x[i:i + m] += bp(noise(m), 2500, 7000) * np.exp(-np.arange(m) / 90) * rng.uniform(0.4, 1)
        return x
    if name in ('swish', 'dash'):
        dur = 0.3 if name == 'swish' else 0.2
        n = int(dur * SR)
        t = tt(n)
        x = sweep_lp(hp(noise(n), 400), 600, 7000, 1.5)
        return x * np.sin(np.pi * t / dur) ** 2 * 1.3
    if name == 'slash':
        n = int(0.9 * SR)
        t = tt(n)
        swoosh = sweep_lp(hp(noise(n), 800), 9000, 1500, 0.5) * np.exp(-t / 0.12)
        ring = (np.sin(2 * np.pi * 3150 * t) + 0.7 * np.sin(2 * np.pi * 4870 * t)
                + 0.4 * np.sin(2 * np.pi * 7210 * t)) * np.exp(-t / 0.22) * 0.35
        body = np.tanh(3 * np.sin(2 * np.pi * np.cumsum(60 + 200 * np.exp(-t / 0.03)) / SR)) * np.exp(-t / 0.2)
        return 1.3 * swoosh + ring + 0.8 * body
    if name == 'sparkle':
        n = int(0.7 * SR)
        x = np.zeros(n)
        for k, m in enumerate((84, 88, 91, 96, 100, 103, 108)):
            i = int(k * 0.045 * SR)
            m_n = int(0.12 * SR)
            x[i:i + m_n] += pulse(mf(m), m_n, 0.25) * expd(m_n, 0.05) * 0.4
        return lp(x, 9000)
    if name == 'chime':
        n = int(1.5 * SR)
        x = np.zeros(n)
        for k, m in enumerate((74, 81, 86, 90)):
            i = int(k * 0.03 * SR)
            x[i:] += inst_bell(m, 1.5)[:n - i] * 0.45
        return x
    if name == 'magic_beam':
        n = int(0.9 * SR)
        t = tt(n)
        f = 300 * 2 ** (t * 3)
        x = sine(f, n) * (0.6 + 0.4 * np.sin(2 * np.pi * 18 * t))
        return x * np.sin(np.pi * t / 0.9) * 0.6 + hp(noise(n), 5000) * 0.15 * np.sin(np.pi * t / 0.9)
    if name == 'bass_drop':
        n = int(1.6 * SR)
        t = tt(n)
        f = 28 + 110 * np.exp(-t / 0.25)
        return np.tanh(2.2 * np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-t / 0.7)
    if name == 'rumble':
        n = int(1.6 * SR)
        t = tt(n)
        return lp(noise(n), 160, 4) * 3.0 * np.sin(np.pi * t / 1.6)
    if name == 'wind':
        n = int(1.9 * SR)
        t = tt(n)
        return bp(noise(n), 500, 1500) * (0.4 + 0.3 * np.sin(2 * np.pi * 1.3 * t)) * np.sin(np.pi * t / 1.9)
    if name == 'puff':
        n = int(0.35 * SR)
        return lp(noise(n), 1500) * np.exp(-tt(n) / 0.08)
    if name == 'boom_low':
        n = int(2.0 * SR)
        t = tt(n)
        return np.tanh(1.5 * np.sin(2 * np.pi * np.cumsum(30 + 40 * np.exp(-t / 0.2)) / SR)) * np.exp(-t / 0.8)
    if name == 'heartbeat':
        n = int(0.8 * SR)
        x = np.zeros(n)
        for i in (0, int(0.22 * SR)):
            k = kick(0.6)[:n - i]
            x[i:i + len(k)] += lp(k, 300) * (1 if i == 0 else 0.7)
        return x
    if name == 'blip':
        n = int(0.035 * SR)
        return pulse(mf(rng.choice([81, 83, 84])), n, 0.5) * expd(n, 0.012)
    if name == 'whip':
        n = int(0.8 * SR)
        t = tt(n)
        crack = hp(noise(n), 2500, 4) * np.exp(-np.maximum(t - 0.18, 0) / 0.012) * (t > 0.18)
        swoosh = sweep_lp(hp(noise(n), 300), 500, 6000, 0.7) * np.sin(np.pi * np.clip(t / 0.22, 0, 1)) * (t < 0.22)
        return 1.6 * crack + 0.9 * swoosh + 0.6 * sfx('fire_whoosh')[:n]
    if name == 'lance':
        n = int(1.1 * SR)
        t = tt(n)
        whoosh = sweep_lp(hp(noise(n), 500), 2000, 12000, 0.4) * np.exp(-t / 0.2)
        ring = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((880, 0.5), (1320, 0.35), (2210, 0.25))) * np.exp(-t / 0.5)
        boom = np.tanh(2 * np.sin(2 * np.pi * np.cumsum(40 + 160 * np.exp(-t / 0.04)) / SR)) * np.exp(-t / 0.3)
        return 1.2 * whoosh + 0.5 * ring + 0.9 * boom
    if name == 'glass':
        n = int(0.9 * SR)
        x = np.zeros(n)
        for k in range(26):
            i = int(rng.uniform(0, 0.35) * SR)
            m = int(rng.uniform(0.05, 0.3) * SR)
            f = rng.uniform(2500, 9000)
            seg = np.sin(2 * np.pi * f * tt(m)) * np.exp(-tt(m) / rng.uniform(0.02, 0.1))
            x[i:i + m] += seg[:n - i] * rng.uniform(0.2, 0.6)
        return x + 0.8 * hp(noise(n), 3000) * np.exp(-tt(n) / 0.05)
    if name == 'step':              # 硬底鞋踩石地
        n = int(0.16 * SR)
        t = tt(n)
        return (bp(noise(n), 120, 900) * np.exp(-t / 0.018) * 1.4 + lp(noise(n), 2500) * np.exp(-t / 0.006) * 0.6)
    if name == 'sword_draw':        # 拔剑：金属摩擦上扬 + 出鞘的"锵"
        n = int(0.9 * SR)
        t = tt(n)
        scrape = bp(noise(n), 2500, 9000) * (0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 37 * t))) * \
            np.clip(t / 0.3, 0, 1) * (t < 0.35)
        ring = sum(np.sin(2 * np.pi * f * t) * a for f, a in ((2340, .5), (3510, .35), (5270, .2))) * \
            np.exp(-np.maximum(t - 0.35, 0) / 0.35) * (t >= 0.35)
        return scrape * 0.6 + ring * 0.7
    if name == 'card_flip':         # 扑克牌掠过 + 翻面
        n = int(0.5 * SR)
        t = tt(n)
        fl = sweep_lp(hp(noise(n), 1500), 2000, 9000, 1.2) * np.sin(np.pi * np.clip(t / 0.42, 0, 1)) ** 2 * 0.7
        snap_i = int(0.42 * SR)
        m = n - snap_i
        fl[snap_i:] += bp(noise(m), 2000, 6000) * np.exp(-tt(m) / 0.012) * 1.3
        return fl
    if name == 'curtain':           # 厚重帷幕拉开
        n = int(1.6 * SR)
        t = tt(n)
        return lp(noise(n), 900) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * np.sin(np.pi * t / 1.6) * 1.2
    if name == 'applause':          # 远处的掌声
        n = int(3.0 * SR)
        x = np.zeros(n)
        for k in range(900):
            i = int(rng.uniform(0, 2.8) * SR)
            m = int(0.01 * SR)
            x[i:i + m] += bp(noise(m), 800, 4000) * np.exp(-np.arange(m) / 60) * rng.uniform(0.2, 1)
        t = tt(n)
        return x * np.clip(t / 0.4, 0, 1) * np.clip((3.0 - t) / 1.2, 0, 1) * 0.5
    if name == 'tick':              # 怀表 / 古钟的一下
        n = int(0.08 * SR)
        return bp(noise(n), 3000, 8000) * np.exp(-tt(n) / 0.004) + np.sin(2 * np.pi * 2800 * tt(n)) * np.exp(-tt(n) / 0.01) * 0.3
    if name == 'wing':              # 大鸟振翅
        n = int(0.7 * SR)
        t = tt(n)
        return lp(noise(n), 700) * np.sin(np.pi * np.clip(t / 0.45, 0, 1)) ** 3 * 2.0
    if name == 'charge':            # 蓄力：上扬的嗡鸣 + 噪声
        n = int(1.2 * SR)
        t = tt(n)
        f = 90 * 2 ** (t * 2.2)
        tone = np.tanh(2 * np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.4
        air = sweep_lp(noise(n), 300, 8000, 1.5) * 0.6
        return (tone + air) * (t / 1.2) ** 1.5
    if name == 'crunch':            # 顿帧时的硬冲击
        n = int(0.5 * SR)
        t = tt(n)
        return np.tanh(4 * lp(noise(n), 5000) * np.exp(-t / 0.03)) + np.tanh(2 * np.sin(2 * np.pi * 55 * t)) * np.exp(-t / 0.12)
    if name == 'crumble':           # 石头崩落
        n = int(1.4 * SR)
        x = np.zeros(n)
        for k in range(60):
            i = int(rng.uniform(0, 1.1) * SR)
            m = int(rng.uniform(0.02, 0.08) * SR)
            x[i:i + m] += lp(noise(m), rng.uniform(400, 2500)) * np.exp(-np.arange(m) / (m / 4)) * rng.uniform(0.3, 1)
        return x * 1.2 + lp(noise(n), 120) * np.exp(-tt(n) / 0.5) * 1.5
    if name == 'crackle':           # 余火噼啪（循环用，1.5 秒）
        n = int(1.5 * SR)
        x = lp(noise(n), 500) * 0.25
        for i in rng.integers(0, n - 400, 45):
            x[i:i + 300] += hp(noise(300), 2500) * np.exp(-np.arange(300) / 35) * rng.uniform(0.2, 0.9)
        return x
    raise KeyError(name)
