"""第四幕 · 怪盗B：月夜美术馆。

原作：执行者 No.X「怪盗绅士」布卢布兰，自称美的崇拜者，只偷他认为美的东西；擅长幻术与分身，
行窃前后留下预告卡 / 名片（署名 B）。台词「怪盗とは、すなわち美の崇拝者。」

场景：深夜的美术馆大厅。玻璃天窗外一轮月亮，月光斜斜地切过大厅；大理石柱、金框名画、红丝绒围栏，
正中的展台上是玻璃展柜，柜里一颗大红宝石（「玫瑰之泪」）在顶灯下闪。门口警卫的探照灯在墙上扫来扫去。

分镜（6 小节 @144，小节从 1 数、拍从 0 数）
  1.0  远景，镜头缓缓向左横移：黑暗的大厅，两道探照灯在墙上来回扫；1.1 / 1.3 宝石闪光。
       一片紫玫瑰花瓣从天窗飘下，穿过月光。
  2.0  花瓣落地 —— 地面卷起紫色花瓣旋风，旋风里浮出人影。
  2.1  硬切中景（2x）：花瓣炸开，他半蹲落地、披风张开；名牌滑入。2.1.5 起身，披风慢慢落定。
  2.3  一道探照灯扫过来正好罩住他（被发现也毫不在意）。
  3.0  台词。3.0 行礼（手按胸口、低头），披风一甩带出花瓣；3.1.5 起身扶面具；3.2 面具一闪。
  4.0  硬切特写（立绘 3x，坏笑）；4.1 镜孔里的眼睛一闪。
  4.2  切回中景：举杖旋转（杖在手上转两圈，拖出光弧）；4.3 左手响指 —— 展柜四周
       弹出四面金框镜子，镜中走出四个分身（紫 / 青色、闪烁）。音乐在 4.3 一记重音后停顿。
  4.3.5 全员压低重心（预备）。4.3.75 同时冲向展柜（拖影 + 速度线）。
  5.0  命中：两帧反相剪影 + 三帧顿帧 → 玻璃炸碎，分身化作花瓣炸开，紫光照亮整个大厅；宝石不见了，
       垫子上只插着一张卡片。
  5.2  硬切远景：展台上开出一朵巨大的紫玫瑰；5.3 玫瑰炸成花瓣雨。警报红灯闪，探照灯乱扫。
  6.0  硬切：美术馆穹顶上，巨大的满月。他立在穹顶尖上高举宝石，披风猎猎；6.1 一张写着 B 的名片
       旋转着朝镜头飘来。最后两拍定格（扑克牌飞入）。
"""
from functools import lru_cache

import numpy as np

import sfx as S
import ui
from gfx import BAYER, H, W, XX, YY, blit, ease_out, hexc, scale_img, smooth


def clamp01(x):
    return np.clip(x, 0.0, 1.0)

LETTERBOX = 0
MUSIC_SHAKE = False          # 顿帧时画面要静止，震屏全部自己管
SAYS = [('bleublanc', 3, 0.0, 'dialog', (4, 3.5))]     # 4.3.5 预备时收起，命中时画面干净
CUES = [
    (1, 0.0, 'night_air', -6),
    (1, 0.5, 'searchlight', -12, 0.6), (1, 2.5, 'searchlight', -13, 0.4),
    (1, 1.0, 'gem_twinkle', -8), (1, 3.0, 'gem_twinkle', -9),
    (2, 0.0, 'petal_swirl', -2),                   # 花瓣落地卷起旋风
    (2, 0.5, 'swish', -8),                         # 切中景，旋风卷紧
    (2, 1.0, 'appear', -1), (2, 1.0, 'step', -6),  # 花瓣炸开，落地
    (2, 3.0, 'spot_on', -3, 0.5),                  # 探照灯罩住他
    (3, 0.0, 'cape_flap', -4), (3, 0.0, 'swish', -10),     # 行礼，披风一甩
    (3, 2.0, 'gem_twinkle', -5),                   # 面具一闪
    (4, 0.0, 'swish', -14),                        # 切特写
    (4, 1.0, 'gem_twinkle', -3),                   # 眼睛一闪
    (4, 2.0, 'whip', -6), (4, 2.5, 'whip', -7),    # 杖转两圈
    (4, 3.0, 'snap', 0), (4, 3.0, 'phantom', -2),  # 响指，分身出现
    (4, 3.5, 'inhale', -4),                        # 预备（音乐停顿里的一口气）
    (4, 3.75, 'dash', -2),                         # 冲出
    (5, 0.0, 'big_impact', 0), (5, 0.0, 'shatter', 0), (5, 0.0, 'crunch', -4),
    (5, 0.25, 'petal_swirl', -5),                  # 分身化作花瓣
    (5, 2.0, 'boom_low', -2), (5, 2.0, 'alarm', -9),       # 大玫瑰绽放，警报
    (5, 3.0, 'sparkle', -2), (5, 3.0, 'petal_swirl', -6),  # 玫瑰炸成花瓣雨
    (6, 0.0, 'wind', -6), (6, 0.0, 'gem_twinkle', -3),     # 穹顶，高举宝石
    (6, 1.0, 'card_flip', -4),                     # 名片飘向镜头
]
ACCENTS = [(2, 1.0, 'hit'), (4, 3.0, 'stop'), (5, 0.0, 'hit'), (5, 2.0, 'drop'), (7, 0.0, 'riser')]


# ================================================================ 合成音效
def _st(x, pan=0.0):
    x = np.asarray(x, np.float64)
    return np.stack([x * np.sqrt(0.5 * (1 - pan)), x * np.sqrt(0.5 * (1 + pan))], 1).astype(np.float32)


def _night_air():
    """空荡大厅的夜：极轻的风声 + 远处座钟两下。"""
    n = int(1.7 * S.SR)
    t = S.tt(n)
    air = S.bp(S.noise(n), 200, 900) * 0.25 * np.clip(t / 0.4, 0, 1) * np.exp(-np.clip(t - 1.2, 0, None) / 0.2)
    tick = np.zeros(n)
    for k in (0.1, 0.52):
        i = int(k * S.SR)
        m = int(0.25 * S.SR)
        tt_ = S.tt(m)
        tick[i:i + m] += (np.sin(2 * np.pi * 1400 * tt_) * 0.4 + np.sin(2 * np.pi * 2900 * tt_) * 0.2) * \
            np.exp(-tt_ / 0.03)
    return _st(air + 0.5 * S.lp(tick, 5000))


def _searchlight():
    """探照灯转动：电机嗡声滑音 + 轻微的金属轴承声。"""
    n = int(0.9 * S.SR)
    t = S.tt(n)
    f = 90 + 40 * np.sin(np.pi * t / 0.9)
    hum = np.sign(np.sin(2 * np.pi * np.cumsum(f) / S.SR)) * 0.25 + np.sin(2 * np.pi * np.cumsum(2 * f) / S.SR) * 0.2
    hum = S.lp(hum, 900) * np.sin(np.pi * t / 0.9) ** 0.7
    squeak = S.bp(S.noise(n), 3000, 5000) * 0.1 * np.sin(np.pi * t / 0.9)
    return _st(hum + squeak)


def _gem_twinkle():
    """宝石闪光：两三个很高的玻璃泛音，带一点亮晶晶的颤音。"""
    n = int(0.9 * S.SR)
    t = S.tt(n)
    x = np.zeros(n)
    for k, (m, a) in enumerate(((100, 0.4), (107, 0.3), (112, 0.22), (104, 0.18))):
        i = int(k * 0.035 * S.SR)
        f = S.mf(m)
        seg = np.sin(2 * np.pi * f * t[:n - i]) * np.exp(-t[:n - i] / 0.25) * (1 + 0.3 * np.sin(2 * np.pi * 9 * t[:n - i]))
        x[i:] += a * seg
    return np.stack([x, np.roll(x, 300)], 1).astype(np.float32) * 0.8


def _petal_swirl():
    """花瓣旋风：绕着转的气流（声相左右摇摆）+ 上扬的风铃。"""
    n = int(1.1 * S.SR)
    t = S.tt(n)
    wind = S.sweep_lp(S.noise(n), 500, 5000, 0.8) * np.sin(np.pi * t / 1.1) ** 0.8 * 0.7
    pan = np.sin(2 * np.pi * 3.5 * t)
    chime = np.zeros(n)
    for k, m in enumerate((86, 89, 93, 96, 98, 101)):
        i = int((0.1 + k * 0.08) * S.SR)
        seg = np.sin(2 * np.pi * S.mf(m) * t[:n - i]) * np.exp(-t[:n - i] / 0.18)
        chime[i:] += 0.18 * seg
    L = wind * np.sqrt(0.5 * (1 - pan)) + chime
    R = wind * np.sqrt(0.5 * (1 + pan)) + np.roll(chime, 200)
    return np.stack([L, R], 1).astype(np.float32)


def _appear():
    """现身：反向的空气声吸入 + 一记柔和的低音 + 闪亮的和弦。"""
    n = int(1.2 * S.SR)
    t = S.tt(n)
    boom = np.sin(2 * np.pi * np.cumsum(48 + 90 * np.exp(-t / 0.05)) / S.SR) * np.exp(-t / 0.35)
    air = S.hp(S.noise(n), 2000) * np.exp(-t / 0.12) * 0.4
    shine = sum(np.sin(2 * np.pi * S.mf(m) * t) * a for m, a in ((74, .3), (81, .25), (86, .22), (93, .15)))
    shine *= np.exp(-t / 0.5) * np.clip(t / 0.01, 0, 1)
    return _st(0.8 * boom + air + 0.35 * shine)


def _spot_on():
    """大功率探照灯"咔哒"一声打开：继电器 + 低频嗡。"""
    n = int(0.8 * S.SR)
    t = S.tt(n)
    clunk = S.lp(S.noise(n), 1500) * np.exp(-t / 0.02) * 1.2 + np.sin(2 * np.pi * 70 * t) * np.exp(-t / 0.08)
    hum = np.sin(2 * np.pi * 120 * t) * 0.15 * np.clip((t - 0.03) / 0.05, 0, 1) * np.exp(-t / 0.5)
    return _st(clunk + hum)


def _cape_flap():
    """披风一甩：低沉的布料扑动。"""
    n = int(0.45 * S.SR)
    t = S.tt(n)
    x = S.sweep_lp(S.noise(n), 300, 1800, 0.5) * np.sin(np.pi * np.clip(t / 0.3, 0, 1)) ** 1.5
    x += 0.5 * S.lp(S.noise(n), 250) * np.exp(-t / 0.05)
    return _st(x)


def _phantom():
    """分身出现：四个错开的、失谐的镜面嗡鸣（像玻璃在唱）+ 左右散开。"""
    n = int(1.2 * S.SR)
    out = np.zeros((n, 2))
    for k, (d, pan, m) in enumerate(((0.0, -0.7, 74), (0.05, 0.7, 77), (0.1, -0.3, 81), (0.15, 0.4, 86))):
        i = int(d * S.SR)
        t = S.tt(n - i)
        f = S.mf(m) * (1 + 0.004 * np.sin(2 * np.pi * 6 * t))
        x = (np.sin(2 * np.pi * np.cumsum(f) / S.SR) + 0.4 * np.sin(2 * np.pi * np.cumsum(f * 2.01) / S.SR))
        x *= np.clip(t / 0.02, 0, 1) * np.exp(-t / 0.35) * 0.3
        out[i:] += _st(x, pan)
    return out.astype(np.float32)


def _inhale():
    """预备：短促的上扬吸气声（给 5.0 的冲击留出反差）。"""
    n = int(0.3 * S.SR)
    t = S.tt(n)
    x = S.sweep_lp(S.noise(n), 300, 6000, 1.5) * (t / 0.3) ** 2
    return _st(0.8 * x)


def _shatter():
    """展柜玻璃整面炸碎：一记硬的碎裂 + 大量碎片散落。"""
    n = int(1.6 * S.SR)
    t = S.tt(n)
    x = S.hp(S.noise(n), 1800) * np.exp(-t / 0.06) * 1.2
    rng = np.random.default_rng(9)
    for k in range(70):
        i = int(rng.uniform(0, 1.0) ** 1.6 * S.SR)
        m = int(rng.uniform(0.03, 0.2) * S.SR)
        if i + m > n:
            continue
        f = rng.uniform(2500, 11000)
        x[i:i + m] += np.sin(2 * np.pi * f * S.tt(m)) * np.exp(-S.tt(m) / rng.uniform(0.01, 0.06)) * \
            rng.uniform(0.1, 0.5) * (1 - i / n)
    L = x
    R = np.roll(x, 150)
    return np.stack([L, R], 1).astype(np.float32)


def _alarm():
    """远处的警报铃：两音交替。"""
    n = int(1.6 * S.SR)
    t = S.tt(n)
    f = np.where((t * 4).astype(int) % 2 == 0, 880, 740)
    x = np.sign(np.sin(2 * np.pi * np.cumsum(f) / S.SR)) * 0.2
    x = S.bp(x, 500, 3000) * np.clip(t / 0.05, 0, 1) * np.exp(-np.clip(t - 1.0, 0, None) / 0.2)
    return _st(x, 0.5)


SFX = {'night_air': _night_air, 'searchlight': _searchlight, 'gem_twinkle': _gem_twinkle,
       'petal_swirl': _petal_swirl, 'appear': _appear, 'spot_on': _spot_on, 'cape_flap': _cape_flap,
       'phantom': _phantom, 'inhale': _inhale, 'shatter': _shatter, 'alarm': _alarm}


# ================================================================ 调色
def _c(h):
    return hexc(h)


VIOLET, VIOLET_L, MINT = _c('#9b5bd6'), _c('#e0c0ff'), _c('#8fe3d4')
WHITE, BLACK = _c('#ffffff'), _c('#0a0610')
MOON_BLUE = _c('#b8c8ff')
RUBY = (_c('#ff6a7a'), _c('#e0203a'), _c('#7a0a20'))


def _night(img):
    """亮版 → 暗版：每个颜色映射成一个偏靛蓝的暗色（调色板仍是一一对应）。"""
    return img * np.array([0.26, 0.26, 0.40], np.float32) + np.array([0.015, 0.01, 0.045], np.float32)


# ================================================================ 美术馆（世界坐标，1x）
GW = 720                     # 世界宽
FLOOR_Y = 196                # 墙脚
FEET_Y = 222                 # 角色脚底
X_CASE = 400                 # 展柜中心
X_REAL = 336                 # 他现身的位置
POSTS = (364, 436)           # 围栏立柱
CLONES = [(290, True), (452, False), (480, False), (262, True)]   # (x, 是否朝右) —— 分身
DASH_END = {-1: -14, 0: -6, 1: 6, 2: 16, 3: -22}                   # 冲刺终点相对展柜中心
MOON_W = (470, 17, 11)       # 天窗外的月亮（世界坐标）
COLS = (110, 250, 550, 690)
PAINTS = [(144, 80, 214, 132, 'sea'), (286, 70, 336, 140, 'lady'), (364, 58, 436, 122, 'rose'),
          (466, 70, 516, 140, 'lord'), (586, 80, 656, 132, 'hill')]

YG, XG = np.mgrid[0:H, 0:GW].astype(np.float32)
BAYER_G = np.tile(BAYER[:4, :4], (H // 4 + 1, GW // 4 + 1))[:H, :GW]


def _rect(img, x0, y0, x1, y1, col):
    img[max(0, int(y0)):max(0, int(y1)), max(0, int(x0)):max(0, int(x1))] = col


def _painting(img, x0, y0, x1, y1, kind):
    g0, g1, g2 = _c('#ffe08a'), _c('#d0a040'), _c('#6a4a18')
    _rect(img, x0, y0, x1, y1, g1)                              # 金框
    _rect(img, x0, y0, x1, y0 + 1, g0)
    _rect(img, x0, y0, x0 + 1, y1, g0)
    _rect(img, x0, y1 - 1, x1, y1, g2)
    _rect(img, x1 - 1, y0, x1, y1, g2)
    for k in range(0, int(x1 - x0), 5):                         # 框上的雕花点
        img[int(y0) + 2, int(x0) + k] = g0
        img[int(y1) - 3, int(x0) + k] = g2
    ix0, iy0, ix1, iy1 = x0 + 5, y0 + 5, x1 - 5, y1 - 5
    _rect(img, ix0 - 1, iy0 - 1, ix1 + 1, iy1 + 1, g2)
    w, h = int(ix1 - ix0), int(iy1 - iy0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bay = np.tile(BAYER[:4, :4], (h // 4 + 1, w // 4 + 1))[:h, :w]
    if kind in ('sea', 'hill'):
        sky = np.where((bay < yy / h * 1.4)[..., None], _c('#e8a870')[None, None], _c('#6a8ac8')[None, None])
        pic = sky.copy()
        hz = h * (0.6 if kind == 'sea' else 0.5)
        if kind == 'sea':
            pic[yy > hz] = _c('#2a4a8a')
            pic[(yy > hz) & ((xx + yy * 3) % 9 < 1)] = _c('#8ab0e8')
            pic[np.hypot(xx - w * 0.7, yy - hz * 0.7) < 5] = _c('#fff0c0')
        else:
            ridge = hz + 6 * np.sin(xx * 0.12) + 3 * np.sin(xx * 0.31)
            pic[yy > ridge] = _c('#3a6a3a')
            pic[yy > ridge + 8] = _c('#2a4a2a')
            pic[(np.abs(xx - w * 0.3) < 2) & (yy > ridge - 10) & (yy < ridge)] = _c('#1a2a1a')
    elif kind == 'rose':
        pic = np.zeros((h, w, 3), np.float32) + _c('#2a1030')
        pic[bay < 0.3 + 0.3 * (yy / h)] = _c('#3a1840')
        d = np.hypot(xx - w / 2, (yy - h * 0.45) * 1.1)
        a = np.arctan2(yy - h * 0.45, xx - w / 2)
        petal = d < h * 0.32 * (0.8 + 0.2 * np.abs(np.cos(2.5 * a)))
        pic[petal] = _c('#c02858')
        pic[petal & (d < h * 0.2 * (0.75 + 0.25 * np.abs(np.cos(3 * a + 1))))] = _c('#e8506e')
        pic[d < h * 0.07] = _c('#ff9ab0')
        pic[petal & ((np.abs(np.sin(a * 5 + d * 0.3)) < 0.12))] = _c('#701838')
        stem = (np.abs(xx - w / 2 - (yy - h * 0.7) * 0.2) < 1) & (yy > h * 0.7)
        pic[stem] = _c('#3a7a3a')
        pic[(np.abs(xx - w * 0.6) < 4) & (np.abs(yy - h * 0.82) < 2)] = _c('#3a7a3a')
    else:                     # 肖像：暗底 + 半身剪影 + 亮脸
        pic = np.zeros((h, w, 3), np.float32) + _c('#2a2018')
        pic[bay < 0.35] = _c('#3a2a20')
        body = (np.hypot((xx - w / 2) / (w * 0.42), (yy - h * 1.05) / (h * 0.45)) < 1)
        pic[body] = _c('#5a1a2a' if kind == 'lady' else '#1a2a4a')
        face = np.hypot((xx - w / 2) / 6, (yy - h * 0.36) / 8) < 1
        hair = np.hypot((xx - w / 2) / 8, (yy - h * 0.32) / 10) < 1
        pic[hair] = _c('#6a4020' if kind == 'lady' else '#2a2a2a')
        pic[face] = _c('#e8c0a0')
        if kind == 'lady':
            pic[(np.abs(xx - w / 2) < 1) & (np.abs(yy - h * 0.6) < 2)] = _c('#ffe0a0')
    img[int(iy0):int(iy0) + h, int(ix0):int(ix0) + w] = pic


@lru_cache(None)
def room_layers():
    """美术馆大厅（亮版, 暗版），(H, GW, 3)。展柜 / 宝石 / 分身之类会动的东西不在这里。"""
    X, Y = XG, YG
    img = np.zeros((H, GW, 3), np.float32)
    # --- 天窗：夜空 + 星 + 铁框玻璃格
    sky = Y < 40
    img[sky] = _c('#1a2250')
    img[sky & (BAYER_G < (40 - Y) / 60)] = _c('#101640')
    rng = np.random.default_rng(4)
    for x, y in zip(rng.uniform(0, GW, 90), rng.uniform(2, 36, 90)):
        img[int(y), int(x)] = _c('#c8d0ff')
    mx, my, mr = MOON_W
    md = np.hypot(X - mx, Y - my)
    img[sky & (md < mr)] = _c('#fff6dc')
    img[sky & (md < mr) & (np.hypot(X - mx - 3, Y - my + 2) < 3)] = _c('#e8dcc0')
    img[sky & (md >= mr) & (md < mr + 3) & (BAYER_G < 0.5)] = _c('#4a5a9a')
    frame = sky & ((((X - 8) % 40) < 2) | (((Y - 2) % 12) < 1) | (Y < 2))
    img[frame] = _c('#302838')
    # 玻璃上的斜反光
    img[sky & ~frame & ((X + Y * 1.5) % 40 < 1.5)] = _c('#3a4a8a')
    # --- 檐口：金色线脚
    _rect(img, 0, 40, GW, 48, _c('#6a5a70'))
    _rect(img, 0, 40, GW, 41, _c('#ffe08a'))
    _rect(img, 0, 44, GW, 45, _c('#c8a050'))
    _rect(img, 0, 47, GW, 48, _c('#2a2030'))
    for x in range(0, GW, 6):                                   # 齿饰
        _rect(img, x, 45, x + 3, 47, _c('#8a7a90'))
    # --- 墙：深紫红丝绒墙面 + 竖向暗纹 + 护墙板
    wall = (Y >= 48) & (Y < FLOOR_Y)
    img[wall] = _c('#6a2a48')
    img[wall & ((X % 18) < 1)] = _c('#5a2240')
    img[wall & (((X + (Y // 9) * 9) % 18) == 9) & ((Y % 9) == 4)] = _c('#8a3a5a')    # 墙纸小花
    _rect(img, 0, 164, GW, FLOOR_Y, _c('#4a1a30'))
    _rect(img, 0, 164, GW, 166, _c('#e0b860'))
    _rect(img, 0, 166, GW, 167, _c('#8a6428'))
    for x in range(8, GW, 36):                                  # 护墙板框
        _rect(img, x, 172, x + 28, 173, _c('#6a2a44'))
        _rect(img, x, 172, x + 1, 190, _c('#6a2a44'))
        _rect(img, x, 189, x + 28, 190, _c('#2a0a18'))
        _rect(img, x + 27, 172, x + 28, 190, _c('#2a0a18'))
    _rect(img, 0, FLOOR_Y - 2, GW, FLOOR_Y, _c('#2a0a18'))
    # --- 名画
    for x0, y0, x1, y1, kind in PAINTS:
        _painting(img, x0, y0, x1, y1, kind)
    # --- 大理石柱
    for cx in COLS:
        x0, x1 = cx - 13, cx + 13
        _rect(img, x0 - 5, 48, x1 + 5, 58, _c('#e8e0f0'))       # 柱头
        _rect(img, x0 - 5, 56, x1 + 5, 58, _c('#a098b8'))
        _rect(img, x0 - 3, 58, x1 + 3, 62, _c('#c8c0d8'))
        _rect(img, x0, 62, x1, FLOOR_Y - 10, _c('#d8d0e8'))     # 柱身
        for k in range(2, 26, 4):                               # 凹槽
            _rect(img, x0 + k, 64, x0 + k + 1, FLOOR_Y - 12, _c('#a8a0c0'))
        _rect(img, x0, 62, x0 + 2, FLOOR_Y - 10, _c('#fff8ff'))
        _rect(img, x1 - 4, 62, x1, FLOOR_Y - 10, _c('#8a82a8'))
        _rect(img, x0 - 4, FLOOR_Y - 10, x1 + 4, FLOOR_Y - 4, _c('#c8c0d8'))   # 柱础
        _rect(img, x0 - 6, FLOOR_Y - 4, x1 + 6, FLOOR_Y, _c('#a098b8'))
        # 大理石纹
        for k in range(6):
            yv = 70 + k * 20
            _rect(img, x0 + 4 + (k * 7) % 14, yv, x0 + 8 + (k * 7) % 14, yv + 1, _c('#b8b0d0'))
    # --- 地面：透视棋盘大理石 + 立柱倒影
    fl = Y >= FLOOR_Y
    depth = 1.0 / np.maximum(Y - 150, 1)
    u = (X - 360) * depth * 18
    v = depth * 900
    chk = ((np.floor(u) + np.floor(v)) % 2 == 0)
    img[fl & chk] = _c('#b8b0cc')
    img[fl & ~chk] = _c('#7a7092')
    img[fl & chk & ((np.floor(u * 3) + np.floor(v * 2)) % 7 == 0)] = _c('#a8a0bc')
    img[fl & (np.abs(Y - FLOOR_Y - 1) < 1)] = _c('#5a5070')
    for cx in COLS:                                             # 倒影
        refl = fl & (np.abs(X - cx) < 12) & (Y < FLOOR_Y + 26) & (BAYER_G < 0.45 - (Y - FLOOR_Y) / 60)
        img[refl] = img[refl] * 0.6 + _c('#e8e0f0') * 0.4
    # --- 展台（大理石）
    _rect(img, 378, 172, 422, 177, _c('#f0e8f8'))
    _rect(img, 378, 176, 422, 177, _c('#a098b8'))
    _rect(img, 382, 177, 418, 210, _c('#d8d0e8'))
    _rect(img, 382, 177, 384, 210, _c('#fff8ff'))
    _rect(img, 413, 177, 418, 210, _c('#8a82a8'))
    _rect(img, 386, 184, 414, 202, _c('#c0b8d4'))               # 嵌板
    _rect(img, 386, 184, 414, 185, _c('#fff8ff'))
    _rect(img, 386, 201, 414, 202, _c('#8a82a8'))
    _rect(img, 376, 209, 424, 216, _c('#b8b0c8'))
    _rect(img, 376, 209, 424, 210, _c('#f0e8f8'))
    _rect(img, 376, 215, 424, 216, _c('#6a6280'))
    lit = img
    dark = _night(lit)
    return lit.astype(np.float32), dark.astype(np.float32), ((lit + dark) * 0.5 - 0.03).astype(np.float32)


@lru_cache(None)
def stanchion_layer():
    """红丝绒围栏：两根黄铜立柱 + 下垂的绳（RGBA，世界坐标宽 GW）。"""
    img = np.zeros((H, GW, 4), np.float32)

    def put(x, y, col):
        if 0 <= y < H and 0 <= x < GW:
            img[y, x, :3] = col
            img[y, x, 3] = 1
    for px in POSTS:
        for y in range(198, 228):
            put(px, y, _c('#ffe08a'))
            put(px + 1, y, _c('#b88a30'))
        for dx in range(-2, 4):                                  # 顶球 / 底座
            put(px + dx - 0, 197 if abs(dx - 0.5) < 2 else 198, _c('#ffe08a'))
            put(px + dx, 227, _c('#8a6420'))
            put(px + dx, 228, _c('#6a4a18'))
        for dx in (-1, 0, 1, 2):
            put(px + dx, 196, _c('#fff0b0') if dx in (0, 1) else _c('#c89a40'))
    a, b = POSTS
    for x in range(a + 2, b):
        k = (x - a) / (b - a)
        y = int(203 + 9 * np.sin(np.pi * k))
        put(x, y, _c('#e02848'))
        put(x, y + 1, _c('#8a0a28'))
    return img


def _to_dark(rgba):
    out = rgba.copy()
    out[..., :3] = _night(rgba[..., :3])
    return out


@lru_cache(None)
def stanchion_dark():
    return _to_dark(stanchion_layer())


# ================================================================ 光照（世界坐标）
def _in_quad(X, Y, pts):
    ok1 = np.ones(X.shape, bool)
    ok2 = np.ones(X.shape, bool)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        c = (x1 - x0) * (Y - y0) - (y1 - y0) * (X - x0)
        ok1 &= c >= 0
        ok2 &= c <= 0
    return ok1 | ok2


MOONBEAM = [(440, 40), (500, 40), (392, 240), (296, 240)]
SEARCH_SRC = (740, -30)             # 探照灯在右上方的回廊里


def search_spots(t, mode='patrol', target=None, k_lock=0.0):
    """两盏探照灯打在墙 / 地上的光斑中心 (x, y)（世界坐标）。"""
    s1 = (330 + 190 * np.sin(t * 0.9), 110 + 30 * np.sin(t * 1.7))
    s2 = (520 + 150 * np.sin(t * 0.7 + 2.0), 130 + 25 * np.sin(t * 1.3 + 1))
    if mode == 'panic':
        s1 = (380 + 220 * np.sin(t * 3.1), 120 + 50 * np.sin(t * 4.3))
        s2 = (420 + 200 * np.sin(t * 2.6 + 2.0), 150 + 40 * np.sin(t * 3.7 + 1))
    if target is not None:
        s1 = (s1[0] + (target[0] - s1[0]) * k_lock, s1[1] + (target[1] - s1[1]) * k_lock)
    return [s1, s2]


def light_mask(X, Y, t, spots, spot_r=(30, 24), moon=0.55, case=0.75, extra=None):
    """每个像素的受光量 0..1：月光（带天窗格子影）、展柜顶灯、探照灯光斑。"""
    L = np.zeros(X.shape, np.float32)
    mb = _in_quad(X, Y, MOONBEAM)
    # 天窗铁框在月光里投下的格子影：沿光的方向平移回天窗平面
    sx = X + (Y - 40) * 0.52
    grid = ((sx - 8) % 40 < 3) | ((Y * 0.6 + sx * 0.1) % 22 < 2)
    L[mb] = moon
    L[mb & grid] = moon * 0.3
    # 顶灯锥 + 地上的光池
    cone = _in_quad(X, Y, [(396, 48), (404, 48), (428, 176), (372, 176)])
    L = np.maximum(L, cone * case * 0.55)
    pool = np.hypot((X - X_CASE) / 44, (Y - 219) / 7) < 1
    L = np.maximum(L, pool * case)
    top = np.hypot((X - X_CASE) / 30, (Y - 150) / 34) < 1
    L = np.maximum(L, top * case)
    for (cx, cy) in spots:
        d = np.hypot((X - cx) / spot_r[0], (Y - cy) / spot_r[1])
        L = np.maximum(L, clamp01(1.25 - d) ** 0.6)
    if extra is not None:
        L = np.maximum(L, extra)
    return L


def beam_haze(dst, ox, oy, zoom, src, spot, t, col=_c('#fff8e0'), k=1.0):
    """探照灯的光柱：从光源到光斑的一条抖动光带（加色，画在屏幕上）。"""
    (x0, y0), (x1, y1) = src, spot
    sx0, sy0 = (x0 - ox) * zoom, (y0 - oy) * zoom
    sx1, sy1 = (x1 - ox) * zoom, (y1 - oy) * zoom
    dx, dy = sx1 - sx0, sy1 - sy0
    ln = np.hypot(dx, dy) + 1e-6
    u = ((XX - sx0) * dx + (YY - sy0) * dy) / ln ** 2
    perp = np.abs((XX - sx0) * dy - (YY - sy0) * dx) / ln
    wid = (4 + 24 * u) * zoom
    m = (u > 0) & (u < 1) & (perp < wid)
    lv = clamp01(1 - perp / np.maximum(wid, 1)) * (0.15 + 0.25 * u) * k
    sel = m & (BAYER < lv)
    dst[sel] = np.minimum(dst[sel] + col * 0.35, 1.0)
    # 光柱里的浮尘
    dust = sel & ((np.floor(XX * 0.7 + YY * 1.3 + t * 30) % 29) == 0)
    dst[dust] = np.minimum(dst[dust] + col * 0.6, 1.0)


def moon_haze(dst, ox, oy, zoom, t):
    """月光的体积感：沿光束方向的淡淡斜纹。"""
    X = XX / zoom + ox
    Y = YY / zoom + oy
    mb = _in_quad(X, Y, MOONBEAM) & (Y < FLOOR_Y + 6)
    streak = mb & (((X + (Y - 40) * 0.52) * 0.5 + t * 2) % 7 < 1) & (BAYER < 0.35)
    dst[streak] = np.minimum(dst[streak] + MOON_BLUE * 0.18, 1.0)
    motes = mb & ((np.floor(X * 1.7 + Y * 0.9 - t * 12) % 53) == 0) & (BAYER < 0.5)
    dst[motes] = np.minimum(dst[motes] + MOON_BLUE * 0.5, 1.0)


# ================================================================ 展柜 / 宝石（每帧画，世界坐标 1x）
def draw_case(img, ox, oy, state='whole', crack=0.0, gem=True, t=0.0, card=False, lit=1.0):
    """img: 1x 画布（视口坐标）。state: whole / gone（玻璃碎了，只剩垫子）。"""
    def P(x, y, col):
        x, y = int(x - ox), int(y - oy)
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            img[y, x] = col
    def R(x0, y0, x1, y1, col, a=1.0):
        xa, ya, xb, yb = int(x0 - ox), int(y0 - oy), int(x1 - ox), int(y1 - oy)
        xa, ya, xb, yb = max(0, xa), max(0, ya), min(img.shape[1], xb), min(img.shape[0], yb)
        if xb > xa and yb > ya:
            img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a) + col * a
    k = lit
    # 丝绒垫
    R(386, 164, 414, 172, _c('#a0183a') * k + _night(_c('#a0183a')) * (1 - k))
    R(386, 164, 414, 165, _c('#e0406a') * k + _night(_c('#e0406a')) * (1 - k))
    R(388, 171, 412, 172, _c('#5a0a20'))
    if gem:
        cx, cy = X_CASE, 157
        rows = ['...rRRr...', '..rRWWRr..', '.rRRWRRRr.', 'rRRRRRRRRx', '.xRRRRRrx.', '..xrRRrx..',
                '...xrrx...', '....xx....']
        pal = {'W': _c('#ffe8f0'), 'R': RUBY[1], 'r': RUBY[0], 'x': RUBY[2]}
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                if ch != '.':
                    P(cx - 5 + i, cy - 4 + j, pal[ch])
    if card:                  # 被留下的名片（斜插在垫子上）
        for j in range(9):
            for i in range(6):
                x, y = 396 + i + j * 0.35, 156 + j
                P(x, y, _c('#f6f0e0') if 0 < i < 5 and 0 < j < 8 else _c('#c8a050'))
        P(398.5 + 1.5, 160, VIOLET)
        P(399.5 + 1.5, 160, VIOLET)
        P(398.5 + 1.8, 161, VIOLET)
        P(399.5 + 1.8, 162, VIOLET)
        P(398.5 + 2.1, 163, VIOLET)
    if state == 'whole':
        g = _c('#c8e0ff')
        R(383, 134, 417, 172, _c('#8ab0e8'), 0.14)              # 玻璃的淡蓝
        for x in (382, 417):
            R(x, 134, x + 1, 172, g, 0.8)
        R(382, 133, 418, 134, WHITE, 0.9)                        # 顶盖
        R(381, 131, 419, 133, _c('#e0d8f0'))
        R(381, 131, 419, 132, WHITE)
        for j in range(30):                                      # 斜反光
            P(387 + j * 0.3, 136 + j, WHITE if j % 6 < 3 else g)
            P(392 + j * 0.3, 136 + j, g) if j < 12 else None
        if crack > 0:                                            # 裂纹：从撞击点放射
            rng = np.random.default_rng(3)
            for a in rng.uniform(0, 2 * np.pi, 9):
                ln = crack * rng.uniform(10, 22)
                for r in range(int(ln)):
                    P(X_CASE + np.cos(a) * r + np.sin(r * 0.7) * 1.2, 152 + np.sin(a) * r, WHITE)


# ================================================================ 角色图（着色、缩放后缓存）
def _edge(a, dx, dy):
    return a & ~np.roll(np.roll(a, dy, 0), dx, 1)


@lru_cache(None)
def spr(ch, pose, scale=1, look='night', flip=False):
    """night = 月夜里（压暗偏蓝 + 右上月光轮廓光）；spot = 被探照灯照亮（原色 + 暖白轮廓）；
    violet / cyan = 幻影分身；white / black = 冲击帧；moonback = 背对满月的剪影（冷白轮廓光）。"""
    img = ch.body(pose).copy()
    if flip:
        img = img[:, ::-1].copy()
    a = img[..., 3] > 0
    if look == 'night':
        img[..., :3] = img[..., :3] * np.array([0.62, 0.62, 0.8], np.float32) + np.array([0.02, 0.02, 0.06])
        rim = (_edge(a, -1, 0) | _edge(a, 0, 1)) & a
        img[rim, :3] = img[rim, :3] * 0.4 + MOON_BLUE * 0.6
    elif look == 'spot':
        rim = (_edge(a, -1, 0) | _edge(a, 1, 0)) & a
        img[rim, :3] = img[rim, :3] * 0.5 + _c('#fff4d8') * 0.5
    elif look == 'violet':
        lum = img[..., :3].mean(-1, keepdims=True)
        img[..., :3] = lum * VIOLET_L * 0.9 + VIOLET * 0.25
        rim = (_edge(a, -1, 0) | _edge(a, 1, 0) | _edge(a, 0, 1) | _edge(a, 0, -1)) & a
        img[rim, :3] = _c('#f0e0ff')
    elif look == 'cyan':
        lum = img[..., :3].mean(-1, keepdims=True)
        img[..., :3] = lum * _c('#d0fff4') * 0.9 + MINT * 0.2
        rim = (_edge(a, -1, 0) | _edge(a, 1, 0) | _edge(a, 0, 1) | _edge(a, 0, -1)) & a
        img[rim, :3] = WHITE
    elif look == 'white':
        img[a, :3] = WHITE
    elif look == 'black':
        img[a, :3] = BLACK
    elif look == 'moonback':
        # 背对满月：几乎全黑的剪影，外轮廓一圈冷白月光；衣服的明暗只留一点点（看得出白礼服和披风的分界）
        lum = img[..., :3].mean(-1, keepdims=True)
        img[..., :3] = _c('#120a22') + lum * _c('#3a3060') * 0.5
        rim = (_edge(a, 1, 0) | _edge(a, -1, 0) | _edge(a, 0, 1)) & a
        img[rim, :3] = _c('#d8d8ff')
    return scale_img(img, scale) if scale > 1 else img


def put_char(dst, ch, pose, x, y, scale=1, look='night', alpha=1.0, flip=False, tint=None, amt=0.0):
    img = spr(ch, pose, scale, look, flip)
    blit(dst, img, x, y, anchor='bottom', alpha=alpha, tint=tint, tint_amt=amt)
    return img


def pt(ch, pose, name, x, y, scale, flip=False):
    """姿势关键点（手 / 杖头 / 宝石 / 眼）→ 屏幕坐标。(x, y) = 脚底中心。"""
    import chars.bleublanc as B
    px, py = B.points(pose)[name]
    if flip:
        px = B.BW - px
    return x - B.BW * scale / 2 + px * scale, y - B.BH * scale + py * scale


# ================================================================ 小特效
def star(dst, x, y, r, col=WHITE, k=1.0):
    """四角星闪光（十字 + 斜向短芒）。"""
    x, y = int(x), int(y)
    for d in range(-r, r + 1):
        f = 1 - abs(d) / (r + 1)
        for px, py in ((x + d, y), (x, y + d)):
            if 0 <= px < W and 0 <= py < H and BAYER[py, px] < f * k + 0.2:
                dst[py, px] = np.minimum(dst[py, px] * 0.3 + col, 1)
    for d in range(-(r // 3), r // 3 + 1):
        for px, py in ((x + d, y + d), (x + d, y - d)):
            if 0 <= px < W and 0 <= py < H:
                dst[py, px] = np.minimum(dst[py, px] + col * 0.6 * k, 1)


def glow(dst, cx, cy, r, col, k=1.0, flat=1.0):
    x0, x1 = max(0, int(cx - r)), min(W, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r * flat)), min(H, int(cy + r * flat) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    d = np.hypot(XX[y0:y1, x0:x1] - cx, (YY[y0:y1, x0:x1] - cy) / flat) / r
    lv = clamp01(1 - d) ** 1.5 * k
    m = BAYER[y0:y1, x0:x1] < lv
    dst[y0:y1, x0:x1][m] = np.minimum(dst[y0:y1, x0:x1][m] + col * 0.6, 1.0)


def tint_scene(dst, col, k):
    """特效的光把整个画面染色（亮部保留）。"""
    if k <= 0:
        return
    lum = dst.mean(-1, keepdims=True)
    dst[:] = dst * (1 - 0.5 * k) + (lum * 0.6 + 0.25) * col * k * 0.9


def spin_staff(dst, gx, gy, ang, zoom=2, trail=True, t=0.0):
    """握点 (gx, gy) 处旋转的导力杖：金杆 + 蓝色杖身 + 红宝珠；身后拖一道扇形光弧。"""
    L1, L2 = 17 * zoom, 30 * zoom          # 握点到杖头 / 杖尖
    if trail:
        for k in range(1, 7):
            a = ang - k * 0.16
            ux, uy = np.cos(a), np.sin(a)
            for r in range(8, L2, 2):
                for sgn, lim in ((1, L1), (-1, L2)):
                    if r > lim:
                        continue
                    px, py = int(gx + sgn * ux * r), int(gy + sgn * uy * r)
                    if 0 <= px < W and 0 <= py < H and BAYER[py, px] < 0.5 - k * 0.07:
                        dst[py, px] = np.minimum(dst[py, px] + (VIOLET_L if k < 3 else VIOLET) * 0.5, 1)
    ux, uy = np.cos(ang), np.sin(ang)
    for r in range(-L2, L1 + 1):
        px, py = gx + ux * r, gy + uy * r
        col = _c('#4a5a94') if r > L1 - 9 * zoom else (_c('#ffd66b') if (r // 3) % 4 == 0 else _c('#c08a2e'))
        for w in range(zoom):
            x, y = int(px - uy * w), int(py + ux * w)
            if 0 <= x < W and 0 <= y < H:
                dst[y, x] = col
    ox, oy = gx + ux * (L1 + 2 * zoom), gy + uy * (L1 + 2 * zoom)
    yy0, xx0 = int(oy - 2 * zoom), int(ox - 2 * zoom)
    for j in range(4 * zoom):
        for i in range(4 * zoom):
            if (i - 2 * zoom + .5) ** 2 + (j - 2 * zoom + .5) ** 2 <= (2 * zoom) ** 2:
                x, y = xx0 + i, yy0 + j
                if 0 <= x < W and 0 <= y < H:
                    dst[y, x] = RUBY[0] if i + j < 2 * zoom else RUBY[1]
    star(dst, ox, oy, 5, _c('#ffc0c8'), 0.7)


def mirror_frame(dst, x, y, w, h, k, col=_c('#ffe08a')):
    """镜子分身出现时的金框椭圆镜（k: 0..1 出现 → 消散）。"""
    if k <= 0 or k >= 1:
        return
    a = np.sin(np.pi * k)
    d = np.hypot((XX - x) / w, (YY - y) / h)
    ring = (np.abs(d - 1) < 0.035) & (BAYER < a + 0.1)
    dst[ring] = col
    inner = (d < 0.97) & (BAYER < 0.25 * a)
    dst[inner] = np.minimum(dst[inner] + _c('#c0d8ff') * 0.4, 1)
    sheen = (d < 0.97) & (np.abs((XX - x) + (YY - y) * 0.6 + (k - 0.5) * w * 3) < 3)
    dst[sheen] = np.minimum(dst[sheen] + WHITE * 0.5 * a, 1)


def vortex(dst, cx, fy, height, r0, t, zoom, k, bits=None):
    """花瓣旋风：三股螺旋丝带（前半圈亮、后半圈暗）+ 从旋风里甩出去的花瓣。k: 0..1 越转越高越快。"""
    for strand in range(3):
        for j in range(70):
            u = j / 69
            a = t * (10 + 10 * k) + u * 9 + strand * 2.094
            r = r0 * (0.35 + 0.65 * u) * zoom * (1 + 0.4 * k)
            x, y = cx + np.cos(a) * r, fy - u * height * zoom * (0.4 + 0.6 * k)
            front = np.sin(a) > 0
            col = VIOLET_L if front else VIOLET * 0.9
            for w in range(zoom):
                px, py = int(x) + w, int(y)
                if 0 <= px < W and 0 <= py < H and (front or BAYER[py, px] < 0.5):
                    dst[py, px] = col
    glow(dst, cx, fy - height * zoom * 0.4, r0 * zoom * 1.4, VIOLET, 0.35 + 0.4 * k, flat=1.6)
    if bits is not None:
        for _ in range(3):
            a = np.random.uniform(0, 2 * np.pi)
            u = np.random.uniform(0.1, 1)
            bits.emit(1, cx + np.cos(a) * r0 * zoom, fy - u * height * zoom * (0.4 + 0.6 * k),
                      -np.sin(a) * 160 * zoom / 2, np.cos(a) * 30 - 60, 0.5, 'vpetal', scale=(zoom - 1 or 1, zoom))


# ================================================================ 预告卡（带 B 的名片）
@lru_cache(None)
def calling_card():
    rows = [
        'GGGGGGGGGGGGGGGG',
        'GwwwwwwwwwwwwwwG',
        'GwggggggggggggwG',
        'GwgwwwwwwwwwwgwG',
        'GwgwwVVVVVwwwgwG',
        'GwgwwVvwwVVwwgwG',
        'GwgwwVvwwVVwwgwG',
        'GwgwwVVVVVwwwgwG',
        'GwgwwVvwwwVVwgwG',
        'GwgwwVvwwwVVwgwG',
        'GwgwwVvwwwVVwgwG',
        'GwgwwVVVVVVwwgwG',
        'GwgwwwwwwwwwwgwG',
        'GwgwwwwpPpwwwgwG',
        'GwgwwwwPPPwwwgwG',
        'GwgwwwwwqwwwwgwG',
        'GwgwwwwqwwwwwgwG',
        'GwgwwwwwwwwwwgwG',
        'GwggggggggggggwG',
        'GwwwwwwwwwwwwwwG',
        'GGGGGGGGGGGGGGGG',
    ]
    pal = {'G': '#c8a050', 'g': '#e8c870', 'w': '#fbf6ea', 'V': '#7a3ab8', 'v': '#c8a0f0', 'P': '#c02858',
           'p': '#ff7a9a', 'q': '#3a8a4a'}
    h, w = len(rows), len(rows[0])
    img = np.zeros((h, w, 4), np.float32)
    for y, r in enumerate(rows):
        for x, c in enumerate(r):
            img[y, x, :3] = _c(pal[c])
            img[y, x, 3] = 1
    return img


def card_img(sx, back=False):
    img = calling_card()
    h, w = img.shape[:2]
    nw = max(1, int(round(abs(sx) * w)))
    xs = (np.arange(nw) * w / nw).astype(int)
    out = img[:, xs].copy()
    if back or sx < 0:
        out[1:-1, 1:-1, :3] = _c('#2a1040')
        out[3:-3:3, 2:-2, :3] = _c('#4a2070')
    return out


# ================================================================ 穹顶夜景（第 6 小节）
@lru_cache(None)
def rooftop_layers():
    """(天空 + 月, 远处城市, 近处穹顶)，都是 (H+40, W) 的 RGBA；纵向多 40 像素给镜头上摇。"""
    HH = H + 40
    yy, xx = np.mgrid[0:HH, 0:W].astype(np.float32)
    bay = np.tile(BAYER[:4, :4], (HH // 4 + 1, W // 4 + 1))[:HH, :W]
    sky = np.zeros((HH, W, 4), np.float32)
    sky[..., 3] = 1
    pal = [_c(h) for h in ('#0a0620', '#140a34', '#1e1048', '#2c1660', '#3c2078', '#50308e')]
    mx, my, mr = 300, 118, 84
    d = np.hypot(xx - mx, yy - my)
    v = clamp01(1 - d / 330) * (len(pal) - 1)
    i0 = np.floor(v).astype(int)
    idx = np.clip(i0 + (bay < v - i0), 0, len(pal) - 1)
    sky[..., :3] = np.stack(pal)[idx]
    rng = np.random.default_rng(12)
    for x, y in zip(rng.uniform(0, W, 160), rng.uniform(0, HH * 0.7, 160)):
        if np.hypot(x - mx, y - my) > mr + 20:
            sky[int(y), int(x), :3] = _c('#c8c0ff') if rng.random() < 0.7 else _c('#ffffff')
    # 满月：亮盘 + 月海 + 抖动的光晕
    halo = (d < mr + 24) & (d >= mr) & (bay < clamp01((mr + 24 - d) / 24) * 0.6)
    sky[halo, :3] = _c('#6a5aa8')
    disc = d < mr
    sky[disc, :3] = _c('#f6f0ff')
    for cx, cy, r in ((-30, -25, 22), (18, -40, 14), (25, 10, 26), (-20, 30, 16), (-45, 5, 10), (40, 45, 9)):
        m = disc & (np.hypot(xx - mx - cx, yy - my - cy) < r)
        sky[m & (bay < 0.7), :3] = _c('#dcd4f0')
    rim = disc & (d > mr - 4) & (xx > mx) & (bay < 0.5)
    sky[rim, :3] = _c('#d8cce8')
    # 横穿月面的薄云
    for cy, x0, x1, th in ((236, 30, 150, 3), (228, 390, 470, 2), (250, 200, 330, 2)):   # 地平线上的几缕薄云
        for x in range(x0, x1):
            p = th * np.sin(np.pi * (x - x0) / (x1 - x0)) ** 0.5
            y0, y1 = int(cy - p * 0.5), int(cy + p * 0.5) + 1
            if np.hypot(x - mx, cy - my) > mr + 2:
                sky[y0:y1, x, :3] = _c('#2a1a50')
    # 远处的城市：屋顶、尖塔、钟楼，零星窗灯
    city = np.zeros((HH, W, 4), np.float32)
    top = np.full(W, 250.0)
    x = 0
    while x < W:
        bw = int(rng.integers(10, 30))
        hgt = rng.uniform(14, 40)
        kind = rng.random()
        for i in range(bw):
            if x + i >= W:
                break
            yv = 262 - hgt
            if kind < 0.25:                                  # 尖屋顶
                yv -= (bw / 2 - abs(i - bw / 2)) * 0.9
            elif kind < 0.33 and abs(i - bw // 2) < 2:       # 尖塔
                yv -= 30
            top[x + i] = yv
        x += bw
    m = yy >= top[None, :]
    city[m, :3] = _c('#120a26')
    city[m, 3] = 1
    city[m & ~np.roll(m, 1, 0), :3] = _c('#3a2a6a')
    for _ in range(140):
        wx, wy = int(rng.uniform(0, W)), int(rng.uniform(230, 290))
        if wy < HH and m[wy, wx]:
            city[wy, wx, :3] = _c('#ffd070') if rng.random() < 0.7 else _c('#ff9a50')
    # 近处：美术馆的玻璃穹顶（顶尖在 x=300），铁肋 + 月光反射
    dome = np.zeros((HH, W, 4), np.float32)
    apex_x, apex_y = 300, 232
    dy_ = yy - apex_y
    shape = dy_ >= ((xx - apex_x) / 230.0) ** 2 * 120
    dome[shape, :3] = _c('#1c1636')
    dome[shape, 3] = 1
    ang = np.arctan2(xx - apex_x, (dy_ + 60))
    ribs = shape & (np.abs(np.sin(ang * 9)) < 0.07)
    dome[ribs, :3] = _c('#4a4070')
    rings = shape & (np.abs(np.sin(np.hypot(xx - apex_x, (dy_ + 60) * 1.6) * 0.16)) < 0.08)
    dome[rings, :3] = _c('#3a3260')
    shine = shape & (np.abs((xx - apex_x) + dy_ * 0.8 + 60) < 10) & (bay < 0.35)
    dome[shine & ~ribs, :3] = _c('#6a78c0')
    edge = shape & ~np.roll(shape, 1, 0)
    dome[edge, :3] = _c('#a8b0f0')
    # 顶尖的小灯笼亭（他站在上面）
    for j in range(10):
        for i in range(-6 + j // 3, 7 - j // 3):
            y_, x_ = apex_y - 2 - j, apex_x + i
            dome[y_, x_, :3] = _c('#3a3260') if abs(i) < 4 - j // 4 else _c('#a8b0f0')
            dome[y_, x_, 3] = 1
    return sky, city, dome


# ================================================================ 渲染：美术馆各镜头
def gallery_view(ox, oy, vw, vh, t, spots, extra_light=None, moon=0.55, case=0.75, alarm=0.0):
    """世界区域 [ox, ox+vw) × [oy, oy+vh) 的 1x 画面（已打光），返回 (vh, vw, 3)。"""
    lit, dark, mid = room_layers()
    ox, oy = int(ox), int(oy)
    X = XG[oy:oy + vh, ox:ox + vw]
    Y = YG[oy:oy + vh, ox:ox + vw]
    L = light_mask(X, Y, t, spots, moon=moon, case=case, extra=extra_light)
    bay = BAYER_G[oy:oy + vh, ox:ox + vw]
    # 三档光（暗 / 月光 / 亮），只在档与档交界的窄带里抖动，大面积保持纯色
    lv = L * 2.0
    base = np.floor(np.minimum(lv, 1.999))
    frac = clamp01((lv - base - 0.5) * 3.0 + 0.5)
    idx = (base + (bay < frac)).astype(int)
    sl = (slice(oy, oy + vh), slice(ox, ox + vw))
    out = np.where((idx == 0)[..., None], dark[sl], np.where((idx == 1)[..., None], mid[sl], lit[sl])).copy()
    if alarm > 0:              # 警报红灯：整个大厅泛红
        out[:] = out * (1 - 0.45 * alarm) + _c('#ff2040') * out.mean(-1, keepdims=True) * 1.3 * alarm + \
            _c('#400010') * 0.4 * alarm
    return out, L


def draw_stanchions(img, ox, oy, L):
    """围栏（按光照在亮 / 暗版之间选）。img/L 都是 1x 视口。"""
    h, w = img.shape[:2]
    sl, sd = stanchion_layer(), stanchion_dark()
    a = sl[oy:oy + h, ox:ox + w, 3] > 0
    bay = BAYER_G[oy:oy + h, ox:ox + w]
    lit = bay < (L * 2 - 0.6)
    img[a & lit] = sl[oy:oy + h, ox:ox + w, :3][a & lit]
    img[a & ~lit] = sd[oy:oy + h, ox:ox + w, :3][a & ~lit]


def fg_columns(dst, cam_x, zoom=1):
    """前景的两根大柱子剪影（比背景多 0.4 的视差），给远景加纵深。"""
    for base in (6, 474):
        sx = int(base - (cam_x - 160) * 0.4)
        w = 30 * zoom
        x0, x1 = max(0, sx - w // 2), min(W, sx + w // 2)
        if x1 > x0:
            dst[:, x0:x1] = _c('#0a0612')
            if sx + w // 2 - 2 < W and sx + w // 2 - 2 >= 0:
                dst[:, sx + w // 2 - 2:min(W, sx + w // 2)] = _c('#2a2040')


# ================================================================ 场景状态
def _petal_path(k):
    """第一片花瓣从天窗飘到他现身的位置（世界坐标），k: 0..1。"""
    x = 402 - (402 - X_REAL) * k + 10 * np.sin(k * 9)
    y = 40 + (FEET_Y - 2 - 40) * k
    return x, y


def _clone_pos(i, k):
    """第 i 个分身（i = -1 是本体）在冲刺进度 k 时的脚底位置（世界坐标）。"""
    if i < 0:
        x0, fl = X_REAL, True
    else:
        x0, fl = CLONES[i]
    tx = X_CASE + DASH_END[i]
    e = k ** 2.2
    return x0 + (tx - x0) * e, FEET_Y - 4 * np.sin(np.pi * k), fl


FIG = [-1, 0, 1, 2, 3]
LOOKS = {-1: 'spot', 0: 'violet', 1: 'cyan', 2: 'violet', 3: 'cyan'}


# ================================================================ 主渲染
def render(dst, ctx):
    lt, t = ctx.lt, ctx.t
    ch = ctx.C['bleublanc']
    st = ctx.state
    at = ctx.at
    P, P2, bits = ctx.P, ctx.P2, ctx.bits
    dt = ctx.dt
    fps = 1 / dt if dt else 30

    # ---------------------------------------------------------------- 1.0–2.0.5：远景
    if lt < at(2, 0.5):
        k = ctx.prog(1, 0, 2, 1)
        cam = 186 - 26 * ease_out(k)
        spots = search_spots(t)
        # 花瓣落地前那一下，月光里的光斑
        kp = clamp01(lt / at(2, 0))
        px, py = _petal_path(kp)
        swirl = ctx.since(2, 0)
        extra = None
        if swirl > 0:
            extra = (np.hypot((XG[:, int(cam):int(cam) + W] - X_REAL) / 20,
                              (YG[:, int(cam):int(cam) + W] - 219) / 5) < 1).astype(np.float32) * 0.8
        img, L = gallery_view(cam, 0, W, H, t, spots, extra)
        draw_case(img, cam, 0, 'whole', t=t)
        draw_stanchions(img, int(cam), 0, L)
        dst[:] = img
        moon_haze(dst, cam, 0, 1, t)
        for sp in spots:
            beam_haze(dst, cam, 0, 1, SEARCH_SRC, sp, t)
        # 宝石闪光（1.1 / 1.3）
        gx, gy = X_CASE - cam, 155
        for b in (1, 3):
            s = ctx.since(1, b)
            if 0 <= s < 0.35:
                star(dst, gx + 2, gy - 2, int(7 * (1 - s / 0.35)) + 2, _c('#ffe0e8'))
        glow(dst, gx, gy, 6, RUBY[0], 0.5)
        # 飘落的花瓣（在月光里发亮）
        if swirl < 0:
            sx, sy = px - cam, py
            inbeam = _in_quad(np.array([px]), np.array([py]), MOONBEAM)[0]
            from fx import VPETAL
            blit(dst, VPETAL, sx, sy, scale=2, anchor='center', flip=int(t * 6) % 2 == 1)
            glow(dst, sx, sy, 8 if inbeam else 4, VIOLET_L, 0.8 if inbeam else 0.4)
            if ctx.rng.random() < 0.5:
                P.emit(1, sx, sy, (-6, 6), (-10, 0), (0.4, 0.8), 'violet', size=1, bright=0.7)
        else:
            # 2.0：花瓣旋风 + 人影浮现
            k2 = clamp01(swirl / (at(2, 1) - at(2, 0)))
            cx = X_REAL - cam
            if ctx.crossed(2, 0):
                ctx.shake(1, 0.2)
                P.emit_radial(30, cx, FEET_Y - 2, (20, 80), (0.3, 0.6), 'violet', size=1, drag=3)
            vortex(dst, cx, FEET_Y, 80, 14, t, 1, k2, bits)
            P.emit(6, (cx - 14, cx + 14), (FEET_Y - 70 * k2, FEET_Y), (-30, 30), (-120, -40), (0.2, 0.4), 'violet',
                   size=1, bright=0.9)
        fg_columns(dst, cam)
        bits.step(dt)
        P.step(dt)
        bits.render(dst, t)
        P.render(dst)
        return dst

    # 中景的视口（1x 世界区域 240×135，放大 2x）
    MX, MY = X_CASE - 150, FEET_Y - 95

    def medium(camx=0.0, spots=None, extra=None, moon=0.55, case=0.75, gem=True, state='whole', crack=0.0,
               card=False, alarm=0.0):
        ox = MX + camx
        sp = spots if spots is not None else search_spots(t)
        img, L = gallery_view(ox, MY, 240, 135, t, sp, extra, moon, case, alarm)
        draw_case(img, ox, MY, state, crack, gem, t, card)
        draw_stanchions(img, int(ox), MY, L)
        dst[:] = scale_img(img, 2)
        moon_haze(dst, ox, MY, 2, t)
        return ox, sp

    def to_scr(wx, wy, ox):
        return (wx - ox) * 2, (wy - MY) * 2

    # ---------------------------------------------------------------- 2.0.5–2.1：硬切中景，旋风里浮出人影
    if lt < at(2, 1):
        k2 = clamp01(ctx.since(2, 0) / (at(2, 1) - at(2, 0)))
        ox, sp = medium(-6)
        fx_, fy = to_scr(X_REAL, FEET_Y, ox)
        tint_scene(dst, VIOLET, 0.2 * k2)
        vortex(dst, fx_, fy, 80, 14, t, 2, k2, bits)
        img = spr(ch, 'land', 2, 'violet').copy()
        h_, w_ = img.shape[:2]
        keep = np.tile(BAYER[:4, :4], (h_ // 4 + 1, w_ // 4 + 1))[:h_, :w_] < (k2 - 0.4) * 1.6
        img[~keep, 3] = 0
        blit(dst, img, fx_, fy, anchor='bottom')
        vortex(dst, fx_, fy, 80, 14, t + 0.13, 2, k2)
        bits.step(dt)
        P.step(dt)
        bits.render(dst, t)
        P.render(dst)
        return dst

    # ---------------------------------------------------------------- 2.1–4.0：中景（现身、行礼、扶面具）
    if lt < at(4, 0):
        s21 = ctx.since(2, 1)
        k = ctx.prog(2, 1, 4, 0)
        camx = -6 + 10 * ease_out(k)
        # 探照灯：2.2 开始扫过来，2.3 锁定他
        klock = smooth(at(2, 2), at(2, 3), lt)
        target = (X_REAL, 186)
        spots = search_spots(t, target=target, k_lock=klock)
        lit_him = lt >= at(2, 3)
        ox, sp = medium(camx, spots)
        if lit_him:
            beam_haze(dst, ox, MY, 2, SEARCH_SRC, sp[0], t, k=1.3)
        # 姿势时间表
        if s21 < 0.12:
            pose = 'land'
        elif lt < at(3, 0):
            amp = int(np.exp(-(s21 - 0.12) / 0.6) * 3.5)
            pose = f'idle:{int(lt * 12) % 4}' if amp > 0 else 'idle'
        elif lt < at(3, 1.5):
            pose = 'bow'
        else:                            # 扶面具站定；披风被大厅里的穿堂风吹得轻轻摆
            pose = f'idle:{int(lt * 6) % 4}'
        fx_, fy = to_scr(X_REAL, FEET_Y, ox)
        look = 'spot' if lit_him else 'night'
        if pose == 'bow':
            fy += 0
        put_char(dst, ch, pose, fx_, fy, 2, look)
        # 2.1：花瓣炸开
        if ctx.crossed(2, 1):
            ctx.shake(4, 0.3)
            ctx.flash('#c89af0', 0.07, 0.45)
            bits.emit(40, (fx_ - 30, fx_ + 30), (fy - 100, fy - 10), (-260, 260), (-240, 60), (0.6, 1.4), 'vpetal',
                      scale=(2, 3), grav=120, drag=1.5)
            P2.emit_radial(120, fx_, fy - 60, (60, 280), (0.3, 0.8), 'violet', size=(1, 2), drag=2.5)
            P2.emit(40, (fx_ - 50, fx_ + 50), fy, (-160, 160), (-40, -5), (0.4, 0.8), 'white', size=1, drag=3,
                    bright=0.5)
        if 0 <= s21 < 0.15:
            glow(dst, fx_, fy - 70, 60 * (1 - s21 / 0.15), VIOLET_L, 1.0)
        if ctx.crossed(2, 3):
            ctx.flash('#fff8e0', 0.08, 0.35)
        # 3.0 行礼：披风一甩，带出一串花瓣
        if ctx.crossed(3, 0):
            bits.emit(18, (fx_ - 50, fx_ - 10), (fy - 110, fy - 50), (-200, -40), (-80, 40), (0.8, 1.6), 'vpetal',
                      scale=(2, 2), grav=60, drag=1.2)
        # 3.2 面具一闪
        s32 = ctx.since(3, 2)
        if 0 <= s32 < 0.4:
            ex, ey = pt(ch, 'idle', 'eye', fx_, fy, 2)
            star(dst, ex + 1, ey, int(9 * (1 - s32 / 0.4)) + 2, _c('#d8fff8'))
        # 宝石一直在闪
        gx, gy = to_scr(X_CASE, 155, ox)
        if int(t * 3) % 5 == 0:
            star(dst, gx + 3, gy - 3, 5, _c('#ffe0e8'), 0.6)
        if lt >= at(2, 1) and ctx.since(2, 1) < 2.2:
            ui.name_tag(dst, ch, ctx.since(2, 1))
        bits.step(dt)
        P2.step(dt)
        bits.render(dst, t)
        P2.render(dst)
        return dst

    # ---------------------------------------------------------------- 4.0–4.2：特写（立绘 3x，坏笑）
    if lt < at(4, 2):
        s40 = ctx.since(4, 0)
        ox = MX + 20
        img, L = gallery_view(ox + 30, MY - 70, 240, 135, t, search_spots(t, target=(X_REAL, 186), k_lock=1.0))
        dst[:] = scale_img(img, 2)
        dst[:] = dst * 0.45 + _c('#1a0a2a') * 0.55
        por = ch.portrait('smirk')
        big = scale_img(por, 3)
        px, py = 150 - 6 * s40 / 0.83, 12
        blit(dst, big, px, py)
        # 面具上划过的一道反光；4.1 镜孔里的眼睛一闪
        sw = clamp01(s40 / 0.3)
        for j in range(40):
            x = int(px + (30 + 70 * sw) * 3 / 3 + j * 0.6 + 60)
            y = int(py + 70 + j * 1.2)
            if 0 <= x < W and 0 <= y < H and sw < 1:
                dst[y, x:x + 3] = np.minimum(dst[y, x:x + 3] + 0.5, 1)
        s41 = ctx.since(4, 1)
        if s41 >= 0:
            ex, ey = px + 42 * 3, py + 32 * 3
            r = int(22 * np.exp(-s41 / 0.25)) + 3
            star(dst, ex, ey, r, _c('#d8fff8'))
            glow(dst, ex, ey, 14 * np.exp(-s41 / 0.3) + 3, MINT, 1.0)
            if ctx.crossed(4, 1):
                P2.emit_radial(24, ex, ey, (40, 140), (0.2, 0.5), 'mint', size=1, drag=3)
        for _ in range(2):
            bits.emit(1, W + 4, (0, H - 80), (-200, -120), (-20, 30), 2.0, 'vpetal', scale=(2, 3))
        bits.step(dt)
        P2.step(dt)
        bits.render(dst, t)
        P2.render(dst)
        return dst

    # ---------------------------------------------------------------- 4.2–5.0：转杖、响指、分身、冲刺
    if lt < at(5, 0):
        k = ctx.prog(4, 2, 5, 0)
        camx = 4 + 6 * k
        s43 = ctx.since(4, 3)
        dim = smooth(at(4, 3.5), at(4, 3.75), lt)
        spots = search_spots(t, target=(X_REAL, 186), k_lock=1.0 - dim)
        ox, sp = medium(camx, spots, moon=0.55 - 0.2 * dim, case=0.75)
        if dim < 0.5:
            beam_haze(dst, ox, MY, 2, SEARCH_SRC, sp[0], t, k=1.3 * (1 - 2 * dim))
        tint_scene(dst, VIOLET, 0.25 * clamp01(s43 / 0.2) if s43 >= 0 else 0.0)
        s375 = ctx.since(4, 3.75)
        # 分身：4.3 从镜中走出（闪烁），4.3.5 预备，4.3.75 冲刺
        figs = [-1] + ([0, 1, 2, 3] if s43 >= 0 else [])
        for i in figs:
            if s375 < 0:
                x0 = X_REAL if i < 0 else CLONES[i][0]
                fl = True if i < 0 else CLONES[i][1]
                wx, wy = x0, FEET_Y
                if lt < at(4, 3.5):
                    pose = 'twirl'
                else:
                    pose = 'dash_ready'
                    fl = not fl if pose == 'dash_ready' and False else fl
            else:
                kk = clamp01(s375 / (at(5, 0) - at(4, 3.75)))
                wx, wy, fl = _clone_pos(i, kk)
                pose = 'dash'
            sx, sy = to_scr(wx, wy, ox)
            look = LOOKS[i] if i >= 0 else ('spot' if dim < 0.5 else 'night')
            alpha = 1.0
            if i >= 0:
                a = clamp01(s43 / 0.25)
                flick = (int(t * 30) + i) % 4 == 0
                alpha = (0.35 + 0.4 * a) * (0.5 if flick else 1.0)
                if s43 < 0.25:
                    mirror_frame(dst, sx, sy - 70, 36, 78, s43 / 0.3)
            # 冲刺拖影
            if pose == 'dash':
                kk = clamp01(s375 / (at(5, 0) - at(4, 3.75)))
                for j in range(1, 5):
                    kj = max(0.0, kk - j * 0.09)
                    tx_, ty_, _ = _clone_pos(i, kj)
                    qx, qy = to_scr(tx_, ty_, ox)
                    put_char(dst, ch, 'dash', qx, qy, 2, 'violet' if i != 1 and i != 3 else 'cyan',
                             alpha=0.35 - j * 0.07, flip=fl)
                # 速度线
                for j in range(6):
                    yy = int(sy - 20 - j * 18)
                    x0_ = int(sx + (60 if fl else -60) * (1 + j % 3 * 0.3))
                    xa, xb = sorted((x0_, int(sx + (140 if fl else -140))))
                    xa, xb = max(0, xa), min(W, xb)
                    if 0 <= yy < H and xb > xa:
                        dst[yy, xa:xb] = np.minimum(dst[yy, xa:xb] * 0.5 + 0.55, 1)
            put_char(dst, ch, pose, sx, sy, 2, look, alpha=alpha, flip=fl)
            # 转杖：杖绕右手转，拖光弧（本体和分身都转）
            if pose == 'twirl':
                gx, gy = pt(ch, 'twirl', 'hand_r', sx, sy, 2, fl)
                ang = (lt - at(4, 2)) * 2 * np.pi / (at(4, 1) - at(4, 0)) * 1.0 * (-1 if fl else 1) + 0.6
                if i < 0:
                    spin_staff(dst, gx, gy, ang, 2, trail=True, t=t)
            if i < 0 and pose == 'twirl' and 0 <= s43 < 0.25:     # 响指的火花（左手）
                hx, hy = pt(ch, 'twirl', 'hand_l', sx, sy, 2, fl)
                star(dst, hx, hy - 4, int(10 * (1 - s43 / 0.25)) + 2, VIOLET_L)
        if ctx.crossed(4, 3):
            ctx.flash('#e0c0ff', 0.1, 0.4)
            for i in range(4):
                sx, sy = to_scr(CLONES[i][0], FEET_Y, ox)
                P2.emit(30, (sx - 20, sx + 20), (sy - 140, sy), (-40, 40), (-60, 20), (0.3, 0.7),
                        'violet' if i % 2 == 0 else 'mint', size=1, drag=2)
        if ctx.crossed(4, 3.75):
            ctx.shake(3, 0.15)
        gx, gy = to_scr(X_CASE, 155, ox)
        glow(dst, gx, gy, 10, RUBY[0], 0.6)
        P2.step(dt)
        bits.step(dt)
        bits.render(dst, t)
        P2.render(dst)
        return dst

    # ---------------------------------------------------------------- 5.0–5.2：命中，玻璃炸碎（中景）
    if lt < at(5, 2):
        s5 = ctx.since(5, 0)
        hit = 5 / fps                 # 2 帧冲击帧 + 3 帧顿帧
        ox = MX + 10
        if ctx.crossed(5, 0):
            ctx.shake(9, 0.6)
        e = s5 - hit
        # 顿帧：分身停在撞上玻璃的那一刻，玻璃满是裂纹
        state = 'whole' if e < 0 else 'gone'
        vio = clamp01(1 - max(e, 0) / 0.8)
        ox, sp = medium(0, search_spots(t, 'panic'), gem=False if e >= 0 else True, state=state,
                        crack=1.0 if e < 0 else 0.0, card=e >= 0.05, case=0.75)
        tint_scene(dst, VIOLET, 0.35 * vio ** 2)
        if e < 0:
            for i in FIG:
                wx, wy, fl = _clone_pos(i, 1.0)
                sx, sy = to_scr(wx, wy, ox)
                put_char(dst, ch, 'dash', sx, sy, 2, LOOKS[i] if i >= 0 else 'violet', flip=fl)
            cx, cy = to_scr(X_CASE, 152, ox)
            glow(dst, cx, cy, 50, WHITE, 1.0)
            if s5 < 2 / fps:          # 冲击帧：整张画面反相成黑白两色（第二帧再叠放射线）
                lum = dst.mean(-1)
                neg = np.where((lum > 0.42)[..., None], BLACK, WHITE)
                neg[(lum > 0.25) & (lum <= 0.42)] = _c('#b89ae0')
                dst[:] = neg
                if s5 >= 1 / fps:
                    a_ = np.arctan2(YY - cy, XX - cx)
                    rays = (np.abs(np.sin(a_ * 11)) < 0.06) & (np.hypot(XX - cx, YY - cy) > 60)
                    dst[rays] = BLACK
            return dst
        cx, cy = to_scr(X_CASE, 152, ox)
        if not st.get('burst'):
            st['burst'] = True
            ctx.flash('#e8d8ff', 0.05, 0.4)
            bits.emit(46, (cx - 30, cx + 30), (cy - 40, cy + 30), (-420, 420), (-380, 120), (0.8, 1.6), 'shard',
                      scale=(1, 3), grav=520, drag=0.6)
            bits.emit(70, (cx - 20, cx + 20), (cy - 60, cy + 40), (-360, 360), (-320, 160), (1.0, 2.0), 'vpetal',
                      scale=(2, 3), grav=90, drag=1.3)
            P2.emit_radial(260, cx, cy, (80, 460), (0.4, 1.1), 'violet', size=(1, 3), drag=1.8)
            P2.emit_radial(80, cx, cy, (60, 300), (0.3, 0.8), 'white', size=(1, 2), drag=2.2)
            P2.emit(60, (cx - 90, cx + 90), (cy + 60, cy + 70), (-60, 60), (-40, -5), (0.4, 0.9), 'white', size=1,
                    drag=3, bright=0.5)
        # 分身化作花瓣：残像在 0.3 秒里被抖动吃掉
        dis = clamp01(e / 0.3)
        if dis < 1:
            for i in FIG:
                wx, wy, fl = _clone_pos(i, 1.0)
                sx, sy = to_scr(wx, wy, ox)
                img = spr(ch, 'dash', 2, LOOKS[i] if i >= 0 else 'violet', fl).copy()
                img[..., 3] *= 0.8
                h_, w_ = img.shape[:2]
                cut = np.tile(BAYER[:4, :4], (h_ // 4 + 1, w_ // 4 + 1))[:h_, :w_] < dis
                img[cut, 3] = 0
                blit(dst, img, sx, sy - 10 * dis, anchor='bottom')
                if ctx.rng.random() < 0.8:
                    bits.emit(1, (sx - 30, sx + 30), (sy - 130, sy), (-80, 80), (-160, -40), (0.8, 1.4), 'vpetal',
                              scale=(2, 2), grav=60, drag=1)
        glow(dst, cx, cy, 30 * np.exp(-e / 0.15) + 6, VIOLET_L, 0.8)
        # 地面冲击环
        if e < 0.5:
            from fx import shockwave
            shockwave(dst, cx, cy + 128, 20 + e * 600, _c('#f0e0ff'), w=3, flat=0.15)
        bits.step(dt)
        P2.step(dt)
        bits.render(dst, t)
        P2.render(dst)
        return dst

    # ---------------------------------------------------------------- 5.2–6.0：远景，巨大紫玫瑰绽放 → 花瓣雨，警报
    if lt < at(6, 0):
        s52 = ctx.since(5, 2)
        cam = 160
        alarm = max(0.0, np.cos(np.pi * s52 / ctx.sec.beat)) ** 6 * 0.4 if s52 >= 0 else 0.0
        img, L = gallery_view(cam, 0, W, H, t, search_spots(t, 'panic'), alarm=alarm, case=0.6)
        draw_case(img, cam, 0, 'gone', gem=False, card=True)
        draw_stanchions(img, int(cam), 0, L)
        dst[:] = img
        for sp in search_spots(t, 'panic'):
            beam_haze(dst, cam, 0, 1, SEARCH_SRC, sp, t, col=_c('#ffe0e0'))
        rx, ry = X_CASE - cam, 140
        s53 = ctx.since(5, 3)
        if s53 < 0:
            g = ease_out(clamp01(s52 / 0.3))
            tint_scene(dst, VIOLET, 0.5 * g)
            from fx import rose
            rose(dst, rx, ry, 8 + 46 * g, t)
            glow(dst, rx, ry, 70 * g, VIOLET_L, 0.5)
            if ctx.crossed(5, 2):
                ctx.shake(5, 0.4)
                P2.emit_radial(120, rx, ry, (40, 200), (0.4, 1.0), 'violet', size=(1, 2), drag=2)
        else:
            tint_scene(dst, VIOLET, 0.5 * np.exp(-s53 / 0.4))
            if ctx.crossed(5, 3):
                ctx.shake(6, 0.4)
                ctx.flash('#f0d8ff', 0.15, 0.6)
                bits.emit(110, (rx - 50, rx + 50), (ry - 50, ry + 50), (-320, 320), (-340, 120), (1.4, 2.6), 'vpetal',
                          scale=(1, 2), grav=70, drag=1.2)
                P2.emit_radial(240, rx, ry, (80, 400), (0.4, 1.2), 'violet', size=(1, 2), drag=1.6)
                P2.emit_radial(60, rx, ry, (60, 260), (0.4, 0.9), 'mint', size=1, drag=2)
            glow(dst, rx, ry, 30 * np.exp(-s53 / 0.2) + 2, WHITE, 1.0)
        # 远处门口跑进来的警卫（剪影 + 手电）
        if s52 > 0.2:
            for j, gx0 in enumerate((560, 610)):
                gx = gx0 - cam - (s52 - 0.2) * 90 + j * 10
                gy = FEET_Y
                step_ = int((t * 8 + j) % 2)
                dst[int(gy - 26):int(gy - 4), int(gx - 3):int(gx + 3)] = _c('#0a0612')
                dst[int(gy - 31):int(gy - 25), int(gx - 2):int(gx + 3)] = _c('#0a0612')
                dst[int(gy - 33):int(gy - 30), int(gx - 3):int(gx + 4)] = _c('#1a1a3a')
                dst[int(gy - 4):int(gy), int(gx - 3 + step_ * 2):int(gx - 1 + step_ * 2)] = _c('#0a0612')
                dst[int(gy - 4):int(gy), int(gx + 1 - step_ * 2):int(gx + 3 - step_ * 2)] = _c('#0a0612')
                glow(dst, gx - 6, gy - 18, 5, _c('#fff0c0'), 1.0)
        fg_columns(dst, cam)
        bits.step(dt)
        P2.step(dt)
        bits.render(dst, t)
        P2.render(dst)
        return dst

    # ---------------------------------------------------------------- 6：穹顶，满月下高举宝石；名片飘向镜头
    s6 = ctx.since(6, 0)
    k6 = ctx.prog(6, 0, 7, 0)
    sky, city, dome = rooftop_layers()
    tilt = 26 - 22 * ease_out(k6)            # 镜头缓缓上摇（背景往下走）
    blit(dst, sky, 0, -40 + tilt * 0.3)
    blit(dst, city, 0, -40 + tilt * 0.6)
    blit(dst, dome, 0, -40 + tilt)
    feet = 232 - 11 - 40 + tilt              # 站在穹顶尖的小亭子顶上
    fx_ = 300
    pose = f'jewel:{int(lt * 10) % 4}'
    put_char(dst, ch, pose, fx_, feet, 2, 'moonback')
    gx, gy = pt(ch, 'jewel', 'gem', fx_, feet, 2)
    glow(dst, gx, gy, 12 + 3 * np.sin(t * 9), RUBY[0], 0.9)
    star(dst, gx, gy, int(6 + 4 * (1 - clamp01(s6 / 0.3))) if s6 < 0.3 else 5 + int(t * 8) % 2, _c('#ffd0d8'))
    if ctx.crossed(6, 0):
        P2.emit_radial(40, gx, gy, (40, 160), (0.3, 0.7), 'rose', size=1, drag=2.5)
    # 风里的花瓣
    if ctx.rng.random() < 0.7:
        bits.emit(1, -6, (40, 230), (160, 260), (-30, 30), 3.0, 'vpetal', scale=(1, 2))
    # 名片：6.1 从他手边弹出，旋转着朝镜头飘来（越来越大），最后定在左下
    s61 = ctx.since(6, 1)
    if s61 >= 0:
        kc = clamp01(s61 / 0.7)
        e = ease_out(kc)
        cx = 330 - 236 * e + 10 * np.sin(s61 * 7) * (1 - kc)
        cy = 70 + 20 * e + 14 * np.sin(s61 * 5) * (1 - kc)
        scale = 1 if kc < 0.3 else (2 if kc < 0.65 else 3)
        sx = np.cos(s61 * 13) if kc < 0.9 else np.cos((1 - kc) * 10)
        if abs(sx) < 0.08:
            sx = 0.08
        img = card_img(sx)
        blit(dst, img, cx, cy, scale=scale, anchor='center')
        if kc >= 0.95:
            glow(dst, cx + 12, cy - 20, 8, _c('#fff0c0'), 0.8)
    bits.step(dt)
    P2.step(dt)
    bits.render(dst, t)
    P2.render(dst)
    return dst
