"""第五幕 · 苍之深渊 薇塔·克洛蒂尔德 —— 帝都海姆达尔的帝国歌剧院。

原作：她的另一重身份是帝都的当红歌姬「蜜斯缇（Misty）」；闪之轨迹 II 里她唤起幻焰城，
使魔是苍色巨鸟「古里亚诺丝（Grianos）」。

分镜（本幕 6 小节 @144，小节从 1 数、拍从 0 数）：
  1.0  全景：金色镜框台口、红丝绒帷幕、巨型水晶吊灯、三层包厢；聚光灯「咔」地打亮，她在台中央高歌
       （合扇向外扬起、左手按胸，两个呼吸姿势来回），每拍从嘴边飘出音符
  2.0  硬切中景 + 缓慢推近，前景观众剪影向下滑出画面；名牌
  2.3  转身：侧仰的歌唱脸 -> 正面 -> 面向镜头的招牌姿势，2.3.5 折扇「啪」地展开
  3.0  硬切特写（头像 2x，smirk）+ 台词；3.2 聚光灯转蓝，歌剧院开始碎成蓝色羽毛，夜空与巨月从裂缝里露出
  4.0  硬切回中景：剧场只剩残片，她站在消散的舞台上；4.1 魔杖在左手现形，4.1-4.3 举杖蓄力（脚下与杖头魔法阵、
       光点向杖尖汇聚），4.3 下蹲预备
  5.0  命中：冲击帧（白底黑剪影 -> 反色）+ 顿帧 4 帧；杖尖苍蓝光柱冲天，苍鸟古里亚诺丝从右向左掠过巨月，
       羽毛炸开，蓝 -> 青 -> 黄绿音波环扩散，整个画面被染蓝
  5.2  扇子下扫放出第二波羽毛与音波环
  6.0  苍鸟从左侧折返，绕到月前张开双翼悬停；她浮起，羽毛落下 —— 定格给扑克牌转场
"""
from functools import lru_cache

import numpy as np

import fx
from gfx import BAYER, H, W, XX, YY, blit, clamp01, dilate, dither_mask, ease_out, hexc, smooth
import ui

LETTERBOX = 0
SAYS = [('vita', 3, 0.0, 'dialog', (4, 2.5))]    # 举杖蓄力前收起对话框
ACCENTS = [(5, 0.0, 'hit'), (7, 0.0, 'riser')]
CUES = [
    (1, 0.0, 'spot_on', -3),        # 聚光灯打亮
    (2, 3.0, 'swish', -8),          # 转身
    (2, 3.5, 'fan_snap', -2),       # 折扇展开
    (3, 2.0, 'shimmer', -6),        # 聚光灯转蓝
    (3, 2.0, 'feathers', -5),       # 剧场碎成羽毛
    (4, 0.0, 'glass', -10),         # 硬切：墙面崩成光
    (4, 1.0, 'sparkle', -4),        # 魔杖现形
    (4, 1.0, 'charge', -3),         # 举杖蓄力（1.2 s，落在 5.0 前）
    (4, 3.0, 'wind', -8),           # 下蹲吸气
    (5, 0.0, 'big_impact', 0),      # 命中
    (5, 0.0, 'azure_burst', -1),
    (5, 0.0, 'magic_beam', -4),
    (5, 0.25, 'bird_cry', -3),      # 苍鸟掠月
    (5, 1.0, 'wing', -2),
    (5, 2.0, 'feathers', -5),       # 第二波
    (5, 2.0, 'chime', -9),
    (6, 0.0, 'wing', -3),           # 苍鸟折返
    (6, 0.5, 'bird_cry', -10),
    (6, 1.0, 'wing', -6),
    (6, 2.0, 'shimmer', -10),
]

TAU = 2 * np.pi
_R = np.random.default_rng(55)


def C(h):
    return hexc(h)


# ================================================================= 静态图层
def _poly(pts):
    return fx.polygon_mask(pts)


@lru_cache(None)
def opera_world():
    """帝国歌剧院（从观众席看向舞台）。世界坐标 480x270，镜头在此基础上缩放取样。返回 RGB。"""
    img = np.zeros((H, W, 3), np.float32)
    img[:] = C('#1c0810')
    x, y = XX, YY
    # ---- 天花板：暗色藻井 + 金色花格
    ceil = y < 26
    img[ceil] = C('#12060c')
    img[ceil & ((x % 40) < 2)] = C('#5a3a14')
    img[(np.abs(y - 25) < 1)] = C('#8a5a1a')
    img[(np.abs(y - 23) < 0.5)] = C('#c8902e')
    ros = ceil & (np.hypot((x % 40) - 20, y - 12) < 3)
    img[ros] = C('#8a5a1a')
    img[ceil & (np.hypot((x % 40) - 20, y - 12) < 1.2)] = C('#f0c860')
    # ---- 两侧包厢（三层，每侧三个，越靠台口越窄）
    tiers = [(34, 84), (92, 140), (148, 194)]
    cols_l = [0, 30, 54, 76]
    rng = np.random.default_rng(3)
    for side in (0, 1):
        for ti, (ty0, ty1) in enumerate(tiers):
            # 层间金色檐口
            band = (y >= ty0 - 6) & (y < ty0 - 1)
            xs = (x < 80) if side == 0 else (x >= 400)
            img[band & xs] = C('#8a5a1a')
            img[(np.abs(y - (ty0 - 5)) < 0.5) & xs] = C('#f0c860')
            img[(np.abs(y - (ty0 - 2)) < 0.5) & xs] = C('#4a2e0e')
            for bi in range(3):
                a, b = cols_l[bi], cols_l[bi + 1]
                if side == 1:
                    a, b = W - cols_l[bi + 1], W - cols_l[bi]
                box = (x >= a + 2) & (x < b - 1) & (y >= ty0) & (y < ty1)
                img[box] = C('#2e0812')
                img[box & (y < ty0 + 14)] = C('#3a0c18')
                # 包厢帘幔
                sc = box & (y < ty0 + 4 + 2 * np.abs(np.sin((x - a) * np.pi / max(1, (b - a - 3)) * 2)))
                img[sc] = C('#8a1424')
                img[box & (np.abs(y - ty0) < 0.6)] = C('#d8a440')
                # 观众（头 + 肩），坐在栏杆后
                n = rng.integers(1, 3)
                for k in range(n):
                    hx = rng.uniform(a + 6, b - 6)
                    hy = ty1 - 18
                    head = (np.hypot(x - hx, (y - hy) * 0.9) < 3.2)
                    sh = (np.hypot((x - hx) * 0.5, y - (hy + 7)) < 3.6)
                    img[(head | sh) & box] = C('#0a0306')
                    img[head & box & (y < hy - 1.5) & (x < hx + 1)] = C('#4a2418')
                    if rng.random() < 0.4:
                        img[int(hy), int(hx + 2)] = C('#f0c860')      # 观剧镜的反光
                # 栏杆：红丝绒面板 + 金色扶手
                par = box & (y >= ty1 - 12)
                img[par] = C('#6a1020')
                img[par & (y < ty1 - 10)] = C('#f0c860')
                img[par & (np.abs(y - (ty1 - 9)) < 0.5)] = C('#8a5a1a')
                img[par & (y >= ty1 - 2)] = C('#c8902e')
                img[par & (((x - a) % 6) < 1) & (y > ty1 - 9) & (y < ty1 - 3)] = C('#c8902e')
            # 包厢之间的金色壁柱
            for c in cols_l:
                cx_ = c if side == 0 else W - c
                pil = (np.abs(x - cx_) < 1.6) & (y >= ty0 - 1) & (y < ty1)
                img[pil] = C('#c8902e')
                img[pil & (np.abs(x - cx_ + 0.8) < 0.6)] = C('#f0c860')
    # ---- 台口（金色镜框 + 拱顶）
    def arch_top(xx):
        return 46 + 14 * ((xx - 240) / 144) ** 2
    opening = (x >= 96) & (x < 384) & (y >= arch_top(x)) & (y < 196)
    frame = (x >= 82) & (x < 398) & (y >= arch_top(x) - 14) & (y < 200) & ~opening
    img[frame] = C('#c8902e')
    img[frame & ((y < arch_top(x) - 12) | (x < 84) | (x >= 396))] = C('#5a3a10')
    img[frame & (((y >= arch_top(x) - 3) & (y < arch_top(x) - 1)) | ((x >= 93) & (x < 95)) | ((x >= 385) & (x < 387)))] = C('#8a5a1a')
    img[frame & (((y >= arch_top(x) - 11) & (y < arch_top(x) - 10)) | ((x >= 85) & (x < 86)) | ((x >= 394) & (x < 395)))] = C('#fff0b0')
    orn = frame & ((((x // 1) + (y // 1)) % 7) == 0) & (((y >= arch_top(x) - 9) & (y < arch_top(x) - 4)) |
                                                     ((x >= 87) & (x < 92)) | ((x >= 388) & (x < 393)))
    img[orn] = C('#f0c860')
    # 台口上方墙面的金色涡纹
    wall = (y >= 26) & (y < arch_top(x) - 14) & (x >= 80) & (x < 400)
    img[wall & ((np.sin(x * 0.35) * 3 + 36 - y) ** 2 < 1.2)] = C('#8a5a1a')
    # ---- 台内：彩绘夜空布景（预示后面的真月亮）
    sky = opening & (y < 182)
    t = np.clip((y - 50) / 130, 0, 1)
    bd = (C('#0c1230')[None, None] * (1 - t[..., None]) + C('#243468')[None, None] * t[..., None])
    img[sky] = bd[sky]
    img[sky & ((y // 3) % 9 == 0) & (dither_mask(0.5))] = C('#1a2654')
    st = sky & (rng.random((H, W)) < 0.012)
    img[st] = C('#8a9ad0')
    moon = sky & (np.hypot(x - 300, y - 92) < 15)
    img[moon] = C('#c8d4f0')
    img[moon & (np.hypot(x - 306, y - 88) < 13)] = C('#243468')
    hills = sky & (y > 160 + 8 * np.sin(x * 0.03) + 4 * np.sin(x * 0.11))
    img[hills] = C('#0e1634')
    # 城堡剪影（幻焰城的伏笔）
    castle = sky & (((np.abs(x - 190) < 14) & (y > 138)) | ((np.abs(x - 182) < 3) & (y > 124)) |
                    ((np.abs(x - 198) < 3) & (y > 128)) | ((np.abs(x - 190) < 2) & (y > 116)))
    img[castle] = C('#0a1028')
    # ---- 红丝绒侧幕（束起）
    def drape(x0, x1, tie_y, flip):
        u = (x - x0) / (x1 - x0)
        if flip:
            u = 1 - u
        narrow = 0.55 + 0.45 * np.clip(np.abs(y - tie_y) / 60, 0, 1)
        m = opening & (u >= 0) & (u < narrow) & (y < 196)
        fold = np.sin((u / narrow) * 9 * np.pi + (y - tie_y) * 0.02)
        col = np.where(fold[..., None] > 0.6, C('#d23a4a'), np.where(fold[..., None] > -0.1, C('#a81e32'),
                       np.where(fold[..., None] > -0.7, C('#7a1222'), C('#4a0a16'))))
        img[m] = col[m]
        edge = m & ~np.roll(m, 1 if not flip else -1, 1)
        img[edge] = C('#2a060c')
        tie = m & (np.abs(y - tie_y) < 1.5)
        img[tie] = C('#f0c860')
        img[tie & (y < tie_y)] = C('#fff0b0')
    drape(96, 136, 120, False)
    drape(344, 384, 120, True)
    # 顶部垂幔（四段花彩 + 金穗）
    u = (x - 96) / 288 * 4
    sw = arch_top(x) + 10 + 12 * np.sin(np.pi * (u % 1))
    swag = opening & (y < sw)
    img[swag] = C('#a81e32')
    img[swag & (y > sw - 4)] = C('#7a1222')
    img[swag & (np.abs(y - (sw - 8)) < 0.5)] = C('#d23a4a')
    fr = opening & (y >= sw) & (y < sw + 2)
    img[fr] = C('#f0c860')
    img[fr & (x % 2 == 0) & (y >= sw + 1)] = C('#8a5a1a')
    # ---- 舞台地板（透视木纹）+ 台唇 + 脚灯
    floor = _poly([(96, 182), (384, 182), (410, 204), (70, 204)])
    fy = np.clip((y - 182) / 22, 0, 1)
    img[floor] = C('#4a2a14')
    img[floor & ((y.astype(int) % 4) == 0)] = C('#3a200e')
    seam = floor & ((((x - 240) / (0.6 + fy * 0.8) + 400) % 18) < 1)
    img[seam] = C('#2a160c')
    img[floor & (y < 184)] = C('#2a160c')
    lip = _poly([(70, 204), (410, 204), (410, 212), (70, 212)])
    img[lip] = C('#2a0e10')
    img[lip & (y < 205)] = C('#c8902e')
    for lx in range(80, 404, 12):
        img[206:208, lx:lx + 3] = C('#fff0b0')
        img[208, lx:lx + 3] = C('#8a5a1a')
    # 乐池 + 栏杆
    pit = y >= 212
    img[pit & (x > 60) & (x < 420)] = C('#0a0406')
    img[pit & (np.abs(y - 222) < 0.6) & (x > 60) & (x < 420)] = C('#8a5a1a')
    img[pit & ((y > 222) & (y < 232)) & ((x % 10) < 1) & (x > 60) & (x < 420)] = C('#5a3a10')
    # ---- 巨型水晶吊灯（观众席上方，挂在台口前）
    cx_, cy_ = 240, 0
    img[(np.abs(x - cx_) < 0.8) & (y < 12)] = C('#8a5a1a')
    rings = [(14, 9, 2.5), (26, 24, 4.5), (36, 15, 3)]
    for ry, rx, rh in rings:
        rim = (np.abs(np.hypot((x - cx_) / rx, (y - ry) / rh) - 1) < 0.14)
        img[rim] = C('#c8902e')
        img[rim & (y < ry)] = C('#f0c860')
        for k in range(-int(rx), int(rx) + 1, 4):
            dy_ = rh * np.sqrt(max(0, 1 - (k / rx) ** 2))
            px, py = int(cx_ + k), int(ry + dy_)
            img[py:py + 5, px] = C('#a0b8e0')
            img[py + 5, px] = C('#e8f0ff')
            img[int(ry - dy_) - 3:int(ry - dy_) - 1, px] = C('#fff0b0')      # 烛火
    body = (np.hypot((x - cx_) / 4, (y - 30) / 12) < 1)
    img[body] = C('#8a5a1a')
    img[body & (x < cx_)] = C('#c8902e')
    img[(np.abs(x - cx_) < 0.8) & (y > 40) & (y < 50)] = C('#e8f0ff')
    return img


CHANDELIER_LIGHTS = [(240 + k, int(ry - rh * np.sqrt(max(0, 1 - (k / rx) ** 2))) - 3)
                     for ry, rx, rh in [(14, 9, 2.5), (26, 24, 4.5), (36, 15, 3)]
                     for k in range(-int(rx), int(rx) + 1, 4)]
FOOTLIGHTS = [(lx + 1, 206) for lx in range(80, 404, 12)]


@lru_cache(None)
def dissolve_field():
    """每个世界像素「碎掉」的先后（0 先 1 后）：外圈、上方先碎，舞台中央最后；3x3 的方块噪声让它一块块剥落。"""
    r = np.random.default_rng(9)
    blk = r.random((H // 3 + 1, W // 3 + 1))
    n = np.repeat(np.repeat(blk, 3, 0), 3, 1)[:H, :W]
    d = np.hypot((XX - 240) / 260, (YY - 196) / 200)
    order = 1 - np.clip(d, 0, 1)
    return (0.62 * order + 0.38 * n).astype(np.float32)


@lru_cache(None)
def night_sky():
    """歌剧院碎开后露出的夜空：深蓝渐层、巨月、横过月面的薄云。"""
    img = np.zeros((H, W, 3), np.float32)
    t = np.clip(YY / H, 0, 1)[..., None]
    img[:] = C('#02040f') * (1 - t) + C('#0c1c4a') * t
    img[dither_mask(0.5) & (YY > 170)] = C('#10245a')
    mx, my, mr = MOON
    d = np.hypot(XX - mx, YY - my)
    for rr, col, lvl in ((mr + 26, '#0e2250', 0.35), (mr + 14, '#16306a', 0.55), (mr + 6, '#2a4a8a', 0.7)):
        img[(d < rr) & dither_mask(lvl)] = C(col)
    moon = d < mr
    img[moon] = C('#dfe8ff')
    img[moon & (np.hypot(XX - mx - 10, YY - my - 8) > mr - 4)] = C('#b8c8ec')
    rr = np.random.default_rng(4)
    for k in range(9):
        a, rad = rr.uniform(0, TAU), rr.uniform(0, mr * 0.75)
        cr = rr.uniform(4, 12)
        c = moon & (np.hypot(XX - mx - np.cos(a) * rad, YY - my - np.sin(a) * rad) < cr)
        img[c] = C('#c4d2f2')
        img[c & (np.hypot(XX - mx - np.cos(a) * rad + 1.5, YY - my - np.sin(a) * rad + 1.5) > cr - 1.5)] = C('#a8b8e0')
    for cy_, amp, th in ((my + 22, 5, 3), (my + 40, 4, 2), (my - 30, 3, 1.5)):
        band = (np.abs(YY - cy_ - amp * np.sin(XX * 0.02)) < th) & (np.abs(XX - mx) < 150 - np.abs(YY - cy_) * 2)
        img[band & dither_mask(0.7)] = C('#1a2a5a')
        img[band & (d < mr) & dither_mask(0.5)] = C('#8ea0cc')
    return img


MOON = (334, 92, 70)


@lru_cache(None)
def audience(size):
    """前景观众剪影（两排头和肩，有高帽、发髻），size=1 全景 / 2 中景。返回 RGBA（高 60*size）。"""
    h = 60 * size
    img = np.zeros((h, W + 80, 4), np.float32)
    yy, xx = np.mgrid[0:h, 0:W + 80].astype(np.float32)
    r = np.random.default_rng(12 + size)
    for row, (base, col, rim) in enumerate(((22, '#140810', '#5a2a22'), (34, '#070306', '#3a1a18'))):
        x = r.uniform(0, 14) * size
        while x < W + 80:
            s = size * (1.0 if row else 0.8) * r.uniform(0.9, 1.15)
            hy = (base + r.uniform(-2, 2)) * size
            head = np.hypot(xx - x, (yy - hy) * 0.85) < 6.5 * s
            sh = (np.hypot((xx - x) / 2.4, yy - hy - 14 * s) < 7 * s) | ((np.abs(xx - x) < 15 * s) & (yy > hy + 14 * s))
            kind = r.random()
            extra = np.zeros_like(head)
            if kind < 0.2:
                extra = (np.abs(xx - x) < 5 * s) & (yy > hy - 15 * s) & (yy < hy - 4 * s)
                extra |= (np.abs(xx - x) < 8 * s) & (np.abs(yy - hy + 5 * s) < 1.2 * s)
            elif kind < 0.45:
                extra = np.hypot(xx - x + 3 * s, yy - hy + 6 * s) < 3.6 * s
            m = head | sh | extra
            img[m, :3] = C(col)
            img[m, 3] = 1
            top = m & ~np.roll(m, 1, 0)
            img[top & (yy < hy + 2 * s), :3] = C(rim)
            x += r.uniform(15, 22) * size
    return img


# ---------------------------------------------------------------- 苍鸟古里亚诺丝
BIRD_PAL = {'O': '#0a0a24', 'd': '#0e2a5c', 'b': '#164896', 'B': '#1f68c2', 'A': '#3a9ce6', 'C': '#6cd6f4',
            'F': '#e8fbff', 'G': '#f8d67c', 'g': '#c89040', 'E': '#fff4a0'}


# 标准翅膀轮廓（翅膀自己的坐标系：x 沿前缘从肩到翼尖，y 指向后缘）；初级飞羽一根根分开
WING = [(0, 0), (30, -4), (62, 2), (60, 10), (53, 8), (56, 16), (48, 13), (50, 21), (42, 17), (43, 24), (36, 19),
        (32, 25), (27, 21), (23, 26), (17, 22), (12, 24), (5, 17), (0, 9)]


def _wing(cv, S, u, v, sx=1.0, sy=1.0, dark=False):
    """按仿射 (S + x*u*sx + y*v*sy) 画一只翅膀；按翅膀坐标分三段上色：覆羽亮蓝、中段蓝、飞羽深蓝 + 青色羽尖。"""
    u, v = np.asarray(u, float), np.asarray(v, float)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    M = np.array([u * sx, v * sy]).T
    pts = [tuple(np.array(S) + M @ np.array(p, float)) for p in WING]
    m = cv.polym(pts)
    cv.sep(m, 'O')
    Mi = np.linalg.inv(M)
    qx = Mi[0, 0] * (cv.px - S[0]) + Mi[0, 1] * (cv.py - S[1])
    qy = Mi[1, 0] * (cv.px - S[0]) + Mi[1, 1] * (cv.py - S[1])
    edge = qy - np.interp(qx, [0, 30, 62], [0, -4, 2])
    cols = ('b', 'd', 'd', 'b') if dark else ('A', 'B', 'b', 'C')
    cv.put(m, cols[1])
    cv.put(m & (edge < 4.5), cols[0])
    cv.put(m & (edge > 11), cols[2])
    cv.put(m & (edge > 17) & (qx > 34), cols[3])
    # 飞羽分隔线
    for x0, x1 in ((53, 56), (48, 50), (42, 43), (36, 32), (27, 23), (17, 12)):
        a_ = tuple(np.array(S) + M @ np.array([x0 + 3, 5.0]))
        b_ = tuple(np.array(S) + M @ np.array([x0, 12.0 if x0 > 30 else 16.0]))
        cv.put(cv.linem([a_, b_]) & m, 'O' if dark else 'd')
    cv.put(cv.strokem(pts[:3], 1.6, 1.0) & m, 'B' if dark else 'C')
    return m


def _tail(cv, root, dirs, length=70):
    for k, (dx, dy) in enumerate(dirs):
        end = (root[0] + dx * length, root[1] + dy * length)
        mid = (root[0] + dx * length * 0.5 - dy * 6, root[1] + dy * length * 0.5 + dx * 6)
        m = cv.strokem([root, mid, end], 5, 2.2)
        cv.sep(m, 'O')
        cv.put(m, 'B' if k % 2 else 'b')
        cv.put(cv.strokem([root, mid, end], 1.4, 0.8) & m, 'A')
        cv.ell(end[0], end[1], 3.4, 3.4, 'C')
        cv.ell(end[0], end[1], 1.6, 1.6, 'd')


@lru_cache(None)
def bird(frame):
    """苍鸟古里亚诺丝。frame 0..2 = 侧面朝左飞的振翅循环（上举 / 平展(透视缩短) / 下扑），3 = 滑翔，
    4 = 正面展翅悬停（定格用）。返回 RGBA。"""
    from chars.vita import Cv
    cv = Cv(110, 160, BIRD_PAL)
    if frame == 4:
        cx, cy = 80, 50
        _tail(cv, (cx, cy + 12), [(-0.35, 1), (-0.12, 1), (0.12, 1), (0.35, 1)], 44)
        for sgn in (-1, 1):
            _wing(cv, (cx + 5 * sgn, cy - 6), (sgn, -0.42), (-0.2 * sgn, 1), 1.12, 1.25)
        body = cv.ellm(cx, cy + 2, 9, 14)
        cv.sep(body, 'O')
        cv.put(body, 'B')
        cv.put(body & (cv.py < cy - 4), 'A')
        cv.put(body & (np.abs(cv.px - cx - 0.5) < 3) & (cv.py > cy), 'C')
        head = cv.ellm(cx, cy - 15, 6.5, 6)
        cv.sep(head, 'O')
        cv.put(head, 'B')
        cv.put(head & (cv.py < cy - 17), 'A')
        for dx in (-5, 0, 5):
            cv.stroke([(cx + dx * 0.4, cy - 19), (cx + dx, cy - 27), (cx + dx * 1.6, cy - 32)], 'C' if dx == 0 else 'A', 2.4, 0.8)
        cv.poly([(cx - 2, cy - 14), (cx + 2.5, cy - 14), (cx + 0.5, cy - 8)], 'G')
        cv.dots([(cx - 4, cy - 16), (cx + 4, cy - 16)], 'E')
        return cv.rgba(BIRD_PAL['O'])
    S = (72, 52)
    wings = {0: ((0.25, -1), (1, 0.15), 1.0), 1: ((0.1, -1), (1, 0.1), 0.35), 2: ((-0.05, 1), (1, -0.25), 0.95),
             3: ((0.55, -0.85), (1, 0.3), 1.0)}
    u, v, sx = wings[frame]
    sx *= 1.3
    fu = (u[0] + 0.35, u[1]) if frame != 2 else (u[0] - 0.3, u[1])
    _wing(cv, (S[0] + 8, S[1] - 3), fu, v, sx * 0.8, 1.0, dark=True)           # 远侧翅膀（错开一点角度）
    _tail(cv, (S[0] + 16, S[1] + 4), [(1, -0.12), (1, 0.05), (1, 0.22)], 70)
    body = cv.ellm(S[0] + 2, S[1] + 3, 18, 8)
    cv.sep(body, 'O')
    cv.put(body, 'B')
    cv.put(body & (cv.py < S[1]), 'A')
    cv.put(body & (cv.py > S[1] + 7), 'b')
    head = cv.ellm(S[0] - 18, S[1] - 2, 7, 6)
    cv.sep(head, 'O')
    cv.put(head, 'B')
    cv.put(head & (cv.py < S[1] - 4), 'A')
    for k, (ax, ay) in enumerate(((14, -12), (20, -7), (22, -1))):
        cv.stroke([(S[0] - 17, S[1] - 6), (S[0] - 17 + ax * 0.5, S[1] - 6 + ay * 0.8), (S[0] - 17 + ax, S[1] - 6 + ay)],
                  'C' if k == 0 else 'A', 2.4, 0.8)
    beak = cv.polym([(S[0] - 24, S[1] - 4), (S[0] - 33, S[1] - 1), (S[0] - 27, S[1] + 0.5), (S[0] - 24, S[1] + 1)])
    cv.sep(beak, 'O')
    cv.put(beak, 'G')
    cv.put(beak & (cv.py > S[1] - 1.2), 'g')
    cv.dots([(S[0] - 21, S[1] - 3)], 'E')
    _wing(cv, S, u, v, sx, 1.25)                                                  # 近侧翅膀
    return cv.rgba(BIRD_PAL['O'])


# ---------------------------------------------------------------- 羽毛（本幕自带的小精灵）
FEATHER_PAL = {'W': '#e8fbff', 'C': '#6cd6f4', 'A': '#3a9ce6', 'B': '#1f68c2', 'd': '#0e2a5c'}
_FEATHERS = [
    ["..WC", ".CAB", "CAB.", "d..."],
    ["WCCAd", ".ABB."],
    ["d...", ".BAC", "BACW", "..C."],
    ["W.", "CC", "AB", "AB", ".d"],
]


@lru_cache(None)
def feather_img(k, big=False):
    rows = _FEATHERS[k % 4]
    img = np.zeros((len(rows), max(len(r) for r in rows), 4), np.float32)
    for yy_, r in enumerate(rows):
        for xx_, ch in enumerate(r):
            if ch != '.':
                img[yy_, xx_, :3] = C(FEATHER_PAL[ch])
                img[yy_, xx_, 3] = 1
    return img


class Feathers:
    """飘落的羽毛：带摆动与翻转，size 2 的在最前景。"""

    def __init__(self, seed=0):
        self.r = np.random.default_rng(seed)
        self.items = []

    def emit(self, n, x, y, vx, vy, life=(1.5, 3.0), size=1, grav=18.0, drag=1.2):
        u = lambda v: self.r.uniform(*v) if isinstance(v, tuple) else v
        for _ in range(n):
            self.items.append([u(x), u(y), u(vx), u(vy), u(life), u(life), self.r.uniform(0, TAU),
                               self.r.uniform(3, 7), int(u(size) + 0.5), grav, drag])

    def radial(self, n, x, y, speed, **kw):
        for _ in range(n):
            a = self.r.uniform(0, TAU)
            s = self.r.uniform(*speed)
            self.emit(1, x, y, np.cos(a) * s, np.sin(a) * s - 20, **kw)

    def step(self, dt, t):
        keep = []
        for it in self.items:
            x, y, vx, vy, life, ml, ph, fr, sz, g, dr = it
            vy += g * dt
            k = np.exp(-dr * dt)
            vx *= k
            vy *= k
            x += (vx + np.sin(t * fr + ph) * 14) * dt
            y += vy * dt
            life -= dt
            it[:5] = x, y, vx, vy, life
            if life > 0 and -20 < x < W + 20 and -40 < y < H + 20:
                keep.append(it)
        self.items = keep

    def render(self, dst, t, size=None):
        for x, y, vx, vy, life, ml, ph, fr, sz, g, dr in self.items:
            if size is not None and sz != size:
                continue
            k = int((t * fr + ph) / 1.2) % 4
            a = clamp01(life / 0.4) * clamp01((ml - life) / 0.08 + 0.3)
            blit(dst, feather_img(k), x, y, scale=sz, anchor='center', alpha=float(a))


# ================================================================= 渲染工具
def cam_sample(img, z, cx, cy):
    """把世界图像按镜头 (缩放 z、中心 cx,cy) 最近邻取样到屏幕。"""
    xs = np.clip(np.floor((np.arange(W) + 0.5 - W / 2) / z + cx).astype(int), 0, W - 1)
    ys = np.clip(np.floor((np.arange(H) + 0.5 - H / 2) / z + cy).astype(int), 0, H - 1)
    return img[ys[:, None], xs[None, :]], xs, ys


def w2s(x, y, z, cx, cy):
    return (x - cx) * z + W / 2, (y - cy) * z + H / 2


def rim_light(spr, color, amt, dirs=((0, 1), (-1, 0))):
    """精灵朝光源一侧的边缘像素染成光色（dirs = 光从哪边来：(dy,dx) 邻居为空即为受光边）。"""
    out = spr.copy()
    a = spr[..., 3] > 0
    edge = np.zeros_like(a)
    for dy, dx in dirs:
        nb = np.zeros_like(a)
        h, w = a.shape
        ys = slice(max(dy, 0), h + min(dy, 0))
        yd = slice(max(-dy, 0), h + min(-dy, 0))
        xs = slice(max(dx, 0), w + min(dx, 0))
        xd = slice(max(-dx, 0), w + min(-dx, 0))
        nb[yd, xd] = a[ys, xs]
        edge |= a & ~nb
    c = np.asarray(color, np.float32)
    out[edge, :3] = out[edge, :3] * (1 - amt) + c * amt
    return out


def tint(spr, color, amt):
    out = spr.copy()
    out[..., :3] = out[..., :3] * (1 - amt) + np.asarray(color, np.float32) * amt
    return out


def spotlight(dst, x_top, x_floor, y_floor, w_top, w_floor, color, strength):
    """从画面上方打下来的像素聚光：梯形光柱（抖动边缘）+ 地面光斑。"""
    if strength <= 0:
        return
    tt = np.clip(YY / max(1, y_floor), 0, 1)
    xc = x_top + (x_floor - x_top) * tt
    hw = w_top + (w_floor - w_top) * tt
    d = np.abs(XX - xc) / hw
    cone = (d < 1) & (YY < y_floor)
    lvl = np.clip((1 - d) * 1.6, 0, 1) * (0.25 + 0.35 * tt) * strength
    dst[cone & (BAYER < lvl)] += np.asarray(color, np.float32) * 0.35
    pool = np.hypot((XX - x_floor) / (w_floor * 1.2), (YY - y_floor) / (w_floor * 0.28)) < 1
    dst[pool & (BAYER < 0.7 * strength)] += np.asarray(color, np.float32) * 0.3


def darken(dst, k):
    dst *= (1 - k)


def pillar(dst, x, y_top, width, color_core='#e8fbff', color_edge='#3a9ce6', prog=1.0):
    if width <= 0.3:
        return
    m = (np.abs(XX - x) < width) & (YY < y_top)
    dst[m & (BAYER < 0.75)] = dst[m & (BAYER < 0.75)] * 0.3 + C(color_edge)
    core = (np.abs(XX - x) < width * 0.45) & (YY < y_top + 2)
    dst[core] = C(color_core)
    glow = (np.abs(XX - x) < width * 2.2) & (YY < y_top) & ~m
    dst[glow & (BAYER < 0.35 * prog)] += C(color_edge) * 0.6


# ================================================================= 姿势时间线
FPS_POSE = 12


def _V():
    from chars import vita
    return vita


def pose_at(ctx, lt):
    """本幕内 lt 时刻的关节 dict（关键姿势 + in-between，按 12fps 取帧）。"""
    V = _V()
    at = ctx.at
    b = ctx.sec.beat
    q = np.floor(lt * FPS_POSE) / FPS_POSE
    # 1.0 - 2.3：歌唱，两个呼吸姿势每两拍往返
    if q < at(2, 3):
        ph = (q / (2 * b)) % 2
        k = 0.5 - 0.5 * np.cos(np.pi * min(ph, 2 - ph))
        J = V.lerp_pose('sing', 'sing_b', k)
        if at(2, 2) <= q < at(2, 3):                    # 2.2 最高音：手臂举到最高
            J = V.lerp_pose(J, 'sing_b', 0.9)
        return J
    # 2.3 - 3.0：转身，折扇在 2.3.5 展开
    if q < at(3, 0):
        k = clamp01((q - at(2, 3)) / (0.5 * b))
        J = V.lerp_pose('sing', 'idle_smirk', ease_out(k))
        J['head'] = 'sing_l' if k < 0.25 else ('front' if k < 0.6 else 'smirk')
        if q < at(2, 3.5):
            fa = dict(V.POSES['sing']['fan'])
            fa['pivot'] = J['fan']['pivot']
            J['fan'] = fa
        return J
    # 3.x 特写期间也保持轻微的动作；4.0 起回中景
    if q < at(4, 1):
        J = dict(V.POSES['idle_smirk'])
        J['sway'] = 0.6 * np.sin(q * 3)
        J['flow'] = 0.8 * np.sin(q * 2.2)
        return J
    if q < at(4, 2):
        return V.lerp_pose('raise_a', 'raise', clamp01((q - at(4, 1.4)) / (0.5 * b)))
    if q < at(5, 0):
        J = dict(V.POSES['raise'])
        J['sway'] = 1.2 + 0.8 * clamp01((q - at(4, 3)) / b)
        return J
    # 5.0 命中 -> 5.2 扇子下扫 -> 5.3 漂浮
    if q < at(5, 1.5):
        J = dict(V.POSES['cast'])
        w = np.sin(q * 9) * 0.8
        J['sway'] += w
        J['flow'] += w
        return J
    if q < at(5, 3):
        return V.lerp_pose('cast', 'cast_b', ease_out(clamp01((q - at(5, 1.5)) / (0.5 * b))))
    if q < at(6, 0):
        return V.lerp_pose('cast_b', 'float', ease_out(clamp01((q - at(5, 3)) / b)))
    ph = (q - at(6, 0)) / (2 * b)
    k = 0.5 - 0.5 * np.cos(np.pi * ph)
    return V.lerp_pose('float', 'float_b', k)


# ================================================================= 主渲染
def render(dst, ctx):
    V = _V()
    lt = ctx.lt
    at = ctx.at
    b = ctx.sec.beat
    st = ctx.state
    ch = ctx.C['vita']
    if 'feathers' not in st:
        st['feathers'] = Feathers(5)
        st['dis_p'] = -1.0
    FE = st['feathers']
    P = ctx.P
    t5 = at(5, 0)
    HOLD = 4 / 30
    # 命中后的顿帧：动画时钟停住 HOLD 秒
    frozen = t5 <= lt < t5 + HOLD
    la = lt - clamp01((lt - t5) / HOLD) * HOLD if lt >= t5 else lt
    dt = 0.0 if frozen else ctx.dt

    J = pose_at(ctx, la)
    spr = V.render_joints(J)
    A = V.anchors(J)

    # ---------------- 镜头
    if lt < at(2, 0):
        shot, z, cx, cy, sc = 'wide', 1.0, 240, 135, 1
    elif lt < at(3, 0):
        k = smooth(at(2, 0), at(3, 0), lt)
        shot, z, cx, cy, sc = 'mid', 1.5 + 0.25 * k, 240, 150 - 3 * k, 2
    elif lt < at(4, 0):
        shot, z, cx, cy, sc = 'close', 3.0 + 0.2 * smooth(at(3, 0), at(4, 0), lt), 240, 140, 2
    elif lt < t5:
        shot, z, cx, cy, sc = 'mid', 1.6, 240, 150, 2
    else:
        shot, z, cx, cy, sc = 'sky', 1.6, 240, 150, 2

    # 聚光灯颜色：暖白 -> 3.2 起转苍蓝
    blue_k = smooth(at(3, 2), at(3, 3), lt)
    spot_col = C('#fff0c8') * (1 - blue_k) + C('#6cc8ff') * blue_k
    # 剧场碎裂进度
    dis_p = -0.05 + 1.15 * smooth(at(3, 2), at(4, 2.5), lt) ** 0.9
    skill_k = clamp01(1 - (lt - t5) / (1.2)) if lt >= t5 else 0.0

    # ---------------- 背景：夜空（远景）+ 歌剧院（镜头取样，按碎裂进度挖空）
    dst = night_sky().copy()
    fx.stars(dst, ctx.t, 0.7)
    if shot != 'sky':
        world = opera_world()
        scr, xs, ys = cam_sample(world, z, cx, cy)
        dis = dissolve_field()[ys[:, None], xs[None, :]]
        keep = dis > dis_p
        burn = keep & (dis < dis_p + 0.035)
        if blue_k > 0:                       # 蓝光把剧场染冷
            scr = scr * (1 - 0.35 * blue_k) + scr.mean(-1, keepdims=True) * C('#3a7aff') * 0.5 * blue_k
        if shot == 'wide' or lt < at(3, 0):
            scr = scr * 0.9
        dst[keep] = scr[keep]
        dst[burn] = C('#6cd6f4')
        dst[burn & (dis < dis_p + 0.012)] = C('#e8fbff')
        # 碎片化作羽毛：从刚刚碎掉的位置飞出
        if dis_p > st['dis_p'] and dis_p > 0 and not frozen:
            newly = (dis > st['dis_p']) & (dis <= dis_p)
            ys_, xs_ = np.nonzero(newly)
            if len(xs_):
                pick = ctx.rng.choice(len(xs_), min(len(xs_), 7), replace=False)
                for i in pick:
                    FE.emit(1, float(xs_[i]), float(ys_[i]), (-30, 30), (-70, -20), life=(1.2, 2.4),
                            size=2 if shot == 'close' else 1, grav=8)
                P.emit_arrays(xs_[pick].astype(np.float32), ys_[pick].astype(np.float32),
                              ctx.rng.normal(0, 20, len(pick)), ctx.rng.uniform(-60, -10, len(pick)),
                              (0.4, 0.9), 'blue', size=1, drag=1.5)
        st['dis_p'] = dis_p
        # 吊灯与脚灯的闪烁（世界坐标 -> 屏幕）
        if dis_p < 0.6:
            for lx, ly in CHANDELIER_LIGHTS + FOOTLIGHTS:
                sx, sy = w2s(lx + 0.5, ly + 0.5, z, cx, cy)
                if 0 <= sx < W and 0 <= sy < H and dissolve_field()[ly, lx] > dis_p:
                    tw = 0.5 + 0.5 * np.sin(ctx.t * 7 + lx * 1.3)
                    r_ = 2 + z
                    m = (np.abs(XX - sx) + np.abs(YY - sy) < r_) & (BAYER < 0.5 * tw)
                    dst[m] += C('#ffd890') * 0.5 * (1 - blue_k)
    else:
        # 命中后：画面整体被苍蓝染色
        dst = dst * (1 - 0.3 * skill_k) + C('#1f68c2') * 0.35 * skill_k

    # ---------------- 角色位置
    feet_w = (240, 196)
    fx_, fy_ = w2s(*feet_w, z, cx, cy)
    if shot in ('mid',):
        fy_ = min(fy_, 206)
    if shot == 'sky':
        fx_, fy_ = 206, 222
    if at(4, 3) <= la < t5:
        fy_ += 2 * sc * smooth(at(4, 3), at(4, 3.3), la)            # 下蹲预备
    if la >= at(5, 3):
        fy_ -= 16 * ease_out(clamp01((la - at(5, 3)) / (4 * b))) + 2 * np.sin((la - at(5, 3)) * 3.2)   # 漂浮
    fx_, fy_ = int(round(fx_)), int(round(fy_))
    h, w = spr.shape[:2]
    ox, oy = fx_ - w * sc / 2, fy_ - h * sc                          # 精灵左上角（屏幕）

    def S(p):
        return ox + p[0] * sc, oy + p[1] * sc

    # ---------------- 聚光灯（1-4 小节）
    if shot in ('wide', 'mid') and lt < t5:
        on = smooth(0.0, 0.06, lt)
        spot_on = on * (1 - smooth(at(4, 1), at(4, 3), lt) * 0.6)
        darken(dst, 0.15 * on * (1 - dis_p))
        spotlight(dst, fx_ + 60 * sc, fx_, fy_, 8 * sc, 22 * sc, spot_col, spot_on)

    # ---------------- 蓄力：画面四周压暗（抖动暗角），光往杖尖收
    if at(4, 2) <= lt < t5:
        k = smooth(at(4, 2), at(4, 3.6), lt)
        vig = np.hypot((XX - fx_) / 260, (YY - fy_ + 80) / 170)
        dst[BAYER < np.clip((vig - 0.45) * 2.2, 0, 1) * k] *= 0.35
    # ---------------- 魔法阵（4.1 起）与汇聚光点
    if at(4, 1) <= lt < t5 + 1.2:
        g = ease_out(clamp01((lt - at(4, 1)) / (1.5 * b)))
        fade_ = 1 - clamp01((lt - t5) / 1.2)
        fx.magic_circle(dst, fx_, fy_ + 2, 44 * sc / 2 * g, -lt * 1.5, C('#3163d4') * 0.9 * fade_, flat=0.28)
        tx, ty = S(A.get('staff_tip', (43.5, 0.5)))
        fx.magic_circle(dst, tx, ty + 3, 14 * g * fade_ + 1, lt * 3, C('#57cfc6') * fade_, star=5, dots=8)
        if lt < t5 and not frozen:
            n = 4
            ang = ctx.rng.uniform(0, TAU, n)
            rad = ctx.rng.uniform(40, 90, n)
            P.emit_arrays(tx + np.cos(ang) * rad, ty + np.sin(ang) * rad, -np.cos(ang) * rad * 2.2,
                          -np.sin(ang) * rad * 2.2, 0.45, 'blue', size=1)
    # 杖尖光团（蓄力时越来越亮）
    if 'staff_tip' in A and lt >= at(4, 1):
        tx, ty = S(A['staff_tip'])
        pw = clamp01((lt - at(4, 1)) / (t5 - at(4, 1))) if lt < t5 else max(0.3, skill_k)
        r_ = 2 + 5 * pw + np.sin(lt * 30) * 0.8
        m = np.hypot(XX - tx, YY - ty - 2) < r_
        dst[m & (BAYER < 0.6)] += C('#6cd6f4') * 0.8
        dst[np.hypot(XX - tx, YY - ty - 2) < r_ * 0.4] = C('#ffffff')

    # ---------------- 苍鸟（命中后）：5.0 近景大鸟从右向左掠过月面；6.0 远处绕回，在月前正面展翅悬停
    bird_pos = None
    if la >= t5:
        u = la - t5
        cyc = (0, 1, 2, 1)
        bx = None
        if u < 0.95:
            k = u / 0.95
            bx, by = 600 - 800 * k, 100 - 22 * np.sin(k * np.pi)
            bf, bsc, bflip = cyc[int(u * 12) % 4], 2, False
        elif la >= at(6, 0):
            v = la - at(6, 0)
            k = clamp01(v / (2.0 * b))
            e = ease_out(k)
            if k < 1:
                ang = np.pi * (1.1 - 1.1 * e)                       # 从左下沿弧线飞到月前
                bx = MOON[0] + np.cos(ang) * 170 * (1 - e)
                by = MOON[1] + 30 + np.sin(ang) * 40 * (1 - e) - 20 * (1 - e)
                bf, bsc, bflip = cyc[int(v * 12) % 4], 1, True
                if k > 0.8:
                    bf = 4
            else:
                bx, by = MOON[0], MOON[1] + 2 + np.sin(v * 3.2) * 2
                bf, bsc, bflip = 4, 1, False
        if bx is not None:
            bimg = bird(bf)
            bird_pos = (bx, by, bimg, bsc, bflip)
            blit(dst, rim_light(bimg, C('#e8fbff'), 0.5, dirs=((-1, 0),)), bx, by, scale=bsc, anchor='center', flip=bflip)
            if u < 0.95 and not frozen:
                FE.emit(2, bx + 90, (by - 30, by + 30), (40, 90), (-10, 30), life=(1.5, 2.5), size=1, grav=20)

    # ---------------- 光柱 + 音波环（命中）
    if t5 <= lt < t5 + 1.0:
        u = la - t5
        tx, ty = S(A.get('staff_tip', (43.5, 0.5)))
        wdt = (18 if frozen else 18 * (1 - ease_out(clamp01(u / 0.7)))) + 1
        pillar(dst, tx, ty + 2, wdt, prog=1 - clamp01(u / 0.8))
    for i, (t0, cols) in enumerate(((t5, ('#3a7aff', '#57cfc6', '#c4e878')), (at(5, 2), ('#57cfc6', '#c4e878', '#e8fbff')))):
        u = la - t0
        if 0 <= u < 1.2:
            px_, py_ = S(A.get('staff_tip', (43.5, 0.5))) if i == 0 else S(A['fan_tip'])
            for j, col in enumerate(cols):
                r = u * 320 - j * 28
                if r > 0:
                    fx.shockwave(dst, px_, py_, r, C(col) * (1 - u / 1.2), w=3, flat=0.85)

    # ---------------- 一次性触发
    if ctx.crossed(1, 0.0):
        ctx.flash('#fff0c8', 0.12, 0.35)
    if lt < at(3, 0):
        for bb in range(8):
            if ctx.crossed(1 + bb // 4, bb % 4):
                mx, my = S(A['mouth'])
                ctx.bits.emit(1 if bb % 4 else 2, mx - 4, my, (-60, -20), (-50, -30), 1.6, 'note', scale=(1, 2))
    if ctx.crossed(2, 2.0):
        mx, my = S(A['mouth'])
        ctx.bits.emit(4, mx - 4, my, (-90, -10), (-80, -30), 1.6, 'note', scale=(1, 2))
        P.emit_radial(30, mx, my, (20, 70), (0.3, 0.7), 'white', size=1, drag=2)
    if ctx.crossed(2, 3.5):                          # 折扇展开
        px_, py_ = S(A['fan_tip'])
        P.emit_radial(24, px_, py_, (30, 90), (0.2, 0.5), 'mint', size=1, drag=3)
        ctx.shake(1, 0.1)
    if ctx.crossed(4, 1.0):                          # 魔杖现形
        tx, ty = S(A['hand_l'])
        P.emit_radial(40, tx, ty - 20, (30, 110), (0.3, 0.6), 'blue', size=(1, 2), drag=3)
    if ctx.crossed(5, 0.0):
        ctx.shake(7, 0.6)
        tx, ty = S(A.get('staff_tip', (43.5, 0.5)))
        st['impact_xy'] = (tx, ty)
    if lt >= t5 + HOLD and not st.get('burst'):
        st['burst'] = True
        tx, ty = st.get('impact_xy', S(A.get('staff_tip', (43.5, 0.5))))
        FE.radial(70, tx, ty, (60, 260), life=(1.6, 3.2), size=(1, 2), grav=26, drag=1.4)
        P.emit_radial(260, tx, ty, (80, 380), (0.5, 1.4), 'blue', size=(1, 2), drag=2)
        P.emit_radial(80, tx, ty, (40, 160), (0.6, 1.4), 'mint', size=1, drag=1.5)
        ctx.flash('#bff0ff', 0.12, 0.3)
    if ctx.crossed(5, 2.0):
        px_, py_ = S(A['fan_tip'])
        FE.radial(30, px_, py_, (40, 180), life=(1.5, 3.0), size=(1, 2), grav=22)
        ctx.bits.emit(6, px_, py_, (-120, 60), (-120, -20), 1.6, 'note', scale=(1, 2))
        ctx.shake(3, 0.3)
    if ctx.crossed(6, 0.0):
        FE.emit(24, (0, W), -10, (-20, 20), (20, 60), life=(2.0, 3.2), size=(1, 2), grav=10)
        ctx.shake(2, 0.3)
    if lt >= at(5, 3) and int(lt * 10) != int((lt - ctx.dt) * 10):
        FE.emit(2, (0, W), -8, (-10, 10), (20, 50), life=(2.5, 3.5), size=(1, 2), grav=6)

    # ---------------- 粒子（角色身后的一层）
    if not frozen:
        P.step(dt)
        ctx.bits.step(dt)
        FE.step(dt, ctx.t)
    P.render(dst)
    FE.render(dst, ctx.t, size=1)

    # ---------------- 角色（特写时换头像）
    if shot == 'close':
        por = ch.portrait('smirk')
        por = rim_light(por, spot_col, 0.4 + 0.4 * blue_k, dirs=((-1, 0), (0, 1)))
        if blue_k > 0:
            por = tint(por, C('#3a7aff'), 0.18 * blue_k)
        bob = int(round(np.sin(lt * 2.4)))
        blit(dst, por, 160, 40 + bob, scale=2)
    else:
        s2 = spr
        if lt < t5:
            s2 = rim_light(s2, spot_col, 0.55, dirs=((-1, 0), (0, 1)))
        else:
            s2 = rim_light(tint(s2, C('#3a7aff'), 0.25 * skill_k), C('#bff0ff'), 0.7, dirs=((-1, 0), (0, 1), (0, -1)))
        if at(4, 1) <= lt < at(4, 1) + 0.07:
            s2 = tint(s2, C('#e8fbff'), 0.6)
        blit(dst, s2, ox, oy, scale=sc)
        if at(3, 0) > lt >= at(2, 0) or shot == 'mid':
            pass
    ctx.bits.render(dst, ctx.t)
    FE.render(dst, ctx.t, size=2)

    # ---------------- 前景观众（1-2 小节）
    if shot == 'wide':
        blit(dst, audience(1), -40 + np.sin(lt * 0.7) * 2, H - 60)
    elif shot == 'mid' and lt < at(3, 0):
        k = smooth(at(2, 0), at(3, 0), lt)
        blit(dst, audience(2), -40 - 20 * k, H - 90 + 40 * k)

    # ---------------- 名牌
    if at(2, 0) <= lt < at(3, 0):
        ui.name_tag(dst, ch, lt - at(2, 0))

    # ---------------- 冲击帧：白底黑剪影 -> 反色
    if t5 <= lt < t5 + 2 / 30:
        first = lt < t5 + 1 / 30
        if first:
            out = np.empty_like(dst)
            out[:] = C('#e8fbff')
            sil = spr[..., 3] > 0
            silb = np.repeat(np.repeat(sil, sc, 0), sc, 1)
            y0, x0 = int(round(oy)), int(round(ox))
            hh, ww = silb.shape
            ya, xa = max(0, y0), max(0, x0)
            yb, xb = min(H, y0 + hh), min(W, x0 + ww)
            region = out[ya:yb, xa:xb]
            region[silb[ya - y0:yb - y0, xa - x0:xb - x0]] = C('#02040f')
            tx, ty = S(A.get('staff_tip', (43.5, 0.5)))
            out[(np.abs(XX - tx) < 8) & (YY < ty)] = C('#02040f')
            if bird_pos:
                bx, by, bimg, bsc, bflip = bird_pos
                bs = np.zeros((H, W, 3), np.float32)
                blit(bs, np.dstack([np.ones(bimg.shape[:2] + (3,), np.float32), bimg[..., 3:]]), bx, by,
                     scale=bsc, anchor='center', flip=bflip)
                out[bs[..., 0] > 0.5] = C('#02040f')
            dst = out
        else:
            lum = np.clip(dst, 0, 1).mean(-1)
            out = np.empty_like(dst)
            out[:] = C('#02040f')
            out[lum > 0.28] = C('#3a9ce6')
            out[lum > 0.5] = C('#e8fbff')
            dst = out
    return np.clip(dst, 0, 1.5)


# ================================================================= 本幕音效
def _sfx():
    import sfx as S_
    SR = S_.SR

    def stereo(x, pan):
        pan = np.broadcast_to(pan, x.shape)
        return np.stack([x * np.sqrt(0.5 * (1 - pan)), x * np.sqrt(0.5 * (1 + pan))], 1).astype(np.float32)

    def spot_on():
        n = int(0.9 * SR)
        t = S_.tt(n)
        clunk = S_.lp(S_.noise(n), 1800) * np.exp(-t / 0.02) * 1.4
        thump = np.sin(2 * np.pi * np.cumsum(55 + 90 * np.exp(-t / 0.03)) / SR) * np.exp(-t / 0.12)
        hum = (np.sin(2 * np.pi * 120 * t) * 0.08 + np.sin(2 * np.pi * 240 * t) * 0.04) * np.exp(-t / 0.5)
        return stereo(clunk + thump + hum, 0.0)

    def fan_snap():
        n = int(0.35 * SR)
        t = S_.tt(n)
        x = S_.sweep_lp(S_.hp(S_.noise(n), 800), 1500, 7000, 0.7) * np.exp(-t / 0.08) * 0.6
        for k in range(7):                        # 扇骨一根根弹开
            i = int((0.012 + k * 0.009) * SR)
            m = int(0.01 * SR)
            x[i:i + m] += S_.bp(S_.noise(m), 2500, 8000) * np.exp(-np.arange(m) / 60) * (0.6 + 0.1 * k)
        i = int(0.08 * SR)
        m = n - i
        x[i:] += S_.bp(S_.noise(m), 1200, 5000) * np.exp(-S_.tt(m) / 0.01) * 1.4
        return stereo(x, -0.3)

    def feathers():
        n = int(1.6 * SR)
        t = S_.tt(n)
        am = 0.5 + 0.5 * np.sin(2 * np.pi * (22 + 6 * np.sin(2 * np.pi * 0.7 * t)) * t)
        x = S_.bp(S_.noise(n), 500, 4000) * am * np.sin(np.pi * np.clip(t / 1.6, 0, 1)) ** 2
        return stereo(x * 0.9, np.sin(2 * np.pi * 0.5 * t) * 0.5)

    def bird_cry():
        n = int(1.0 * SR)
        t = S_.tt(n)
        f = 2300 - 800 * np.clip(t / 0.55, 0, 1) ** 0.7 + 60 * np.sin(2 * np.pi * 28 * t)
        ph = 2 * np.pi * np.cumsum(f) / SR
        tone = np.tanh(2.5 * np.sin(ph)) * 0.5 + np.sin(2 * ph) * 0.2
        e = np.clip(t / 0.04, 0, 1) * np.exp(-np.maximum(t - 0.25, 0) / 0.25)
        breath = S_.bp(S_.noise(n), 2000, 6000) * 0.25
        x = S_.lp((tone + breath) * e, 7000)
        return stereo(x, 0.7 - 1.4 * np.clip(t / 0.9, 0, 1))     # 从右飞到左

    def azure_burst():
        n = int(2.2 * SR)
        t = S_.tt(n)
        sweep = S_.sweep_lp(S_.noise(n), 9000, 300, 0.5) * np.exp(-t / 0.5) * 0.8
        boom = np.tanh(2 * np.sin(2 * np.pi * np.cumsum(38 + 120 * np.exp(-t / 0.05)) / SR)) * np.exp(-t / 0.4)
        glass = sum(np.sin(2 * np.pi * f * t + k) * np.exp(-t / (0.4 + 0.15 * k))
                    for k, f in enumerate((1568, 2489, 3136, 3951, 4978))) * 0.12
        x = (sweep + boom + glass) * 0.5
        return np.stack([x, np.roll(x, 300)], 1).astype(np.float32)

    return dict(spot_on=spot_on, fan_snap=fan_snap, feathers=feathers, bird_cry=bird_cry, azure_burst=azure_burst)


SFX = {name: (lambda name=name: _sfx()[name]()) for name in
       ('spot_on', 'fan_snap', 'feathers', 'bird_cry', 'azure_burst')}
