"""Leonhardt / 莱恩哈特（剑帝·雷韦）—— 执行者 No.II「剑帝」。

参照：refs/leonhardt/bust.png（SC Evo 半身）与 full.png（SC Evo 全身，服装一致），
关键点标注见 data/landmarks/leonhardt.json，用 tools/measure.py 对照。

识别点：银灰色刺头（头顶一撮翘起、侧发垂到下颌）、紫瞳、冷峻的压低眉眼；
灰褐色（greige）高立领长风衣（深色前襟包边 + 银铆钉 + 袖章）、深色 V 领内衫、
红色项圈挂银环（环里一个 8 字）、红腰带 + 垂下的红色长穗、灰色长裤、黑鞋红袜；
魔剑「噬岩者 / Kernhitter」：金色华丽的宽刃（刃面铜橙、金边、锯齿剑脊、蓝宝石）。
身高约 180cm（全身 72px ≈ 180cm 的比例尺，脚底在画布最后一行附近）。

结构：肖像 = 逐像素对照 bust.png 画成的 80×80 ASCII 底图（PORTRAIT_BASE，专用色 PAL_P）+ 表情贴片（EXPR）；
全身 = 可复用的部件（头贴片 / 躯干 / 手臂 / 腿 / 风衣下摆 / 剑），由少量关节坐标摆出姿势（见 POSES）。
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

import pixelkit as pk

PAL = {
    'O': '#1c1218',                                                  # 描边
    'S': '#fff2e4', 's': '#fcdcc2', 'k': '#e5ac8c', 'K': '#ae6e5c',  # 肤
    'A': '#fbf8f6', 'a': '#ddd5d3', 'c': '#ada1a0', 'C': '#766a6e',  # 银灰发（暖灰）
    'V': '#c872c6', 'v': '#6e2a6c', 'W': '#f6f0f6',                  # 紫瞳 / 眼白
    'B': '#dccdc3', 'b': '#bcaaa0', 'd': '#8f7d77', 'D': '#63534f',  # 灰褐风衣
    'L': '#4c4446', 'l': '#2c2428',                                  # 深色包边 / 衬里
    'n': '#5c5050', 'm': '#3c3234',                                  # 深色内衫
    'R': '#d0444a', 'r': '#922c32', 'q': '#561a22',                  # 红（项圈 / 腰带 / 穗 / 袜）
    'G': '#f6f6fb', 'g': '#a9a9bb', 'u': '#5e6a82',                  # 银（环 / 铆钉）
    'P': '#86808a', 'p': '#605b64', 'Q': '#423e46',                  # 灰裤
    'Y': '#ffe68a', 'y': '#d8a23c', 'o': '#b0642a', 'x': '#6e3418',  # 剑：金 / 暗金 / 铜橙刃面 / 深铜
    'U': '#58c8ff', 'F': '#3a2e30',                                  # 蓝宝石 / 鞋
    'E': '#6a615e', 'h': '#7d706b', 'i': '#a39792', 'w': '#6e6670',  # 头发最暗 / 深阴影 / 中间调 / 受光外描边（只用于肖像）
}
KEYS = ['.'] + list(PAL)
KID = {k: i for i, k in enumerate(KEYS)}
_RGB = np.zeros((len(KEYS), 3), np.float32)
for _k, _v in PAL.items():
    _RGB[KID[_k]] = pk.hexc(_v)


# ---------------------------------------------------------------- 画布工具
class Cv:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.a = np.zeros((h, w), np.uint8)

    def _mask(self, fn):
        im = Image.new('L', (self.w, self.h), 0)
        fn(ImageDraw.Draw(im))
        return np.array(im) > 0

    def m_poly(self, pts):
        return self._mask(lambda d: d.polygon([tuple(p) for p in pts], fill=1, outline=1))

    def m_line(self, pts, width=1):
        return self._mask(lambda d: d.line([tuple(p) for p in pts], fill=1, width=width))

    def is_(self, keys):
        return np.isin(self.a, [KID[k] for k in keys])

    def fill(self, m, c, where=None):
        if where is not None:
            m = m & (self.is_(where) if isinstance(where, str) else where)
        self.a[m] = KID[c]
        return m

    def poly(self, c, pts, where=None):
        return self.fill(self.m_poly(pts), c, where)

    def line(self, c, pts, width=1, where=None):
        return self.fill(self.m_line(pts, width), c, where)

    def px(self, c, pts, where=None):
        for x, y in pts:
            if 0 <= x < self.w and 0 <= y < self.h:
                if where is None or KEYS[self.a[y, x]] in where:
                    self.a[y, x] = KID[c]

    def stamp(self, x0, y0, rows, flip=False):
        """ASCII 贴片：'.'/' ' 跳过，'_' 擦成透明。flip=True 时左右镜像贴。"""
        for dy, r in enumerate(rows):
            if flip:
                r = r[::-1]
            for dx, ch in enumerate(r):
                x, y = x0 + dx, y0 + dy
                if ch in '. ' or not (0 <= x < self.w and 0 <= y < self.h):
                    continue
                self.a[y, x] = 0 if ch == '_' else KID[ch]

    def shade(self, base, dark, dy=1, dx=1, keep=None, where=None):
        """base 色区域中朝右下（背光）的边缘带换成 dark。"""
        m = self.is_(base)
        region = self.is_(keep or base)
        sh = np.zeros_like(region)
        sh[:self.h - dy, :self.w - dx] = region[dy:, dx:]
        edge = m & ~sh
        if where is not None:
            edge &= where
        self.a[edge] = KID[dark]
        return edge

    def light(self, base, lit, dy=1, dx=1, keep=None, where=None):
        """base 色区域中朝左上（受光）的边缘换成 lit。"""
        m = self.is_(base)
        region = self.is_(keep or base)
        up = np.zeros_like(region)
        up[dy:, dx:] = region[:self.h - dy, :self.w - dx]
        edge = m & ~up
        if where is not None:
            edge &= where
        self.a[edge] = KID[lit]
        return edge

    def border(self, inner, outer_keys, c):
        """outer_keys 色块中与 inner 相邻（4 邻域）的一圈像素涂成 c（选择性内描边）。"""
        g = _grow(inner)
        self.a[g & ~inner & self.is_(outer_keys)] = KID[c]

    def outline(self, c='O'):
        a = self.a > 0
        self.a[_grow(a) & ~a] = KID[c]

    def rgba(self, rgb=None):
        out = np.zeros((self.h, self.w, 4), np.float32)
        out[..., :3] = (_RGB if rgb is None else rgb)[self.a]
        out[..., 3] = (self.a > 0).astype(np.float32)
        return out


def _grow(m):
    g = m.copy()
    g[1:] |= m[:-1]
    g[:-1] |= m[1:]
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    return g


# ================================================================ 肖像 80×80
# 逐像素翻译 refs/leonhardt/bust.png（SC Evo 半身像，脸朝画面左的 3/4 侧，近侧 = 画面右、露耳）。
# 取景 = measure.py fidelity 的固定取景（两眼中心 + 下巴三点，缩放 0.211）：头骨顶 y≈12，眼 y31..34，下巴 (37,53)。
# 做法：官方图按上述取景搬到 80×80，按材质（发 / 肤 / 风衣 / 深色前襟与内衫 / 项圈 / 银饰）分区后逐格取色，
# 再按像素画整理：头发按官方图的大块明暗分成干净色带（亮 a / 暗 c / 背光 i·C·h，无孤立色点），
# 发束分缝线两头收尖、背光侧带一像素中间调，冠部 / 右侧各一道高光弧，分缝 V 与发旋；
# 外描边 O（头顶受光的上缘用浅描边 w），头发压在衣领上的下缘描暗线；近侧立领压暗一档，与同色的背光头发按明度分开；
# 五官重画（眼、近侧眉、鼻梁高光、嘴）、衣领 / 前襟 / 铆钉 / 项圈 / 吊坠按官方图逐行读出的坐标重画。
# 头发：A 高光 / a 亮 / c 中 / i 中暗 / C 暗 / h 深阴影 / E 发束缝与暗描边；
# 风衣：B 受光 / d 背光 / D 折痕与近侧立领；深色前襟 L / l；内衫 n / m；项圈 R / r / q；银饰 G / g / u；O 眼线与外描边。
# 注意：官方图里背光的银发与风衣阴影几乎同色（Lab 差 < 3），发色与衣色是按官方图的聚类中心精确取的，
# 并且肖像里不用风衣过渡色 b —— 否则 measure.py 的材质分类会把阴影里的头发算成衣服。
PAL_P = dict(PAL, **{
    'O': '#2a2226', 'w': '#6e6670',                                  # 五官线与外描边 / 头发受光面的浅描边
    'S': '#fff0dc', 's': '#fbe3cb', 'k': '#e1b193', 'K': '#a57961',  # 肤
    'A': '#f3efed', 'a': '#e8e2e0', 'c': '#bab1ac', 'i': '#a39792', 'C': '#8d7d76', 'h': '#786a64', 'E': '#6a615e',  # 银灰发
    'V': '#b974ac', 'v': '#6e3662', 'W': '#f0e8ee',                  # 紫瞳 / 眼白
    'B': '#c8b8b2', 'b': '#af9d95', 'd': '#86756f', 'D': '#63544f',  # 灰褐风衣
    'L': '#655d5e', 'l': '#322c2d', 'n': '#5f585a', 'm': '#413b3b',  # 深色前襟 / 内衫
    'R': '#b64c49', 'r': '#7a2b2b', 'q': '#582526',                  # 红项圈
    'G': '#f6f6f7', 'g': '#ced3dc', 'u': '#989ea8',                  # 银饰
})
_RGB_P = _RGB.copy()
for _k, _v in PAL_P.items():
    _RGB_P[KID[_k]] = pk.hexc(_v)
# 材质表（data/landmarks/leonhardt.json 的 _classes 引用这些名字）
P_SKIN = [PAL_P[k] for k in 'SskK']
P_HAIR = [PAL_P[k] for k in 'AaciChE']
P_EYES = [PAL_P[k] for k in 'VvW']
P_COAT = [PAL_P[k] for k in 'BbdD']
P_DARK = [PAL_P[k] for k in 'LlnmRrqGgu']

PORTRAIT_BASE = [
    '................................................................................',  # 0
    '.......................................ww.......................................',  # 1
    '......................................Oicw.....w................................',  # 2
    '.......................................OccwwwwOcw...............................',  # 3
    '....................................wwwwwcccccwccw..............................',  # 4
    '...........................wwwwwwwwwccaaccccCccwccO.............................',  # 5
    '..........................OccccaaaEccaaaacccCcccccw.............................',  # 6
    '...........................OccccEEEAAAaaaacCEccccccw............................',  # 7
    '........................wwwwccEEcccaaaaaaaccEcccccccw...........................',  # 8
    '.....................wwwcccaaaccaaaaaaaaccccEcccccwccwww........................',  # 9
    '....................wcccccaaaaaaaaaaaaEEccccEccccccaaaccwww.....................',  # 10
    '...................OCcccaaaaaaccaaEEEEcccccccECcccaaaaacciCO....................',  # 11
    '....................wwcaaaaaacccEEEccccccccccCcCEEEaaaaaccO.....................',  # 12
    '...................wccaaaaaaccEEcccaaccccccccccaAAEEEaaaacw.....................',  # 13
    '..................wccccaaaaEEEcccaaaaaaaccccccaAAaccEAAAaccw....................',  # 14
    '.................wccccaaaaccccaaaaaaaaaacccccaaaaaaacaaaacccO...................',  # 15
    '................wcccccaAAAAAAAAAAaaaaaaccccccaaaaaaaaaaaacccw...................',  # 16
    '...............OccOOcAAccaaaaaaaaAAAaacciicccaaaaaaaaaaaaccccO..................',  # 17
    '................OO.waacaAAAAAAaaaaaaacEiCCiiEEEaaaaaaacaaccccO..................',  # 18
    '..................OaaacaaaaaaaaaaaaaacEChhCCEcccaacccccccccccw..................',  # 19
    '..................waaacaaaaaaaaaaaacccEEEhEEEcccacccEcccccccccO.................',  # 20
    '.................OcaaccaaaaacaaccccccEkiEiiskcccacccEcccccccccO.................',  # 21
    '.................OcaaccaaaaCcacccEccEksssssssscEccccEcccccccccO.................',  # 22
    '................OccccEcaaacCcccccEccCksssssssscEcccccEcccccciiO.................',  # 23
    '................OccccEcaacCCccccEEcckssssssssscEcccccEcCccciiEiO................',  # 24
    '...............OiOcccEccccEcccccEccksssssssssscEccccccECccciCEiO................',  # 25
    '................OOcccEccccEccccEEccksssssssssscEccccccECEcciCEEO................',  # 26
    '.................OcccEccccEccccEcckssssssssssscEccccccCcEcccChEhO...............',  # 27
    '.................OcccEccccEccckscckssssssssccccEccccciiiEicciCOhO...............',  # 28
    '.................OcccEccccEccksccksssssccccssscEccccciiiEiiciCOhhO..............',  # 29
    '.................OcccEcccEEiksCkcksssccsssssssiccccciCCiiEiiiCOOO...............',  # 30
    '.................OcccEcccEiiOOOOOssssssOOOOOOOOicccciKCCiEiiiCCO................',  # 31
    '.................OccccEccEiiWvVVkssssssOWWvVWvOiccccCKKCCECiCCCO................',  # 32
    '.................OccccEccEiikvVkssssssssskvVVkscccciKKKKKCECCCChO...............',  # 33
    '.................OccccEciEiiskkssssssssssskkkskcccciKKKkKCCCCCChO...............',  # 34
    '.................OccccEciiissssssSsssssssssssscckciCKKkkKCCCCCCChO..............',  # 35
    '.................OiOccEciisssssssSssssssssssssckciCKKkkkiCCCCCCChO..............',  # 36
    '................OiiOccCciisssssssSssssssssssssckciCKKkkkCCCCCChhhhO.............',  # 37
    '................OiCOccCcccsssssssSssssssssssssskciiKKKkCCCCCChhhhhO.............',  # 38
    '.................OOOccCcccsssssssssssssssssssssckciKKkCCCCCCChhOOhhO............',  # 39
    '..................OcccCccccsssssssssssssssssssskkciKkkEEEEEEEhhO.OhhO...........',  # 40
    '...................OcccccccsssssssssssssssssssskkEEKKldddddddEEO..OO............',  # 41
    '...................OcOcciiEcsssssskkkssssssssskkkllllLdddddddDDO................',  # 42
    '...................OcOciiiEisssssskksssssssssskkkllllLDDDDDDDdO.................',  # 43
    '.................OOOiOiOOiEEsssssssssssssssssskkkllllLDDDDDDDDO.................',  # 44
    '................OllOOOhOOiiCEsssssssssssssssskkkillllLDDDDDDDDDO................',  # 45
    '................OdllllEOOCiCEEssssKKKKKsssssskkkilllLLddDDDDDDDO................',  # 46
    '................OdlllllEhCiCCiEsssskkssssssskkKkilllLLDDddDDDDDDO...............',  # 47
    '.................OllllldECCCCCEEssskkkssssskkKkkillllLDDDDddDDDDO...............',  # 48
    '.................OdllllddCCCCCCCcssssssssskKKKkkillllLDDDDDDddDDO...............',  # 49
    '.................OdllllldECCCCCCCsssssssskKKKkkkillllLDDDDDDDDddO...............',  # 50
    '..................OlllllddCCCCCCCKsssssskKKKkkkkCllllLDDDDDDDDDDO...............',  # 51
    '..................OllllldLEChhhhCKKssssKKKKkkkkkEllllLDDDDDDDDDDO...............',  # 52
    '..................OdlllldLdChhhhhCKKKKKKKKkkkkkBdllllLDDDDDDDDDO................',  # 53
    '..................OdlllllLdEhhhhhCkKKKKKKkkkkkkBdllLlLDDDDDDDDDO................',  # 54
    '...................OllllldddhhhhhCkkKKKKkkkkkkkddllLlLDDDDDDDDDO................',  # 55
    '...................OllllldddEhhhhCkkkkkkkkkkkkkLdllllLDDDDDDDDDO................',  # 56
    '...................OdllllddddhhhhhskkkkkkkkkkkkLdllllLDDDDDDDDDDO...............',  # 57
    '....................OllllldddEhhhRRkkkkkkkRRRRRLLLLLlLDDDDDDDDDDOO..............',  # 58
    '....................OlllllddddEEhrrRllRRRRrrrrrLLLLLlLDDDDDDDDDDDdOOO...........',  # 59
    '.....................OlllllddLllkqqqllrrrrrrqqqlLLLLlDDDDDDDDDDDDdddlOO.........',  # 60
    '.....................OllllldLKsssskrglqqqqqkkkllLLLLlddddddddddddddddddOOOO.....',  # 61
    '...................OOOllllnnkssssssussssssssksLlLLdLlddddddddddddddddddddddOOO..',  # 62
    '...............OOOOBBBdLllnnkssssgGGGgssssssksLlLLdLddddddddddddddddddddddddddOO',  # 63
    '.............OOBBBBBBBBLllnnksssGssLssgssssksslLGLdldddddddddddddddddddddddddDDD',  # 64
    '...........OOBBBBBBBBBBLllnnkssGssLsLssgsssksslLuLdLddddddddddddddddddddddDDDBBB',  # 65
    '.........OOBBBBBBBBBBBBLllmnkssGsssLsssgsssksnlLLLddddddddddddddddddddddddBBBBBB',  # 66
    '........OBBBBBBBBBBBBBBLllmnkssGssLsLssusssssnlLLLddddddddddddddddddddddBBBBBBBB',  # 67
    '.......OBBBBBBBBBBBBBBBLllmnnksgssLsLssusssssnlLLLddddddddddddddddBBBBBBBBBBBBBB',  # 68
    '......ODBBBBBBBBBBBBBBBLllmnnkssgssLssussssssLlLLLddBBBBBBBBBBBBBBBBBBBBBBBBBBBB',  # 69
    '......ODBBBBBBBBBBBBBBBLllmnnnkssugggusssssnmlLLLLdBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',  # 70
    '......ODBBBBBBBBBBBBBBBLLllnnnksssssssssnnmmnlLLLLdBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',  # 71
    '......ODBBBBBBBBBBBBBBBLGllnnnnkssssssnnmmnnnlLLLLdBBBBBBBBBBBBBBBBdBBBBBBBBBBBB',  # 72
    '......ODBBBBBBBBBBBBBBBLullmnnmkssssssmmnnnnnlGLLLdBBBBBBBBBBBBBBBBBdBBBBBBBBBBB',  # 73
    '......ODBBBBBBBBBBBBBBBLLllmnnmksssssnmnnnnnnluLLLdBBBBBBBBBBBBBBBBBdBBBBBBBBBBB',  # 74
    '......ODBBBBBBBBBBBBBBBLLllmnnmskssssnmnnnnnnlLLLLdBBBBBBBBBBBBBBBBBBdBBBBBBBBBB',  # 75
    '......ODBBBBBBBBBBBBBBBLLllmnnnmkssssmnnnnnnlLLLLLdBBBBBBBBBBBBBBBBBBdBBBBBBBBBB',  # 76
    '......ODBBBBBBBBBBBBBBdLLllmnnnmkssssmnnnnnnlLLLLLdBBBBBBBBBBBBBBBBBBBdBBBBBBBBB',  # 77
    '......ODBBBBBBBBBBBBBBdLLllmnnnmksssnmnnnnnnlLLLLLdBBBBBBBBBBBBBBBBBBBdBBBBBBBBB',  # 78
    '......ODBBBBBBBBBBBBBLdLLllmnnnmssssnmnnnnnnlLLLLLdBBBBBBBBBBBBBBBBBBBBdBBBBBBBB',  # 79
]

# 表情贴片（盖在 PORTRAIT_BASE 上；'.' = 不动）。底图本身就是 neutral：冷脸、眼睛半睁、嘴抿成一字。
EXPR = {
    # 瞪视：两眼上睑压低、眼缝变窄，近侧眉内端下压成倒八字
    'glare': [
        (28, 31, ['kkkkk',
                  'OOOOO',
                  'kvVvk']),
        (37, 28, ['......sss.',
                  '..sssEEE..',
                  '.EEEE.....',
                  '..OOOOOOOO',
                  '..OOvVVvOO',
                  '....kkkkk.']),
    ],
    # 冷笑：近侧嘴角上挑一格，远侧仍是一字
    'smirk': [
        (33, 45, ['.....KK',
                  'kKKKKss']),
    ],
}


def portrait(expr=None):
    """80×80 头像。expr：None / 'neutral'（默认冷脸）、'glare'（眯眼瞪视）、'smirk'（嘴角微挑）。"""
    c = Cv(80, 80)
    c.stamp(0, 0, PORTRAIT_BASE)
    for x0, y0, rows in EXPR.get(expr, []):
        c.stamp(x0, y0, rows)
    return c.rgba(_RGB_P)


LANDMARKS = {
    # 从 `python tools/measure.py sprite leonhardt portrait` 的网格上读出（像素索引坐标）
    'portrait': {
        # 眼 / 下巴三点决定 fidelity 的取景，改了会让整张图对不上官方图，别动
        'head_top': (40, 12),
        'chin': (37, 53),
        'eye_far_out': (27, 32), 'eye_far_in': (34, 32), 'eye_far_top': (30, 31.5), 'eye_far_bot': (30, 34),
        'eye_near_in': (38, 32), 'eye_near_out': (48, 31), 'eye_near_top': (43.5, 31.5), 'eye_near_bot': (43.5, 34),
        'brow_near': (42.5, 28.5),
        'nose': (33.5, 41.5), 'mouth': (35.5, 45.5),
        'cheek_far': (26.5, 34), 'jaw_far': (29.5, 45), 'jaw_near': (48, 45),
        'neck_far': (33, 57), 'neck_near': (47, 57),
        'x_ear_top': (52, 32), 'x_ear_bot': (52, 40.5), 'x_pendant': (35, 66.5),
        # 轮廓 / 服装点（与 data/landmarks/leonhardt.json 的 bust 组同名）
        'x_spike_tall': (47, 3), 'x_spike_thin': (39, 2), 'x_lock_tip_far': (16, 17),
        'x_part': (42, 21), 'x_whorl': (46.5, 11.5), 'x_hair_near_tip': (67, 40),
        'x_collar_far_tip': (17.5, 44.5), 'x_collar_near_top': (50.5, 41.5),
        'x_rivet_collar_hi': (52, 46.5), 'x_rivet_collar_lo': (51, 54.5), 'x_choker_hook': (36.5, 59.5),
        'x_rivet_chest_near': (48, 64.5), 'x_rivet_chest_near2': (46, 73.5), 'x_rivet_chest_far': (24, 72.5),
    },
    # 从 `python tools/measure.py sprite leonhardt idle` 读出；关节点与 POSES['idle'] 一致
    'idle': {
        'head_top': (29, 2), 'chin': (28.5, 10), 'neck': (29, 13),
        'shoulder_r': (24, 14), 'shoulder_l': (37, 12), 'elbow_l': (42, 18), 'wrist_l': (45, 22),
        'waist': (26, 26), 'crotch': (28, 38),
        'knee_r': (17, 53), 'knee_l': (40, 54), 'ankle_r': (12, 67), 'ankle_l': (46, 67),
        'x_collar_tip_r': (23, 8), 'x_collar_tip_l': (34, 7),
        'x_belt_buckle': (29, 26.5), 'x_tassel_end': (26, 42),
        'x_coat_flare_r': (7, 49), 'x_coat_hem_r': (9, 53), 'x_coat_flare_l': (58, 42), 'x_coat_hem_l': (54, 52),
        'x_sword_hilt': (44, 25), 'x_blade_r': (46, 29), 'x_blade_b': (43.5, 32.5), 'x_blade_l': (38, 26.5),
    },
}


# ================================================================ 全身 64×72
# 比例取自 full.png（SC Evo 全身立绘，仰视、腿长）：缩放 70/380，头骨顶 (29,2)、下巴 (29,10)、脚底 y71，
# 头高 8px ≈ 9.3 头身；身高 70px ≈ 175–180cm（72px ≈ 180cm 的比例尺）。
# 所有部件都由少数关节点摆放（见 POSES），方便以后加走路 / 冲刺 / 挥剑的关键帧。

def limb_poly(pts, widths):
    """沿折线生成带宽度的多边形（每个节点给一个宽度），用于袖子 / 裤腿 / 风衣下摆。"""
    p = np.array(pts, float)
    w = np.array(widths, float)
    tang = np.zeros_like(p)
    tang[1:-1] = p[2:] - p[:-2]
    tang[0] = p[1] - p[0]
    tang[-1] = p[-1] - p[-2]
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1])[:, None], 1e-6)
    nor = np.stack([-tang[:, 1], tang[:, 0]], 1)
    left = p + nor * w[:, None] / 2
    right = p - nor * w[:, None] / 2
    return [tuple(v) for v in np.r_[left, right[::-1]]]


HEAD_B = [   # x0=hx-6, y0=hy-2（hx,hy = 头骨顶）；脸朝左的 3/4 侧：远侧眼 (x0+4)、近侧眼 (x0+7)、耳 (x0+8)
    "...a..a..c..",
    "..aAaaAaaac.",
    ".caAAAaaaaac",
    "caAaaaacaaac",
    ".cacSssacacc",
    "cacassOOkcac",
    ".cccOsvskCac",
    ".c.cskssk.cc",
    "c...sKssk.c.",
    "....ssskk...",
    ".....kk.....",
]
HAND_GRIP = [  # 握拳（手背朝外），x0,y0 = 拳头左上
    ".ss.",
    "sSsk",
    "sskk",
    ".kk.",
]


def _head(c, hx, hy, flip=False):
    c.stamp(hx - 6, hy - 2, HEAD_B, flip=flip)


def _collar(c, nx, ny):
    """高立领：远侧（画面左）领尖翻起、近侧领片高到耳下；红项圈 + 银环。nx,ny = 锁骨中点。"""
    c.poly('L', [(nx - 6, ny - 5), (nx - 4, ny - 3), (nx - 2, ny), (nx - 3, ny + 1), (nx - 6, ny)])
    c.line('b', [(nx - 6, ny - 5), (nx - 6, ny)])
    c.poly('L', [(nx + 2, ny - 1), (nx + 3, ny - 5), (nx + 5, ny - 6), (nx + 6, ny - 2), (nx + 3, ny + 1)])
    c.line('d', [(nx + 5, ny - 5), (nx + 6, ny - 2)])
    # 脖子 + 项圈 + V 领胸口 + 银环
    c.poly('k', [(nx - 1, ny - 3), (nx + 1, ny - 3), (nx + 1, ny - 2), (nx - 1, ny - 2)])
    c.line('R', [(nx - 1, ny - 2), (nx + 1, ny - 2)])
    c.px('r', [(nx + 1, ny - 2)])
    c.px('s', [(nx, ny - 1), (nx - 1, ny), (nx, ny), (nx, ny + 1)])
    c.px('G', [(nx - 1, ny - 1)])


def _torso(c, nx, ny, wx, wy):
    """深色 V 领内衫 + 前襟包边 + 红腰带。nx,ny 锁骨中点；wx,wy 腰带中点。"""
    c.poly('m', [(nx - 4, ny), (nx + 2, ny - 1), (wx + 3, wy), (wx - 4, wy)])
    c.poly('n', [(nx - 3, ny + 3), (nx, ny + 2), (wx + 1, wy - 1), (wx - 2, wy - 1)])
    c.line('m', [(nx - 1, ny + 4), (wx - 1, wy - 2)])
    # 前襟：远侧窄包边、近侧宽包边带铆钉
    c.line('L', [(nx - 5, ny + 1), (wx - 4, wy + 1)])
    c.line('L', [(nx + 2, ny - 1), (wx + 3, wy)])
    c.line('l', [(nx + 1, ny), (wx + 2, wy)])
    for t in (0.2, 0.5, 0.8):
        c.px('g', [(round(nx + 2 + (wx + 3 - nx - 2) * t), round(ny + (wy - ny) * t))])
    # 红腰带（微微向右下斜）+ 银扣
    c.poly('R', [(wx - 5, wy - 2), (wx + 3, wy - 1), (wx + 3, wy + 1), (wx - 5, wy)])
    c.line('r', [(wx - 5, wy), (wx + 3, wy + 1)])
    c.px('G', [(wx + 2, wy)])
    c.px('g', [(wx + 2, wy + 1), (wx + 1, wy)])


def _tassel(c, x, y, sway=0):
    """腰带扣下垂下的红色长穗（三股，末端散开）。"""
    for dx0, col, end in [(1, 'r', 13), (0, 'R', 14), (-1, 'q', 12)]:
        c.line(col, [(x + dx0, y), (x + dx0, y + 6), (x + dx0 - 2 + sway, y + 10), (x + dx0 * 2 - 3 + sway * 2, y + end)])


def _coat_back(c, wx, wy, hem_y, dx=0):
    """两腿之间露出的风衣衬里（深色）。dx = 下摆被风 / 冲刺带向右后方的偏移。"""
    c.poly('D', [(wx - 6, wy + 1), (wx + 6, wy + 1), (wx + 11 + dx, hem_y - 2), (wx + 6 + dx, hem_y),
                 (wx - 3 + dx, hem_y + 1), (wx - 10 + dx // 2, hem_y - 2)])
    c.line('F', [(wx - 3 + dx, hem_y + 1), (wx + 5 + dx, hem_y)])


def _coat_panel(c, top_in, top_out, hem_out, hem_in, flare=None, lining=True, seams=()):
    """风衣的一片前摆：top_in/top_out 腰部内外点，hem_out/hem_in 下摆外/内点，flare=外缘中间的鼓出点。"""
    flare = [flare] if flare and np.isscalar(flare[0]) else (flare or [])
    pts = [top_in, top_out] + list(flare) + [hem_out, hem_in]
    m = c.poly('b', pts)
    if lining:
        c.line('L', [top_in, hem_in])
    for sp in seams:
        c.line('d', sp, where=m)
    return m


def _leg(c, hip, knee, ankle, near=True):
    """裤腿（宽松直筒、裤脚卷边）+ 红袜 + 黑鞋。"""
    m = c.poly('P', limb_poly([hip, knee, ankle], [6, 5, 4]))
    ax, ay = ankle
    c.line('p', [(knee[0] + 1.5, knee[1]), (ax + 1.5, ay - 1)], where=m)
    c.line('Q', [(hip[0] + 2, hip[1] + 2), (knee[0] + 2, knee[1])], where=m)
    c.poly('P', [(ax - 3, ay - 3), (ax + 3, ay - 3), (ax + 3, ay - 2), (ax - 3, ay - 2)])      # 裤脚卷边
    c.line('p', [(ax - 3, ay - 2), (ax + 3, ay - 2)])
    c.poly('R', [(ax - 2, ay - 1), (ax + 2, ay - 1), (ax + 2, ay), (ax - 2, ay)])
    c.px('r', [(ax + 2, ay), (ax + 1, ay)])
    # 鞋：鞋尖朝外侧（远腿朝左、近腿朝右）
    if near:
        c.poly('F', [(ax - 2, ay + 1), (ax + 3, ay + 1), (ax + 6, ay + 3), (ax + 6, ay + 4), (ax - 2, ay + 4)])
    else:
        c.poly('F', [(ax - 4, ay + 1), (ax + 2, ay + 1), (ax + 2, ay + 4), (ax - 6, ay + 4), (ax - 6, ay + 3)])
    c.line('Q', [(ax - 1, ay + 1), (ax + 1, ay + 1)])


def _arm(c, sh, el, wr, cuff=True):
    """袖子（上臂宽、袖口深色）+ 袖章。sh 肩、el 肘、wr 腕。"""
    m = c.poly('b', limb_poly([sh, el, wr], [5, 4, 4]))
    c.line('d', [(sh[0] + 1, sh[1] + 2), (el[0] + 1, el[1] + 1), (wr[0], wr[1])], where=m)
    # 袖章（上臂一圈深色 + 银钉）
    bx, by = sh[0] + (el[0] - sh[0]) * 0.3, sh[1] + (el[1] - sh[1]) * 0.3
    c.line('L', [(bx - 1, by - 1), (bx + 1, by + 1)], where=m)
    c.px('g', [(round(bx), round(by))])
    if cuff:
        ux, uy = wr[0] - el[0], wr[1] - el[1]
        L = np.hypot(ux, uy)
        ux, uy = ux / L, uy / L
        cx, cy = wr[0] - ux * 1.5, wr[1] - uy * 1.5
        c.poly('L', limb_poly([(cx - ux * 1.5, cy - uy * 1.5), (cx + ux * 1.5, cy + uy * 1.5)], [5, 5]))
    return m


def sword(c, hand, tip, flip=False, hidden=None, horn=True):
    """魔剑「噬岩者 / Kernhitter」。hand = 握把中心，tip = 剑尖；按像素中心投影到剑轴上逐像素着色。
    结构：金色柄头 / 黑握把 / 金色护手 / 黑剑镡嵌蓝宝石 / 金边宽刃（刃面铜橙、一道深铜血槽）/
    剑脊一侧的金色梳齿 / 刃根一枚后弯的尖角。flip 翻转刃口朝向；hidden 为遮挡掩码（被风衣挡住的部分不画）。"""
    hx, hy = hand
    tx, ty = tip
    dx, dy = tx - hx, ty - hy
    Lb = float(np.hypot(dx, dy))
    ux, uy = dx / Lb, dy / Lb
    nx, ny = -uy, ux
    if flip:
        nx, ny = -nx, -ny
    yy, xx = np.mgrid[0:c.h, 0:c.w]
    px = xx + 0.5 - hx
    py = yy + 0.5 - hy
    u = px * ux + py * uy
    v = px * nx + py * ny
    out = np.zeros((c.h, c.w), np.uint8)

    def put(m, k):
        out[m & (out == 0)] = KID[k]

    B0 = 4.0                                   # 刃根
    tip0 = Lb - 7                              # 剑尖段
    t = np.clip((u - tip0) / 7, 0, 1)
    root = np.clip(1 - (u - B0) / 8, 0, 1)                    # 刃根更宽（华丽的宽刃）
    edge_w = np.where(u < tip0, 3.0 + 2.4 * root, 3.0 * (1 - t))   # 刃口一侧（宽）
    back_w = np.where(u < tip0, 1.6 + 1.2 * root, 1.6 * (1 - t))   # 剑脊一侧（刃根处带梳齿，也更宽）
    blade = (u >= B0) & (u <= Lb) & (v >= -back_w) & (v <= edge_w)
    horn_m = (u >= B0) & (u < B0 + 3) & (v > edge_w) & (v < edge_w + 3.2 - (u - B0))   # 刃根后弯尖角
    if horn:
        put(horn_m, 'Y')
    put(blade & (v >= edge_w - 1.0), 'Y')                     # 刃口金边
    put(blade & (v <= -back_w + 1.0) & ((np.floor(u) % 2) == 0), 'Y')   # 剑脊梳齿
    put(blade & (v <= -back_w + 1.0), 'y')
    put(blade & (np.abs(v - 0.6) < 0.5), 'x')                 # 血槽
    put(blade, 'o')
    # 剑镡（黑）+ 蓝宝石、金护手、握把、柄头
    put((u >= 2.5) & (u < B0) & (np.abs(v) <= 2.2) & (np.abs(u - 3.7) < 0.7) & (np.abs(v - 0.3) < 0.8), 'U')
    put((u >= 2.5) & (u < B0) & (np.abs(v) <= 2.2), 'l')
    put((u >= 1.5) & (u < 2.5) & (np.abs(v) <= 3.0), 'y')
    put((u >= -3) & (u < 1.5) & (np.abs(v) <= 1.0), 'l')
    put((u >= -4.5) & (u < -3) & (np.abs(v) <= 1.4), 'y')
    m = out > 0
    if hidden is not None:
        m &= ~hidden
    c.a[m] = out[m]
    return m


# ---------------------------------------------------------------- 背影部件
HEAD_BACK = [   # x0=hx-6, y0=hy-2：后脑勺（逆光，外缘一圈亮边）；第 7 行起是会被风吹动的发梢
    "...A..A..A..",
    "..AacaAcaaA.",
    ".AacaacaacaA",
    "Aacaacaacaac",
    ".caacaacaacc",
    "Acaacaacacac",
    ".cacCacCacC.",
    ".cCacCacCcc.",
    "c.cCc.cCcC..",
    "..C...C..C..",
]
HEAD_BACK_TURN = [   # 头微微转向画面左：露出一点脸颊和耳朵
    "...A..A..A..",
    "..AacaAcaaA.",
    ".AacaacaacaA",
    "Aacaacaacaac",
    "saaacaacaacc",
    "skcacaacacac",
    "sscCacCacC..",
    ".sCacCacCcc.",
    "..cCc.cCcC..",
    "..C...C..C..",
]


def _head_back(c, hx, hy, turn=False, sway=0):
    rows = HEAD_BACK_TURN if turn else HEAD_BACK
    c.stamp(hx - 6, hy - 2, rows[:7])
    c.stamp(hx - 6 + sway, hy - 2 + 7, rows[7:])


def _body_back(c, P):
    """背影：长风衣背面（逆光，边缘一道暖色轮廓光）、垂下的袖子、立领、后脑。"""
    hx, hy = P['head']
    sway = P.get('sway', 0)
    sa, sb = P['shoulder_r'], P['shoulder_l']          # 背影里 _r 在画面右、_l 在画面左（仍是角色自己的左右）
    sa, sb = (sb, sa) if sa[0] > sb[0] else (sa, sb)
    _leg(c, P['hip_l'], P['knee_l'], P['ankle_l'], near=False)
    _leg(c, P['hip_r'], P['knee_r'], P['ankle_r'], near=True)
    ha, hb = P['hem_a'], P['hem_b']
    coat = c.poly('d', [(sa[0] - 1, sa[1]), (hx - 4, sa[1] - 2), (hx + 5, sb[1] - 2), (sb[0] + 1, sb[1]),
                        (sb[0], hy + 26), (sb[0] + 3 + sway // 2, hy + 40), (hb[0] + sway, hb[1]),
                        (hx + 1 + sway, hb[1] + 1), (hx - 2 + sway, hb[1] + 1), (ha[0] + sway, ha[1]),
                        (sa[0] - 3 + sway // 2, hy + 40), (sa[0], hy + 26)])
    c.line('D', [(hx, hy + 38), (hx - 1 + sway, hb[1] + 1)])            # 背后开衩
    c.line('d', [(hx - 6, hy + 30), (ha[0] + 4 + sway, ha[1] - 2)], where=coat)   # 两道长褶
    c.line('d', [(hx + 7, hy + 30), (hb[0] - 4 + sway, hb[1] - 2)], where=coat)
    c.line('D', [(ha[0] + sway, ha[1]), (hx - 2 + sway, hb[1] + 1), (hx + 1 + sway, hb[1] + 1),
                 (hb[0] + sway, hb[1])])                                  # 下摆暗边
    c.line('D', [(sa[0] + 2, hy + 25), (sb[0] - 2, hy + 25)])            # 腰线
    # 逆光轮廓：左右外缘与肩线一像素暖亮
    edge = coat & ~(np.roll(coat, 1, 1) & np.roll(coat, -1, 1))
    c.fill(edge, 'b', where='d')
    top = coat & ~np.roll(coat, 1, 0)
    c.fill(top, 'B', where='db')
    # 袖子
    arms = []
    for side, (sh, el, wr) in (('a', (sa, P['elbow_a'], P['wrist_a'])), ('b', (sb, P['elbow_b'], P['wrist_b']))):
        m = _arm(c, (sh[0], sh[1] + 2), el, wr)
        c.fill(m, 'd', where='b')
        c.fill(m & ~(np.roll(m, 1, 1) & np.roll(m, -1, 1)), 'b', where='d')
        arms.append(m)
        if P.get('grip') == side:
            c.stamp(round(wr[0]) - 1, round(wr[1]) - 1, HAND_GRIP)
        else:
            x, y = round(wr[0]), round(wr[1]) + 1
            c.px('k', [(x - 1, y), (x, y), (x - 1, y + 1)])
            c.px('K', [(x, y + 1)])
    for m in arms:
        c.border(m, 'dDbB', 'O')
    if P.get('sword') and P.get('sword_layer') == 'front':
        sword(c, *P['sword'], flip=P.get('sword_flip', False), horn=P.get('horn', True))
        c.stamp(round(P['wrist_b'][0]) - 1, round(P['wrist_b'][1]) - 1, HAND_GRIP)
    # 立领（背面）
    c.poly('D', [(hx - 5, hy + 8), (hx + 5, hy + 8), (hx + 6, hy + 13), (hx - 6, hy + 13)])
    c.line('L', [(hx - 5, hy + 8), (hx + 5, hy + 8)])
    c.px('b', [(hx - 6, hy + 12), (hx + 6, hy + 12), (hx - 5, hy + 10), (hx + 5, hy + 10)])
    _head_back(c, hx, hy, P.get('turn', False), sway)


def _body_front(c, P):
    """正面 3/4（朝画面左）的通用画法：姿势完全由关节点和几个开关决定。"""
    hx, hy = P['head']
    nx, ny = P['neck']
    wx, wy = P['waist']
    layer = P.get('sword_layer', 'front')
    arms = P.get('arms', ['l'])
    grips = P.get('grips', arms)

    def draw_sword():
        if P.get('sword'):
            sword(c, *P['sword'], flip=P.get('sword_flip', False), horn=P.get('horn', True))

    # ---- 后层：风衣衬里、远侧前摆
    _coat_back(c, wx, wy, P.get('back_hem', wy + 19), P.get('back_dx', 0))
    _coat_panel(c, (wx - 4, wy), (wx - 7, wy + 2), P['hem_r'], (P['hem_r'][0] + 3, P['hem_r'][1] - 1),
                flare=P['flare_r'], seams=[[(wx - 6, wy + 6), (P['hem_r'][0] + 1, P['hem_r'][1] - 3)]])
    if layer == 'behind':
        draw_sword()
    # ---- 腿
    _leg(c, P['hip_r'], P['knee_r'], P['ankle_r'], near=False)
    _leg(c, P['hip_l'], P['knee_l'], P['ankle_l'], near=True)
    c.poly('P', [(wx - 4, wy + 1), (wx + 6, wy + 1), (P['hip_l'][0] + 2, P['hip_l'][1] + 2),
                 (wx + 1, wy + 11), (P['hip_r'][0] - 2, P['hip_r'][1] + 2)])
    c.line('p', [(wx + 1, wy + 4), (wx + 1, wy + 10)])
    # ---- 近侧前摆
    fl = P['flare_l']
    _coat_panel(c, (wx + 4, wy), (wx + 10, wy - 2), P['hem_l'], (P['hem_l'][0] - 6, P['hem_l'][1] - 1),
                flare=fl,
                seams=[[(wx + 10, wy + 4), (P['hem_l'][0] - 2, P['hem_l'][1] - 3)],
                       [(fl[-1][0] - 5, fl[-1][1]), (P['hem_l'][0] + 1, P['hem_l'][1] - 2)]])
    for sh in P.get('coat_shadow', []):
        c.poly('d', sh, where='b')
    # ---- 上身
    sr, sl = P['shoulder_r'], P['shoulder_l']
    c.poly('b', [(sr[0] - 2, sr[1]), (nx - 2, ny - 1), (nx + 4, ny - 2), (sl[0] + 1, sl[1]),
                 (wx + 10, wy - 2), (wx + 4, wy + 1), (wx - 4, wy + 1), (wx - 7, wy + 2)])
    for sh in P.get('coat_shadow', [])[:1]:
        c.poly('d', sh, where='b')
    _torso(c, nx, ny, wx, wy)
    _tassel(c, *P['tassel'])
    if layer == 'mid':
        draw_sword()
    # ---- 手臂（与躯干同色，用选择性描边分开）
    ms = []
    for side in arms:
        ms.append(_arm(c, P['shoulder_' + side], P['elbow_' + side], P['wrist_' + side]))
    for m in ms:
        c.border(m, 'bBdD', 'O')
    if layer == 'front':
        draw_sword()
    for side in grips:
        w = P['wrist_' + side]
        c.stamp(round(w[0]) - 1, round(w[1]) - 1, HAND_GRIP)
    # ---- 明暗
    c.shade('b', 'd', 1, 1, keep='bBdL')
    c.light('b', 'B', 1, 1, keep='bBdL')
    _collar(c, nx, ny)
    _head(c, hx, hy)


# 画布：S = 64×72（原点即身体坐标）；L = 160×104，身体坐标 (0,0) 在画布 (64,32)。
# 所有姿势都以「身体坐标 (32,72)」为脚底根点：场景里把根点放到地面上即可（见 ROOT / pose_info）。
CANVAS = {'S': (64, 72, 0, 0), 'L': (160, 104, 64, 32)}
ROOT = (32, 72)

_BACK = dict(view='back', head=(29, 2), shoulder_l=(21, 15), shoulder_r=(38, 15),
             elbow_a=(19, 25), wrist_a=(18, 33), elbow_b=(40, 25), wrist_b=(41, 33),
             hip_l=(26, 36), hip_r=(33, 36), knee_l=(25, 53), knee_r=(34, 53), ankle_l=(24, 67), ankle_r=(35, 67),
             hem_a=(16, 58), hem_b=(43, 58))
_LUNGE = dict(hem_r=(46, 56), flare_r=[(38, 46)], hem_l=(70, 48), flare_l=[(62, 36)], back_hem=50, back_dx=6)

# 姿势 = 关节坐标（身体坐标）。键名与 LANDMARKS 的身体点一致（角色自己的左右：正面时 _r 在画面左）。
POSES = {
    'idle': dict(   # full.png：分腿站立，脸朝左，左手（画面右）在腰侧反握魔剑、剑身藏在风衣后
        head=(29, 2), neck=(29, 13), waist=(27, 27),
        shoulder_r=(24, 14), shoulder_l=(37, 12), elbow_l=(42, 18), wrist_l=(45, 22),
        hip_r=(24, 33), hip_l=(33, 34), knee_r=(17, 53), knee_l=(40, 54), ankle_r=(12, 67), ankle_l=(46, 67),
        sword=((45.5, 24.0), (41.0, 40.0)), sword_layer='behind',
        hem_r=(9, 53), flare_r=(7, 49), hem_l=(54, 52), flare_l=[(39, 31), (47, 35), (58, 42)],
        tassel=(29, 28, 0), back_hem=46,
        coat_shadow=[[(35, 16), (37, 15), (38, 20), (36, 25), (35, 22)],             # 臂下躯干的阴影
                     [(36, 36), (44, 40), (51, 50), (47, 51), (40, 44), (35, 39)],   # 近侧前摆内侧的暗面
                     [(20, 33), (22, 36), (14, 47), (11, 52), (10, 50)]],            # 远侧前摆内侧
    ),
    'skill': dict(  # 鬼炎斩蓄势：弓步前压，双手在右腰握剑、剑身斜指右上（剑尖 = weapon_tip），风衣向右后扬起
        head=(23, 6), neck=(24, 17), waist=(27, 31),
        shoulder_r=(19, 18), shoulder_l=(32, 16), elbow_r=(24, 25), wrist_r=(29, 29),
        elbow_l=(36, 23), wrist_l=(32, 28), arms=['r', 'l'],
        hip_r=(23, 36), hip_l=(31, 37), knee_r=(12, 49), knee_l=(42, 51), ankle_r=(11, 67), ankle_l=(52, 67),
        sword=((31.5, 30.5), (61.5, 1.5)), sword_flip=True, horn=False,
        hem_r=(15, 56), flare_r=(12, 49), hem_l=(62, 50), flare_l=[(62, 40)],
        tassel=(28, 32, 3), back_hem=48,
        coat_shadow=[[(35, 36), (46, 40), (60, 49), (52, 50), (40, 44)]],
    ),
    # ---- 背影（哈梅尔的墓前）。风衣下摆 / 发梢三帧循环：back → back_w1 → back_w2
    'back': dict(_BACK, sway=0, sword=((11.5, 40.5), (12.5, 80)), sword_layer='behind'),
    'back_w1': dict(_BACK, sway=1, sword=((11.5, 40.5), (12.5, 80)), sword_layer='behind'),
    'back_w2': dict(_BACK, sway=2, sword=((11.5, 40.5), (12.5, 80)), sword_layer='behind'),
    'back_turn': dict(_BACK, sway=1, turn=True, sword=((11.5, 40.5), (12.5, 80)), sword_layer='behind'),
    # 结尾：背对镜头，右手垂剑（剑尖指向右下的地面）
    'back_sword': dict(_BACK, canvas='L', sway=2, grip='b', elbow_b=(43, 24), wrist_b=(46, 31),
                       sword=((46.5, 32.5), (68, 62)), sword_layer='front', horn=False),
    # ---- 转身：左手（画面左）按在插进地里的剑柄上
    'rest': dict(
        head=(28, 2), neck=(28, 13), waist=(27, 27),
        shoulder_r=(23, 14), shoulder_l=(35, 13), elbow_r=(18, 20), wrist_r=(14, 25),
        elbow_l=(38, 20), wrist_l=(38, 27), arms=['r', 'l'],
        hip_r=(24, 33), hip_l=(32, 34), knee_r=(21, 52), knee_l=(36, 52), ankle_r=(19, 67), ankle_l=(39, 67),
        sword=((14.5, 28.5), (13.0, 76.0)), sword_layer='mid',
        hem_r=(14, 56), flare_r=(16, 44), hem_l=(46, 57), flare_l=[(40, 40)], tassel=(29, 28, 0), back_hem=46,
    ),
    # ---- 拔剑：把插在地里的剑拔起（draw1），抡过头顶（draw2），落进蓄势架势（stance）
    'draw1': dict(
        canvas='L', head=(27, 5), neck=(27, 16), waist=(27, 30),
        shoulder_r=(22, 17), shoulder_l=(34, 16), elbow_r=(17, 13), wrist_r=(16, 7),
        elbow_l=(38, 23), wrist_l=(37, 30), arms=['r', 'l'], grips=['r', 'l'],
        hip_r=(24, 36), hip_l=(32, 37), knee_r=(19, 53), knee_l=(38, 53), ankle_r=(16, 67), ankle_l=(42, 67),
        sword=((16.5, 9.5), (15.0, 58.0)), sword_layer='front',
        hem_r=(14, 57), flare_r=(15, 46), hem_l=(50, 56), flare_l=[(44, 40)], tassel=(29, 31, 1), back_hem=49,
    ),
    'draw2': dict(
        canvas='L', head=(25, 4), neck=(25, 15), waist=(27, 29),
        shoulder_r=(20, 16), shoulder_l=(32, 14), elbow_r=(22, 22), wrist_r=(28, 17),
        elbow_l=(36, 20), wrist_l=(31, 15), arms=['r', 'l'],
        hip_r=(23, 35), hip_l=(31, 36), knee_r=(15, 51), knee_l=(40, 52), ankle_r=(12, 67), ankle_l=(48, 67),
        sword=((29.5, 16.5), (42.0, -24.0)), sword_layer='front', sword_flip=True, horn=False,
        hem_r=(14, 56), flare_r=(13, 47), hem_l=(58, 52), flare_l=[(56, 40)], tassel=(28, 30, 2), back_hem=48,
    ),
    'stance': dict(  # = skill，画在 L 画布上，重心再压低一点
        canvas='L', head=(22, 8), neck=(23, 19), waist=(27, 33),
        shoulder_r=(18, 20), shoulder_l=(31, 18), elbow_r=(24, 27), wrist_r=(29, 31),
        elbow_l=(36, 25), wrist_l=(32, 30), arms=['r', 'l'],
        hip_r=(23, 38), hip_l=(31, 39), knee_r=(11, 51), knee_l=(43, 53), ankle_r=(9, 67), ankle_l=(55, 67),
        sword=((31.5, 32.5), (64.0, 2.0)), sword_flip=True, horn=False,
        hem_r=(15, 58), flare_r=(12, 50), hem_l=(64, 52), flare_l=[(64, 42)], tassel=(28, 34, 3), back_hem=50,
        coat_shadow=[[(35, 38), (46, 42), (62, 51), (54, 52), (40, 46)]],
    ),
    # ---- 冲刺：身体压成一条斜线，剑拖在身后，风衣水平向后飞
    'dash': dict(
        canvas='L', head=(12, 19), neck=(16, 28), waist=(27, 37),
        shoulder_r=(12, 28), shoulder_l=(23, 25), elbow_r=(22, 36), wrist_r=(30, 40),
        elbow_l=(29, 32), wrist_l=(33, 38), arms=['r', 'l'],
        hip_r=(24, 43), hip_l=(31, 44), knee_r=(12, 55), knee_l=(44, 57), ankle_r=(6, 67), ankle_l=(58, 66),
        sword=((32.5, 39.5), (76.0, 50.0)), sword_flip=True, horn=False,
        hem_r=(56, 52), flare_r=[(42, 44)], hem_l=(80, 44), flare_l=[(68, 32)], back_hem=50, back_dx=12,
        tassel=(29, 38, 6),
    ),
    # ---- 斩击三帧：举过右肩（slash1）→ 命中，剑身扫向左前下方（slash2）→ 收势贴地（slash3）
    'slash1': dict(
        _LUNGE, canvas='L', head=(15, 10), neck=(18, 21), waist=(26, 34),
        shoulder_r=(13, 22), shoulder_l=(25, 19), elbow_r=(20, 12), wrist_r=(26, 5),
        elbow_l=(29, 11), wrist_l=(28, 3), arms=['r', 'l'],
        hip_r=(23, 40), hip_l=(30, 41), knee_r=(12, 52), knee_l=(42, 55), ankle_r=(6, 67), ankle_l=(55, 67),
        sword=((27.5, 4.5), (58.0, -22.0)), sword_flip=True, horn=False, tassel=(28, 35, 5),
    ),
    'slash2': dict(
        _LUNGE, canvas='L', head=(12, 14), neck=(15, 24), waist=(25, 36),
        shoulder_r=(10, 25), shoulder_l=(22, 22), elbow_r=(3, 29), wrist_r=(-4, 31),
        elbow_l=(12, 28), wrist_l=(-1, 30), arms=['r', 'l'],
        hip_r=(22, 41), hip_l=(29, 42), knee_r=(10, 53), knee_l=(42, 56), ankle_r=(3, 67), ankle_l=(56, 67),
        sword=((-4.5, 31.5), (-44.0, 48.0)), horn=False, tassel=(27, 37, 5),
        hem_l=(72, 50), flare_l=[(64, 36)],
    ),
    'slash3': dict(
        canvas='L', head=(10, 20), neck=(14, 29), waist=(25, 39),
        shoulder_r=(9, 30), shoulder_l=(20, 27), elbow_r=(4, 38), wrist_r=(-1, 45),
        elbow_l=(11, 37), wrist_l=(1, 46), arms=['r', 'l'],
        hip_r=(22, 44), hip_l=(29, 45), knee_r=(8, 55), knee_l=(42, 58), ankle_r=(2, 67), ankle_l=(57, 67),
        sword=((-0.5, 46.5), (-34.0, 70.0)), horn=False,
        hem_r=(40, 60), flare_r=[(34, 50)], hem_l=(60, 58), flare_l=[(56, 44)], back_hem=54, back_dx=4,
        tassel=(27, 40, 3),
    ),
    # ---- 收势起身：单手垂剑向左前下方
    'recover': dict(
        canvas='L', head=(20, 4), neck=(21, 15), waist=(25, 29),
        shoulder_r=(16, 16), shoulder_l=(28, 15), elbow_r=(14, 24), wrist_r=(15, 31),
        elbow_l=(29, 23), wrist_l=(22, 30), arms=['r', 'l'], grips=['l'],
        hip_r=(22, 35), hip_l=(30, 36), knee_r=(14, 51), knee_l=(40, 52), ankle_r=(9, 67), ankle_l=(49, 67),
        sword=((21.5, 30.5), (-10.0, 60.0)), horn=False,
        hem_r=(12, 58), flare_r=[(12, 48)], hem_l=(52, 59), flare_l=[(46, 44)], tassel=(28, 30, 2), back_hem=48,
    ),
}
# 按剑站立时的风吹帧：风衣下摆与腰穗向右后方飘
POSES['rest_w1'] = dict(POSES['rest'], hem_l=(48, 56), flare_l=[(42, 40)], hem_r=(15, 56), tassel=(29, 28, 1))
POSES['rest_w2'] = dict(POSES['rest'], hem_l=(50, 55), flare_l=[(43, 39)], hem_r=(16, 55), tassel=(29, 28, 2),
                        back_dx=2)
# 动画用的姿势顺序（场景按时间挑帧）
ANIM = {
    'wind': ['back', 'back_w1', 'back_w2', 'back_w1'],
    'rest': ['rest', 'rest_w1', 'rest_w2', 'rest_w1'],
    'draw': ['rest', 'draw1', 'draw2', 'stance'],
    'slash': ['stance', 'dash', 'slash1', 'slash2', 'slash3', 'recover'],
}
_NOSHIFT = {'view', 'canvas', 'sway', 'turn', 'grip', 'grips', 'arms', 'sword_layer', 'sword_flip', 'horn',
            'back_hem', 'back_dx'}


def _shift(v, ox, oy):
    if isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], (list, tuple)):
        return type(v)(_shift(p, ox, oy) for p in v)
    if isinstance(v, tuple) and len(v) in (2, 3) and all(isinstance(q, (int, float)) for q in v):
        return (v[0] + ox, v[1] + oy) + tuple(v[2:])
    return v


def _pose(pose):
    P = POSES.get(pose, POSES['idle'])
    w, h, ox, oy = CANVAS[P.get('canvas', 'S')]
    Q = {k: (v if k in _NOSHIFT else _shift(v, ox, oy)) for k, v in P.items()}
    if 'back_hem' in P:
        Q['back_hem'] = P['back_hem'] + oy
    return Q, (w, h, ox, oy)


def pose_info(pose):
    """场景用的每帧数据（画布像素坐标）：size、root（脚底根点）、hand / tip（剑的握把与剑尖）、eyes。"""
    Q, (w, h, ox, oy) = _pose(pose)
    sw = Q.get('sword')
    hd = Q.get('head')
    if Q.get('view') == 'back':
        eyes = []
    else:
        eyes = [(hd[1] + 4, hd[0] - 2), (hd[1] + 4, hd[0])]
    return dict(size=(w, h), root=(ROOT[0] + ox, ROOT[1] + oy),
                hand=sw[0] if sw else None, tip=sw[1] if sw else None, eyes=eyes)


@lru_cache(None)
def _body_cached(pose):
    Q, (w, h, ox, oy) = _pose(pose)
    c = Cv(w, h)
    if Q.get('view') == 'back':
        if Q.get('sword') and Q.get('sword_layer') == 'behind':
            sword(c, *Q['sword'], horn=Q.get('horn', True))
        _body_back(c, Q)
    else:
        _body_front(c, Q)
    c.outline()
    for edge in (c.a[0], c.a[:, 0], c.a[:, -1]):
        edge[edge > 0] = KID['O']
    return c.rgba()


def body(pose='idle'):
    """全身像。朝画面左的 3/4 侧：idle（full.png 的分腿站姿、左手反握魔剑）、skill（弓步蓄势，剑斜指右上）；
    另有场景用的动画帧（见 POSES / ANIM）：back* 背影与风吹帧、rest 按剑、draw1/draw2/stance 拔剑入架势、
    dash 冲刺、slash1/2/3 斩击、recover 收势、back_sword 背影垂剑。S 画布 64×72，L 画布 160×104；
    所有姿势的脚底根点见 pose_info(pose)['root']。"""
    return _body_cached(pose).copy()


META = dict(
    key='leonhardt',
    name_zh='莱恩哈特',
    title_zh='剑帝',
    rank_zh='执行者 No.II',
    skill_zh='鬼炎斩',
    quote_zh='修罗之道，唯剑而已。',
    fx=dict(
        primary='#ff4a2a',
        secondary='#ffc45a',
        description=(
            '鬼炎斩：蓄势时从护手到剑尖（weapon_tip）整把「噬岩者」被赤红鬼火包裹，火舌沿剑身向后拖曳，'
            '剑脊的金色梳齿处迸出火星；随后一记由右上向左下的巨大斩击，留下一道宽弧形的火焰刀光'
            '（外缘 primary 赤红、内芯 secondary 金橙，可夹一丝银白残像），刀光划过后炸开余烬。'
            '风衣下摆与红色腰穗被气浪吹向右后方。'
        ),
    ),
    # 全身图中两只眼的像素 (y, x)：远侧眼、近侧眼（虹膜）。由 HEAD_B 贴片位置 + POSES 的 head 推出
    eyes_body={'idle': [(6, 27), (6, 29)], 'skill': [(10, 21), (10, 23)]},
    weapon_tip=(2, 60),              # skill 姿势剑尖像素 (y, x)（POSES['skill']['sword'] 的剑尖）
)

