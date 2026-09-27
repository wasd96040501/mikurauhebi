"""Campanella / 肯帕雷拉 —— 执行者 No.0「道化师」。

v3：按官方立绘量过比例重画（tools/measure.py）。
- 立绘 portrait()：以 SC Evo 半身像为底稿（缩到 80x80 后按材质分色块、手工清理），朝画面左 3/4，
  刘海盖住角色左眼（近侧），可见的是远侧的红眼 + 眼下红色脸纹。眼、嘴、脸纹、鼻按表情另行盖章。
- 全身 body()：按 Sen III 立绘比例（1 px ≈ 45 官方像素，身高 64 px ≈ 160 cm）。
  由部件拼装：腿 → 后手臂 → 躯干（外套 / 衬衫下摆 / 黑翻领白边 / 领带 / 金扣）→ 头 → 前手臂，
  每个姿势只是一组关节坐标（POSES），方便以后加走路 / 攻击帧。
光源左上。每种材质 3~4 阶，外轮廓按材质用深色描边（头发描深橄榄，皮肤描深棕，其余描近黑）。
"""
import math

import numpy as np
from PIL import Image, ImageDraw

import pixelkit as pk

PAL = {
    'O': '#1c1418',                                                    # 通用描边
    'o': '#2a2c0c',                                                    # 头发描边
    'x': '#6e3428',                                                    # 皮肤线 / 描边
    'H': '#eef07a', 'h': '#cacc36', 'g': '#969a2a', 'd': '#5e641c',    # 发（黄绿）
    's': '#fcdcc4', 'k': '#f0a67a', 'K': '#b86e4e',                    # 肤
    'W': '#f2f2ec', 'R': '#ff6a80', 'r': '#c8203c', 'q': '#4a1422',    # 眼
    'M': '#ff4a78',                                                    # 脸纹
    'J': '#e8829e', 'j': '#c8607e', 'i': '#8e4052', 'I': '#5e2434',    # 外套 / 裤（覆盆子红）
    'L': '#5e5c6c', 'l': '#42404c', 'm': '#2a2832',                    # 黑翻领
    'T': '#f0eef2', 't': '#b4b0bc',                                    # 翻领白边
    'B': '#b4c2f0', 'b': '#7c86b4', 'n': '#545a86',                    # 浅蓝衬衫
    'G': '#f8c850', 'Y': '#c88c20',                                    # 黄领带 / 金扣
    'S': '#5a5668', 'z': '#2e2a38',                                    # 黑皮鞋
}
KEYS = ['.'] + list(PAL)
KID = {k: i for i, k in enumerate(KEYS)}
_RGB = np.zeros((len(KEYS), 3), np.float32)
for _k, _v in PAL.items():
    _RGB[KID[_k]] = pk.hexc(_v)
# 外轮廓描边色：按描边内侧那个像素的材质选
_OUT = np.full(len(KEYS), KID['O'], np.int64)
for _k in 'Hhgd':
    _OUT[KID[_k]] = KID['o']
for _k in 'skK':
    _OUT[KID[_k]] = KID['x']


# ---------------------------------------------------------------- 画布工具
class Cv:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.a = np.zeros((h, w), np.uint8)
        self.post = []  # 描边后再画的像素 (x, y, 色键)
        self.T = None   # 可选坐标变换 (x, y) -> (x, y)，鞠躬时上半身以胯为支点错切前倾

    def tp(self, pts):
        if self.T is None:
            return [tuple(p) for p in pts]
        return [tuple(v for v in self.T(*p)) for p in pts]

    def m_poly(self, pts):
        im = Image.new('L', (self.w, self.h), 0)
        ImageDraw.Draw(im).polygon(self.tp(pts), fill=1, outline=1)
        return np.array(im, bool)

    def m_line(self, pts, width=1):
        im = Image.new('L', (self.w, self.h), 0)
        ImageDraw.Draw(im).line(self.tp(pts), fill=1, width=width)
        return np.array(im, bool)

    def fill(self, m, c, where=None):
        if where is not None:
            m = m & where
        self.a[m] = KID[c]
        return m

    def poly(self, c, pts, where=None):
        return self.fill(self.m_poly(pts), c, where)

    def line(self, c, pts, width=1, where=None):
        return self.fill(self.m_line(pts, width), c, where)

    def px(self, c, pts):
        for x, y in self.tp(pts):
            x, y = int(round(x)), int(round(y))
            if 0 <= x < self.w and 0 <= y < self.h:
                self.a[y, x] = KID[c]

    def stamp(self, rows, x0, y0):
        """ASCII 盖章：'.'/' ' 不动，'_' 擦成透明。有 T 时只平移到变换后的位置（像素不旋转）。"""
        if self.T is not None:
            tx, ty = self.T(x0, y0)
            x0, y0 = int(round(tx)), int(round(ty))
        for dy, r in enumerate(rows):
            for dx, ch in enumerate(r):
                if ch in '. ':
                    continue
                y, x = y0 + dy, x0 + dx
                if 0 <= x < self.w and 0 <= y < self.h:
                    self.a[y, x] = 0 if ch == '_' else KID[ch]

    def has(self, keys):
        return np.isin(self.a, [KID[k] for k in keys])

    @property
    def opaque(self):
        return self.a > 0

    def rgba(self, outline=True):
        img = np.zeros((self.h, self.w, 4), np.float32)
        img[..., :3] = _RGB[self.a]
        img[..., 3] = (self.a > 0)
        if outline:
            op = self.a > 0
            src = np.zeros_like(self.a)
            for dy, dx in ((-1, 0), (0, 1), (0, -1), (1, 0)):   # 优先取下方像素的材质
                nb = shift(self.a, dy, dx)
                src = np.where((src == 0) & ~op, nb, src)
            ring = (src > 0) & ~op
            img[ring, :3] = _RGB[_OUT[src[ring]]]
            img[ring, 3] = 1
        for x, y, ch in self.post:
            if 0 <= x < self.w and 0 <= y < self.h:
                img[y, x, :3] = _RGB[KID[ch]]
                img[y, x, 3] = 1
        return img


def shift(m, dy, dx):
    """m 平移 (dy, dx)，越界补 0/False。"""
    out = np.zeros_like(m)
    h, w = m.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = m[ys, xs]
    return out


def edge(m, dy, dx, n=1):
    """m 中距 (dy,dx) 方向外侧 n 像素以内的点。"""
    inner = m.copy()
    for i in range(1, n + 1):
        inner &= shift(m, -dy * i, -dx * i)
    return m & ~inner


def border(m, other):
    """m 中与 other（m 以外）4 邻接的像素。"""
    o = other & ~m
    g = shift(o, 1, 0) | shift(o, -1, 0) | shift(o, 0, 1) | shift(o, 0, -1)
    return m & g


# ================================================================ 立绘 80x80
# 底稿：官方半身像按相似变换（缩放 0.21，下巴落在 (37,55)）缩到 80x80，按材质分色块后手工清理。
# 眼 / 嘴 / 脸纹 / 鼻不在底稿里，由 portrait() 按表情盖章。
PORTRAIT_BASE = [
    '................................................................................',
    '.......................................hhhhhhh..................................',
    '.....................................hhh.....hh.................................',
    '....................................hh........h.................................',
    '...................................hh...........................................',
    '..................................hh............................................',
    '.................................hh.............................................',
    '.................................h..............................................',
    '.................................h.hhhhhhhhhhhhh................................',
    '......................hhhh......hhhhhgggghhhhhhhhhh.............................',
    '..........................hh..hhghhhgghhhhhhhhhhhhhhh...........................',
    '............................hhhhghhhhhhhhhhhhhhhhhhhhh..........................',
    '...........................hhhhhhhhhhhhhhhhhhhhhhhhhhhh.........................',
    '..........................hhhhhhhgghhhhhhhhhhhhhhhhhhhhhh.......................',
    '.........................hhhhhhhgghhhhhhhhhhhhhhhhhhhhhhhh......................',
    '........................hhhhhggggggghhhhhhhhhhhhhhhhhhhhhh......................',
    '........................hhhhgggghhgggghhhhhhhhhhhhhhhhhhhhg.....................',
    '.......................hhhhgghhhhhhhhhhhhhhhhhhhhhhhhhhhhggg....................',
    '......................hhhhhghhhhhhhhhhhhhhhhhhhhhghhhhhhhggg....................',
    '.....................hhhhhhhhhhhhhhhhhhhhhhhhhhhhghhhhhhhhggg...................',
    '.....................hhhhhhhhhhhhhhhhhhhhhhhhhhhggghhhhghgggg...................',
    '....................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhggghhhghgggg...................',
    '....................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhggghhhgggggg..................',
    '...................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhggghghgggggg..................',
    '...................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhggggghgggggg..................',
    '...................hhhhhhhghkhhhhhhhhhhhhhhhhhhhgggggggggggdggd.................',
    '...................hhhhhhhghkshhhhhhhhhhhhhhhhghgggggggggggdggd.................',
    '..................hhhhhhhhghkshhhhhhhhhhhhghhhghhggggdggggggdgd.................',
    '..................hhhhhhhhghkshhhhhhhhhhhhgghhhhhggggdggggggdgg.................',
    '..................hhhhhhhgghsshhhhhhhhhhhhhgghhghggggdgggggdddgh................',
    '..................hhhhhhhgghssshhhhhhhhhhhhhgghghggggddggggdddgh................',
    '..................hhhhhhhgkhsssgghhhghhhhhhhhggghggggddggggddddhh...............',
    '..................hhhhhghgkssssggghhgghhhhgghggghhgggddgggddddddg...............',
    '..................hghhgghgssssssgghhhgghhhhgggggghhggdddggdddddggg..............',
    '..................hghhgghgsssssssgggggggghhgggggghhggddddgdddddddd..............',
    '..................gghgggggssssssssggggggggggggggghhggddddggddgddgg.gg...........',
    '..................ghggggggsssssssssggggggggggggdghhgddKkddgddggddg..............',
    '..................dhgggdggssssssssssgggggggggggddghgddKkddgdddgddg..............',
    '..................d.gggdggssssssssssssggggkgggggdgggddKkddddddggdd..............',
    '....................gggddgssssssssssssssgskkgggggggdddKkdddddddgg.dd............',
    '....................hdgdddsssssssssssssssssskkkgggddddKkddddddddd..d............',
    '....................hdggddsssssssssssssssssssskgggddggKkdddddddddd..............',
    '...................hhhggddddssssssssssssssssssskgddgggdddddddddddd..............',
    '...................hhhgggdssssssssssssssssssssskgdggggddddddddd..d..............',
    '...................hghgggddssssssssssssssssssssddgggggdddd.dddd.................',
    '....................ghghggdsssssssssssssssssssskkgggggdddd.d.d..................',
    '.....................gghggdssssssssssssssssssssssgggggdd.d......................',
    '.......................hhggdsssssssssssssssssssskgggggdd........................',
    '.......................hhggdsssssssssssssssssssskgggggd.........................',
    '........................hgggdsssssssssssssssssskkggggg..........................',
    '........................ggggddssssssssssssssskkkKKgggg..........................',
    '........................ggggdd.sssssssssssskkkKKhgggggg.........................',
    '........................ggggdd..sssssssssskkKKkkhgggddg.........................',
    '.........................gggdd....sssssskkKKkkhggggdd.g.........................',
    '........................hgggggd....ssskkKKkkkkhggggdd...........................',
    '........................hgggggd....xxkKKkkkkkBhggggd............................',
    '.........................hgggggd.B.kkKkkkkkkBhgggggdbb..........................',
    '.........................hgggggd.BBBGGYGGYBBBhggggdbblt.........................',
    '..........................hggggdBBBBGGYGGYBBBhgggdbbllltt.......................',
    '..........................hggggdBBBBBBGbbbBBBhgggdbblllllt......................',
    '...........................gggdBBBBBbGYbGYBBBhggdbbllllllltt....................',
    '............................hgdBBBBBbGYbGYnBBBhdbbllllllllllt...................',
    '...........................ThgdLBBBbGYBbGYnBBBhdblllllllllllltt.................',
    '.........................TTLLdLLBBBbGYBbGYnnBBdbllllllllllllllltj...............',
    '........................TLLLLdLLBBbGYBBbbGYnBbdmlllllllllllllllTjj..............',
    '......................jjTLLLLLLLBBbGYBBbbGYBnbbmllllllllllllllTjjjji............',
    '.....................jjTLLLLLLLLLbGYBBBbbGYnnbmmllllllllllllllTjjjjii...........',
    '...................jjjjTLLLLLLLLLbGYBBBbbGYnbbmllllllllllllllTjjjjjiiii.........',
    '..................jjjjTLLLLLLLLLLbGYBBBbbGYnnnmllllllllllllllTjjjjjiiiii........',
    '.................jjjjjTLLLLLLLLLLGYbBBBbbGYnnnllllllllllllllTjjjjjjiiiiiii......',
    '................JjjjjTLLLLLLLLLLLGYbBBBbbGYnnnllllllllllllllTjjjjjjiiiiiiii.....',
    '...............JJjjjjTLLLLLLLLLLGYbbBBbbbbGYnnllllllllllllllTjjjjjjiiiiiiiiii...',
    '..............JJJjjjjTLLLLLLLLLLGYbBBBbbbbGYnnllllllllllllllTjjjjjjjiiiiiiiiii..',
    '..............JJJjjjTLLLLLLLLLLLGYbBBBbbbbGYnnlllllllllllllTjjjjjjjjiiiiiiiiii..',
    '..............JJJjjjTLLLLLLLLLLGYLbBBBbbbbGYnllllllllllllllTjjjjjjjjiiiiiiiiiii.',
    '..............JJJjjjTLLLLLLLLLLGYLbBBBbbbbGYnllllllllllllllTjjjjjjjjiiiiiiiiiii.',
    '..............JJJjjjTLLLLLLLLLLLLLbBBBbbbbnnnllllllllllllllTjjjjjjjjiiiiiiiiiiii',
    '.............JJJJJjTLLLLLLLLLLLLLLLBBBbbbbnnnlllllllllllllTjjjjjjjjjiiiiiiiiiiii',
    '.............JJJJJjTLLLLLLLLLLLLLLLBBBbbbbnnnlllllllllllllTjjjjjjjjjiiiiiiiiiiii',
    '.............JJJJJjTLLLLLLLLLLLLLLLBBBbbbbnnnlllllllllllllTjjjjjjjjjiiiiiiiiiiii',
]

# 可见的是远侧（角色右眼）：外眼角在左、上挑的睫毛，眼白在外侧，红虹膜靠内侧，左上一点高光
EYES = {  # 盖章左上角 (25, 34)
    'smirk': [
        'OO......',
        '.OOOOOO.',
        '.WWqWqqO',
        '..Wqqrr.',
        '..Wrrrr.',
        '....RR..',
    ],
    'wink': [  # 眯成向上的弧（得意地闭眼）
        '........',
        'O.......',
        '..OOOO..',
        '.O....O.',
        '........',
        '........',
    ],
}
MOUTHS = {  # 盖章左上角 (34, 48)
    'smirk': [
        'x.....x',
        '.xxxxx.',
        '..kk...',
    ],
    'wink': [
        'x.....x',
        '.xWWWx.',
        '..xxx..',
    ],
}
# 脸纹：眼下红点 + 托住它的弯月 + 向下的一道尖
MARK = [(29, 42), (27, 43), (28, 44), (29, 44), (30, 44), (31, 43), (30, 45), (31, 46)]


def portrait(expr=None):
    expr = expr if expr in EYES else 'smirk'
    c = Cv(80, 80)
    for y, r in enumerate(PORTRAIT_BASE):
        for x, ch in enumerate(r):
            if ch not in '. ':
                c.a[y, x] = KID[ch]
    skin = c.has('skK')
    hair = c.has('Hhgd')
    # 发梢压在脸上：与皮肤相接的头发描一道暗线
    c.fill(border(hair, skin) & c.has('Hhg'), 'd')
    # 近侧下颌线（下巴 -> 刘海下）与下巴底
    c.line('x', [(37, 55), (47, 49)])
    # 头发高光（天使环一带的几道短光）
    c.px('H', [(27, 25), (28, 24), (37, 23), (38, 22), (39, 22), (46, 22), (47, 21), (52, 24), (53, 23),
               (22, 30), (22, 31), (33, 16), (34, 15)])
    # 脸部
    c.stamp(EYES[expr], 25, 34)
    for i, p in enumerate(MARK):
        c.px('r' if i == len(MARK) - 1 else 'M', [p])
    c.px('K', [(35, 46)])
    c.px('k', [(36, 46)])
    c.stamp(MOUTHS[expr], 34, 48)
    return c.rgba()


# ================================================================ 全身 48x72
# 比例：官方立绘 (x, y) -> (0.0222x + 10, 0.0222y + 3.4)。身高 64 px（发顶 y=5，鞋底描边 y=71）。
# 头：12x13 章（头骨顶到下巴 ≈ 9 px，约 7 头身，与官方一致），HEAD_CHIN = 章内下巴像素。
HEAD_B = [
    '...hhhh.....',
    '..hHHhhhg...',
    '.hHhhhhhgg..',
    '.hhhhhhgggd.',
    '.hhhhhgggddg',
    '.hhhhhhggdd.',
    '.hsOOhhgdskd',
    '.gsrshgdkkd.',
    '.gsMsssgd.d.',
    '.ggssKskd...',
    '.g..ksk.g...',
    '.g......g...',
    '.g......g...',
]
HEAD_CHIN = (5, 10)
AHOGE = [(-3, -11, 'g'), (-4, -12, 'g'),                                        # 朝左的短呆毛
         (-1, -11, 'h'), (-1, -12, 'h'), (0, -13, 'h'), (1, -13, 'h'), (2, -12, 'h'), (2, -11, 'g')]  # 高的一根向右弯下


def stamp_at(c, rows, anchor, ax, ay):
    """把章内 (ax, ay) 对准世界坐标 anchor 盖章；有 c.T 时只变换锚点，章本身不变形。"""
    x, y = anchor
    if c.T is not None:
        x, y = c.T(x, y)
    x0, y0 = int(round(x)) - ax, int(round(y)) - ay
    T, c.T = c.T, None
    c.stamp(rows, x0, y0)
    c.T = T
    return x0, y0


def draw_head(c, chin, sway=0):
    """chin = 下巴像素（世界坐标）。呆毛在描边之后画（细线不加外框）；sway = 呆毛尖向左(-1)/右(1)甩。"""
    x0, y0 = stamp_at(c, HEAD_B, chin, *HEAD_CHIN)
    x, y = x0 + HEAD_CHIN[0], y0 + HEAD_CHIN[1]
    tip = int(round(sway))
    c.post += [(x + dx + (tip if dy <= -13 or (dx >= 2) else 0), y + dy, k) for dx, dy, k in AHOGE]


MAT = {  # 亮 / 中 / 暗 / 分隔线
    'jk': ('J', 'j', 'i', 'I'),
    'pt': ('j', 'j', 'i', 'I'),     # 前腿
    'pb': ('i', 'i', 'I', 'I'),     # 后腿（背光）
}


def part(c, mat, m, lit=1, dark=1, sep=True):
    """一块材质（m 为掩膜）：中间色 + 左上亮边 + 右下暗边 + 与已画部分相接处的分隔线。"""
    hi, mid, lo, line = MAT[mat]
    prev = c.opaque.copy()
    c.fill(m, mid)
    if lit:
        c.fill(edge(m, -1, -1, lit) & ~edge(m, 1, 1, 1), hi)
    if dark:
        c.fill((edge(m, 1, 1, dark) | edge(m, 0, 1, dark)) & ~edge(m, -1, 0, 1), lo)
    if sep:
        c.fill(border(m, prev), line)
    return m


def limb_mask(c, pts, widths):
    """关节折线 + 每段宽度 -> 掩膜（每段是一条粗线，关节处补圆）。"""
    m = np.zeros((c.h, c.w), bool)
    for (p0, p1), w in zip(zip(pts[:-1], pts[1:]), widths):
        m |= c.m_line([p0, p1], int(round(w)))
    return m


HANDS = {   # 小手章：(ASCII, 锚点 = 章内对准手腕的位置, 指尖 = 特效定位点)
    'palm_up':    (['.sss..', 'ssssss', '.kkkk.'], (0, 1), (5, 1)),        # 手心向上托在胸前（指尖朝画面右）
    'open_down':  (['ss.', 'sss', 'sks', 'skk', '.sk', '..k'], (1, 0), (2, 5)),   # 自然张开下垂
    'snap':       (['.s..', 'ss.s', 'sss.', 'skk.', '.k..'], (1, 4), (3, 1)),  # 捻指：食指竖起，拇指与中指捏合
    'snap_done':  (['s.s.', 'ssss', 'sss.', '.kk.'], (1, 3), (1, 0)),      # 打完响指：手指弹开
    'fist':       (['ss.', 'ssk', 'kk.'], (1, 0), (1, 1)),
    'open_out':   (['..ss', 'ssss', 'ssk.', '.k..'], (3, 2), (0, 1)),      # 手掌向外甩开（指尖朝画面左）
    'open_right': (['ss..', 'ssss', '.kss', '..k.'], (0, 1), (3, 2)),      # 手掌向外甩开（指尖朝画面右）
    'open_up':    (['s.s.', 'ssss', '.kk.'], (1, 2), (1, 0)),              # 手心朝上举起（抛牌）
    'palm_flat':  (['ssss', '.kkk'], (3, 0), (0, 0)),                      # 撑在台沿上
    'card':       (['WWW', 'WrW', 'WWW', 'ss.', 'ssk'], (1, 4), (1, 1)),    # 两指夹着一张牌
    'pocket':     (['k'], (0, 0), (0, 0)),                                  # 插兜：只露一点
}


def _hand_anchor(kind, wrist, cuff_dir):
    """手章锚点对准的世界坐标（手腕再往手的方向走 1 px）。"""
    return wrist[0] + cuff_dir[0], wrist[1] + cuff_dir[1]


def hand(c, kind, wrist, cuff_dir):
    rows, (ax, ay), _ = HANDS[kind]
    return stamp_at(c, rows, _hand_anchor(kind, wrist, cuff_dir), ax, ay)


def arm(c, shoulder, elbow, wrist, kind, cuff_dir, widths=(3, 3), sep=True):
    """袖子（外套色）+ 浅蓝袖口 + 手。cuff_dir = 袖口朝向手的单位向量，用来放袖口。"""
    m = limb_mask(c, [shoulder, elbow, wrist], widths)
    part(c, 'jk', m, sep=sep)
    ux, uy = cuff_dir
    cx, cy = wrist[0] - ux * 0.5, wrist[1] - uy * 0.5
    c.line('b', [(cx - uy, cy + ux), (cx + uy, cy - ux)])
    c.px('B', [(cx - uy, cy + ux)])
    return hand(c, kind, wrist, cuff_dir)


def _seg_quad(p0, p1, w0, w1):
    """线段 p0->p1 两端各宽 w0 / w1 的四边形。"""
    ux, uy = _unit(p0, p1)
    nx, ny = -uy, ux
    return [(p0[0] + nx * w0 / 2, p0[1] + ny * w0 / 2), (p1[0] + nx * w1 / 2, p1[1] + ny * w1 / 2),
            (p1[0] - nx * w1 / 2, p1[1] - ny * w1 / 2), (p0[0] - nx * w0 / 2, p0[1] - ny * w0 / 2)]


SHOES = {
    'r': (['.zzz.', 'Szzzz', 'zzzzz'], 3, 0),     # 脚尖朝画面左下（前脚）
    'l': (['zzz.', 'zzSz', 'zzzz'], 1, 0),        # 脚尖朝画面右（后脚）
    'back': (['zz', 'zS', '.z'], 1, 0),           # 脚跟抬起 / 跪地时的后脚
    'dangle': (['zz.', 'Szz', '.zz'], 1, 0),      # 坐着时垂下的脚
}


def leg(c, hip, knee, ankle, front=True, flare=1.0, shoe='r'):
    """裤腿：大腿 + 小腿两段（膝盖可弯），裤脚略喇叭；鞋子盖在脚踝下。"""
    m = c.m_poly(_seg_quad(hip, knee, 4, 3)) | c.m_poly(_seg_quad(knee, ankle, 3, 3 + 1.5 * flare))
    part(c, 'pt' if front else 'pb', m, lit=1 if front else 0, dark=1, sep=not front)
    rows, ax, ay = SHOES[shoe]
    c.stamp(rows, int(round(ankle[0])) - ax, int(round(ankle[1])) - ay)


TORSO = [  # x 14..30, y 16..43（站姿世界坐标）
    '.......BkkkB.....',
    '...JTllBGGYBllTi.',
    '..JjTLGGbGnblTjji',
    '..JjTGYbbGnllTjji',
    '..JjTLllGbblTjjji',
    '..JjTLllYnllTjji.',
    '..JjjTTTTTTTjjji.',
    '...Jjjijjjjjjii..',
    '...JjGYjjjGYjii..',
    '...Jjjijjjjjii...',
    '...Jjjijjjjjii...',
    '...Jjjijjjjjii...',
    '....Jjijjjjjii...',
    '....Jjijjjjjii...',
    '....JGYjjjGYii...',
    '....Jjijjjjjii...',
    '...Jjjijjjjjii...',
    '...Jjjijjjjjjii..',
    '..Jjjjijjjjjjii..',
    '.Jjjjjijjjjjjii..',
    '.JjjjibJjjjjjjib.',
    '.BJjjibJjjjjji.b.',
    '.bJji.bBJjjji.b..',
    '..BJib.b.Jji.b...',
    '..bb.b..B.J.b....',
    '...Bb....Bbb.....',
    '..........b......',
    '.................',
]
TORSO_XY = (14, 16)


def torso(c):
    """外套（双排扣、下摆两片尖角）+ 浅蓝衬衫下摆 + 黑方翻领（白边）+ 衬衫领 + 黄缎带领带。
    按行盖章；上半身前倾（c.T 为错切）时每行各自平移，像素簇不被打碎。"""
    x0, y0 = TORSO_XY
    T, c.T = c.T, None
    for dy, r in enumerate(TORSO):
        tx, ty = T(x0, y0 + dy) if T is not None else (x0, y0 + dy)
        c.stamp([r], int(round(tx)), int(round(ty)))
    c.T = T


def lean_T(px_, py_, deg):
    """上半身前倾：以 (px_, py_) 为支点做水平错切（正值 = 向画面左倾），并略微压低高度。
    错切只平移整行像素，比旋转更不容易把像素簇打碎。"""
    t = math.tan(math.radians(deg))
    cs = math.cos(math.radians(deg))
    return lambda x, y: (x - (py_ - y) * t, py_ - (py_ - y) * cs)


def _unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy) or 1
    return dx / n, dy / n


# 姿势 = 关节坐标（世界坐标，站姿基准；lean 时上半身整体以 pivot 为支点错切前倾）
SHOULDER_R, SHOULDER_L, CHIN = (18.5, 18.5), (29.5, 18.5), (23, 15)
PIVOT = (23, 40)     # 前倾的支点（胯）

# ---------------------------------------------------------------- 姿势
# 一个关键姿势 = dict：
#   leg_r / leg_l = (胯, 膝, 踝, 鞋)       胯是站姿坐标（自动加上 off），膝 / 踝是世界坐标
#   arm_r / arm_l = (肘, 腕, 手型, (上臂宽, 前臂宽))   站姿上半身坐标（随 off / lean 一起动）
#   off = (dx, dy)   上半身（连胯）整体平移；lean = 前倾角度（负值 = 后仰）；back = 画在躯干后面的手臂
#   ahoge = 呆毛摆动（-1 / 0 / 1）
# 角色右 = 画面左（他面朝画面左 3/4）。
_STAND_R = ((21, 39), (22, 55), (21.5, 67.5), 'r')
_STAND_L = ((25, 39), (25.5, 54.5), (25.5, 66.5), 'l')
_POCKET_R = ((15.5, 27), (18.5, 33), 'pocket', (3, 3))
_DOWN_L = ((33, 27.5), (36.5, 34.5), 'open_down', (3, 3))
KEY = {
    # --- 官方 Sen III 立绘：右手屈肘托在胸前，左手向右下伸开，双腿交叉
    'idle': dict(leg_r=_STAND_R, leg_l=_STAND_L,
                 arm_r=((11.5, 25.5), (18.5, 26.5), 'palm_up', (3, 3)), arm_l=_DOWN_L),
    'breath': dict(leg_r=_STAND_R, leg_l=_STAND_L, off=(0, 1), ahoge=1,
                   arm_r=((11.5, 25), (18.5, 26), 'palm_up', (3, 3)), arm_l=((33, 27), (36.5, 34), 'open_down', (3, 3))),
    # --- 响指：抬手（预备）-> 手在脸旁捏住 -> 打响、手指弹开（随势）
    'snap_0': dict(leg_r=_STAND_R, leg_l=((25, 39), (26, 54.5), (26.5, 65.5), 'l'), lean=-2,
                   arm_r=_POCKET_R, arm_l=((34, 26), (31, 21), 'fist', (3, 3))),
    'snap': dict(leg_r=((21, 39), (21.5, 55), (21, 67.5), 'r'), leg_l=((25, 39), (26.5, 54.5), (27, 65.5), 'l'),
                 arm_r=_POCKET_R, arm_l=((35, 22), (33, 14), 'snap', (3, 3))),
    'snap_2': dict(leg_r=((21, 39), (21.5, 55), (21, 67.5), 'r'), leg_l=((25, 39), (26.5, 54.5), (27, 65.5), 'l'),
                   off=(0, -1), lean=-3, ahoge=-1,
                   arm_r=_POCKET_R, arm_l=((37, 19), (38, 10), 'snap_done', (3, 3))),
    # --- 鞠躬：展臂（预备）-> 前倾 -> 深鞠躬（右手抚胸，左手向外展开）-> 回身
    'bow': dict(lean=18, leg_r=((21, 39), (20, 55), (19, 67.5), 'r'), leg_l=((25, 39), (28, 54), (31, 65.5), 'l'),
                arm_r=((14, 25), (19.5, 25.5), 'palm_up', (3, 3)), arm_l=((37, 26), (43, 31), 'open_right', (3, 3))),
    # --- 从火焰中现身：蹲伏 -> 起身展臂 -> 张开双臂亮相
    'appear_0': dict(off=(1, 11), lean=24, ahoge=1,
                     leg_r=((21, 39), (14, 58), (19, 67.5), 'r'), leg_l=((25, 39), (27, 63), (32, 66.5), 'back'),
                     arm_r=((15, 28), (21, 32), 'fist', (3, 3)), arm_l=((32, 27), (26, 32), 'fist', (3, 3))),
    'appear_1': dict(off=(0, 4), lean=8,
                     leg_r=((21, 39), (18, 56), (20, 67.5), 'r'), leg_l=((25, 39), (27, 57), (28, 66.5), 'l'),
                     arm_r=((13, 24), (9, 29), 'open_out', (3, 3)), arm_l=((34, 24), (39, 28), 'open_right', (3, 3))),
    'appear_2': dict(off=(0, -1), lean=-5, ahoge=-1,
                     leg_r=((21, 39), (21.5, 55), (21, 67.5), 'r'), leg_l=((25, 39), (26.5, 54.5), (27.5, 66.5), 'l'),
                     arm_r=((11, 16), (5, 11), 'open_out', (3, 3)), arm_l=((36, 16), (42, 11), 'open_right', (3, 3))),
    # --- 弹牌：牌夹在左肩前（预备）-> 甩出 -> 随势
    'flick_0': dict(lean=-5, ahoge=1, leg_r=_STAND_R, leg_l=((25, 39), (27, 54.5), (28, 66.5), 'l'),
                    arm_r=((22, 25), (27, 19), 'card', (3, 3)), arm_l=_DOWN_L),
    'flick_1': dict(lean=7, ahoge=-1, leg_r=((21, 39), (19, 55), (18, 67.5), 'r'), leg_l=_STAND_L,
                    arm_r=((12, 22), (6, 27), 'open_out', (3, 3)), arm_l=((33, 26), (37, 31), 'open_right', (3, 3))),
    'flick_2': dict(lean=4, leg_r=((21, 39), (20, 55), (19.5, 67.5), 'r'), leg_l=_STAND_L,
                    arm_r=((13, 25), (8, 31), 'open_out', (3, 3)), arm_l=_DOWN_L),
    # --- 坐在高处台沿上，双腿悬空晃，玩一张牌（台沿在 SEAT_Y 行）
    'sit_0': dict(off=(0, 12), back='r',
                  leg_r=((21, 39), (12, 55), (11, 65), 'dangle'), leg_l=((25, 39), (17, 57), (18, 66), 'dangle'),
                  arm_r=((15, 29), (15, 39), 'palm_flat', (3, 3)), arm_l=((32, 26), (28, 19), 'card', (3, 3))),
    'sit_1': dict(off=(0, 12), back='r', ahoge=1,
                  leg_r=((21, 39), (12, 55), (14, 65), 'dangle'), leg_l=((25, 39), (17, 57), (15, 66), 'dangle'),
                  arm_r=((15, 29), (15, 39), 'palm_flat', (3, 3)), arm_l=((32, 25), (29, 18), 'card', (3, 3))),
    'sit_toss': dict(off=(0, 12), back='r', lean=-3,
                     leg_r=((21, 39), (12, 55), (12, 65), 'dangle'), leg_l=((25, 39), (17, 57), (17, 66), 'dangle'),
                     arm_r=((15, 29), (15, 39), 'palm_flat', (3, 3)), arm_l=((33, 22), (31, 14), 'open_up', (3, 3))),
    'sit_flick': dict(off=(0, 12), back='r', lean=6, ahoge=-1,
                      leg_r=((21, 39), (12, 55), (9, 64), 'dangle'), leg_l=((25, 39), (17, 57), (19, 66), 'dangle'),
                      arm_r=((15, 29), (15, 39), 'palm_flat', (3, 3)), arm_l=((24, 22), (13, 20), 'open_out', (3, 3))),
    # --- 施法（旧 PV 的总攻段）：右手向画面左上甩出，左手收在身后握拳，弓步
    'cast': dict(back='l', leg_r=((20, 39), (15, 53), (12, 66.5), 'r'), leg_l=((25, 39), (29, 54), (33, 65.5), 'l'),
                 arm_r=((12, 14), (6, 9), 'open_out', (3, 3)), arm_l=((33, 25), (31, 31), 'fist', (3, 3))),
}
SEAT_Y = 51          # sit_* 姿势里屁股 / 台沿所在的行


def _lerp(a, b, t):
    if isinstance(a, str) or isinstance(b, str):
        return a if t < 0.5 else b
    if isinstance(a, tuple):
        return tuple(_lerp(x, y, t) for x, y in zip(a, b))
    return a + (b - a) * t


def tween(a, b, t):
    """两个关键姿势之间的中间帧（关节线性插值，手型 / 鞋型取较近的一端）。"""
    A, B = KEY[a] if isinstance(a, str) else a, KEY[b] if isinstance(b, str) else b
    out = {}
    for k in set(A) | set(B):
        va = A.get(k, B.get(k))
        vb = B.get(k, A.get(k))
        if k in ('off',):
            va, vb = A.get(k, (0, 0)), B.get(k, (0, 0))
        if k in ('lean', 'ahoge'):
            va, vb = A.get(k, 0), B.get(k, 0)
        out[k] = _lerp(va, vb, t) if k not in ('back',) else (va if t < 0.5 else vb)
    return out


POSES = dict(KEY)
# 稳定的别名（其他场景直接用这些名字）
POSES['sit'] = KEY['sit_0']            # 坐在台沿上，双腿悬空，一手夹着牌
POSES['card_flick'] = KEY['flick_1']   # 把牌甩出去的一瞬（预备帧 flick_0，随势 flick_2）
# 中间帧（名字 = 动作_序号，按播放顺序排）
for _name, _a, _b, _ts in (
        ('appear', 'appear_0', 'appear_1', (0.5,)),            # appear_0 -> appear_0h -> appear_1 -> appear_2
        ('rise', 'appear_1', 'appear_2', (0.5,)),
        ('bowin', 'appear_2', 'bow', (0.35, 0.7)),              # 展臂 -> 深鞠躬
        ('bowout', 'bow', 'idle', (0.35, 0.7)),                 # 深鞠躬 -> 站定
        ('raise', 'idle', 'snap_0', (0.5,)),                    # 站定 -> 抬手
        ('lift', 'snap_0', 'snap', (0.5,)),
        ('flick', 'flick_0', 'flick_1', (0.5,)),
        ('sway', 'sit_0', 'sit_1', (0.5,)),
):
    for _i, _t in enumerate(_ts):
        POSES[f'{_name}_t{_i}'] = tween(_a, _b, _t)

# 常用动作序列（场景按节拍取帧）：[(姿势名, 该帧占的相对时长)]
SEQ = {
    'appear': ['appear_0', 'appear_t0', 'appear_1', 'rise_t0', 'appear_2'],
    'bow': ['appear_2', 'bowin_t0', 'bowin_t1', 'bow'],
    'unbow': ['bow', 'bowout_t0', 'bowout_t1', 'idle'],
    'snap': ['idle', 'raise_t0', 'snap_0', 'lift_t0', 'snap'],
    'flick': ['flick_0', 'flick_t0', 'flick_1', 'flick_2'],
    'sit': ['sit_0', 'sway_t0', 'sit_1', 'sway_t0'],
    'vanish': ['vanish_0', 'vanish_1', 'vanish_2', 'vanish_3', 'vanish_4', 'vanish_5'],
}


def seq_frame(name, p):
    """动作序列在进度 p（0..1）处的帧名。"""
    s = SEQ[name]
    return s[min(len(s) - 1, max(0, int(p * len(s))))]


def _xf(P):
    """上半身坐标变换：先绕胯前倾，再整体平移 off。"""
    ox, oy = P.get('off', (0, 0))
    lean = P.get('lean', 0)
    L = lean_T(*PIVOT, lean) if lean else (lambda x, y: (x, y))
    return lambda x, y: (L(x, y)[0] + ox, L(x, y)[1] + oy)


def _build(pose):
    P = POSES[pose]
    c = Cv(48, 72)
    ox, oy = P.get('off', (0, 0))
    # --- 腿（后腿先画；胯跟着上半身平移）
    for key, front in (('leg_l', False), ('leg_r', True)):
        hip, knee, ankle, shoe = P[key]
        leg(c, (hip[0] + ox, hip[1] + oy), knee, ankle, front=front, shoe=shoe)
    legs = c.opaque.copy()
    # --- 上半身
    c.T = _xf(P)
    hands = {}

    def arm_(which):
        elbow, wrist, kind, w = P['arm_' + which]
        sh = SHOULDER_R if which == 'r' else SHOULDER_L
        x0, y0 = arm(c, sh, elbow, wrist, kind, _unit(elbow, wrist), w)
        tx, ty = HANDS[kind][2]
        hands[which] = (y0 + ty, x0 + tx)

    back = P.get('back')
    if back:
        arm_(back)
    before = c.a.copy()
    torso(c)
    # 外套下摆在裤子上投一道阴影（紧贴下摆下方的一行裤腿像素压暗）
    tm = c.a != before
    c.a[legs & ~tm & shift(tm, 1, 0) & np.isin(c.a, [KID['j'], KID['i']])] = KID['I']
    for which in ('l', 'r'):
        if which != back:
            arm_(which)
    draw_head(c, CHIN, P.get('ahoge', 0))
    c.T = None
    c.hands = hands
    return c


_VANISH_N = 6


def _vanish(k):
    """幻焰消失：以打完响指的姿势为底，从脚往上逐行溶解，溶解边缘烧成橙 / 蓝火。"""
    c = _build('snap_2')
    img = c.rgba()
    a = img[..., 3] > 0
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rng = np.random.default_rng(5)
    noise = rng.random((h, w)) * 0.35
    level = (k + 1) / _VANISH_N * 1.45
    front = (h - 1 - yy) / h + noise            # 越靠下越先烧掉
    gone = front < level - 0.25
    burn = (front < level) & ~gone & a
    out = img.copy()
    out[gone, 3] = 0
    fire = pk.hexc('#ff8a20'), pk.hexc('#ffd060'), pk.hexc('#4ab0ff')
    pick = rng.integers(0, 10, (h, w))
    out[burn, :3] = np.where((pick[burn] < 6)[:, None], fire[0], np.where((pick[burn] < 9)[:, None], fire[1], fire[2]))
    return out


def body(pose='idle'):
    if pose.startswith('vanish_'):
        return _vanish(int(pose.split('_')[1]))
    if pose not in POSES:
        pose = 'idle'
    return _build(pose).rgba()


def hand_points(pose):
    """{'r': (y, x), 'l': (y, x)}：两只手指尖在 body(pose) 里的像素位置（特效从这里发出）。"""
    if pose.startswith('vanish_'):
        pose = 'snap_2'
    return dict(_build(pose if pose in POSES else 'idle').hands)


def _eye_pixels(pose):
    """全身像里可见那只眼（红虹膜）的 (y, x) 像素。"""
    c = _build(pose)
    return [tuple(int(v) for v in p) for p in np.argwhere(c.a[:40] == KID['r'])]


# 立绘尺度的手（特写镜头里和 portrait() 同像素密度）：抬起的左手，捏住准备打响指 / 打完弹开
PORTRAIT_HANDS = {
    'snap': [
        '....xx........',
        '...xssx.......',
        '...xssx.......',
        '...xssx.xx....',
        '...xssxxssx...',
        '.xxxssxssskx..',
        'xsssssssssskx.',
        'xsssKKsssskkx.',
        '.xssssKKsskx..',
        '.xsssssssskx..',
        '..xsssssskx...',
        '..xksssskkx...',
        '...xbBbbbx....',
        '...xBBbbnx....',
        '..OjJjjjjiO...',
        '..OJjjjjjiO...',
        '..OJjjjjiiO...',
        '..OjjjjjiiO...',
    ],
    'snap_done': [
        'xx....xx......',
        'xsx..xssx.xx..',
        'xssx.xssxxssx.',
        '.xssxxssxssx..',
        '..xssxssxssx..',
        '.xxsssssssskx.',
        'xsssssssssskx.',
        'xsssssssssskx.',
        '.xssssssssskx.',
        '.xsssssssskx..',
        '..xsssssskx...',
        '..xksssskkx...',
        '...xbBbbbx....',
        '...xBBbbnx....',
        '..OjJjjjjiO...',
        '..OJjjjjjiO...',
        '..OJjjjjiiO...',
        '..OjjjjjiiO...',
    ],
}


def portrait_hand(kind='snap'):
    """特写用的手（RGBA），与 portrait() 同像素密度。"""
    return pk.from_ascii(PORTRAIT_HANDS[kind], PAL)


META = dict(
    key='campanella',
    name_zh='肯帕雷拉',            # 官方中文多译作「坎佩尼拉」；按需求统一写作肯帕雷拉
    title_zh='道化师',
    rank_zh='执行者 No.0',
    skill_zh='幻焰',                # 他来去时的「幻焰」，也呼应结社的「幻焰计划」
    quote_zh='那么——好戏开场了。',
    fx=dict(
        primary='#ff8a20',
        secondary='#4ab0ff',
        description=('响指一打，指尖炸开一小团橙色火花（带 2~3 颗蓝色星点）；'
                     '随后从抬起的手甩出一道橙黄火焰长鞭，沿弧线横扫，鞭身外缘橙红、芯部亮黄，'
                     '尾迹散落蓝色小火星。登场/退场用「幻焰」：一圈橙色火焰从脚下卷起包住全身，'
                     '火焰散去人已消失（或出现），边缘同样夹带蓝色火星。'),
    ),
    eyes_body={p: _eye_pixels(p) for p in POSES},
    hand_snap=hand_points('snap')['l'],      # body('snap') 中捻指的指尖 (y, x)
    hand_cast=hand_points('cast')['r'],      # body('cast') 中甩出去的那只手 (y, x)
    hands={p: hand_points(p) for p in POSES},   # 每个姿势两只手的指尖 (y, x)
    seat_y=SEAT_Y,
    seq=SEQ,
)

# 关键点：从自己的渲染图上读（python tools/measure.py sprite campanella portrait / idle）
LANDMARKS = {
    'portrait': {
        'head_top': [40, 12], 'chin': [36, 55],
        'eye_far_out': [26, 35], 'eye_far_in': [32, 36], 'eye_far_top': [29, 35], 'eye_far_bot': [29, 39],
        'nose': [35, 46], 'mouth': [37, 49],
        'cheek_far': [26, 40], 'jaw_far': [29, 49], 'jaw_near': [47, 49],
        'neck_far': [34, 56], 'neck_near': [46, 55],
        'x_mark': [29, 42],
        'x_hair_top': [39, 8], 'x_hair_far': [18, 30], 'x_hair_near_tip': [68, 36],
        'x_lock_far_tip': [29, 63], 'x_lock_near_tip': [46, 63],
        'x_collar_tip_far': [33, 67], 'x_collar_tip_near': [44, 67],
        'x_trim_far': [24, 64], 'x_trim_near': [63, 63],
    },
    'idle': {
        'head_top': [22, 6], 'chin': [23, 15], 'neck': [23, 17],
        'shoulder_r': [18, 18], 'shoulder_l': [30, 18],
        'elbow_r': [12, 25], 'elbow_l': [33, 28], 'wrist_r': [18, 26], 'wrist_l': [36, 34],
        'waist': [22, 31], 'crotch': [23, 42],
        'knee_r': [22, 55], 'knee_l': [25, 54], 'ankle_r': [21, 68], 'ankle_l': [25, 67],
        'x_lapel_r': [18, 21], 'x_lapel_bot': [19, 22], 'x_lapel_l': [26, 22],
        'x_hem_r_side': [16, 36], 'x_hem_r_tip': [17, 39], 'x_hem_l_tip': [24, 40], 'x_hem_l_side': [28, 35],
        'x_shirt_r_tip': [18, 42], 'x_shirt_l_tip': [24, 42],
        'x_fingers_r': [25, 27], 'x_fingers_l': [38, 40],
    },
}
