"""布卢布兰（怪盗B / 怪盗绅士），执行者 No.X。

参考：头像按 SC Evo 半身像（refs/bleublanc/bust.png），全身按《闪之轨迹 II》立绘（refs/bleublanc/full.png）。
官方图先在 data/landmarks/bleublanc.json 标关键点，再按相似变换缩到像素分辨率当描图底稿，
形状（多边形顶点 / ASCII 图）都是照底稿读出来的；本文件末尾的 LANDMARKS 是从成图上逐点读回的坐标，
tools/measure.py compare 用它们和官方图对比。

识别要点：银白面具（镜孔里一只青色的眼）+ 面具两侧白色羽翼、蓝紫色长卷发（头顶一根呆毛）、
白色礼服 + 浅蓝 / 薄荷绿镶边、洋红领巾、石板蓝长披风（浅色内衬）、金杆红宝珠的导力杖（杖头两侧水晶羽翼）、
紫玫瑰（花瓣）。身高按 72px ≈ 180cm。

画法：
- 大块面用多边形，按左上光源自动分 3–4 档色阶（Sprite.part_fill）；
- 小而要紧的部位（头、披肩、外套）用手写 ASCII 图（Sprite.stamp_part），逐像素定；
- 前层部件压在后层上时，只在两边亮度接近的地方描 1px 深色分隔线（Sprite.lines(thr)），外轮廓统一描深色；
- 最后清掉孤立的单像素（Sprite.despeckle，眼睛 / 嘴这类故意的单点细节列在保护区里）。
全部在原生分辨率下完成，无抗锯齿。

全身像的部件（给下一轮动画用）：cape(pose) / legs / torso / capelet / tassel / arm_r(deg, hand) / arm_l(kind) /
staff(top, bot) / hair(pose) / head(expr) / petals，各自带位置参数，_body_idle / _body_skill 只是按层序把它们拼起来。
"""
import numpy as np

import pixelkit as pk

# ---------------------------------------------------------------- palette
OUT = '#1a0f1f'
HAIR = ['#a3a8e2', '#7f86c6', '#5c629c', '#3c3f70', '#25274a']   # 蓝紫长卷发，光 → 暗
SKIN = ['#ffe5cc', '#efb496', '#c07866']
WHITE = ['#ffffff', '#dadbe6', '#a9aabf', '#6f7088']             # 白礼服 / 银面具 / 手套 / 羽翼
TEAL = ['#b6f2e6', '#6cc8bc', '#357f86']                         # 薄荷绿荷叶边、袖口
MAG = ['#e45aa8', '#b02c7c', '#6c1850']                          # 洋红领巾
SLATE = ['#8ea6c0', '#5d6f8c', '#3f4c68', '#262e44']             # 石板蓝披风（亮 = 内衬）
GOLD = ['#ffd66b', '#c08a2e']
RED = ['#ff5a4a', '#c0243a', '#6e1428']
LENS = ['#c4faff', '#4ccbd8', '#1d6878']                         # 面具镜孔里的青色眼
ROSE = ['#c48cf0', '#8e4fd0', '#4e2388']                         # 紫玫瑰
TRIM = ['#bfe2ef', '#86b3c9', '#557c96']                         # 肩上浅蓝饰带
EDGE = '#5e5460'                                                 # 立领前缘的棕灰镶边
BOOT = ['#9a9cb4', '#6c6e88', '#474a60', '#2c2e40']              # 长靴
VEST = ['#aab0d8', '#7c82b0', '#565a86']                         # 马甲（灰紫）
NAVY = ['#4a5a94', '#2c3664']                                    # 腰带 / 饰带深蓝
CRYSTAL = ['#d8f4ff', '#9ccde0']                                 # 杖头水晶羽翼
RIBBON = ['#f07a98', '#c04468']                                  # 发带（偏粉的红）


# ---------------------------------------------------------------- raster helpers
def _poly(pts, h, w):
    """偶奇规则多边形填充（按像素中心取样，无抗锯齿）。pts: [(x, y), ...]"""
    yy, xx = np.mgrid[0:h, 0:w]
    px, py = xx + 0.5, yy + 0.5
    inside = np.zeros((h, w), bool)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if y1 == y2:
            continue
        cond = (py >= min(y1, y2)) & (py < max(y1, y2))
        xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (px < xint)
    return inside


def _ell(cx, cy, rx, ry, h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1


def _shift(m, dy, dx):
    """把掩码平移 (dy, dx)，越界补 False。"""
    out = np.zeros_like(m)
    h, w = m.shape
    ys = slice(max(dy, 0), h + min(dy, 0))
    yd = slice(max(-dy, 0), h + min(-dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    xd = slice(max(-dx, 0), w + min(-dx, 0))
    out[ys, xs] = m[yd, xd]
    return out


class Sprite:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.col = np.full((h, w), -1, np.int32)      # 颜色下标
        self.part = np.full((h, w), -1, np.int32)     # 部件编号（用于内描线）
        self.colors = []
        self.parts = []                               # 每个部件的描线颜色（或 None）

    def ci(self, hexs):
        if hexs not in self.colors:
            self.colors.append(hexs)
        return self.colors.index(hexs)

    # 主力：画一个部件并自动分色阶
    def part_fill(self, mask, ramp, hl=1, sh=2, deep=1, line=True, grad=None, lit=None):
        """ramp: 由亮到暗；hl/sh: 左上高光带、右下阴影带宽度；deep: 最深档宽度（需 ramp≥4 或复用最后一档）。
        grad=(y0, y1): y 超过 y1 的像素整体压暗一档（竖向体积感）。lit: 额外高光掩码。"""
        mask = mask & np.ones((self.h, self.w), bool)
        lvl = np.full((self.h, self.w), 1, np.int32)
        if hl:
            edge = np.zeros_like(mask)
            for k in range(1, hl + 1):
                edge |= ~_shift(mask, k, k)
                if k == 1:
                    edge |= ~_shift(mask, 1, 0)
            lvl[mask & edge] = 0
        if sh:
            edge = np.zeros_like(mask)
            for k in range(1, sh + 1):
                edge |= ~_shift(mask, -k, -k)
            lvl[mask & edge] = 2
        if grad is not None:
            y0, y1 = grad
            yy = np.mgrid[0:self.h, 0:self.w][0]
            lvl[mask & (yy >= y1) & (lvl < 2)] += 1
        if deep and len(ramp) > 3:
            edge = np.zeros_like(mask)
            for k in range(1, deep + 1):
                edge |= ~_shift(mask, -k, -k) & ~_shift(mask, -k, 0)
            lvl[mask & edge] = 3
        if lit is not None:
            lvl[mask & lit] = 0
        lvl = np.minimum(lvl, len(ramp) - 1)
        pid = len(self.parts)
        self.parts.append(ramp[-1] if line is True else line)
        for i, c in enumerate(ramp):
            sel = mask & (lvl == i)
            self.col[sel] = self.ci(c)
        self.part[mask] = pid
        return pid

    def flat(self, mask, color, line=None):
        pid = len(self.parts)
        self.parts.append(line)
        self.col[mask] = self.ci(color)
        self.part[mask] = pid
        return pid

    def poly(self, pts):
        return _poly(pts, self.h, self.w)

    def ell(self, cx, cy, rx, ry):
        return _ell(cx, cy, rx, ry, self.h, self.w)

    def lines(self, thr=None):
        """前层部件与后层部件交界处，前层一侧描 1px 深色线。
        thr: 只在两侧亮度差 < thr 时描线（白压白要线，白压深色披风就不用）——选择性描边。"""
        p = self.part
        newcol = self.col.copy()
        lum = np.array([float(pk.hexc(c) @ np.array([0.3, 0.55, 0.15], np.float32)) for c in self.colors] + [0.0])
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = _shift(p + 1, dy, dx) - 1   # 邻居部件号（越界为 -1）
            front = (p >= 0) & (q >= 0) & (p > q)
            if thr is not None:
                qc = _shift(self.col + 1, dy, dx) - 1
                front &= np.abs(lum[self.col] - lum[qc]) < thr
            for pid in np.unique(p[front]):
                lc = self.parts[pid] if pid < len(self.parts) else None
                if lc is None:
                    continue
                newcol[front & (p == pid)] = self.ci(lc)
        self.col = newcol

    def stamp(self, rows, x0, y0, cmap):
        for j, r in enumerate(rows):
            for i, c in enumerate(r):
                if c in '. ':
                    continue
                y, x = y0 + j, x0 + i
                if 0 <= y < self.h and 0 <= x < self.w:
                    if c == '_':               # 擦除
                        self.col[y, x] = -1
                        self.part[y, x] = -1
                    else:
                        self.col[y, x] = self.ci(cmap[c])
                        if self.part[y, x] < 0:
                            self.part[y, x] = 10_000

    def stamp_part(self, rows, x0, y0, cmap=None, line=None):
        """手绘 ASCII 部件：盖上去并登记成一个新部件（line=None：边缘不自动描线，轮廓已经手画好）。"""
        cmap = cmap or CM
        pid = len(self.parts)
        self.parts.append(line)
        for j, r in enumerate(rows):
            for i, c in enumerate(r):
                y, x = y0 + j, x0 + i
                if c in '. ' or not (0 <= y < self.h and 0 <= x < self.w):
                    continue
                if c == '_':
                    self.col[y, x] = -1
                    self.part[y, x] = -1
                else:
                    self.col[y, x] = self.ci(cmap[c])
                    self.part[y, x] = pid
        return pid

    def px(self, pts, color):
        for (x, y) in pts:
            if 0 <= y < self.h and 0 <= x < self.w:
                self.col[y, x] = self.ci(color)

    def line(self, pts, color, part=False):
        """1px 折线（Bresenham）。pts: [(x, y), ...]"""
        ci = self.ci(color)
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            dx, dy = abs(x1 - x0), -abs(y1 - y0)
            sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
            err = dx + dy
            while True:
                if 0 <= y0 < self.h and 0 <= x0 < self.w and (part or self.col[y0, x0] >= 0):
                    self.col[y0, x0] = ci
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

    def despeckle(self, protect=()):
        """清孤点：一个像素 8 邻域里没有同色像素、且 4 邻域里有 ≥2 个同色邻居时，改成那个邻居色。
        protect: [(x0, y0, x1, y1), ...] 半开区间，眼睛 / 嘴 / 宝石这类故意的单点细节不动。"""
        c = self.col
        h, w = c.shape
        keep = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in protect:
            keep[max(0, y0):y1, max(0, x0):x1] = True
        new = c.copy()
        for y in range(h):
            for x in range(w):
                v = c[y, x]
                if v < 0 or keep[y, x]:
                    continue
                nb8 = [c[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                       if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w]
                if v in nb8:
                    continue
                nb4 = [c[y + dy, x + dx] for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                       if 0 <= y + dy < h and 0 <= x + dx < w and c[y + dy, x + dx] >= 0]
                if nb4:
                    vals, cnt = np.unique(nb4, return_counts=True)
                    if cnt.max() >= 2:
                        new[y, x] = vals[cnt.argmax()]
        self.col = new

    def render(self, outline=OUT, remap=None):
        img = pk.blank(self.w, self.h)
        for i, c in enumerate(self.colors):
            pk.paint(img, self.col == i, (remap or {}).get(c, c))
        if outline:
            a = img[..., 3] > 0
            g = a.copy()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                g |= _shift(a, dy, dx)
            pk.paint(img, g & ~a, outline)
        return img


# 通用字符 → 颜色（ASCII 图章用）
CM = {
    'o': OUT,
    'H': HAIR[0], 'h': HAIR[1], 'j': HAIR[2], 'J': HAIR[3],
    'S': SKIN[0], 's': SKIN[1], 'k': SKIN[2],
    'W': WHITE[0], 'w': WHITE[1], 'v': WHITE[2], 'V': WHITE[3],
    'T': TEAL[0], 't': TEAL[1], 'Q': TEAL[2],
    'M': MAG[0], 'm': MAG[1], 'n': MAG[2],
    'B': SLATE[0], 'b': SLATE[1], 'd': SLATE[2], 'D': SLATE[3],
    'G': GOLD[0], 'g': GOLD[1],
    'R': RED[0], 'r': RED[1], 'x': RED[2],
    'L': LENS[0], 'l': LENS[1], 'q': LENS[2],
    'P': ROSE[0], 'p': ROSE[1], 'u': ROSE[2],
    'K': HAIR[4], 'Y': NAVY[0], 'y': NAVY[1], 'E': VEST[0], 'e': VEST[1], 'f': VEST[2],
    'C': CRYSTAL[0], 'c': CRYSTAL[1], '1': BOOT[0], '2': BOOT[1], '3': BOOT[2], '4': BOOT[3],
    'A': TRIM[0], 'a': TRIM[1], '@': TRIM[2], '&': EDGE,
}


# ================================================================ portrait (80×80)
# 按 SC Evo 半身像（refs/bleublanc/bust.png）逐像素描：3/4 侧脸朝画面左，面具盖住眉、鼻和远侧脸颊，
# 近侧耳上是银色羽翼。取景固定为 tools/measure.py fidelity 的三锚点相似变换（两眼中心 + 下巴），
# 头发、领子、肩线都照变换后的官方图落位（细节参考 crop_sen2_scraft.png）。
# 画法：
# - 头发用 H/h/j/J/K 五档；顶上是顺着梳向的发缕（亮缕 H、中间 h、缕间 j），分缝里是 J/K 的暗影；
#   外缘不描黑线，最外一圈只用中间调（sel-out），深色只留在内部；
# - 深色描线（o）只画在官方图有墨线的地方：面具轮廓、镜孔、黑色高领、前襟线、银扣、立领卷纹；
# - 面具 = 远侧面（v/V）+ 鼻梁高光棱（W）+ 正面（w，上缘高光、下半压暗）；
# - 衣服：远侧大翻领（灰）、白披肩 + 浅蓝饰带、中间灰色内衬、近侧立领（深色前缘 + 白色受光楔）、
#   近侧披肩（宽白高光带 + 浅蓝饰带 + 卷草纹），红色发带只用 1px 细线。
# 字符 -> 颜色见 PORTRAIT_CM（通用 CM + 发带 X/Y）。嘴按表情另画，底图里那块是空的皮肤。
PORTRAIT_CM = dict(CM, X=RIBBON[0], Y=RIBBON[1])
PORTRAIT_BASE = [
    '....................................Hhhj....hh..................................',
    '...................................Hhj.......j...............................W..',
    '..................................Hhj.........j............................WWv..',
    '....................v.............Hj......................................Wwwv..',
    '.....................w...........Hhj.....................................Wwwwv..',
    '.....................W...........Hj.....................................Wwwwwv..',
    '.....................W..........Hhj...................................WWwwVwwv..',
    '.....................Wv.........Hj...................................WwwVVvwv...',
    '.....................Wv.........Hj..............................W...WwWVvvwwv...',
    '.....................WWv........HhH.............................W..WWVVvwwwWv...',
    '.....................WWv........Hhj....HHHHHHHHHHHH............WwWWWVvvwwVWwv...',
    '.....................WWv.......hHhH.HHHhhhhHjjhhhHjHHh.........WwwWVvwwVVvwwv...',
    '....................vWWv..H....hHhjhHHHjjhhhHHjhhhHjhhhjj.....WwwVVvwwVvvwwv....',
    '....................wWWv.Hhh...hHhjjJjhHHjjhhhHjjhhHjhhjjjj..WwwVvvwwVvwwwwv....',
    '....................wWWvHhjhhhhhjJJJJhhhhHHjhhhHHjhhHjhjjjjjWwwVvwwVVvwwwwwv....',
    '....................wWWvHhjhhhhjJJKJJjhhhhhHjjhhhHjhhhjjjjJWwwwvwWVvvwwwwwwv....',
    '....................vWwHhhjhhhhjJKKKJjhhhhhhHHjhhhjjhhhjjjWwwwwwWVvwwwwwwwwv....',
    '....................vWwHHhjhhhjJKKKKJJhhhhjhhhHjHhhjjhhjjjWwwwwVVvwwwwwwwwv.....',
    '.....................WHhHhjhhhjJKKKKKJjhhhhjhhhhHhhjHjhjjWwwwWVvvwwwwWwwv.......',
    '.....................wHhHhjhhhJKKKKKKJjjhhhHjjhhHhhhjhjjjWwwWVvwwwwWWwVV........',
    '.....................wHhHhjhhjJKKKKKKKJjhhhhHhhhHhhhjhhjWwwWVvwwwWWwVVvvWW......',
    '.....................vHhHhjhhjJJKKKKKKJJjhhhhhhhHhhhjhhjWwwVvwwwWwVVvvwwwwWWW...',
    '......................HhHhjhhJJJJKKKJJJJJjjhhhhhHhhhjjhjWwVvwwWWVVvvwwwwwwv.....',
    '.....................HhhHhjhJKssssssssssssssJhhhHhhhjhjWoVVwwWwVvvwwwwwWW.......',
    '.....................HhhhhjjJSSSSSSSSSSSSSSSjoohHhhhjhoVwwVwWVVvwwwwwWWwwVVV....',
    '......................HhhhjjJSSSSSSSSSSoooooohohHhhjJhowWwVwVvvwwwwWWVVVVv......',
    '......................HhhjjjJSSSSooooooWWWWWWWohHhhjjhowWwVVvwwwwWWVVv..........',
    '......................HhhjjjoooooWWWWWWwwwwwwwohHhhjjhowWwVvwwwWVVVvv...........',
    '......................HhhjjjovvvvWwwwwwwwwWWwwohHhhjjhowWwVwwwVVvvvwwWWW........',
    '......................HhhjjjoVvvvWwVVwwwwWwwwwohHhhjjjowWwVwwVvvwwwVV...........',
    '.......................HhjjjoVvvvWwwwVwwwoooooohHhhjjjowWVwwwvwv................',
    '.......................HhjjjoVvvvWwwwwwoooLLLLohHhhjjjowVVwv....................',
    '.......................HhjhjoqvvvWwwwwwoLLllllloHhhjjjJoJJJJj..j................',
    '........................HjhjoqvvvWwwwwwollllqqwoHhhjjjJksskjJjjo................',
    '........................HjhjovqvvWwwwwwwqqqqwwwoHhhjjjJkSskJJJj.................',
    '........................HjhjovqvvWwwwwwWwwwwwwwoHhjJjjJkSskJJjj.................',
    '.......................HhjhjovvvvWwVwwWwwwwwwwwoHhjJjjjksskjj...................',
    '.......................HhjhjovvvvWwwwWvvvvvvvoooHhjJjjjkskj.jj..................',
    '......................HhjhhjovvvvWVvWvvvvvvoosssHhjJjjjjkk.jj...................',
    '.....................HhhjhjjJovvWWvvvvvvvoosssssHhjjJjjJJJj.h...................',
    '....................HhhhjjjJJovvWvvvvvvossssSSssHhjjJjjJJJj.h...................',
    '....................HhhhjjjJJjovWvvvvosssSSSSSSsshjjJjjJJj..j.........YXX.......',
    '...................hhhJjjjJJjJoWvvvosssSSSSSSSSSshjjJjjJj...j......YYY.Y........',
    '....................hhjJjJJjJJJoWoksSSSSSSSSSSSSsjjjJjjj..jjj....YY...Y.........',
    '.....................jjJJJJjJJJookSSSSSSSSSSSSSSsjjjJjJJjjJj..YYY.....Y.........',
    '......................jjJJJjJJJJJkSSSSSSSSSSSSSSsjjjJJJJJJVVYYVV......Y.........',
    '........................hjJJjJJJJkSSSSSSSSSSSSSssjjjJJJJJVYY&&&&&&&&&&Y.........',
    '.........................hjJJjJJJkSSSSSSSSSSSSSssjJ&&&&&&&&&&&&&&&&&&&&.........',
    '........................VwHjjVwwwwkSSSSSSSSSSSskjj&&&&&&&&&&&&&&&&&&&&&.........',
    '.....................VwwvvwHjVvvvvkSSSSSSSSSSsskjjvvooovvvvvvvvvvvvvvvV.........',
    '...................VwvvvvvvHjVvvvvwkSSSSSSSSsskjjjvovvovvvvvvvvvvvvvvvV.........',
    '...................VvvvwvvvHJJVvvvvkkSSSSSSsskhjjjWovovvvvvvvvvvvvvvvvV.........',
    '...................VvvvwvvHhjJVvvvvskkSSSSsskhhjjjWvovvvvvvvvvvvvvvvvVjh........',
    '...................VvvvwvvHjJVvvvvvsSkkkkkkkshjjjWWvvvvvvvvvvvvvvvvvvVjjj.......',
    '....................VvvvwvHjVvvvvvvsskksssssssssjWWvvvvvvvvvvvvvvvvvvVjjjjh.....',
    '....................VvvvwvhjVvvvvwooooooooooo&oojWWvvvvvvvvvvvvvvvvvvVjjjjjj....',
    '....................VvvvwvjjVvvvvwooooooooooo&&jWWWvvvvvvvvvvvvvvvvvVJJjjjjjj...',
    '.....................Vvvwvwwvvvvvwwwwoowwwwww&&WWWvvvvvvvvvvvvvvvvvvVJJjjjjjjjh.',
    '.....................Vvvvwvvvvvvwwwwwoowwwwww&&WWWvvvvvvvvvvvvvvvvvvVJj..jjjjjjh',
    '.....................Vvvvwvvvvvvwwwwwoowwwwwv&&WWWvvvvvvvvvvvvvvvVVV.vvvvvvjjjjj',
    '......................Vvvvwvvvvvwwwwwoowwwwwv&&WWWvvvvvvvvvvvvVVV.vvvwwwwwwvvjjj',
    '......................Vvvvwvvvvvwwwwwoowwwwvw&&WWWvvvvvvvvvVVV.vvvwwwwwwwwwwwvvj',
    '.......................Vvvvvvvv.wwwwoowwwwvww&&WWvvvvvvvvVV..vvwwwwwwwwWWWWWWWWv',
    '.......................Vvvvvvvv.vwwwoowwwwvw&&&WWvvvvvVVV.vvvwwwwWWWWWWWWWWWWWWW',
    '........................Vvvvvv...VVVVVVVwvww&&&WWvvvvV.vvvwwWWWWWWWWWwwwwwWWWWWW',
    '...................WWWWWWVvvv...VWMMMMmnVwww&&&WWvvvV.vwwWWWWWwwwwwwwWWWWWWWWWWW',
    '................WWWwwwwwwWWWW...VMWMMmmnVwww&&&WWvvV.vWWWWwwwwWWWWWWWWWWWWWWWWWW',
    '..............WWwwwwwwwwwwwwwW...VMMmmnVwwww&&&WvvV.vWWwwwWWWWWWWWWWWWWWWWWWWWWW',
    '............WWwwwwwwwwwwwwvwwwW..wVMmnVwwwww&&vWvV.vWWWWWWWWWWWWWWWWWWWWWWWWWWWW',
    '...........Wwwwwwwwwwwwwwwwvwww..wwVnVwwwwww&&vWV.vWWWWWWWWWWWWWWWWWWWWWWWWWWWWW',
    '..........Wwwwwwwwwwwwwwwwwwvww..wwoVwwwwwwwoooW.vWWWWWWWWWWWWWWWWWWWWWWWWWWWwww',
    '...........WwwwvvvwwwwooowwwvwwWwwwoVwwwwwwoWWwovWWWWWWWWWWWWWWWWWWWWWwwwwwwwwww',
    '.........WWwvvvwwvwwwoWWwowwwvwwwwwoVwwwwwoWWwwvoWWWWWWWWWWWwwwwwwwwwwwwwwwwwwww',
    '........WwvvwwwwwvwwoWWwwvowwvwwwwwoVwwwwwoWwwvVowwWWWWwVVVVVwwwwwwwwwwwwwwwwwww',
    '........WvwwwwwwwvwwoWwwvVowwwvwwwwoVwwwwwovvvVVoVVVVVVVAAAAAVVVVVVVVVwwwwwwwwww',
    '.......WwwwwwwwwVvwwovvvVVowwwvwwwoVwwwwwwwoVVVowAAAAAAAaaaaaAAAAAAAAAVVVVVVVwww',
    '.......VVVVVVVVVavwwvoVVVowwwwvwwwoVwwwwwwwwooowwaaaaaaaaaaaaaaaaaaaaaAAAAAAAVVV',
    '........aa@@aaaaavwwvwooowwwwwwvwwoVwwwwwwww&&vwwaaaa@@aaaa@@aaaa@@aaaaaaaaaaAAA',
    '........a@aa@aaaavwwvwwwwwwwwwwvwwoVwwwwwwww&&wwVaaa@aa@aa@aa@aa@aa@aaa@@aaaa@@a',
    '.......VaaaaaaaaavwwvwwwwwwwwwwwvvoVvvvvvvvv&VwwAaaaaa@aaaaa@aaaaa@aaa@aa@aa@aa@',
]
PORTRAIT_MOUTH = {   # (x0, y0, 图章)；'.' = 不动
    None: (34, 44, ['........k',           # 浅浅的微笑：近侧嘴角微微上挑
                    '.kkkkkkk.',
                    '.........',
                    '...sss...']),
    'smirk': (34, 43, ['.........k',        # 坏笑：近侧嘴角明显上挑，远侧压平
                       '........k.',
                       '.kkkkkk...',
                       '..........',
                       '...ss.....']),
    'laugh': (34, 44, ['k.......k',           # 张嘴大笑：上排牙、口腔、下唇影
                       'oWWWWWWWo',
                       '.onmmmno.',
                       '..okkko..',
                       '...sss...']),
}


def portrait(expr=None):
    """expr: None（微笑）/ 'smirk'（坏笑）/ 'laugh'（大笑）；别的名字按 None 处理。"""
    s = Sprite(80, 80)
    s.stamp(PORTRAIT_BASE, 0, 0, PORTRAIT_CM)
    x0, y0, rows = PORTRAIT_MOUTH.get(expr, PORTRAIT_MOUTH[None])
    s.stamp(rows, x0, y0, PORTRAIT_CM)
    return s.render(outline=None)


# 从上面这张像素图上逐点读出（像素中心坐标），和 data/landmarks/bleublanc.json 的 bust 同名。
# 两眼 + 下巴是取景锚点，改了会让整张官方图重新对位，别动。
LM_PORTRAIT = {
    'head_top': (40, 10), 'chin': (36.5, 53),
    'eye_near_in': (38.5, 33), 'eye_near_out': (45.5, 31), 'eye_near_top': (42, 31.5), 'eye_near_bot': (42, 33.5),
    'eye_far_out': (29, 31.5), 'eye_far_in': (30.2, 35.3),
    'mouth': (38, 45), 'jaw_far': (33, 45),
    'neck_far': (36.5, 54), 'neck_near': (45, 54),
    'x_mask_tip': (31.5, 44), 'x_mask_far_top': (28, 27), 'x_mask_ridge_top': (32.5, 27.5),
    'x_mask_near_top': (45.5, 24.5), 'x_mask_near_bot': (47, 37),
    'x_fringe': (36, 22.5), 'x_ear': (56.5, 35.5),
    'x_wing_near_tip': (77, 1), 'x_wing_far_tip': (20.5, 3.5),
    'x_collar_near_tip': (70.5, 46.5), 'x_collar_far_tip': (18.5, 50.5),
    'x_gem': (36, 66.5), 'x_button_far': (23, 74), 'x_button_near': (45, 73),
    'x_ahoge_tip': (46, 2), 'x_curl_far': (19, 42),
}

# ================================================================ body (56×72)
# 按《闪之轨迹 II》全身立绘描：右手（画面左）握杖斜伸，左手扶面具，双腿交叉（左腿藏在右腿后，只露出靴尖），
# 石板蓝披风向两侧张开。比例尺 72px ≈ 180cm：头顶 y≈3、脚底 y=71（翅膀尖占掉最上面 3 行）。
#
# 结构（下一轮做动画用）：每个部件是一个函数，画在同一张 Sprite 上，位置由 Rig 里的几个参数决定：
#   cape   —— 披风（按姿势换多边形）
#   legs   —— 裤子 + 两只靴子（dx, dy 平移）
#   torso  —— 衣摆、外套、马甲、腰带、领巾、流苏（dx, dy）
#   arm_r / arm_l —— 手臂：肩点 + 角度（多边形绕肩旋转），每只手有几种手型
#   staff  —— 握点 + 方向 + 长度（杖头图章自动沿杖身摆放）
#   head   —— ASCII 图章（表情换嘴），hair —— 披在身后的长发
# 多边形坐标 = 像素角坐标（像素 i 覆盖 [i, i+1]），由官方立绘按 0.0738 倍对位后读出。
BW, BH = 56, 72
BODY_REMAP = {}
SLATE_FOLD = '#71849f'      # 披风褶的亮边（介于内衬亮色和外层之间）


def _rot(pts, pivot, deg):
    """把多边形绕 pivot 旋转 deg 度（画面坐标，顺时针为正）。"""
    if not deg:
        return pts
    a = np.radians(deg)
    c, s_ = np.cos(a), np.sin(a)
    px, py = pivot
    return [(px + (x - px) * c - (y - py) * s_, py + (x - px) * s_ + (y - py) * c) for x, y in pts]


def _mv(pts, dx=0, dy=0):
    return [(x + dx, y + dy) for x, y in pts]


# ---------------------------------------------------------------- 披风
CAPE = {
    'idle': dict(
        out=[(27.8, 19.4), (23.5, 20.3), (17.7, 20.7), (10.3, 21.6), (4.4, 23.3), (0.0, 26.0), (3.3, 28.1),
             (7.0, 30.7), (9.2, 32.4), (7.0, 33.3), (4.4, 34.0), (3.0, 34.4), (1.5, 36.6), (0.4, 38.8), (0.1, 40.0),
             (2.6, 42.8), (6.3, 45.1), (10.8, 45.4), (13.1, 46.3), (14.2, 46.7), (13.9, 51.9), (16.8, 53.7),
             (19.8, 54.5), (23.1, 54.7), (24.3, 54.4), (24.5, 52.0), (25.5, 50.3), (27.0, 48.8), (28.9, 48.2),
             (31.1, 48.1), (33.0, 48.3), (39.9, 49.5), (42.1, 49.9), (45.8, 50.2), (49.4, 49.9), (52.9, 48.8),
             (53.0, 44.0), (52.4, 41.4), (50.6, 39.5), (48.3, 38.1), (49.4, 36.6), (53.1, 35.1), (55.7, 33.3),
             (56.1, 31.4), (55.0, 27.4), (52.4, 22.9), (49.4, 18.1), (47.6, 14.4), (44.3, 14.1), (39.6, 15.7),
             (33.0, 18.3)],
        lining=[[(0.0, 26.0), (4.4, 23.3), (10.3, 21.6), (17.7, 20.7), (21.5, 23.1), (17.5, 26.8), (13.1, 29.7),
                 (9.2, 32.4), (7.0, 30.7), (3.3, 28.1)],
                [(47.6, 14.4), (49.4, 18.1), (52.4, 22.9), (55.0, 27.4), (56.1, 31.4), (55.7, 33.3), (53.1, 35.1),
                 (50.3, 33.4), (49.2, 28.2), (48.5, 20.9)],
                [(3.0, 34.4), (7.0, 33.3), (9.2, 32.4), (10.8, 35.6), (9.4, 40.0), (6.3, 43.0), (2.6, 42.8),
                 (0.1, 40.0), (0.4, 38.8), (1.5, 36.6)]],
        folds=[[(20, 24), (18, 34), (19, 46)], [(24, 30), (24, 44), (22, 53)], [(47, 24), (46, 36), (47, 48)],
               [(44, 30), (43, 40)]],
    ),
    'skill': dict(     # 右臂甩开把披风带起：左半边向上翻出三道尖角，露出大片浅色内衬
        out=[(28, 18.5), (21, 14.5), (13, 12), (6, 11.5), (0, 12.5), (4, 16), (6, 19.5), (2, 22.5), (0, 27),
             (4, 30), (7, 33), (3, 37.5), (0.5, 42.5), (5, 44.5), (10, 46.5), (14, 50.5), (18, 55), (23, 56),
             (25.5, 52.5), (29, 49.5), (34, 49), (40, 50.5), (45.8, 51.5), (51, 50), (54.5, 46), (53.5, 40),
             (55.5, 33), (54.5, 25.5), (51.5, 19.5), (48, 15), (44.3, 14.1), (39.6, 15.7), (33, 18.3)],
        lining=[[(28, 18.5), (21, 14.5), (13, 12), (6, 11.5), (0, 12.5), (4, 16), (6, 19.5), (8, 17.5),
                 (14, 16), (21, 17.5)],
                [(0, 27), (4, 30), (7, 33), (3, 37.5), (0.5, 42.5), (3, 35), (2, 30)],
                [(51.5, 19.5), (54.5, 25.5), (55.5, 33), (53.5, 40), (51, 36), (51.5, 28)]],
        folds=[[(9, 24), (10, 34), (12, 44)], [(17, 24), (18, 38), (19, 52)], [(46, 26), (45, 38), (47, 48)]],
    ),
}


def cape(s, pose='idle', dx=0, dy=0):
    c = CAPE[pose]
    out = s.poly(_mv(c['out'], dx, dy))
    s.part_fill(out, SLATE[1:], hl=1, sh=1, deep=1)
    for pts in c['lining']:
        s.part_fill(out & s.poly(_mv(pts, dx, dy)), [SLATE[0], SLATE[0], SLATE[1]], hl=0, sh=1, deep=0, line=SLATE[2])
    for pts in c['folds']:                  # 褶：一道暗线 + 右侧一道亮线
        s.line(_mv(pts, dx + 1, dy), SLATE_FOLD)
        s.line(_mv(pts, dx, dy), SLATE[3])
    # 下摆一圈深色厚边（官方图的披风边是深藏青）
    edge = out & ~_shift(out, -1, 0)
    s.col[edge] = s.ci(SLATE[3])


# ---------------------------------------------------------------- 腿（右腿在前，左腿交叉藏在后面只露出靴子）
def legs(s, dx=0, dy=0):
    P = lambda pts: s.poly(_mv(pts, dx, dy))
    s.part_fill(P([(33.6, 60.2), (33.7, 65.1), (33.4, 68.1), (33.0, 69.2), (31.6, 69.0), (30.2, 65.1), (29.9, 62.9),
                   (30.8, 61.1)]), BOOT, hl=1, sh=1, deep=0)                                  # 后面那只靴
    s.part_fill(P([(32.6, 37.2), (42.7, 37.8), (41.5, 44.1), (39.2, 50.4), (39.2, 52.0), (32.9, 50.6), (33.4, 44.1)]),
                WHITE, hl=1, sh=2, deep=1, line=WHITE[3])                                         # 白裤
    s.part_fill(P([(32.6, 50.2), (39.0, 50.4), (38.3, 52.5), (37.5, 54.8), (37.1, 60.0), (37.3, 67.4), (37.2, 71),
                   (37.0, 72.0), (35.0, 72.0), (33.7, 69.6), (33.9, 65.1), (33.6, 57.8), (32.8, 53.7)]),
                BOOT, hl=1, sh=1, deep=1)                                                          # 前面那只长靴
    for y in (55, 58, 61):
        s.px([(35 + dx, y + dy)], GOLD[0])                                                         # 靴扣
    s.line(_mv([(33, 51), (38, 51)], dx, dy), TEAL[1])                                              # 靴口纹饰
    s.line(_mv([(33, 52), (37, 52)], dx, dy), TEAL[2])


# ---------------------------------------------------------------- 躯干：手绘 ASCII（外套前身 + 马甲 + 斜腰带 + 锯齿前摆 + 右侧后摆）
TORSO_BACK = [   # 原点 (26, 19)：外套前身（白）| 黑色门襟线 | 灰紫马甲（两粒扣）；斜腰带 + 金扣；白/薄荷绿锯齿前摆；右侧后摆
    '....WWWWWwwvo.........',
    '....WWWWWwwvo.........',
    '....WWWWWwwvo.........',
    '....WWWWWwwvoEeef.....',
    '....WWWWWwwvoEeef.....',
    '....WWWWWwwvoEeef.....',
    '....WWWWWwwvoEWef.....',
    '....WWWWWwwvoEeef.....',
    '.....yyWWwwvoEWef.....',
    '....WyYyyWwvoEeef.....',
    '....WWWyYyyvoEeef.....',
    '....WWWWWyYyyyyyyW....',
    '...WyWWyWWywyGgGgWv...',
    '...ytyytyytyy....Wv...',
    '..tTttTttTtQ.....Wwv..',
    '..tTttTttTtQ.....WwV..',
    '.TTttTTttTTt......Wv..',
    '..QQ.QQ.QQ.......TTtt.',
    '.................QQ...',
]


def torso(s, dx=0, dy=0):
    s.stamp_part(TORSO_BACK, 26 + dx, 19 + dy)


# ---------------------------------------------------------------- 披肩 + 立领 + 洋红领巾（盖在手臂根部上面）
CAPELET = [   # 原点 (20, 10)：立领（黑边）、白披肩、深蓝饰带 + 青宝石、下缘深蓝镶边 + 薄荷绿荷叶边、洋红领巾
    '...........wWoW.........',
    '...........wWoW.........',
    '.........WWWwownMMn.....',
    '......WWWWWWwwvmMMmn....',
    '....WWWWyyyyWwvLMmmn....',
    '..WWWWWWWWWwyyylMmmnn...',
    '.yyyWWWWWWwwwvvmMMmnn...',
    'TTttyyyyyyywwvvmMmmnn...',
    'QQ.TTttTTttTyyynMmmmn...',
    '..QQ.QQ.QQ.TTttnMMmmn...',
    '...........QQ.QnMmMmmn..',
    '...............nMMmMmmn.',
    '...............nmMmMmnn.',
    '................n.n.n...',
]


def capelet(s, dx=0, dy=0):
    s.stamp_part(CAPELET, 20 + dx, 10 + dy)


def tassel(s, dx=0, dy=0):
    """腰间金穗：金绳 + 红珠 + 向左飘的一大把流苏（两行金色）。"""
    s.stamp_part(['....g',
                  '...g.',
                  '..gG.',
                  '.RRr.',
                  'Rrrx.',
                  '.xx..'], 37 + dx, 34 + dy)
    s.stamp_part(['.....GGggg',
                  '...GGgg...',
                  '.GGgg.....',
                  'Ggg.......',
                  'g.........'], 28 + dx, 39 + dy)


# ---------------------------------------------------------------- 手臂
# 右臂（画面左）：一套袖子 / 袖口 / 手套多边形（idle 的形状），整条手臂绕肩点 SHOULDER_R 旋转 deg 度得到别的姿势；
# 手可以是握杖的拳（多边形）或张开的手掌（图章，贴在旋转后的手腕上）。
SHOULDER_R = (30.8, 16.8)
ARM_R = dict(
    sleeve=[(27.8, 20.1), (23.5, 20.5), (21.1, 22.5), (18.9, 24.5), (17.6, 25.9), (19.6, 26.8), (20.4, 29.9),
            (22.6, 27.4), (25.5, 23.8), (29.3, 23.1), (30.8, 21.6)],
    cuff=[(17.0, 26.0), (18.2, 25.9), (19.6, 26.7), (19.9, 30.1), (19.4, 33.6), (17.9, 33.8), (16.9, 31.1),
          (16.2, 27.9)],
    glove=[(15.6, 28.6), (16.8, 29.3), (17.4, 31.2), (16.8, 33.4), (15.1, 33.8), (13.8, 32.7), (13.4, 30.5),
           (14.2, 29.3)],
    wrist=(16.8, 30.3),
)
HAND_R = {   # 张开的手掌（甩出扑克牌），原点 = 手腕左上角的偏移
    'open': ((-5, -3), ['W.W...',
                        '.WWW.W',
                        'WWWwW.',
                        '.WwwwT',
                        '..wvvt']),
}


def arm_r(s, deg=0, hand='grip', part='arm', dx=0, dy=0):
    """part='arm' 画袖子 + 袖口；part='hand' 只画手（握杖时手套要盖在杖上，所以分开画）。"""
    R = lambda pts: _mv(_rot(pts, SHOULDER_R, deg), dx, dy)
    if part == 'arm':
        s.part_fill(s.poly(R(ARM_R['sleeve'])), WHITE, hl=1, sh=1, deep=1, line=WHITE[3])
        s.part_fill(s.poly(R(ARM_R['cuff'])), TEAL, hl=1, sh=1, deep=0, line=TEAL[2])
    elif hand == 'grip':
        s.part_fill(s.poly(R(ARM_R['glove'])), WHITE, hl=1, sh=1, deep=0, line=WHITE[3])
    else:
        (wx, wy), = R([ARM_R['wrist']])
        (ox, oy), rows = HAND_R[hand]
        s.stamp_part(rows, int(wx) + ox, int(wy) + oy)
    (wx, wy), = R([ARM_R['wrist']])
    return wx, wy


# 左臂（画面右）：每个姿势一套上臂 / 袖口多边形 + 手型图章
ARM_L = {
    'mask': dict(      # 扶面具
        upper=[(42.5, 13.5), (46, 13.8), (48, 15.2), (48.8, 17.8), (47.2, 18.6), (44, 17.8), (42.5, 17)],
        fore=[(39.3, 13.6), (41.5, 12.6), (43.9, 13.8), (44.2, 16.5), (42.5, 17.6), (39.8, 17.2), (39, 15.5)],
        hand=((38, 8), ['.W..',       # 食指顶住面具右下角
                        '.Ww.',
                        'WWw.',
                        'wWWv',
                        '.wWv',
                        '..wv'])),
    'cane': dict(      # 垂手拄杖
        upper=[(42.3, 13.8), (45, 14.2), (47.4, 18.5), (48.2, 21.5), (46, 22), (44, 19), (42.2, 17)],
        fore=[(45.4, 20.6), (48.6, 20.4), (49.2, 23.6), (46.2, 24)],
        hand=((46, 23), ['WWwv',
                         'wWwv',
                         '.vv.'])),
}


def arm_l(s, kind='mask', part='all', dx=0, dy=0):
    a = ARM_L[kind]
    if part in ('all', 'upper'):
        s.part_fill(s.poly(_mv(a['upper'], dx, dy)), WHITE, hl=1, sh=1, deep=1, line=WHITE[3])
    if part in ('all', 'fore'):
        s.part_fill(s.poly(_mv(a['fore'], dx, dy)), TEAL, hl=1, sh=1, deep=0, line=TEAL[2])
        (hx, hy), rows = a['hand']
        s.stamp_part(rows, hx + dx, hy + dy)


# ---------------------------------------------------------------- 导力杖（金杆、蓝色杖身、红宝珠、两侧水晶羽翼）
def staff(s, top=(9.6, 16.1), bot=(27.1, 60.7), wings=True):
    (x0, y0), (x1, y1) = top, bot
    n = int(max(abs(x1 - x0), abs(y1 - y0)))
    ux, uy = (x1 - x0) / n, (y1 - y0) / n
    at = lambda t: (int(np.floor(x0 + ux * t)), int(np.floor(y0 + uy * t)))
    s.line([at(9), at(n - 1)], GOLD[1], part=True)                  # 金棕杖杆
    s.px([at(n)], GOLD[0])                                           # 杖尖
    for t in range(2, 10):                                           # 杖身：蓝色金属，2px 宽
        x, y = at(t)
        s.px([(x, y)], NAVY[0])
        s.px([(x + 1, y)], NAVY[1])
    for t in (2, 9):                                                 # 金色饰环
        x, y = at(t)
        s.px([(x - 1, y), (x, y)], GOLD[0])
        s.px([(x + 1, y)], GOLD[1])
    if wings:                                                        # 杖头两侧的水晶羽翼（各三根）
        x, y = at(4)
        for (ex, ey), c in (((-8, -3), CRYSTAL[0]), ((-7, 0), CRYSTAL[1]), ((-6, 3), CRYSTAL[1]),
                            ((5, -7), CRYSTAL[0]), ((7, -4), CRYSTAL[1]), ((6, -1), CRYSTAL[1])):
            sx = -1 if ex < 0 else 2
            s.line([(x + sx, y), (x + ex, y + ey)], c, part=True)
            s.px([(x + ex, y + ey)], CRYSTAL[0])
    x, y = at(1)
    s.px([(x - 1, y), (x, y), (x + 1, y), (x + 2, y)], GOLD[0])        # 金冠
    x, y = at(0)
    s.stamp(['.RR.', 'RRrr', 'rrrx', '.xx.'], x - 1, y - 2, CM)       # 红宝珠
    return at


# ---------------------------------------------------------------- 头（ASCII）+ 身后长发
HAIR_BACK = {   # 披在身后的长卷发：左边两大绺甩到披肩上，右边一绺垂到左臂外侧
    'idle': [[(34.8, 4.0), (32, 5.5), (28.5, 6.8), (25, 7.8), (22.5, 9.2), (21.8, 11), (23.2, 10.6), (25.5, 9.8),
              (28, 9.6), (31, 9), (33.5, 8)],
             [(33.5, 7.5), (30.5, 9), (27, 10.6), (24, 12), (21.8, 13.8), (20.2, 16.2), (19.5, 18.5), (20, 20.8),
              (21.4, 21.2), (21, 19), (21.8, 16.8), (23.6, 14.8), (26.5, 13.3), (30, 11.8), (33.8, 10.2)],
             [(39.6, 5.4), (41.5, 6.5), (43, 8.5), (44.5, 10.5), (46.5, 12), (47.6, 13.4), (46, 13.6), (44.2, 12.4),
              (42.5, 10.8), (41, 9)]],
}
HAIR_BACK['skill'] = [_mv(HAIR_BACK['idle'][0], -1, -1), _mv(HAIR_BACK['idle'][1], -1, -2), HAIR_BACK['idle'][2]]


def hair(s, pose='idle', dx=0, dy=0):
    for pts in HAIR_BACK[pose]:
        s.part_fill(s.poly(_mv(pts, dx, dy)), HAIR[1:], hl=1, sh=1, deep=0)


HEAD = [   # 原点 (28, 0)
    '..W......jj.....',
    '.WwW.......j....',
    'WwWw.......j...W',
    'vWwWw......h..Ww',
    '.vwWWvjhHHhj.Wwv',
    '...vwVhHhWWojwv.',
    '....VjhhWWwohj..',
    '...jhhSwLWvoh...',
    '..jhjSSswwo.j...',
    '.jhj.SSSsoo.....',
    'hj...sSWWk......',
    '......ksSk......',
]

HEAD_MOUTH = {None: [], 'laugh': [(10, 7, 'WWo'), (11, 7, 'oo')]}   # (行, 列, 像素)


def head(s, x0=28, y0=0, expr=None):
    s.stamp_part(HEAD, x0, y0)
    for (r, c, t) in HEAD_MOUTH.get(expr, []):
        s.stamp([t], x0 + c, y0 + r, CM)


# ---------------------------------------------------------------- 组装
def _body_idle():
    s = Sprite(BW, BH)
    cape(s, 'idle')
    staff(s)
    legs(s)
    arm_l(s, 'mask', part='upper')
    torso(s)
    arm_r(s)
    capelet(s)
    hair(s)
    head(s)
    arm_l(s, 'mask', part='fore')
    arm_r(s, part='hand')
    tassel(s)
    s.lines(thr=0.3)
    petals(s, PETALS_IDLE)
    return s


def _body_skill():
    """死亡魔术：右臂向左甩开、张手撒牌；左手垂下拄杖；披风被带起向左鼓开；大笑。"""
    s = Sprite(BW, BH)
    cape(s, 'skill')
    legs(s)
    torso(s)
    arm_l(s, 'cane', part='upper')
    staff(s, top=(47.8, 10.5), bot=(49.6, 59.5))
    hair(s, 'skill')
    arm_r(s, deg=SKILL_ARM_DEG)
    capelet(s)
    head(s, expr='laugh')
    arm_l(s, 'cane', part='fore')
    arm_r(s, deg=SKILL_ARM_DEG, hand='open', part='hand')
    tassel(s)
    s.lines(thr=0.3)
    petals(s, PETALS_SKILL)
    # 手里飞出的几张白色扑克牌
    for (x, y) in ((3, 9), (6, 6), (2, 14)):
        s.stamp_part(['WW', 'Wr', 'WW'], x, y)
    return s


SKILL_ARM_DEG = 52

# 紫玫瑰花瓣（官方立绘里散落在披风前的那几片），每片 2×2
PETALS_IDLE = [(16, 21), (14, 42), (50, 29), (48, 23), (21, 40)]
PETALS_SKILL = [(12, 22), (6, 30), (24, 44), (50, 32), (15, 38)]


def petals(s, spots):
    for x, y in spots:
        s.stamp_part(['pP', 'up'], x, y)


# ================================================================ 第二轮：美术馆一幕要用的姿势
# 通用拼装 _rig(spec)：spec 里写披风、腿、上半身偏移、左右臂、杖、表情，按固定层序画。
# 姿势名可以带帧号："jewel:2" = jewel 姿势的第 2 帧披风飘动（0..3），"idle:1" 同理。
LEGS = {
    'crouch': dict(   # 落地半蹲：右腿（前）屈膝踏地，左膝跪地、脚尖向后
        back=[[(37, 45), (43, 45.5), (44, 56), (44.5, 64.5), (40.5, 66), (39.5, 57), (37, 50)]],
        back_boot=[[(40, 63.5), (44.5, 63), (49, 67.5), (52, 70.5), (52, 72), (42, 72), (40, 69)]],
        front=[[(32, 45), (39, 45), (36.5, 50), (31, 54.5), (26.5, 53.5), (26.5, 50.5), (31, 47.5)]],
        front_boot=[[(26.5, 52.5), (31, 53), (31, 60), (31.5, 68), (32.5, 71.5), (24.5, 72), (25, 69.5), (27, 67),
                     (27, 59)]],
        cuff=[(27, 54), (31, 54)], buttons=[(28, 58), (28, 61), (28, 64)]),
    'lunge': dict(    # 冲刺弓步：右腿（前）向左跨出，左腿向右后蹬直
        back=[[(33, 42), (40, 43), (44, 52), (47, 60), (43.5, 62), (38.5, 53), (33, 48)]],
        back_boot=[[(43.5, 59.5), (47.5, 58.5), (51, 65.5), (55.5, 70), (55.5, 72), (49, 72), (45.5, 67.5)]],
        front=[[(29, 42), (36, 43), (30.5, 52), (25.5, 57.5), (21, 55.5), (23, 50)]],
        front_boot=[[(21, 55), (26, 57), (24.5, 64), (24.5, 70), (25.5, 72), (15, 72), (16, 69.5), (20, 68),
                     (20.5, 62)]],
        cuff=[(21, 56), (25, 57)], buttons=[(22, 61), (22, 64), (22, 67)]),
}


def legs_pose(s, kind):
    L = LEGS[kind]
    for pts in L['back']:
        s.part_fill(s.poly(pts), WHITE, hl=1, sh=2, deep=1, line=WHITE[3])
    for pts in L['back_boot']:
        s.part_fill(s.poly(pts), BOOT, hl=1, sh=1, deep=0)
    for pts in L['front']:
        s.part_fill(s.poly(pts), WHITE, hl=1, sh=2, deep=1, line=WHITE[3])
    for pts in L['front_boot']:
        s.part_fill(s.poly(pts), BOOT, hl=1, sh=1, deep=1)
    s.line(L['cuff'], TEAL[1])
    s.px(L['buttons'], GOLD[0])


CAPE.update({
    'land': dict(      # 从天窗落下：披风向上兜满风、两侧张开
        out=[(29, 27), (20, 22), (10, 17), (2, 13.5), (0, 17), (3, 22), (1, 28), (4, 35), (2, 42), (7, 48), (13, 55),
             (20, 60), (28, 62.5), (36, 62.5), (44, 60), (50, 55), (54, 48), (52, 42), (55.5, 34), (53, 27),
             (55.5, 19), (52, 14.5), (46, 19), (40, 23.5), (34, 26)],
        lining=[[(29, 27), (20, 22), (10, 17), (2, 13.5), (0, 17), (3, 22), (10, 23.5), (18, 25), (24, 27)],
                [(55.5, 19), (52, 14.5), (46, 19), (40, 23.5), (34, 26), (44, 25.5), (51, 24)]],
        folds=[[(12, 28), (11, 40), (14, 52)], [(20, 30), (21, 44), (24, 58)], [(46, 30), (47, 42), (45, 54)]]),
    'dash': dict(      # 冲刺：披风水平甩向身后（画面右）
        out=[(25, 22), (31, 20), (37, 19), (44, 17), (51, 15), (56, 13), (56, 19), (53, 22), (56, 26), (56, 33),
             (52, 35), (55, 40), (49, 43), (45, 49), (39, 52), (33, 50), (28, 44), (25, 36), (24, 28)],
        lining=[[(56, 13), (56, 19), (53, 22), (56, 26), (56, 33), (52, 35), (49, 30), (50, 22)]],
        folds=[[(34, 24), (42, 26), (50, 27)], [(33, 32), (41, 36), (48, 38)]]),
})


def _flutter(pts, k, amp, drift=0.0, y0=24.0):
    """披风飘动：y0 以下的顶点按帧号 k 做横向正弦摆动（越往下摆得越大），drift = 整体被风吹向右的量。"""
    out = []
    for x, y in pts:
        w = min(1.0, max(0.0, (y - y0) / 28.0))
        out.append((x + (amp * np.sin(k * np.pi / 2 + y * 0.28) + drift) * w,
                    y - abs(amp) * 0.5 * w * (1 + np.cos(k * np.pi / 2 + x * 0.2)) * 0.5))
    return out


def cape_pose(s, name, k=None, amp=0.0, drift=0.0, dx=0, dy=0):
    c = CAPE[name]
    f = (lambda pts: _mv(_flutter(pts, k, amp, drift), dx, dy)) if k is not None else (lambda pts: _mv(pts, dx, dy))
    out = s.poly(f(c['out']))
    s.part_fill(out, SLATE[1:], hl=1, sh=1, deep=1)
    for pts in c['lining']:
        s.part_fill(out & s.poly(f(pts)), [SLATE[0], SLATE[0], SLATE[1]], hl=0, sh=1, deep=0, line=SLATE[2])
    for pts in c['folds']:
        q = [(round(x), round(y)) for x, y in f(pts)]
        s.line([(x + 1, y) for x, y in q], SLATE_FOLD)
        s.line(q, SLATE[3])
    edge = out & ~_shift(out, -1, 0)
    s.col[edge] = s.ci(SLATE[3])


ARM_L.update({
    'out': dict(       # 向右下方伸开（落地 / 鞠躬时平衡、拿杖）
        upper=[(42, 13.5), (45.5, 14), (49.5, 21), (50.5, 25), (48, 26), (45, 20), (42, 17.5)],
        fore=[(48, 24.5), (51, 24), (52, 27.5), (49, 28.5)],
        hand=((49, 27), ['.WW.', 'WWwv', '.wv.'])),
    'snap': dict(      # 抬到肩旁打响指
        upper=[(42.5, 13.5), (46, 13), (49, 14.5), (49, 17), (46, 17.5), (42.5, 17)],
        fore=[(46.5, 11), (49.5, 11), (49.8, 14.5), (46.5, 14.8)],
        hand=((46, 6), ['.W.W', 'WWW.', 'wWWv', '.wWv', '..wv'])),
    'raise': dict(     # 高举宝石
        upper=[(42.5, 13.5), (46, 12.5), (50, 10.5), (51.5, 13), (47, 16), (42.5, 17)],
        fore=[(49.5, 7.5), (52.5, 7.5), (52.8, 11.5), (49.5, 12)],
        hand=((49, 4), ['WWWw', 'wWWv', '.wv.'])),
    'back': dict(      # 冲刺时甩向身后
        upper=[(42.5, 14), (46, 14.5), (51, 17), (52.5, 19.5), (50, 20.5), (45.5, 18), (42.5, 17.5)],
        fore=[(50.5, 17.5), (53.5, 18), (53.5, 21.5), (50.5, 21)],
        hand=((52, 20), ['WW', 'wv'])),
})
GEM = ['.rRr.',        # 盗来的宝石（红色的“玫瑰之泪”），高光一点白
       'rRWRr',
       '.xrx.',
       '..x..']
# 右臂的弯曲姿势（旋转直臂做不出来的）：鞠躬时手按胸口
ARM_R_BENT = {
    'chest': dict(
        sleeve=[(29, 17), (31.5, 19), (30.5, 24.5), (34, 21.5), (36, 22), (36, 24.5), (31, 27.5), (27.5, 27),
                (27, 20)],
        cuff=[(33.5, 20.5), (36, 20.5), (36.5, 23.5), (34, 24)],
        hand=((35, 20), ['WWw', 'wWv', '.wv'])),
}


def _staff_from(grip, ang, above=16.0, below=32.0):
    """握点 + 杖身方向（度，0 = 竖直向下，正 = 杖尖往右偏）→ (杖头, 杖尖)。"""
    a = np.radians(ang)
    ux, uy = np.sin(a), np.cos(a)
    return (grip[0] - ux * above, grip[1] - uy * above), (grip[0] + ux * below, grip[1] + uy * below)


# 姿势表。up = 上半身（躯干 / 披肩 / 头 / 手臂）相对腿的偏移
POSES = {
    'idle': dict(cape='idle', legs='cross', up=(0, 0), arm_r=('rot', 0, 'grip'), arm_l='mask', staff='grip',
                 staff_ang=21),
    'land': dict(cape='land', legs='crouch', up=(0, 9), arm_r=('rot', 18, 'grip'), arm_l='out', staff='grip',
                 staff_ang=38, hair='idle'),
    'bow': dict(cape='idle', legs='cross', up=(-1, 3), arm_r=('bent', 'chest'), arm_l='out', staff='left',
                staff_ang=-18, head_dy=3, head_dx=-2, expr='smile'),
    'twirl': dict(cape='idle', legs='cross', up=(0, 0), arm_r=('rot', 85, 'grip'), arm_l='snap', staff=None),
    'dash_ready': dict(cape='land', legs='crouch', up=(2, 7), arm_r=('rot', -25, 'open'), arm_l='back', staff=None),
    'dash': dict(cape='dash', legs='lunge', up=(-6, 6), arm_r=('rot', 62, 'open'), arm_l='back', staff=None,
                 expr='laugh'),
    'jewel': dict(cape='idle', legs='cross', up=(0, 0), arm_r=('rot', 0, 'grip'), arm_l='raise', staff='grip',
                  staff_ang=21, gem=True, flutter=(2.5, 3.0)),
}


def _rig(name, k=None):
    P_ = POSES[name]
    s = Sprite(BW, BH)
    ux, uy = P_['up']
    pts = {}
    fl = P_.get('flutter', (1.2, 0.5))
    cape_pose(s, P_['cape'], k, fl[0], fl[1], ux * 0, 0) if k is not None else cape_pose(s, P_['cape'])
    # 右手握杖：杖先画（在手套后面）
    kind_r = P_['arm_r']
    grip = None
    if kind_r[0] == 'rot':
        deg = kind_r[1]
        (wx, wy), = _mv(_rot([ARM_R['wrist']], SHOULDER_R, deg), ux, uy)
        grip = (wx - 1.2, wy + 1.2)
        pts['hand_r'] = (wx - 1, wy + 1)
    else:
        (hx, hy), rows = ARM_R_BENT[kind_r[1]]['hand']
        pts['hand_r'] = (hx + ux + 1.5, hy + uy + 1)
    if P_.get('staff') == 'grip':
        top, bot = _staff_from(grip, P_['staff_ang'])
        staff(s, top, bot)
        pts['orb'] = top
    if P_['legs'] == 'cross':
        legs(s)
    else:
        legs_pose(s, P_['legs'])
    kind_l = P_['arm_l']
    arm_l(s, kind_l, part='upper', dx=ux, dy=uy)
    torso(s, ux, uy)
    if kind_r[0] == 'rot':
        arm_r(s, deg=kind_r[1], dx=ux, dy=uy)
    else:
        a = ARM_R_BENT[kind_r[1]]
        s.part_fill(s.poly(_mv(a['sleeve'], ux, uy)), WHITE, hl=1, sh=1, deep=1, line=WHITE[3])
    capelet(s, ux, uy)
    hair(s, P_.get('hair', 'idle'), ux, uy)
    head(s, 28 + ux + P_.get('head_dx', 0), 0 + uy + P_.get('head_dy', 0), P_.get('expr'))
    if P_.get('staff') == 'left':
        (hx, hy), _ = ARM_L[kind_l]['hand']
        g = (hx + ux + 2, hy + uy + 1.5)
        top, bot = _staff_from(g, P_['staff_ang'], above=14, below=34)
        staff(s, top, bot)
        pts['orb'] = top
    arm_l(s, kind_l, part='fore', dx=ux, dy=uy)
    (hx, hy), rows = ARM_L[kind_l]['hand']
    pts['hand_l'] = (hx + ux + len(rows[0]) / 2, hy + uy + len(rows) / 2)
    if kind_r[0] == 'rot':
        arm_r(s, deg=kind_r[1], hand=kind_r[2], part='hand', dx=ux, dy=uy)
    else:
        a = ARM_R_BENT[kind_r[1]]
        s.part_fill(s.poly(_mv(a['cuff'], ux, uy)), TEAL, hl=1, sh=1, deep=0, line=TEAL[2])
        (hx, hy), rows = a['hand']
        s.stamp_part(rows, hx + ux, hy + uy)
    tassel(s, ux, uy)
    if P_.get('gem'):
        (hx, hy), _ = ARM_L[kind_l]['hand']
        s.stamp_part(GEM, hx + ux, hy + uy - 4)
        pts['gem'] = (hx + ux + 2, hy + uy - 3)
    s.lines(thr=0.3)
    pts['eye'] = (36 + ux + P_.get('head_dx', 0), 7 + uy + P_.get('head_dy', 0))
    return s, pts


def _parse(pose):
    name, _, k = pose.partition(':')
    return name, (int(k) % 4 if k else None)


def build(pose='idle'):
    """-> (Sprite, 关键点 dict)。关键点：hand_r / hand_l / orb（杖头）/ gem / eye，全身像像素坐标 (x, y)。"""
    name, k = _parse(pose)
    if name == 'skill':
        return _body_skill(), {'hand_r': (META['hand_skill'][1], META['hand_skill'][0]), 'eye': (36, 7)}
    if name == 'idle' and k is None:
        return _body_idle(), {'hand_r': (15.4, 31.3), 'hand_l': (40, 10.5), 'orb': (9.6, 16.1), 'eye': (36, 7)}
    return _rig(name if name in POSES else 'idle', k)


def body(pose='idle'):
    s, _ = build(pose)
    s.despeckle(BODY_KEEP if pose in ('idle', 'skill') else BODY_KEEP_RIG)
    return s.render(remap=BODY_REMAP)


def points(pose='idle'):
    """某个姿势的手 / 杖头 / 宝石 / 眼睛在全身像上的像素坐标（场景里的特效从这里发出）。"""
    return build(pose)[1]


BODY_KEEP_RIG = [(20, 0, 50, 22), (0, 0, 56, 30), (27, 33, 43, 50)]


# 清孤点时不动的区域：头（面具 / 眼 / 嘴）、杖头、流苏、靴扣、腰带金扣、马甲扣、skill 的扑克牌
BODY_KEEP = [(27, 0, 45, 13), (0, 10, 20, 27), (27, 33, 43, 45), (34, 54, 37, 62), (38, 29, 45, 33), (39, 23, 42, 29),
             (0, 5, 12, 20), (40, 6, 56, 18)]


META = dict(
    key='bleublanc',
    name_zh='布卢布兰',
    title_zh='怪盗绅士',
    rank_zh='执行者 No.X',
    skill_zh='死亡魔术',
    quote_zh='美，乃值得骄傲之物。',
    fx=dict(
        primary='#9b5bd6',     # 紫玫瑰
        secondary='#8fe3d4',   # 薄荷绿（荷叶边 / 面具镜片色）
        description=('怪盗的“死亡魔术”：从张开的白手套（hand_skill）甩出一扇旋转的白色扑克牌/魔术飞刀，'
                     '夹着大量紫色玫瑰花瓣呈螺旋飞向目标；目标周围闪出 2–3 个半透明的布卢布兰残影（幻术分身，'
                     '可用本角色 body 半透明+偏紫色调），随后花瓣与卡牌在目标处收拢成一朵巨大的紫玫瑰，'
                     '啪地爆散成花瓣雨与薄荷绿的星形闪光。整体节奏像舞台魔术谢幕，偏华丽而非爆炸。'),
    ),
    # 剪影里发光的眼睛 (y, x)：面具镜孔里那只青色的眼（远侧眼被面具挡住）
    eyes_body={'idle': [(7, 36)], 'skill': [(7, 36)]},
    hand_skill=None,          # (y, x)，skill 姿势张开的手掌中心；下面按手臂旋转算出来
)


def _hand_skill():
    s = Sprite(BW, BH)
    arm_r(s, deg=SKILL_ARM_DEG, hand='open', part='hand')
    y, x = np.argwhere(s.col >= 0).mean(0)
    return int(round(y)), int(round(x))


META['hand_skill'] = _hand_skill()

# 从 idle 像素图上逐点读出（像素坐标），和 data/landmarks/bleublanc.json 的 full 同名。
# 被披肩 / 裙摆盖住的关节（肩、裆）按袖筒与裤管的走向估。
LM_IDLE = {
    'head_top': (36.5, 3.9), 'chin': (36.8, 11.5), 'neck': (36.8, 12.8),
    'shoulder_r': (30.3, 16.3), 'shoulder_l': (42.6, 14.8),
    'elbow_r': (23, 23.3), 'elbow_l': (47.6, 17.3),
    'wrist_r': (16.3, 29.8), 'wrist_l': (40.6, 13.3),
    'waist': (38, 30.5), 'crotch': (37.8, 38.5),
    'knee_r': (35.8, 49), 'ankle_r': (35.3, 65.8), 'ankle_l': (31.8, 63.5),
    'x_staff_orb': (9.4, 15.3), 'x_staff_tip': (27, 60),
    'x_cape_tip_r': (0.6, 25.5), 'x_cape_low_r': (-0.3, 39.3), 'x_cape_hem_r': (13.4, 51.4), 'x_cape_hem_c': (22.6, 54.2),
    'x_cape_out_l': (55, 31.5), 'x_cape_low_l': (52.4, 48.3),
    'x_coat_hem_r': (27, 35.5), 'x_coat_hem_l': (46, 36.5),
    'x_tassel_ball': (38.8, 38), 'x_tassel_end': (28, 43),
    'x_hand_l': (39.8, 10.8), 'x_toe_r': (35.5, 71.4), 'x_toe_l': (31.8, 68.5),
}
LANDMARKS = {'portrait': LM_PORTRAIT, 'idle': LM_IDLE}
