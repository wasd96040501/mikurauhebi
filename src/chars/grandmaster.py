"""盟主（Grandmaster）—— 身喰らう蛇 / 噬身之蛇 的最高领袖，全身像素立绘。

参考：《闪之轨迹 IV》盟主立绘（refs/grandmaster/full.png，主参考），辅以《界之轨迹》立绘与《始之轨迹》主视觉。
- 双眼轻闭，上半张脸笼在面纱般的阴影里，只露出下半脸与淡淡的微笑；
- 淡蓝色长发中分，前面两束垂到大腿，右侧（画面右）一大片披散下来，发梢渐变成淡紫；
- 金色项圈 + 深色挂脖领，胸口菱形金纹，胸前金色胸针；
- 裸肩，白色长披风绕过上臂（带粉色镶边），从臂弯一直拖到地面、两侧展成尖角；
- 深色紧身上衣、斜挂的金 V 腰带，下方金线收成 V 字；
- 胸前垂下的酒红色长垂布，白色双蛇杖与衔尾蛇纹，尖角下缘，两侧金边；
- 深紫色长裙 + 两层淡紫荷叶褶，洋红色高跟鞋。
右臂（画面左）微微张开、掌心向下，左臂自然垂下 —— 与官方站姿一致。

画布 110x150（W x H），脚底贴最后一行，render.py 用 anchor='bottom' 贴图。
盟主在 PV 里是巨像式的存在，身高约 148px，不套用 72px≈180cm 的比例。

画法：在字符画布上用无抗锯齿的多边形 / 贝塞尔笔触铺色块（每种材质 3–4 阶），细节用 ASCII 小图章，
最后自动外描边；背后逆光只描头顶与肩上的上缘。
部件（披风、头发、长裙、垂布、上衣、手臂、头部）各自是函数，由 Pose 里的几个参数定位，方便以后加动画帧。
"""
from dataclasses import dataclass, replace

import numpy as np

import pixelkit as pk

W, H = 110, 150
CX = 55

# ----------------------------------------------------------------- 调色板
OUT = '#1a1224'
PAL = {
    'O': OUT,
    # 皮肤
    'S': '#fcebd8', 's': '#e8c2ae', 'q': '#a87070',
    # 面纱阴影（上半张脸）
    'V': '#6c5a72', 'w': '#a08a98', 'v': '#3a2a44',
    # 淡蓝长发（上）→ 淡紫（下）
    'Y': '#eaf4ff', 'H': '#bcd4f0', 'h': '#93a9d8', 'j': '#6874ac', 'L': '#cdc6ea', 'l': '#a59ed0',
    # 白披风
    'W': '#fbf8fe', 'C': '#e2dcf0', 'c': '#b9afd6', 'd': '#8a7eae',
    # 粉色镶边
    'p': '#bc6a94',
    # 深色上衣 / 右袖
    'K': '#5c5a68', 'k': '#3e3c4a', 'x': '#282632',
    # 紫色袖口
    'P': '#8c6cb0', 'U': '#644888', 'u': '#44305e',
    # 深紫长裙
    'N': '#7a6c96', 'n': '#5c5078', 'm': '#42385a', 'e': '#2e2640',
    # 淡紫荷叶褶
    'F': '#e0d6f0', 'f': '#b6a8d4', 't': '#8a7ca6',
    # 酒红垂布
    'M': '#b4648c', 'R': '#8c4468', 'r': '#62304e',
    # 金
    'G': '#f8e4a4', 'g': '#d8ae5c', 'b': '#8c6834',
    # 鞋
    'A': '#c8608e', 'a': '#8c3a62', 'z': '#521e3e',
    # 逆光
    'Z': '#fff2cc',
}


# ----------------------------------------------------------------- 画布
class Cv:
    def __init__(s, h=H, w=W):
        s.h, s.w = h, w
        s.a = np.full((h, w), ' ', dtype='<U1')
        yy, xx = np.mgrid[0:h, 0:w]
        s.px, s.py = xx + .5, yy + .5

    def polym(s, pts):
        m = np.zeros((s.h, s.w), bool)
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if y1 == y2:
                continue
            c = (y1 > s.py) != (y2 > s.py)
            xi = (x2 - x1) * (s.py - y1) / (y2 - y1) + x1
            m ^= c & (s.px < xi)
        return m

    def ellm(s, cx, cy, rx, ry):
        return ((s.px - cx) / rx) ** 2 + ((s.py - cy) / ry) ** 2 <= 1

    def has(s, chars):
        return np.isin(s.a, list(chars))

    def put(s, m, c, on=None, off=None):
        if on is not None:
            m = m & s.has(on)
        if off is not None:
            m = m & ~s.has(off)
        s.a[m] = c

    def poly(s, pts, c, **k):
        m = s.polym(pts)
        s.put(m, c, **k)
        return m

    @staticmethod
    def bez(pts, t):
        p = np.array(pts, float)
        while len(p) > 1:
            p = p[:-1] * (1 - t) + p[1:] * t
        return p[0]

    def strokem(s, pts, w0, w1=None, n=64):
        w1 = w0 if w1 is None else w1
        ts = np.linspace(0, 1, n)
        P = np.array([s.bez(pts, t) for t in ts])
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        Wd = (w0 + (w1 - w0) * ts)[:, None] / 2
        L, R = P + N * Wd, P - N * Wd
        return s.polym([tuple(p) for p in np.concatenate([L, R[::-1]])])

    def stroke(s, pts, c, w0, w1=None, **k):
        m = s.strokem(pts, w0, w1)
        s.put(m, c, **k)
        return m

    def linem(s, pts):
        ln = sum(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
        m = np.zeros((s.h, s.w), bool)
        for t in np.linspace(0, 1, int(ln * 3) + 2):
            x, y = s.bez(pts, t)
            xi, yi = int(np.floor(x)), int(np.floor(y))
            if 0 <= xi < s.w and 0 <= yi < s.h:
                m[yi, xi] = True
        return m

    def line(s, pts, c, **k):
        """1px 贝塞尔细线。"""
        m = s.linem(pts)
        s.put(m, c, **k)
        return m

    def dots(s, pts, c, **k):
        m = np.zeros((s.h, s.w), bool)
        for x, y in pts:
            if 0 <= x < s.w and 0 <= y < s.h:
                m[y, x] = True
        s.put(m, c, **k)

    def patch(s, x0, y0, rows, flip=False):
        for dy, r in enumerate(rows):
            r = r[::-1] if flip else r
            for dx, ch in enumerate(r):
                if ch not in '. ' and 0 <= y0 + dy < s.h and 0 <= x0 + dx < s.w:
                    s.a[y0 + dy, x0 + dx] = ch

    @staticmethod
    def shift(m, dx, dy):
        o = np.zeros_like(m)
        h, w = m.shape
        ys = slice(max(-dy, 0), h + min(-dy, 0))
        yd = slice(max(dy, 0), h + min(dy, 0))
        xs = slice(max(-dx, 0), w + min(-dx, 0))
        xd = slice(max(dx, 0), w + min(dx, 0))
        o[yd, xd] = m[ys, xs]
        return o

    def ring(s, m):
        g = m.copy()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            g |= s.shift(m, dx, dy)
        return g & ~m

    def sep(s, m, c, on=None):
        """在 m 外侧 1px、已有颜色处描分隔线。"""
        s.put(s.ring(m) & (s.a != ' '), c, on=on)

    def edge(s, m, dx, dy):
        """m 中沿 (dx,dy) 方向的邻格不在 m 里的像素。"""
        return m & ~s.shift(m, -dx, -dy)

    def rgba(s):
        img = np.zeros((s.h, s.w, 4), np.float32)
        for ch, col in PAL.items():
            m = s.a == ch
            if m.any():
                img[m, :3] = pk.hexc(col)
                img[m, 3] = 1
        o = s.ring(s.a != ' ')
        img[o, :3] = pk.hexc(OUT)
        img[o, 3] = 1
        return img


# ----------------------------------------------------------------- 姿势参数
@dataclass(frozen=True)
class Pose:
    """几个关键点决定部件位置（画布坐标）。r = 角色右侧（画面左），l = 角色左侧（画面右）。"""
    sh_r: tuple = (38.0, 40.0)      # 右袖从披风下露出的点
    el_r: tuple = (32.5, 50.0)
    wr_r: tuple = (23.5, 63.5)
    hand_r: str = 'down'            # 'down' 掌心朝下张开 / 'up' 举起
    sh_l: tuple = (63.0, 42.0)
    el_l: tuple = (65.5, 55.0)
    wr_l: tuple = (68.5, 74.5)
    hair_dx: float = 0.0            # 发梢左右摆（动画用）
    cape_dx: float = 0.0            # 披风下摆左右摆（动画用）


def pose(name='idle', **kw):
    """在已有姿势上改几个参数得到新帧，例如 pose('idle', hair_dx=1.5, cape_dx=-1)。"""
    return replace(POSES[name], **kw)


POSES = {
    'idle': Pose(),
    'raise': Pose(sh_r=(37.0, 37.0), el_r=(29.0, 34.0), wr_r=(21.0, 25.0), hand_r='up', cape_dx=-2.0),
    # 举手的分解帧：预备（手臂略收）-> 抬到腰高 -> 过肩翻掌 -> raise -> 最高点（随后回落到 raise）
    'raise0': Pose(sh_r=(38.0, 40.0), el_r=(33.5, 50.5), wr_r=(26.0, 63.0)),
    'raise1': Pose(sh_r=(37.5, 39.0), el_r=(31.0, 45.0), wr_r=(23.0, 50.0), cape_dx=-0.5),
    'raise2': Pose(sh_r=(37.0, 38.0), el_r=(29.5, 39.0), wr_r=(21.0, 36.0), hand_r='up', cape_dx=-1.2),
    'raise_hi': Pose(sh_r=(37.0, 36.5), el_r=(28.5, 32.0), wr_r=(20.5, 22.0), hand_r='up', cape_dx=-2.5),
}


# ----------------------------------------------------------------- 部件
def _cloak(cv, P):
    """白色长披风：挂在两臂弯处一直拖到地面，下摆向两侧展开成尖角；上白下淡紫。"""
    ex, ey = P.el_r
    wx, wy = P.wr_r
    dx = P.cape_dx
    # 画面左片：上缘随右臂走
    left = [(38, 42), (ex + 1, ey - 4), (ex - 3, ey + 2), (min(wx + 3, 29), max(wy, 60)), (27.5, 76), (25.5, 94),
            (21.5 + dx * .3, 110), (16 + dx * .6, 125), (11 + dx, 139.5), (17 + dx, 140), (24, 138.5), (30, 136),
            (32, 124), (34, 110), (36, 98), (37, 84), (37.5, 70), (38, 56)]
    right = [(66, 44), (72, 52), (77, 56), (80, 64), (81.5, 80), (84, 100), (88 + dx * .3, 116), (93 + dx * .6, 129),
             (99.5 + dx, 141.5), (92 + dx, 141.5), (84, 140.5), (78, 139), (75, 126), (72, 112), (71, 96), (71, 80),
             (70, 64), (67, 54)]
    for pts, side in ((left, 0), (right, 1)):
        m = cv.polym(pts)
        yy = cv.py
        v = yy + 0.6 * np.abs(cv.px - CX)                  # 外侧先暗：明暗分界线顺着下摆斜下去
        cv.put(m & (v < 100), 'W')
        cv.put(m & (v >= 100) & (v < 124), 'C')
        cv.put(m & (v >= 124), 'c')
        # 竖向褶：从臂弯垂下的暗条
        if side == 0:
            folds = [([(ex + 1.5, ey + 3), (32, 90), (26, 134)], 1.4, 1.2),
                     ([(ex - 3.5, ey + 12), (25, 96), (17, 132)], 1.0, 1.0)]
        else:
            folds = [([(72, 60), (76, 96), (86, 136)], 1.6, 1.2), ([(77, 62), (82, 100), (93, 136)], 1.2, 1.0)]
        for f, w0, w1 in folds:
            fm = cv.strokem(f, w0, w1) & m
            cv.put(fm & cv.has('W'), 'C')
            cv.put(fm & cv.has('C'), 'c')
            cv.put(fm & cv.has('c'), 'd')
        # 靠身体一侧的里衬阴影 + 下摆暗边
        inner = cv.edge(m, 1 if side == 0 else -1, 0) | cv.edge(cv.shift(m, 1 if side == 0 else -1, 0),
                                                              1 if side == 0 else -1, 0)
        cv.put(inner & m & cv.has('WC'), 'c')
        cv.put(inner & m & cv.has('c') & (yy > 100), 'd')
        cv.put(cv.edge(m, 0, 1) & m, 'd')


def _gown(cv, P):
    """深紫长裙：腰部收窄向下展开；外侧转折暗、中间亮，竖褶。"""
    pts = [(40, 57), (62, 54), (64, 66), (67, 84), (69.5, 100), (72, 116), (76, 128), (80, 139), (70, 141),
           (58, 140), (48, 141), (36, 140), (26, 139), (29, 128), (32, 114), (34, 100), (36, 86), (38, 70)]
    m = cv.poly(pts, 'n')
    xs = cv.px
    cv.put(m & (np.abs(xs - 48) < 11) & (cv.py > 70), 'N')
    cv.put(m & ((xs < 37) | (xs > 68)), 'm')
    for f in ([(38, 72), (35, 104), (31, 138)], [(60, 70), (63, 104), (66, 138)], [(64, 70), (68, 104), (73, 138)],
              [(40, 90), (39, 112), (38, 138)]):
        cv.put(cv.linem(f) & m, 'm')
    cv.put(cv.linem([(57, 76), (59, 108), (61, 138)]) & m, 'e')
    # 下摆一圈暗红细线（官方图裙摆上的装饰线）


def _ruffle(cv, pts):
    """一层淡紫荷叶褶：带状多边形，下缘按 2px 一褶起伏成波浪，褶面亮暗相间。"""
    m = cv.polym(pts)
    pl = np.floor(cv.px / 2) % 2 == 0
    m = m | (cv.shift(cv.edge(m, 0, 1), 0, 1) & pl)          # 波浪下缘
    cv.put(m, 't')
    cv.put(m & pl, 'f')
    cv.put(m & cv.edge(m, 0, -1), 'F')
    cv.put(m & cv.shift(cv.edge(m, 0, -1), 0, 1) & pl, 'F')
    cv.put(cv.ring(m) & cv.shift(m, 0, 1) & cv.has('NnmecCWd'), 'e')   # 褶边投在下面的暗线
    return m


def _ruffles(cv, P):
    """两层荷叶褶：从垂布两侧斜挂下来（官方图里的淡紫褶边），裙摆一圈底褶。"""
    dx = P.cape_dx * 0.3
    _ruffle(cv, [(42, 100), (44, 102), (38, 108), (33, 115), (29, 117.5), (30, 113), (35, 107)])
    _ruffle(cv, [(55, 101), (58, 101), (63, 106), (69, 111), (72, 116), (68, 116.5), (63, 112), (58, 107)])
    _ruffle(cv, [(39, 125), (41, 127), (35, 132), (28, 137), (21 + dx, 140.5), (19 + dx, 139), (26, 134), (33, 129)])
    _ruffle(cv, [(56, 127), (60, 128), (67, 132), (75, 136), (84 + dx, 140.5), (80, 141.5), (72, 140), (63, 135),
                 (57, 131)])
    # 裙摆底褶
    _ruffle(cv, [(24, 138), (40, 139), (46, 140), (58, 140), (72, 139.5), (82, 140), (82, 142.5), (58, 143),
                 (46, 143), (26, 142.5)])


def _feet(cv, P):
    """洋红色细带高跟鞋：画面左脚尖朝左下（鞋跟在右），画面右脚朝前。"""
    cv.patch(31, 138, [
        ".........aAa.",
        "........aAAa.",
        ".......aAzAa.",
        "......aAAAza.",
        ".....aAAzAa..",
        "...aAAAAza.a.",
        ".aAAAAAaa..a.",
        "aAAAaa.....a.",
        "aAa........a.",
    ])
    cv.patch(62, 139, [
        "aAAAa",
        "aAzAa",
        "aAAza",
        "azAAa",
        "aAzAa",
        "aAAAa",
        "aAAAa",
        "aAAAa",
        ".aAa.",
        "..a..",
    ])


def _tabard(cv, P):
    """酒红长垂布：胸针下细长一条，腰带下展宽，尖角下缘，两侧金边；白色双蛇杖 + 衔尾蛇纹。"""
    m = cv.poly([(42, 60), (53, 60), (53, 135), (47.5, 145), (42, 135)], 'R')
    top = cv.poly([(47.5, 38), (51.5, 38), (51.5, 60), (47.5, 60)], 'R')
    m = m | top
    cv.put(m & (cv.py < 84), 'M')
    cv.put(m & (cv.py > 120), 'r', on='R')
    cv.put(cv.edge(m, 1, 0), 'r')
    # 两侧金边（1px）+ 内侧暗线；下缘斜边也描金
    gold = (cv.edge(m, -1, 0) | cv.edge(m, 1, 0) | cv.edge(m, 0, 1)) & (cv.py > 64)
    cv.put(gold & (cv.px < 47.5), 'g')
    cv.put(gold & (cv.px >= 47.5), 'b')
    cv.line([(43.5, 66), (43.5, 134)], 'r', on='MR')
    cv.line([(51.5, 66), (51.5, 134)], 'r', on='MR')
    # 双蛇杖：杖头双翼 + 一条细杖 + 缠绕的蛇（1px 波线）
    cv.patch(45, 75, [
        "W.W.W",
        ".WWW.",
        "..W..",
    ])
    cv.line([(47.5, 78), (47.5, 100)], 'C', on='MRr')
    pts = [(47.5 + 1.5 * np.sin((y - 79) * np.pi / 4.5), y) for y in np.arange(79, 100, 0.2)]
    cv.dots({(int(round(x - .5)), int(y)) for x, y in pts}, 'W', on='MRrC')
    # 下方衔尾蛇纹
    cv.patch(45, 126, [
        ".WWW.",
        "W...W",
        "W.W.W",
        "W..WW",
        ".WW..",
    ])


def _bodice(cv, P):
    """深色紧身上衣：挂脖领 + 胸口菱形金纹 + 胸针；斜挂的金腰带，腰下深色前片收成 V。"""
    # 上衣主体（胸口到腰下的 V 尖）
    body = [(46, 29), (49, 27), (55, 27), (58, 29), (60.5, 38), (61.5, 48), (63, 54), (63, 60), (57, 66), (48, 73),
            (39, 62), (40, 55), (44.5, 46), (45, 36)]
    m = cv.poly(body, 'k')
    cv.put(m & (cv.px < 49) & (cv.py < 56), 'K')
    cv.put(cv.edge(m, 1, 0) & (cv.py < 56), 'x')
    cv.put(m & (cv.py > 58), 'x')
    cv.line([(57, 58), (58, 64)], 'k', on='x')
    # 挂脖领（choker 下方的深色竖条）
    cv.poly([(49.5, 21.5), (55, 21.5), (55.5, 29), (49, 29)], 'k')
    # 金纹：项圈、竖线、菱形
    cv.poly([(49.5, 20.6), (55.2, 20.6), (55.2, 22.4), (49.5, 22.4)], 'g')
    cv.line([(50, 21), (55, 21)], 'G')
    cv.dots([(52, 23), (52, 24)], 'g')
    cv.patch(50, 25, [
        "..g..",
        ".g.g.",
        "g.b.g",
        ".g.g.",
        "..g..",
    ])


def _belt(cv, P):
    """胸针、斜挂的金腰带与腰下金线（压在垂布上）。"""
    # 腰下深色前片：压在垂布上，收成 V 尖，金色包边
    fl = cv.polym([(41, 58), (63.5, 53), (63, 59), (57, 64), (48.6, 72), (40, 63)])
    cv.put(fl, 'x')
    cv.put(fl & (cv.px < 45), 'k')
    cv.put(fl & cv.edge(fl, 0, 1) & (cv.px < 48.5), 'g')
    cv.put(fl & cv.edge(fl, 0, 1) & (cv.px >= 48.5), 'b')
    # 胸针 + 下垂金条
    cv.patch(47, 32, [
        ".gGGg.",
        "gGbbGg",
        "bgGGgb",
        ".bgGb.",
        "..gg..",
        "..gb..",
        "..gb..",
    ])
    # 金腰带：画面左低右高斜挂，中间 V 形扣
    belt = cv.polym([(42.5, 56.8), (50, 58), (63.5, 51.5), (64, 55), (50, 61), (42.5, 60.2)])
    cv.put(belt, 'g')
    cv.put(belt & cv.edge(belt, 0, -1), 'G')
    cv.put(belt & cv.edge(belt, 0, 1), 'b')
    cv.patch(47, 59, [
        "gG.Gg",
        "bgGgb",
        ".bgb.",
        "..b..",
    ])
    # 腰下金线：两条斜线收到 V 尖；再向两侧斜出
    for pts, c in (([(42, 60), (36, 71)], 'g'), ([(59, 60), (66, 74)], 'b'), ([(48, 66), (48, 72)], 'b')):
        cv.line(pts, c, on='KkxNnmeWCc')


HAND_DOWN = [   # 画面左手：掌心朝前、手指松松下垂张开（指尖朝左下）
    "......SS.",
    ".....SSSs",
    "....SSSSs",
    "...SSSSs.",
    "..SSsSSs.",
    "..SS.SsS.",
    ".SS.SS.S.",
    ".S..S..s.",
    "S..S.....",
    "S..s.....",
]
HAND_UP = [     # 举起的手：掌心朝外、五指张开
    "S.S.S....",
    "S.S.S.S..",
    "SsSsS.S..",
    "SSSSSsS..",
    ".SSSSSs..",
    ".SSSSs...",
    "..SSs....",
]
HAND_L = [      # 画面右手：侧面垂下
    "SSS.",
    "SSSs",
    "SSSs",
    "SSSs",
    "SsSs",
    "S.Ss",
    "S.S.",
    "S.S.",
    ".S..",
    ".s..",
]


CUFF = [(1.25, -2.8), (-3.2, -1.5), (-9, 0), (-8.7, 3.2), (-2.9, 7), (3, 9.8), (4.4, 6), (3.9, 1.4)]


def _frame(a, b):
    u = np.subtract(b, a, dtype=float)
    u /= np.hypot(*u)
    return u, np.array([u[1], -u[0]])


def _arm_r(cv, P):
    """角色右臂（画面左）：深色长袖从披肩下伸出，大号紫色荷叶袖口（沿前臂方向张开），手张开。"""
    u0, n0 = _frame(P.sh_r, P.el_r)
    u1, n1 = _frame(P.el_r, P.wr_r)
    o = lambda p, n, k: tuple(np.array(p) - k * n)          # 向外侧（-n）偏：垂坠的宽袖
    sl = cv.strokem([o(P.sh_r, n0, 0.5), o(P.el_r, n0, 1.2)], 7, 8) | \
        cv.strokem([o(P.el_r, n1, 1.2), o(P.wr_r, n1, 1.8)], 8, 8)
    cv.put(sl, 'k')
    cv.put(sl & cv.shift(~sl, 1, 0), 'K')
    cv.put(sl & cv.shift(~sl, -1, 0), 'x')
    cv.line([P.sh_r, P.el_r], 'K', on='k')
    wx, wy = P.wr_r
    u, n = _frame(P.el_r, P.wr_r)
    cuff = [tuple(np.array(P.wr_r) + a * u + b * n) for a, b in CUFF]
    cm = cv.poly(cuff, 'U')
    cv.put(cm & cv.edge(cm, 0, -1), 'P')
    # 袖口的竖褶：沿前臂方向的暗条
    for t in (-1.5, 2.5, 6):
        a0 = np.array(P.wr_r) - 7 * u + t * n
        cv.put(cv.linem([tuple(a0), tuple(a0 + 9 * u + 1.5 * n)]) & cm, 'u')
    cv.put(cm & cv.edge(cm, 0, 1), 'u')
    if P.hand_r == 'up':
        cv.patch(int(wx) - 3, int(wy) - 7, HAND_UP)
    else:
        cv.patch(int(wx) - 8, int(wy), HAND_DOWN)


def _arm_l(cv, P):
    """角色左臂（画面右）：深紫灰长袖垂下，手自然垂下。"""
    sl = cv.strokem([P.sh_l, P.el_l], 5, 4.6) | cv.strokem([P.el_l, P.wr_l], 4.6, 4)
    cv.put(sl, 'n')
    cv.put(sl & cv.shift(~sl, 1, 0), 'N')
    cv.put(sl & cv.shift(~sl, -1, 0), 'm')
    wx, wy = P.wr_l
    cm = cv.poly([(wx - 2.5, wy - 1.5), (wx + 2, wy - 1.5), (wx + 2.5, wy + 2), (wx - 2.5, wy + 2)], 'U')
    cv.put(cm & cv.edge(cm, 0, 1), 'u')
    cv.patch(int(wx) - 1, int(wy) + 2, HAND_L)


def _shoulders(cv, P):
    """裸肩与胸口皮肤。"""
    cv.poly([(40.5, 28), (43, 25.5), (46, 24), (49, 24), (49, 30), (41, 31)], 'S')
    cv.poly([(55, 23.5), (59, 23), (62.5, 24.5), (64.5, 27), (64, 31), (59, 31), (55, 30)], 'S')
    cv.dots([(47, 29), (48, 29), (55, 29), (56, 29), (63, 29), (63, 30)], 's')
    cv.dots([(50, 20), (51, 20), (52, 20), (53, 20), (54, 20)], 's')    # 脖子（下巴下）


def _shawl(cv, P):
    """白色披肩：从两肩绕过上臂（肩头裸露），下缘一条粉色镶边。"""
    left = [(46, 29), (49, 29.5), (48.5, 32), (43, 36), (37, 41.5), (33, 44.5), (30, 45), (29.5, 42.5),
            (32, 38), (36.5, 33.5), (41, 30.5)]
    right = [(54, 28.5), (57, 27.5), (61, 29), (64.5, 28.5), (67.5, 31.5), (71.5, 37), (75, 43.5), (77, 49),
             (77.5, 54.5), (74.5, 56), (71.5, 53.5), (68, 48), (63, 41.5), (58.5, 35.5), (54, 31.5)]
    for pts, trim, fold in ((left, [(33.5, 43.5), (39, 38.5), (46.5, 31.5)], [(28.5, 43), (35, 36.5), (42, 30)]),
                            (right, [(57.5, 32.5), (64.5, 40), (70.5, 48), (74, 55)], [(62, 30.5), (69, 37), (75, 46)])):
        m = cv.poly(pts, 'W')
        cv.put(cv.linem(fold) & m, 'C')
        cv.put(m & cv.edge(m, 0, 1), 'c')
        cv.put(cv.linem(trim) & m, 'p')


def _hair_back(cv, P):
    """脑后与画面右侧披散的大片长发（压在披肩与披风上）。"""
    dx = P.hair_dx
    pts = [(58, 6), (60.5, 10), (62, 15), (64.5, 20), (69, 25), (74, 32), (79, 40), (83, 48), (86 + dx * .5, 57),
           (86 + dx * .6, 64), (83, 58), (79, 51), (77, 50), (79, 62), (81, 78), (81.5 + dx * .7, 96),
           (80 + dx, 110), (77.5, 102), (76, 88), (74, 72), (71.5, 60), (67, 50), (63, 42), (60, 33), (58.5, 25),
           (58, 18)]
    m = cv.poly(pts, 'H')
    cv.put(m & (cv.py > 52), 'L')
    for f, c in (([(61, 12), (66, 26), (76, 42), (84, 58)], 'h'), ([(60, 18), (65, 34), (73, 52), (78, 76), (79, 104)], 'h'),
                 ([(63, 12), (70, 28), (80, 46)], 'Y')):
        lm = cv.linem(f) & m
        cv.put(lm & cv.has('H'), c)
        cv.put(lm & cv.has('L'), 'l' if c == 'h' else 'L')
    cv.put(cv.edge(m, 1, 0) & cv.has('H'), 'h')
    cv.put(cv.edge(m, 1, 0) & cv.has('L'), 'l')


def _hair_front(cv, P):
    """垂在胸前的三束长发：画面左两束、正中一束（盖住上衣与垂布）。"""
    dx = P.hair_dx
    strands = [
        ([(46.5, 8), (45, 24), (43.5, 44), (43.2, 64), (42 + dx * .3, 86), (41 + dx * .6, 104)], 4.0, 2.2),
        ([(42.5, 33), (40, 46), (38.5, 64), (37 + dx * .4, 88), (38 + dx * .6, 105)], 2.4, 1.6),
        ([(55.5, 17), (54, 30), (53.6, 50), (54, 70), (55.5 + dx * .3, 88), (57 + dx * .6, 103)], 3.6, 2.0),
    ]
    for pts, w0, w1 in strands:
        m = cv.strokem(pts, w0, w1)
        cv.put(m, 'H')
        cv.put(m & (cv.py > 54), 'L')
        cv.put(m & cv.edge(m, 1, 0) & (cv.py < 54), 'h')
        cv.put(m & cv.edge(m, 1, 0) & (cv.py >= 54), 'l')
        cv.put(m & cv.edge(m, -1, 0) & (cv.py < 44) & (cv.py > 14), 'Y')
    cv.line([(46.5, 22), (45, 40), (44.5, 62), (43.2, 86)], 'j', on='HL')


def _head(cv, P):
    """头：中分的淡蓝长发、上半脸笼在面纱阴影里、闭着的双眼、淡淡的微笑。"""
    hair = [(51, 1), (56.5, 1), (58.5, 2.5), (59.5, 5), (60.5, 9), (61.5, 14), (62.5, 18), (62, 20), (59, 21),
            (57, 21), (57, 18), (48, 18), (48, 26), (44.8, 26), (44.8, 8), (45.5, 5), (47.5, 2.5)]
    m = cv.poly(hair, 'H')
    cv.put(m & (cv.py < 4.5) & (cv.py > 2), 'Y')
    cv.put(m & cv.edge(m, 1, 0), 'h')
    # 脸
    face = [(48, 8.5), (51.5, 8.2), (55, 10), (57, 13), (57.2, 17.5), (55.5, 19.4), (53.5, 20.8), (51.5, 20.8),
            (49.6, 19.8), (48, 18)]
    fm = cv.polym(face)
    cv.put(fm, 'S')
    cv.put(fm & (cv.py < 12), 'V')                     # 面纱阴影
    cv.put(fm & (cv.py >= 12) & (cv.py < 15), 'w')     # 纱的下缘（透出闭着的眼）
    cv.put(fm & cv.edge(fm, 1, 0) & (cv.py > 15), 's')
    cv.put(fm & cv.edge(fm, 0, -1) & (cv.py > 15), 's')
    # 闭着的眼（向下弯的睫毛线）与微笑
    cv.dots([(48, 12), (49, 13), (50, 13)], 'v')
    cv.dots([(53, 13), (54, 13), (55, 12)], 'v')
    cv.dots([(52, 18)], 'q')
    cv.dots([(51, 17), (53, 18)], 's')
    # 横过脸的一绺刘海（从发旋斜到画面右脸颊）与脸左缘的一小绺
    lock = [(51, 3.5), (53.5, 3.5), (56.5, 7.5), (58.3, 12.5), (58.3, 18.5), (57, 18), (56.4, 14), (55, 10.5),
            (52.5, 7.5), (50.5, 5.5)]
    lk = cv.poly(lock, 'H')
    cv.put(lk & cv.edge(lk, -1, 0), 'h')
    cv.put(cv.ring(lk) & fm & (cv.py < 12), 'v')
    cv.put(cv.ring(lk) & fm & (cv.py >= 12), 's')
    cv.stroke([(48.3, 6), (48.4, 9), (48.1, 10.5)], 'H', 1.6, 1.0)
    # 发旋与发丝
    cv.line([(52, 2), (51, 5), (49.5, 8)], 'h', on='HY')
    cv.line([(54, 2.5), (57.5, 6), (59.5, 11)], 'h', on='HY')
    cv.line([(59.5, 5), (61, 12), (61.5, 17)], 'j', on='HY')


def _fill_holes(cv, P):
    """部件之间漏出的透明小洞：用上 / 左 / 右 / 下邻格的颜色补上（避免描边在身体里画出黑点）。"""
    empty = cv.a == ' '
    out = np.zeros_like(empty)
    out[0], out[-1], out[:, 0], out[:, -1] = empty[0], empty[-1], empty[:, 0], empty[:, -1]
    while True:
        grow = out.copy()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grow |= cv.shift(out, dx, dy)
        grow &= empty
        if (grow == out).all():
            break
        out = grow
    holes = empty & ~out
    while holes.any():
        for dx, dy in ((0, 1), (1, 0), (-1, 0), (0, -1)):
            src = np.roll(np.roll(cv.a, dy, 0), dx, 1)
            m = holes & (src != ' ')
            cv.a[m] = src[m]
            holes &= ~m


def _rim(cv, P):
    """背后逆光：头顶、发顶、肩与披肩上缘 1px 暖金。"""
    body = cv.a != ' '
    top = cv.edge(body, 0, -1) & (cv.py < 34) & ~cv.has('SsqGgb')
    top &= cv.shift(top, 1, 0) | cv.shift(top, -1, 0)      # 只留 ≥2px 的横向一段，避免斜边上的孤立亮点
    cv.put(top, 'Z')


# ----------------------------------------------------------------- 组装
LAYERS = [_cloak, _gown, _ruffles, _feet, _bodice, _tabard, _belt, _hair_back, _arm_l, _shoulders, _shawl, _arm_r,
          _hair_front, _head, _fill_holes, _rim]


def draw(P):
    cv = Cv()
    for f in LAYERS:
        f(cv, P)
    return cv


def body(pose='idle'):
    P = POSES.get(pose, POSES['idle']) if isinstance(pose, str) else pose
    img = draw(P).rgba()
    rows = np.nonzero(img[..., 3].any(1))[0]
    return np.roll(img, H - 1 - rows[-1], 0)      # 让鞋底 / 描边贴住最后一行，方便 anchor='bottom'


# 像素图关键点（从 body('idle') 的渲染上读出；角色自己的左右：_r = 画面左）
LANDMARKS = {
    'idle': {
        'head_top': [53, 2], 'chin': [52, 20], 'neck': [52, 24],
        'shoulder_r': [43, 27], 'shoulder_l': [63, 28],
        'elbow_r': [33, 49], 'elbow_l': [66, 55],
        'wrist_r': [22, 64], 'wrist_l': [68, 76],
        'waist': [52, 58],
        'ankle_r': [41, 138], 'ankle_l': [64, 139],
        'x_choker': [52, 21], 'x_belt_r': [43, 57], 'x_belt_l': [63, 52],
        'x_cloak_arm_r': [30, 44], 'x_cloak_arm_l': [76, 55],
        'x_cloak_tip_r': [11, 139], 'x_cloak_tip_l': [99, 141],
        'x_tabard_tip': [47, 144], 'x_toe_r': [31, 146], 'x_toe_l': [64, 148],
        'x_brooch': [49, 34], 'x_bodice_v': [48, 71],
    },
}

META = dict(
    key='grandmaster',
    name_zh='盟主',
    title_zh='盟主',
    rank_zh='身喰らう蛇 盟主',
    note='身喰らう蛇的盟主：闭目、面纱阴影、淡蓝长发、白披风、深色上衣与金 V 腰带、酒红双蛇杖垂布、'
         '深紫长裙与淡紫荷叶褶。原点：脚底中心（anchor=bottom）。',
    poses=list(POSES),
    fx=dict(primary='#fff0c0', secondary='#bcd4f0'),
    # 以下坐标都是 body(pose) 画布（110x150）里的像素坐标
    eyes_body={'idle': [(49, 13), (54, 13)], 'raise': [(49, 13), (54, 13)]},   # 闭着的双眼
    head_body={'idle': (53, 11), 'raise': (53, 11)},                         # 头部中心（光环 / 头顶光）
    hand_r={'idle': (18, 72), 'raise': (21, 19), 'raise0': (20, 72), 'raise1': (18, 58), 'raise2': (21, 30),
            'raise_hi': (21, 16)},                            # 角色右手（画面左）掌心
    hand_l={'idle': (68, 84), 'raise': (68, 84)},                            # 角色左手（画面右）
    hand_skill=(21, 19),                                                     # 举手施法点（raise）
    height_px=149,
)
