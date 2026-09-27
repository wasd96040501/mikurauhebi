"""苍之深渊 薇塔·克洛蒂尔德（Vita Clotilde）像素立绘。

参考：闪之轨迹 III 立绘（refs/vita/full.png，头像取景用同图裁切的 crop_sen3.png）——头向右肩歪、3/4 侧脸朝画面右、
右手（画面左）垂下持大折扇、左手叉腰；灰褐长发（近侧头顶后方盘发髻，发髻下垂几缕 S 形卷发，发梢渐变成浅蓝，
发髻下插一根小银簪），紫眼、远侧眼下泪痣、两侧耳坠（金色月牙环 + 蓝白扇形垂片）；
蓝色高立领（前面开口）+ 两片翼状大翻领（白色滚边、金色月牙纹、下缘黑色荷叶边），白色连身裙 + 远侧深色侧片，
金色锁链腰带（中间青色宝石），蓝色高开衩外裙（白色蕾丝下摆，斜向画面左拖出长尖角），
黑色长手套（右腕金手镯、左前臂金色螺旋臂环），黑丝袜 + 金色高跟鞋。
技能姿势里另保留闪 II 造型的叉形蓝水晶魔杖（特效锚点要用）。

头像 80x80（v3）：官方头部按 0.13 缩放、不旋转（保留歪头），材质分区烘焙在 PORTRAIT_MAT，
头发 / 脸 / 领子 / 五官按底稿坐标手工重画；tools/measure.py fidelity vita 量保真度。
全身 48x72，约 7.5 头身，72px≈180cm；全身由部件拼成（_hair_back / _head / _torso / _collar / _arm_r / _arm_l /
_legs / _skirt / _fan / _staff），每个部件只吃几个关节坐标，下一轮加动作帧时改关节即可。
"""
import numpy as np

import pixelkit as pk

# ----------------------------------------------------------------- 画布工具


class Cv:
    def __init__(s, h, w, pal):
        s.h, s.w, s.pal = h, w, pal
        s.a = np.full((h, w), ' ', dtype='<U1')
        yy, xx = np.mgrid[0:h, 0:w]
        s.px, s.py = xx + .5, yy + .5

    # ---- 掩码
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

    # ---- 上色
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

    def ell(s, cx, cy, rx, ry, c, **k):
        s.put(s.ellm(cx, cy, rx, ry), c, **k)

    @staticmethod
    def bez(pts, t):
        p = np.array(pts, float)
        while len(p) > 1:
            p = p[:-1] * (1 - t) + p[1:] * t
        return p[0]

    def strokem(s, pts, w0, w1=None, n=40, t0=0.0, t1=1.0):
        w1 = w0 if w1 is None else w1
        ts = np.linspace(t0, t1, n)
        P = np.array([s.bez(pts, t) for t in ts])
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        W = (w0 + (w1 - w0) * ts)[:, None] / 2
        L, R = P + N * W, P - N * W
        return s.polym([tuple(p) for p in np.concatenate([L, R[::-1]])])

    def stroke(s, pts, c, w0, w1=None, **k):
        m = s.strokem(pts, w0, w1)
        s.put(m, c, **k)
        return m

    def linem(s, pts, n=None):
        ln = sum(np.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
        n = n or int(ln * 3) + 2
        m = np.zeros((s.h, s.w), bool)
        for t in np.linspace(0, 1, n):
            x, y = s.bez(pts, t)
            xi, yi = int(np.floor(x)), int(np.floor(y))
            if 0 <= xi < s.w and 0 <= yi < s.h:
                m[yi, xi] = True
        return m

    def line(s, pts, c, n=None, **k):
        """1px 贝塞尔细线（按像素取整采样）。"""
        s.put(s.linem(pts, n), c, **k)

    def dots(s, pts, c, **k):
        m = np.zeros((s.h, s.w), bool)
        for x, y in pts:
            if 0 <= x < s.w and 0 <= y < s.h:
                m[int(y), int(x)] = True
        s.put(m, c, **k)

    def patch(s, x0, y0, rows, flip=False):
        """ASCII 部件贴图（'.' / ' ' 透明）。"""
        x0, y0 = int(round(x0)), int(round(y0))
        for dy, r in enumerate(rows):
            if flip:
                r = r[::-1]
            for dx, ch in enumerate(r):
                if ch not in '. ' and 0 <= y0 + dy < s.h and 0 <= x0 + dx < s.w:
                    s.a[y0 + dy, x0 + dx] = ch

    @staticmethod
    def shift(m, dx, dy):
        o = np.zeros_like(m)
        H, W = m.shape
        ys = slice(max(dy, 0), H + min(dy, 0))
        yd = slice(max(-dy, 0), H + min(-dy, 0))
        xs = slice(max(dx, 0), W + min(dx, 0))
        xd = slice(max(-dx, 0), W + min(-dx, 0))
        o[yd, xd] = m[ys, xs]
        return o

    def edge(s, m, dx, dy):
        """m 中沿 (dx,dy) 方向紧贴边界的一圈像素。"""
        return m & ~s.shift(m, dx, dy)

    def ring(s, m):
        g = m.copy()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            g |= s.shift(m, dx, dy)
        return g & ~m

    def sep(s, m, c='O', on=None):
        """在 m 外侧 1px、且已有颜色的地方描一圈分隔线（选择性内描边）。"""
        r = s.ring(m) & (s.a != ' ')
        s.put(r, c, on=on)

    def cast(s, src, dst, c, dx=1, dy=1):
        """src 区域向 (dx,dy) 投下 1px 阴影到 dst 颜色上。"""
        m = s.shift(s.has(src), -dx, -dy) & s.has(dst)
        s.a[m] = c

    def grad(s, m, bands, axis='y'):
        """按 y（或 x）分段上色：bands=[(上限, 颜色), ...]"""
        v = s.py if axis == 'y' else s.px
        prev = np.zeros_like(m)
        for lim, c in bands:
            mm = m & (v < lim) & ~prev
            s.a[mm] = c
            prev |= mm
        s.a[m & ~prev] = bands[-1][1]

    def rgba(s, outline_col, skip_outline=()):
        img = np.zeros((s.h, s.w, 4), np.float32)
        for ch, col in s.pal.items():
            m = s.a == ch
            if m.any():
                img[m, :3] = pk.hexc(col)
                img[m, 3] = 1
        a = s.a != ' '
        o = s.ring(a)
        img[o, :3] = pk.hexc(outline_col)
        img[o, 3] = 1
        return img


# ----------------------------------------------------------------- 调色板（每种材质 3~4 阶）
OUT = '#170d22'
PAL = {
    'O': OUT,
    # 皮肤
    'T': '#fbeedb', 'S': '#f6dcc4', 's': '#e8b49a', 'q': '#c48a7a',
    # 紫眼
    'W': '#ffffff', 'I': '#c49cf0', 'i': '#7a48c0', 'L': '#24142c',
    # 灰褐长发 + 浅蓝发梢
    'Y': '#b89e9a', 'H': '#877274', 'h': '#615357', 'j': '#433a41', 'J': '#2e272d', 'u': '#9cc0dc', 'v': '#6a8cb4',
    # 蓝（立领 / 外裙）
    'C': '#62c8ee', 'A': '#2a88d2', 'B': '#1760aa', 'b': '#103f82', 'd': '#0b2654',
    # 白（裙身 / 滚边 / 蕾丝）
    'F': '#ffffff', 'f': '#d4d2ea', 'x': '#a29cc2',
    # 黑（手套 / 侧片 / 荷叶边）
    'K': '#6a5c86', 'k': '#40365a', 'n': '#241d34',
    # 黑丝袜
    'M': '#9a8290', 'm': '#6a5466', 'N': '#43344a',
    # 金
    'G': '#f8d67c', 'g': '#c89040', 'e': '#7c5226',
    # 扇面（薄荷 -> 青 -> 蓝）
    'P': '#bdeedc', 'Q': '#72d2ce', 'R': '#48a8e4', 'r': '#2468b8',
    # 发梢渐变（灰 -> 灰蓝 -> 浅蓝）用的中间色
    'p': '#8a92aa',
    # 宝石
    'E': '#62f0e8',
    # 描边：头发 / 皮肤线；紫瞳深色
    'z': '#271a23', 'y': '#8e4a48', 'l': '#5c2e8e',
}


# ================================================================= 头像 80x80
# v3：以闪 III 立绘的头部为底稿（refs/vita/full.png，缩放 0.13、不旋转，下巴落在 (42,52)），
# 保留官方的歪头角度与构图。材质分区（PORTRAIT_MAT）由底稿按材质聚类后清理、烘焙在这里；
# 头发按一缕一缕的发束重画（每缕：亮边 / 中间调 / 暗边 + 背光侧暗线 + 高光），
# 脸、立领翻领、耳坠、五官都按底稿坐标手工放置。光源左上。
# 关键点见文件末尾 LANDMARKS['portrait']（tools/measure.py fidelity vita 量保真度）。

PORTRAIT_MAT = [   # 材质底图：h 发 / s 肤 / b 蓝领 / w 白 / k 黑 / g 金 / . 透明
    '................................................................................',
    '................................................................................',
    '................................................................................',
    '................................................................................',
    '...............................................hhhhh............................',
    '....................................hhhhhhhhhhhhhh.hhh..........................',
    '.................................hhhhhhhhhhhhhhhhhh..hh.........................',
    '................................hhhhhhhhhhhhhhhhhhhhhhhh........................',
    '.........................hhh..hhhhhhhhhhhhhhhhhhhhhhhhhhh.......................',
    '.......................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh......................',
    '.....................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh.....................',
    '....................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh....................',
    '...................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh...................',
    '..................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh..................',
    '..................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh.................',
    '.................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh................',
    '................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh................',
    '................hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh................',
    '...............hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh...............',
    '...............hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh..............',
    '..............hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhsshhhhhhhhhhhhhhhhhhhh..............',
    '..............hhhhhhhhhhhhhhhhhhhhhhhhhhhhssssssssssshhhhhhhhhhhhhh.............',
    '.............hhhhhhhhhhhhhhhhhhhhhhhhhhhssssssssssssssssshhhhhhhhhh.............',
    '.............hhhhhhhhhhhhhhhhhhhhhhhhhhhsssssssssssssssssshhhhhhhhh.............',
    '............hhhhhhhhhhhhhhhhhhhhhhhhhhhsssssssssssssssssssshhhhhhhhh............',
    '.............hhhhhhhhhhhhhhhhhhhhhhhhhhssssssssssssssssssssshhhhhhhh............',
    '.............hhhhhhhhhhhhhhhhhhhhhhhhhsssssssssssssssssssssshhhhhhhh............',
    '.............hhhhhhhhhhhhhhhhhhhhhhhhhsssssssssssssssssssssshhhhhhhh............',
    '..............hhhhhhhhhhhhsshhhhhhhhhhsssssssssssssssssssssshhhhhhhh............',
    '..............hhhhhhhhhhhssshhhhhhhhhhsssssssssssssssssssssshhhhhhhh............',
    '..............hhhhhhhhhhhssshhhhhhhhhhsssssssssssssssssssssshhhhhhhh............',
    '..............hhhhhhhhhhhsshhhhhhhhhhhsssssssssssssssssssssshhhhhhhh............',
    '...............hhhhhhhhhhsshhhhhhhhhhhsssssssssssssssssssssshhhhhhhh............',
    '................hhhhhhhhhsshhhhhhhhhhhsssssssssssssssssssssshhhhhhhhh...........',
    '................hhhhhhhhhsshhhhhhhhhhhsssssssssssssssssssssshhhhhhhhh...........',
    '...............hhhhhhhhhhhhhhhhhhhhhhhssssssssssssssssssssshhhhhhhhhh...........',
    '..............hhhhhhhhhhhhhhhhhhhhhhhhssssssssssssssssssshhhhhhhhhhhh...........',
    '.........hhhhhhhhhhhhhhhhhhhhhhhhhhhhhssssssssssssssssssshhhhhhhhhhhh...........',
    '..........hhhhhhhhhhhhhhhhhhhhhhhhhhhhsssssssssssssssssssshhhhhhhhhhh...........',
    '...........hhhhhhhhhhhhhhhhhhhhhhhhhhhsssssssssssssssssssshhhhhhhhhhh...........',
    '.............hhhhhhhhhhhhhhhhhhhhhhhhhsssssssssssssssssssshhhhhhhhhhh...........',
    '............hhhhhhhhhhhhhhhhhhhhhhhhssssssssssssssssssssshhhhhhhhhhhh...........',
    '...........hhhhhhhhhhhhhhhhhhhhhhhhhssssssssssssssssssssshhhhhhhhhhhh...........',
    '...........hhhhhhhhhhhhhhhhhhhhhhhhhsssssssssssssssssssshhhhhhhhhhhhh...........',
    '.............hhhhhhhhhhhhhhhhhhhhhhssssssssssssssssssssshhhhhhhhhhhhh...........',
    '................hhhhhhhhhhhhhhhhhhhssssssssssssssssssssshhhhhhhhhhhhh...........',
    '................hhhhhhhhhhhhhhhhhhhhsssssssssssssssssssshhhhhhhhhhhhh...........',
    '...............hhhhhhhhhhhhhhhhhhhhhsssssssssssssssssssshhhhhhhhhhhhh...........',
    '...............hhhhhhhhhhhhhhhhhhssssssssssssssssssssssshhhhhhhhhhhhh...........',
    '...............hhhhhhhhhhhhhhhhhsssssssssssssssssshhhhsshhhhhhhhhhhhh...........',
    '..............hhhhhhhhhhhhhhhhhkkhhsssssssssssshhhhhhhsshhhhhhhhhhhhh...........',
    '..............hhhhhhhhhhhhhhhbbbbbkhhssssssshhhhhhhhhhkkhhhhhhhhhhhhh...........',
    '.............hhhhhhhhhhhhhhhhbbbbbbbhkhsshhkhhhhhhhhhhkkhhhhhhhhhhhhh...........',
    '.............hhhhhhhhhhhhhhhhbbbbbbbbbhkhhhkhhhkkkkhhhkkhhhhhhhhhhhhh...........',
    '.............hhhhhhhhhhhhhhhbbbbbbbbbbbbhhkbhhhkkkkhhhkkhhhhhhhhhhhhh...........',
    '............hhhhhhhhhhhhhhhhbbbbbbbbbbbbbkkbhhkkkkkhhhhkhhhhhhhhhhhhh...........',
    '............hhhhhhhhhhhhhhhbbbbbbbbbbbbbbkkhhkkkkkkhhhhkhkhhhhhhhhhhh...........',
    '...........hhhhhhhhhhhhhhhbbbbbbbbbbbbbbbkbhhkkkkkkhhhhkhkhhhhhhhhhhh...........',
    '...........hhhhhhhhhhhhhbbbbbbbbbbbbbbbbbkkkbbkhhhhhsshhhkhhhhhhhhhhh...........',
    '...........hhhhhhhhhhhhhbbbbbbbbbbbbbbbbbkkkkbbbbkhhsshhskkhhhhhhhhhh...........',
    '..........hhhhsssssssshhhbbbbbbbbbbbbbbbbbkkkkbbbbhkssshssshhhhhhhhhh...........',
    '.........hhhhssssssssssskkbbbbbbbbbbbbbbbbkkkkbbbbkkhsshsssshhhhhhhhh...........',
    '.........hhhsssssssssssshhhbbbbbbbbbbbbbbbkkkbbbbbbkhhshhssshhhhhhhhh...........',
    '.........hhsssssssssssssshhhbbbbbbbbbbbbbbkkbbbbbbbbkksshssshhhhhhhhh...........',
    '........hhhsssssssssssssshhhbbbbbbbbbbbbbbkkbbbbbbbbkkhshhsshhhhhhhhh...........',
    '........hhsssssssssssssssshhbbbbbbbbbbbbbbbbbbbbbbbbbkksshsshhhhhhhhh...........',
    '.......hhhssssssssssssssssshhbbbbbbbbbbbbbbbbbbbbbbbbbksshhhhhhhhsshhh..........',
    '.......hhhssssssssssssssssshhbbbbbbbbbbbbbbbbbbbbbbbbbkhsshhhhhhhsshhhhh........',
    '......hhhhssssssssssssssssshhkbbbbbbbbbbbbbbbbbbbbbbbbbksshhhhhhhhsshhhhhhw.....',
    '......hhhssssssssssssssssssshkbbbbbbbbbbbbbbbbbbbbbbbbbwssshhhhhhhsshhhhhhkhhw..',
    '......hhhssssssssssssssssssshhbbbbbbbbbbbbbbbbbbbbbbbbbbbssshhhhhhsshhhhhhkkkhhw',
    '.....hhhhssssssssssssssssssshhbbbbbbbbbbbbbbbbbbbbbbbbbbbbbshhhhhhhhhhhhhhkkkkkh',
    '.....kkhhsssssssssssssssssssshkbbbbbbbbbbbbbbbbbbbbbbbbbbbbbhhhhhhhhhhhhhhkkkkkk',
    '....hkkhhsssssssssssssssssssshhbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbhhhhhhhhhhhhhkkkkkk',
    '....hkhhhsssssssssssssssssssshhbbbbbbbbbbbbbbbbbkkkkkbbbbbbbbbhhhhhhhhhhhhkkkkkk',
    '...hkkhhhssssssssssssssssssssshbbbbbbbbbbbbbbbbbkkkhhhhkbbbbbbbhhhhhhhhhhhkkkkkk',
    '...kkhhhhssssssssssssssssssssshhbbbbbbbbbbbbbbbbksshhhhhhhbbbbbhhhhhhhhhhhkkkkkk',
    '..hhhhhhhhsssssssssssssssssssshhhbbbbbbbbbbbbbbbhssshhhhhhhkbbbbbbhhhhhhhhhkkkkk',
    '..hhhh.hhhssssssssssssssssshhhhhhhhbbbbbbbbbbbbbkssshkkkghhhhbbbbbhhhhhhhhhhkkkk',
    '.hhhhh.hhhssssssssssssssssshhhggshhhbbbbbbbbbbbbksshhkksghhhkhbbbbkwwkhhhhhhkkkk',
]


def _lock(cv, pts, w0, w1=None, tones='YHh', line='j', hl=None, clip=None, light=(-0.6, -0.8), n=64,
          taper=(0.0, 0.3)):
    """一大缕头发：沿贝塞尔中线的带状区域，两端收尖（taper = 起 / 止两端收尖所占的比例）。
    横向坐标 v（-1..1，+1 = 迎光侧）：v 大 -> tones[0] 亮，中间 tones[1] 基色，背光侧 tones[2] 暗，
    背光侧最外一圈描 line（与下一缕之间的深缝）。hl=(t0, t1) 时在这段迎光侧点一道高光。返回掩码。"""
    w1 = w0 if w1 is None else w1
    ts = np.linspace(0, 1, n)
    P = np.array([cv.bez(pts, t) for t in ts])
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    Nn = np.stack([-T[:, 1], T[:, 0]], 1)
    lit = np.sign(Nn @ np.array(light)) + (Nn @ np.array(light) == 0)
    W = (w0 + (w1 - w0) * ts)
    if taper[0] > 0:
        W = W * np.clip(0.3 + 0.7 * ts / taper[0], 0.3, 1)
    if taper[1] > 0:
        W = W * np.clip((1 - ts) / taper[1], 0.12, 1) ** 0.8
    Lf, Rt = P + Nn * W[:, None] / 2, P - Nn * W[:, None] / 2
    m = cv.polym([tuple(q) for q in np.concatenate([Lf, Rt[::-1]])])
    if clip is not None:
        m &= clip
    ys, xs = np.nonzero(m)
    if not len(xs):
        return m
    Q = np.stack([xs + .5, ys + .5], 1)
    j = ((Q[:, None, :] - P[None]) ** 2).sum(-1).argmin(1)
    v = ((Q - P[j]) * Nn[j]).sum(1) * lit[j] / np.maximum(W[j] / 2, 0.5)
    cv.a[ys, xs] = np.where(v > 0.55, tones[0], np.where(v < -0.25, tones[2], tones[1]))
    if line:
        sel = v < -0.78
        cv.a[ys[sel], xs[sel]] = line
    if hl is not None:
        sel = (ts[j] >= hl[0]) & (ts[j] <= hl[1]) & (v > 0.1) & (v < 0.7)
        cv.a[ys[sel], xs[sel]] = 'Y'
    return m


def _p_base(cv):
    mat = np.array([list(r) for r in PORTRAIT_MAT])
    cv.mat = mat
    fill = {'h': 'h', 's': 'S', 'b': 'B', 'w': 'F', 'k': 'k', 'g': 'G'}
    for k, c in fill.items():
        cv.a[mat == k] = c
    return mat


def _p_hair(cv, mat):
    """头发：4 阶（高光 Y / 基色 H / 暗 h / 深暗 j），按大块发束画：
    头顶迎光一侧以 Y/H 为主、一道顺头型的宽高光带；发髻圆顶迎光；近侧 4 个胖的 S 形卷（上缘亮、下侧暗，卷与卷之间留空）；
    远侧长发 2~3 缕宽发束、柔和过渡；贴脸的鬓发各是一缕干净收尖的发束。发束内部不画 1px 细纹。"""
    H = mat == 'h'
    P = lambda pts, c: cv.put(H & cv.polym(pts), c)
    cv.put(H, 'H')
    # ---- 大块暗部：远侧外缘、发髻下、耳后、脖子后、近侧身后长发
    P([(62, 19), (72, 19), (72, 80), (61, 80)], 'h')
    P([(64.5, 50), (72, 50), (72, 80), (64, 80)], 'j')
    P([(46, 46), (57, 45), (58, 58), (46, 58)], 'j')                                         # 远侧脖子后
    P([(14.5, 19.5), (22, 22), (30.5, 19), (31, 25.5), (22, 26.5), (14.5, 24.5)], 'h')      # 发髻下
    P([(20, 27), (29.5, 28.5), (30.5, 46), (24, 47), (20, 40)], 'j')                         # 耳后（卷发后面）
    P([(0, 45), (28, 45), (30, 80), (0, 80)], 'h')
    # ---- 头顶：迎光面 H，一道宽高光带顺着头型；远侧和发际线下转暗；一道柔和的发束分界
    P([(55.5, 12), (62, 16), (64, 22), (57.5, 21.5)], 'h')
    P([(36.5, 22.4), (42, 18.8), (49, 17.2), (56, 18.2), (57.5, 21), (52, 20), (45.5, 20.8), (40, 23)], 'h')
    P([(39, 22.4), (44, 20.2), (50, 19.4), (55.5, 20.2), (51, 20.4), (45, 21.2)], 'j')
    band = cv.polym([(33.5, 11.6), (37, 8.6), (42, 6.6), (48, 6.2), (53, 7.4), (56.5, 9.8), (52.5, 10.2),
                     (47.5, 9.4), (42.5, 9.8), (38, 11.6), (35, 13.6)])
    cv.put(H & band, 'Y')
    cv.put(H & cv.polym([(35, 13.6), (38, 11.6), (42.5, 9.8), (47.5, 9.4), (52.5, 10.2), (51, 11.4), (46, 11.2),
                         (41, 12.4), (36.5, 15)]), 'H')
    cv.put(H & cv.linem([(56.5, 17.2), (50, 13.4), (42, 13.6), (35.5, 16.6)]), 'h')
    cv.put(H & cv.linem([(52.5, 15.5), (48, 18.6), (44.5, 20.6)]), 'j')                    # 分线
    # ---- 发髻：圆顶迎光（Y 顶 + H），下半与贴头一侧转暗，底部一道深缝
    bun = H & cv.ellm(23.5, 14.5, 8.6, 6.8)
    cv.put(bun, 'H')
    cv.put(bun & cv.ellm(21.5, 11.6, 6.2, 3.2), 'Y')
    cv.put(bun & cv.ellm(21.5, 11.2, 7.6, 4.6) & ~cv.ellm(21.5, 11.6, 6.2, 3.2) & (cv.py < 11), 'H')
    cv.put(bun & ((cv.py > 17.5) | (cv.px > 29)), 'h')
    cv.put(bun & cv.polym([(20, 19.4), (24, 18.6), (28.5, 19), (27.2, 21), (21, 21.2)]), 'j')
    cv.put(H & cv.linem([(31.6, 8.8), (30.8, 11), (30.4, 14.5), (31, 17)]), 'j')
    # ---- 远侧长发：两缕宽发束（内缕迎光、外缘转暗），一道分界
    _lock(cv, [(56.2, 17), (59.6, 27), (60.2, 46), (61, 80)], 5.2, 4.6, 'HHh', None, (0.05, 0.3), clip=H, taper=(0, 0))
    cv.put(H & cv.linem([(62.2, 20), (63.4, 34), (63.6, 56), (64.2, 80)]), 'j')
    # ---- 贴脸的鬓发（远侧，画面右）：一缕收尖、迎光边一道高光
    fr = _lock(cv, [(56.5, 18.6), (58.6, 27), (58.4, 38), (56.2, 46.5)], 4.2, 3.2, 'HHh', None, None, clip=H,
               taper=(0, 0.3))
    cv.put(fr & cv.edge(fr, -1, 0) & (cv.py > 21) & (cv.py < 40), 'Y')
    # ---- 近侧脸旁的长鬓发：一缕干净收尖，迎光边高光
    nl = _lock(cv, [(37.4, 16), (32.6, 22), (31, 31), (31.8, 40), (34.8, 47.8)], 6.4, 3, 'HHh', 'h', None, clip=H)
    cv.put(nl & cv.edge(nl, -1, 0) & (cv.py > 18) & (cv.py < 36), 'Y')
    _lock(cv, [(39.6, 19), (37.6, 24), (37.2, 31), (37.8, 38), (36.6, 43)], 2.8, 2, 'HHh', None, None, clip=H,
          taper=(0, 0.4))
    # ---- 近侧 S 形卷：卷发区底色压成暗部，上面叠 4 个收尖的 S 形卷（上缘亮 + 外弯高光、下侧暗），
    #      卷与卷之间的深缝用 j 小块表现，外轮廓的缺口保留原画剪影里的空隙
    P([(6, 23), (23, 23), (24, 49), (6, 49)], 'h')
    P([(12.5, 36.2), (17, 34.6), (17.4, 37.4), (13, 39.4)], 'j')      # 卷 1 与卷 2 之间
    P([(12, 42.6), (16.4, 40.6), (18.6, 43.6), (14.6, 45.4)], 'j')    # 卷 2 与卷 3 之间
    P([(15.6, 24.6), (21, 23.8), (18.4, 27), (14.4, 28)], 'j')        # 发髻下、卷 1 起点
    for pts, w0, w1, hl in (
            ([(22.6, 23.4), (16.6, 25.4), (13.4, 29.6), (15.8, 33.2), (8.6, 36.8)], 5, 2.6, (0.02, 0.5)),
            ([(25.6, 28.6), (20.2, 31), (17.8, 35.4), (15.6, 39.6), (10.6, 42.8)], 5, 2.6, (0.05, 0.5)),
            ([(24, 36.4), (22.6, 41), (19.6, 44.6), (15, 47.4)], 4.4, 2.6, (0.1, 0.45)),
            ([(21.5, 45.4), (26, 47.8), (31, 46.8), (33.4, 44.4)], 4, 3, (0.2, 0.6))):
        _lock(cv, pts, w0, w1, 'YHh', None, hl, clip=H, taper=(0, 0.35))


JAW = [(33.5, 41), (36.5, 45.5), (39.5, 49.2), (42, 51.2), (44.5, 50.8), (48, 48.8), (52, 46.3), (55.5, 43.2), (59, 38.5)]


def _p_face(cv, mat):
    """皮肤按官方的暖象牙色：亮 T / 基 S / 影 s / 深影 q 两阶阴影。"""
    Sm = mat == 's'
    top = cv.polym([(30, 14)] + JAW + [(62, 38), (62, 14)])        # 下颌线以上 = 脸
    face = Sm & top & (cv.px > 30)
    hair = mat == 'h'
    near = lambda dx, dy, k=1: face & np.any([cv.shift(hair, dx * i, dy * i) for i in range(1, k + 1)], 0)
    cv.put(face, 'T')
    # 额头：刘海下两道（S -> s），近侧脸颊贴鬓发、远侧脸颊贴发
    cv.put(near(0, 1, 3) & (cv.py < 27), 'S')
    cv.put(near(0, 1, 1) & (cv.py < 27), 's')
    cv.put(near(-1, 0, 2) & (cv.px < 44), 'S')
    cv.put(near(-1, 0, 1) & (cv.px < 44) & (cv.py > 30), 's')
    cv.put(near(1, 0, 3) & (cv.px > 50), 'S')
    cv.put(near(1, 0, 1) & (cv.px > 50), 's')
    cv.put(face & cv.polym([(55, 37), (59, 36), (57, 41.5), (53.5, 45), (51, 46)]), 'S')    # 远侧颊到下颌
    # 脖子（下颌以下在下巴的阴影里；紧贴下颌两像素更深）
    neck = Sm & ~top & (cv.px > 29) & (cv.py < 57)
    cv.put(neck, 's')
    cv.put(neck & (cv.shift(face, 0, 1) | cv.shift(face, 0, 2)), 'q')
    cv.put(neck & cv.polym([(29, 44), (34, 44), (36, 50), (31, 52)]), 'S')
    # 耳朵（发髻卷发与鬓发之间露出）
    ear = cv.polym([(26.6, 25.4), (28.8, 25.2), (29.8, 29), (29.4, 34), (28.2, 37.2), (26.8, 36.4), (26, 31)])
    cv.sep(ear, 'j', on='YHhjJ')
    cv.put(ear, 's')
    cv.put(ear & cv.polym([(27, 26), (28.6, 26), (28.6, 30), (27.2, 31)]), 'S')
    cv.dots([(28, 30), (28, 31), (27, 32)], 'q')


def _p_body(cv, mat):
    """近侧裸肩：球面明暗（左上迎光，贴头发、贴翻领一侧转暗）；远侧肩在阴影里；远侧手套。"""
    Sm = mat == 's'
    sh = Sm & cv.polym([(0, 56), (31, 56), (31, 80), (0, 80)])
    d = np.hypot((cv.px - 13.5) / 10, (cv.py - 69) / 12.5)
    cv.put(sh, 's')
    cv.put(sh & (d < 1.0), 'S')
    cv.put(sh & (d < 0.62), 'T')
    hair = mat == 'h'
    touch = sh & (cv.shift(hair, 1, 0) | cv.shift(hair, 0, 1) | cv.shift(hair, -1, 0))
    cv.put(touch, 'q')
    cv.put(sh & (cv.px > 24 + (cv.py - 58) * 0.3), 's')          # 翻领投下的影
    cv.put(sh & (cv.px > 26 + (cv.py - 58) * 0.3), 'q')
    far = Sm & cv.polym([(45, 56), (64, 56), (64, 80), (45, 80)])
    cv.put(far, 's')
    cv.put(far & cv.polym([(49, 60), (55, 62), (58, 68), (54, 70), (49, 65)]), 'S')
    cv.put(far & (cv.shift(hair, 1, 0) | cv.shift(hair, -1, 0)), 'q')
    glove = cv.has('k') & (cv.px > 66) & (cv.py > 64)
    cv.put(glove, 'k')
    cv.put(glove & cv.edge(glove, 0, -1), 'K')


def _p_collar(cv, mat):
    """高立领（前面开口）+ 两片翼状大翻领：蓝缎、白色滚边、金色月牙纹，外缘黑色荷叶边。"""
    stand_n = cv.polym([(29.5, 51.6), (33, 52.4), (37, 54.6), (38.8, 54.8), (39, 62.2), (33, 62), (29, 59), (27.6, 56.6)])
    stand_f = cv.polym([(41.4, 53.2), (44, 52), (46.8, 53), (47.4, 59), (44.4, 61.2), (41.4, 62.2)])
    gap = cv.polym([(38.8, 54), (41.6, 53.2), (41.6, 62.4), (39, 62.4)])
    wing_n = cv.polym([(21.2, 58.4), (26, 57.2), (29.4, 56.8), (33, 62), (39, 62.2), (41.4, 65.4), (44.2, 70),
                       (46.6, 74), (48.2, 80), (34.4, 80), (31.2, 75), (29.2, 70), (26.6, 64.2), (23.6, 60.6)])
    wing_f = cv.polym([(44.4, 61.2), (47.4, 59), (50.4, 58.4), (54.2, 61), (57.4, 66), (61, 70.6), (63.4, 74),
                       (64.6, 78), (59, 80), (48.2, 80), (46.6, 74), (44.2, 70), (41.4, 65.4)])
    allc = stand_n | stand_f | gap | wing_n | wing_f
    # 黑色荷叶边：翻领外缘（贴着皮肤的一侧）锯齿状，立领上缘一圈
    fr = cv.ring(allc) & (cv.a != ' ') & ~cv.has('YHhjJ')
    fr |= cv.shift(fr, -1, 0) & cv.has('STs') & (cv.py > 57)
    cv.put(fr, 'n')
    cv.put(fr & ((cv.px.astype(int) + cv.py.astype(int)) % 3 == 0) & (cv.py > 57), 'k')
    cv.put(wing_f, 'b')
    cv.put(wing_f & cv.polym([(47.4, 59), (50.4, 58.4), (54.2, 61), (57.4, 66), (56, 68), (51, 63)]), 'B')
    cv.put(wing_f & (cv.py > 76.5), 'd')
    cv.put(wing_n, 'B')
    cv.put(wing_n & cv.polym([(21.2, 58.4), (29.4, 56.8), (33, 62), (36, 63), (30, 64), (25.5, 61)]), 'A')
    cv.put(wing_n & cv.polym([(23, 58.4), (28.4, 57.4), (29.6, 59), (25, 59.6)]), 'C')
    cv.put(wing_n & cv.polym([(35, 70), (44.2, 70), (46.6, 74), (48.2, 80), (38, 80)]), 'b')
    cv.put(stand_n, 'B')
    cv.put(stand_n & (cv.px < 33), 'A')
    cv.put(stand_f, 'b')
    cv.put(stand_f & (cv.px > 45.5), 'd')
    cv.put(gap, 'n')
    # 白色滚边
    F = lambda pts, c='F': cv.put(cv.linem(pts) & allc, c)
    F([(29.6, 52.4), (33, 53.2), (37, 55.4)])
    F([(38.3, 55.4), (38.3, 62)])
    F([(22.4, 59), (25, 60.2), (27.5, 63.5), (29.5, 68), (31.5, 73), (33.5, 77.5), (34.6, 80)])
    F([(39, 62.4), (42.6, 68.6), (45.7, 73.6), (47.6, 79)])
    F([(42.4, 60.6), (42.6, 54)], 'f')
    F([(47.6, 61.4), (50.5, 64.8), (53, 68.6)], 'f')
    F([(50.4, 59.4), (55, 64), (59, 69.4), (62.6, 73.4)])
    # 金色月牙纹
    G = lambda pts, c='G': cv.put(cv.linem(pts) & (wing_n | wing_f), c)
    G([(30.4, 70.4), (31.6, 68.8), (33.4, 68.8)])
    G([(32.4, 71.6), (37, 73.4), (42, 74), (45.4, 76.2)], 'g')
    G([(51.4, 70.4), (53, 68.6), (55, 69.2)])
    G([(53, 71.2), (57, 72.4)], 'g')


EYE_NEAR = [   # 近侧（画面左，大眼）盖章左上角 (38, 26)：外眼角上挑的深色上睫毛线，紫瞳上深下浅 + 1px 高光，
    "L.........",     # 瞳孔右侧露眼白，下睫毛只暗示一下
    "LLLLLLLLL.",
    ".LLllllLLL",
    "..WlWilWW.",
    "..WiIIiW..",
    "...IIII...",
    "....ss....",
]
EYE_NEAR_SMIRK = [row for row in EYE_NEAR]
EYE_NEAR_SMIRK[2] = ".LLLLLLLLL"      # 眼睑压低半格：似笑非笑、意味深长
EYE_FAR = [    # 远侧（画面右，窄）盖章左上角 (51, 33)：外眼角在右
    "..LLLLLL",
    "LLLllLLL",
    "..lWlWW.",
    "..iIiW..",
    "..sIIs..",
]
EYE_FAR_SMIRK = [row for row in EYE_FAR]
EYE_FAR_SMIRK[1] = "LLLLLLLL"
EYE_NEAR_SHUT = [   # 闭眼（歌唱）：向下弯的睫毛弧，外眼角一点上挑
    "..........",
    "..........",
    "L.........",
    ".LL....LL.",
    "...LLLL...",
    "..........",
    "..........",
]
EYE_FAR_SHUT = [
    "........",
    "........",
    "LL...LL.",
    "..LLL...",
    "........",
]


def _p_features(cv, expr):
    shut = expr == 'sing'
    smirk = expr == 'smirk'
    cv.patch(38, 26, EYE_NEAR_SHUT if shut else EYE_NEAR_SMIRK if smirk else EYE_NEAR)
    cv.patch(51, 33, EYE_FAR_SHUT if shut else EYE_FAR_SMIRK if smirk else EYE_FAR)
    # 眉：细长的弧（近侧长，远侧短）
    cv.dots([(40, 24), (41, 23), (42, 23), (43, 22), (44, 22), (45, 22), (46, 23), (47, 23)], 'h')
    cv.dots([(48, 24)], 'H')
    cv.dots([(53, 31), (54, 31), (55, 30), (56, 30), (57, 31)], 'h')
    cv.dots([(56, 40)], 'L')                        # 泪痣（远侧眼下）
    cv.dots([(51, 41)], 's')                        # 鼻（3/4 侧只点一下鼻翼阴影）
    cv.dots([(52, 40)], 'S')
    if shut:
        cv.patch(44, 43, [".qq.", "qyyq", ".qq."])    # 歌唱：张嘴
    elif expr == 'smirk':
        cv.dots([(43, 43), (44, 44), (45, 44), (46, 44), (47, 44)], 'y')     # 单边嘴角上挑
        cv.dots([(48, 43), (49, 42)], 'y')
        cv.dots([(45, 45), (46, 45)], 's')
    else:
        cv.dots([(43, 43)], 'q')
        cv.dots([(44, 44), (45, 44), (46, 44), (47, 44)], 'y')     # 淡淡的笑
        cv.dots([(48, 43)], 'y')
        cv.dots([(45, 45), (46, 45)], 's')
    # 腮红（近侧颊上一点）
    cv.dots([(41, 36)], 'S')


def _p_jewel(cv):
    # 近侧耳坠：金色月牙环 + 蓝白扇形垂片
    cv.dots([(27, 39), (26, 40), (27, 41)], 'G')
    cv.dots([(28, 40)], 'g')
    cv.patch(25, 42, [".FF.", "CAAb", ".Bb."])
    # 远侧耳坠：小扇片 + 金色细条
    cv.patch(51, 47, ["CAb"])
    cv.dots([(52, 48), (52, 49), (52, 50)], 'G')
    # 发髻下插的小银簪
    cv.dots([(21, 27)], 'F')
    cv.dots([(22, 28)], 'f')


def _outline(cv):
    """按材质描外轮廓：头发深棕、皮肤红棕、其余近黑。"""
    img = cv.rgba(OUT)
    a = cv.a != ' '
    ring = cv.ring(a)
    hair = cv.has('YHhjJ')
    skin = cv.has('TSsq')
    near_h = ring & (cv.shift(hair, 1, 0) | cv.shift(hair, -1, 0) | cv.shift(hair, 0, 1) | cv.shift(hair, 0, -1))
    near_s = ring & (cv.shift(skin, 1, 0) | cv.shift(skin, -1, 0) | cv.shift(skin, 0, 1) | cv.shift(skin, 0, -1))
    img[near_s, :3] = pk.hexc(PAL['y'])
    img[near_h, :3] = pk.hexc(PAL['z'])
    return img


def portrait(expr=None):
    cv = Cv(80, 80, PAL)
    mat = _p_base(cv)
    _p_hair(cv, mat)
    _p_body(cv, mat)
    _p_face(cv, mat)
    _p_collar(cv, mat)
    _p_jewel(cv)
    _p_features(cv, expr)
    return _outline(cv)


EXPRS = (None, 'smirk', 'sing')


# ================================================================= 全身 48x72
# 官方立绘按 0.0255 缩放（72px≈180cm；约 7.5 头身，头高≈8px），脚底在最后一行。
# 部件都只吃关节坐标 J（像素坐标，浮点），方便以后做动作帧。

HEAD_TILT = [   # 头向右肩歪（官方立绘的招牌姿态），3/4 侧脸朝画面右；左上是发髻，近侧卷发里露出耳坠
    ".jjj..........",
    "jHYHjjjHYYj...",
    "jHHYHjHHHYHj..",
    ".jHHhjHYHHhHj.",
    ".jhHhHhSTTHHj.",
    ".jHhYhLiTSHhj.",
    ".jhhHhSTLiHhj.",
    ".jHhshSTTSHj..",
    "..jGhhSsSsHj..",
    "..jBhjSsjhhj..",
    "...jhj.j..hj..",
]
HEAD_TILT_SING = [   # 闭眼、张嘴歌唱
    ".jjj..........",
    "jHYHjjjHYYj...",
    "jHHYHjHHHYHj..",
    ".jHHhjHYHHhHj.",
    ".jhHhHhSTTHHj.",
    ".jHhYhSSTSHhj.",
    ".jhhHhLLTTHhj.",
    ".jHhshSTLLHj..",
    "..jGhhSyssHj..",
    "..jBhjSqjhhj..",
    "...jhj.j..hj..",
]
HEAD_FRONT = [   # 转身的中间帧：脸几乎正对镜头
    ".jjj..........",
    "jHYHjjjHYYj...",
    "jHHYHjHHHYHj..",
    ".jHHhjHHHHhHj.",
    ".jhHhHSTTTShj.",
    ".jHhYhSTTTShj.",
    ".jhhHhLiTiLhj.",
    ".jHhshSTTTSj..",
    "..jGhhSTqTsj..",
    "..jBhjhSSshj..",
    "...jhj..j.hj..",
]
HEAD_SMIRK = [row for row in HEAD_TILT]
HEAD_SMIRK[8] = "..jGhhSSqyHj.."      # 嘴角挪到一侧：意味深长的笑
HEAD_ANCHOR = (6, 9)   # 下巴在部件里的 (列, 行)
HEADS = {'tilt': HEAD_TILT, 'sing': HEAD_TILT_SING, 'front': HEAD_FRONT, 'smirk': HEAD_SMIRK,
         'sing_l': HEAD_TILT_SING, 'tilt_l': HEAD_TILT}   # *_l = 镜像（脸朝画面左）


def _head(cv, J, sing=False):
    """头部件。J['head'] 选变体：tilt / smirk / sing / front / sing_l / tilt_l（_l 为镜像，朝画面左）。
    发髻、耳坠都画在章里，镜像时一起翻到另一边。"""
    kind = J.get('head', 'sing' if sing else 'tilt')
    cx, cy = J['chin']
    rows = HEADS[kind]
    mirror = kind.endswith('_l')
    ax = len(rows[0]) - 1 - HEAD_ANCHOR[0] if mirror else HEAD_ANCHOR[0]
    x0, y0 = int(round(cx)) - ax, int(round(cy)) - HEAD_ANCHOR[1]
    cv.patch(x0, y0, rows, flip=mirror)


def _hair_back(cv, J, sway=0.0, lift=0.0):
    """身后长发：近侧（画面左）从头侧披下、在手臂后向左下散开到胯；远侧（画面右）顺背垂下，从叉腰的手臂内外露出。
    发梢渐变成灰蓝 -> 浅蓝（官方立绘的特征）。sway = 发梢整体左右摆动（像素），lift = 发梢上扬（负数 = 向上）。"""
    hx, hy = J['head_top']
    w, L = sway, lift
    tip_bands = lambda y0: [(y0, 'h'), (y0 + 5, 'H'), (y0 + 9, 'p'), (y0 + 13, 'v'), (y0 + 80, 'u')]
    near = cv.polym([(hx - 6, hy + 1), (hx - 9.5, hy + 5), (hx - 11.5, hy + 12), (hx - 14 + w * .3, hy + 19),
                     (hx - 17 + w * .6, hy + 26 + L * .4), (hx - 18.5 + w, hy + 30 + L * .7), (hx - 19.5 + w, hy + 34.5 + L),
                     (hx - 17 + w, hy + 35 + L * .9), (hx - 16 + w * .8, hy + 38 + L), (hx - 13 + w * .6, hy + 32 + L * .5),
                     (hx - 7, hy + 27), (hx - 5, hy + 18), (hx - 4, hy + 10)])
    cv.grad(near, tip_bands(hy + 18))
    for pts in ([(hx - 8, hy + 4), (hx - 9.5, hy + 13), (hx - 12 + w * .4, hy + 22), (hx - 17 + w, hy + 32 + L * .7)],
                [(hx - 7, hy + 10), (hx - 9, hy + 19), (hx - 12 + w * .6, hy + 27 + L * .4), (hx - 15 + w, hy + 34 + L)]):
        cv.line(pts, 'j', on='hH')
        cv.line(pts, 'v', on='pu')
    cv.line([(hx - 9, hy + 6), (hx - 10, hy + 12)], 'Y', on='hH')
    cv.line([(hx - 10.5, hy + 14), (hx - 14 + w * .5, hy + 25 + L * .3), (hx - 19 + w, hy + 34 + L * .9)], 'v', on='pu')
    cv.line([(hx - 11, hy + 21), (hx - 14 + w * .8, hy + 30 + L * .6)], 'j', on='hH')
    _tips(cv, near, hy + 26)
    far = cv.polym([(hx + 1, hy), (hx + 3, hy + 4), (hx + 4, hy + 12), (hx + 6 + w * .3, hy + 20),
                    (hx + 8 + w * .6, hy + 27 + L * .5), (hx + 11 + w, hy + 34 + L), (hx + 8.5 + w, hy + 34.5 + L),
                    (hx + 5.5 + w * .5, hy + 28 + L * .5), (hx + 1, hy + 23), (hx, hy + 12)])
    cv.grad(far, tip_bands(hy + 17))
    cv.line([(hx + 2.5, hy + 3), (hx + 3.2, hy + 12), (hx + 5.5 + w * .3, hy + 21), (hx + 8 + w * .8, hy + 30 + L * .7)], 'j', on='hH')
    cv.line([(hx + 2.5, hy + 13), (hx + 4, hy + 18)], 'Y', on='hH')
    cv.line([(hx + 4.5, hy + 17), (hx + 7 + w * .5, hy + 26 + L * .4), (hx + 10 + w, hy + 33 + L)], 'v', on='pu')
    _tips(cv, far, hy + 24)


def _tips(cv, m, y0):
    """发梢：y0 以下的下缘隔列缺口，散成一缕缕。"""
    bot = cv.edge(m, 0, 1) & (cv.py > y0)
    cut = bot & ((cv.px.astype(int) % 3) == 1)
    cv.a[cut] = ' '
    bot2 = cv.edge(m & ~cut, 0, 1) & (cv.py > y0 + 4) & ((cv.px.astype(int) % 3) == 1)
    cv.a[bot2 & (cv.a == 'u')] = ' '


def _leg(cv, hip, knee, ankle, front=True, toe=(1, 1)):
    """黑丝长腿 + 金色高跟鞋。toe = 脚尖相对脚踝的偏移。"""
    thigh = cv.strokem([hip, knee], 3.4, 2.8)
    shin = cv.strokem([knee, ((knee[0] + ankle[0]) / 2 + 0.3, (knee[1] + ankle[1]) / 2), ankle], 2.6, 1.6)
    m = thigh | shin
    cv.sep(m, 'n', on='ABbdCFfuv')
    cv.put(m, 'm')
    cv.put(cv.edge(m, -1, 0) & (cv.py > knee[1] - 4), 'M' if front else 'm')
    cv.put(cv.edge(m, 1, 0), 'N')
    ax, ay = ankle
    tx, ty = ax + toe[0], ay + toe[1]
    # 鞋：鞋跟 + 鞋面（金），脚踝绑带
    shoe = cv.polym([(ax - 1, ay), (ax + 1, ay), (tx + 1, ty), (tx + 0.6, ty + 1), (ax - 0.4, ty + 1), (ax - 1, ay + 2.6)])
    cv.sep(shoe, 'e', on='mMN')
    cv.put(shoe, 'g')
    cv.put(shoe & (cv.px > ax), 'G')
    cv.put(cv.edge(shoe, 0, 1) & (cv.px > ax), 'g')
    cv.dots([(ax, ay - 1)], 'g')                  # 脚踝绑带


def _skirt(cv, J, flow=0.0, lift=0.0):
    """蓝色外裙：前片从左胯斜拖到左下长尖角（白蕾丝下摆），右片沿开衩垂到右下尖角。
    flow / lift：下摆整体的水平 / 竖直位移（越往下越大），做风吹和漂浮时的二级运动。"""
    def T(pts):
        out = []
        for x, y in pts:
            k = float(np.clip((y - 36) / 28, 0, 1.2)) ** 1.2
            out.append((x + flow * k, y + lift * k))
        return out
    # 前片下方的半透明纱（压在后腿上）
    veil = cv.polym(T([(15, 57), (20, 52), (26, 48.5), (29.5, 48), (29.5, 56), (22, 57.5)]))
    cv.put(veil, 'v')
    cv.grad(veil, [(53, 'v'), (60, 'u')])
    cv.line(T([(16, 57), (22, 57.5), (29.5, 56)]), 'A', on='uv')
    # 右片（画面右）
    rp = cv.polym(T([(34.5, 31), (38.5, 30.5), (39.5, 40), (40.2, 52), (40, 59), (38.6, 63.5), (37.6, 66.5),
                     (37, 60), (37.2, 56), (36, 46), (35, 38)]))
    cv.sep(rp, 'O', on='mMNn')
    cv.put(rp, 'B')
    rx = lambda y: 38.2 + flow * float(np.clip((y - 36) / 28, 0, 1.2)) ** 1.2
    right = cv.px > 38.2 + flow * np.clip((cv.py - 36) / 28, 0, 1.2) ** 1.2
    cv.grad(rp & right, [(46, 'b'), (80, 'd')])
    cv.put(rp & (cv.py > 57 + lift * 0.8) & ~right, 'A')
    cv.line(T([(39, 33), (39.5, 44), (39.6, 58)]), 'G', on='Bbd')
    fr = cv.linem(T([(35, 33), (35.8, 44), (37, 56)])) | cv.linem(T([(35.6, 33), (36.4, 44), (37.6, 56)]))
    cv.put(fr & rp, 'F')
    cv.put(cv.linem(T([(35.6, 36), (36.4, 44), (37.6, 55)])) & rp, 'f')
    # 前片
    fp = cv.polym(T([(25.5, 30.5), (33, 31), (33, 36), (32.3, 40), (30, 45), (27.5, 48.5), (21, 55.5),
                     (16, 60), (12.2, 63.5), (13.5, 59.5), (17, 52), (20, 46), (22.6, 41), (24.2, 37)]))
    cv.sep(fp, 'O', on='mMNnuvABbd')
    cv.put(fp, 'B')
    # 由上到下：深蓝 -> 亮蓝 -> 青
    band = (cv.px - 12 - flow * 0.6) * 0.55 + (cv.py - 30 - lift * 0.6)
    cv.put(fp & (band > 22), 'A')
    cv.put(fp & (band > 27), 'C')
    cv.put(fp & (band < 13), 'b')
    cv.put(fp & (cv.px < 24.2 + (cv.py - 37) * -0.33 + 1.2) & (cv.py < 50), 'd')
    # 金边（前片上缘斜线）+ 白蕾丝下摆
    cv.line([(24.8, 34), (29, 36), (32.6, 36.5)], 'G', on='Bbd')
    hem = cv.linem(T([(32.8, 36), (32.4, 40), (30, 45), (27.4, 48.4), (21, 55.4), (15.8, 59.8), (12.5, 63)]))
    cv.put(hem, 'F')
    cv.put(hem & ((cv.px.astype(int) + cv.py.astype(int)) % 3 == 0) & (cv.py > 44), 'f')   # 蕾丝孔


def _torso(cv, J):
    """白色连身裙上身 + 左侧（画面右）黑色侧片 + 金色锁链腰带（中间青宝石）。"""
    wx, wy = J['waist']
    nx, ny = J['neck']
    # 裸肩（近侧）
    sh = cv.polym([(22.4, 16), (23.4, 13.8), (25.8, 12.8), (28, 13.4), (28, 17.5), (26, 19), (22.8, 18.6)])
    cv.put(sh, 'S')
    cv.put(sh & cv.ellm(24.2, 15, 1.6, 1.8), 'T')
    cv.put(sh & (cv.px > 26.6), 's')
    # 远侧肩
    cv.poly([(33, 13.5), (35.5, 14), (36, 16.5), (33.5, 17)], 'S')
    cv.dots([(35, 16)], 's')
    # 裙身：胸口 -> 细腰 -> 胯
    dress = cv.polym([(26.5, 16.5), (33.5, 16), (35.4, 17.4), (35.8, 21), (34.8, 24.5), (34.3, 28.5), (34.5, 31),
                      (25.8, 31), (26.4, 28), (27.7, 25), (27.5, 22), (26.3, 19)])
    cv.sep(dress, 'x', on='SsT')
    cv.put(dress, 'F')
    side = dress & (cv.px > 32.4 - (cv.py < 22) * 0.6)
    cv.put(side, 'k')
    cv.put(side & cv.edge(side, -1, 0), 'K')
    cv.line([(31.4, 17), (31.3, 22), (31.2, 25), (31.8, 30)], 'G', on='FfkK')
    cv.put(dress & (cv.px < 27.8) & (cv.py > 21), 'f')
    cv.dots([(28, 19), (29, 20)], 'f')                 # 胸下阴影
    # 胯下露出的白裙（前片左上）
    cv.poly([(25.5, 30.5), (29, 31), (29, 34.5), (25, 36.5)], 'F', on='Bbd ')
    cv.poly([(25.2, 34), (27.5, 34), (26.5, 37), (24.8, 37)], 'f', on='BbdF ')
    # 锁链腰带：从左腰斜到右胯，中间方形宝石扣
    belt = cv.linem([(25.8, 27.8), (29, 29.2), (33, 30.3), (36.5, 30.4), (38.8, 30)])
    cv.put(belt, 'G')
    cv.put(cv.shift(belt, 0, -1) & ~belt & (cv.a != ' '), 'g')
    gx, gy = J['belt_gem']
    cv.patch(gx - 1, gy - 1, ["ggg", "gEg", "ggg"])
    cv.dots([(gx - 2, gy), (gx + 2, gy)], 'G')


def _collar(cv, J):
    """蓝色高立领（下巴下）+ 两片向两肩张开的翼状翻领（白色滚边、金色纹），尖端在胸口，下缘黑色荷叶边。"""
    col = cv.polym([(25, 13.2), (27.6, 11.8), (31.2, 11.2), (33, 12.6), (36.2, 14.8), (35, 16.6),
                    (31.4, 19.8), (29.4, 17.8), (26.8, 15.4)])
    cv.sep(col, 'k', on='SsTFfx')
    cv.put(col, 'B')
    cv.put(col & (cv.px < 29) & (cv.py > 12.8), 'A')
    cv.put(col & (cv.px < 27.4) & (cv.py > 13) & (cv.py < 14.6), 'C')
    cv.put(col & (cv.px > 33) & (cv.py > 14.5), 'b')
    cv.put(col & (cv.py < 12.6), 'b')
    cv.line([(25.6, 13.8), (28.2, 16), (30.8, 19)], 'F', on='ABbC')
    cv.line([(35.6, 15.4), (33, 17.4), (31.8, 18.8)], 'f', on='ABbC')
    cv.dots([(30, 12), (30, 13), (30, 14)], 'd')                 # 前襟开口
    cv.dots([(28, 15), (33, 15)], 'g')                            # 金纹


def _arm_r(cv, J):
    """右臂（画面左）自然下垂，黑长手套从上臂开始，腕上金手镯；手握扇轴。"""
    sh, el, wr = J['shoulder_r'], J['elbow_r'], J['wrist_r']
    up = cv.strokem([sh, el], 3.0, 2.8)
    lo = cv.strokem([el, wr], 2.8, 2.2)
    m = up | lo
    cv.sep(m, 'O', on='hHjJpuvABbdSsT')
    cv.put(m, 'k')
    top = cv.polym([(sh[0] - 3, sh[1] - 3), (sh[0] + 3, sh[1] - 3), (sh[0] + 3, sh[1] + 3.4), (sh[0] - 3, sh[1] + 4.0)])
    cv.put(m & top, 'S')
    cv.put(m & top & cv.edge(m, 1, 0), 's')
    cv.put(m & ~top & cv.edge(m, -1, 0), 'K')
    cv.put(m & ~top & cv.edge(m, 1, 0), 'n')
    cv.put(m & cv.ring(top) & ~top, 'K')     # 手套上缘的荷叶边
    # 金手镯 + 青宝石
    bx, by = wr[0] + 0.3, wr[1] - 1.6
    cv.dots([(bx - 1, by), (bx + 1, by)], 'G')
    cv.dots([(bx, by)], 'E')
    cv.dots([(bx - 1, by + 1), (bx, by + 1), (bx + 1, by + 1)], 'g')


def _hand_r(cv, J):
    wx, wy = J['wrist_r']
    cv.patch(wx - 2, wy, [".kK", "kKk", "Kkn", ".n."])


def _arm_l(cv, J):
    """左臂（画面右）叉腰：肩 -> 向外的肘 -> 手按在腰带上；前臂缠金色螺旋臂环。"""
    sh, el, wr, hd = J['shoulder_l'], J['elbow_l'], J['wrist_l'], J['hand_l']
    up = cv.strokem([sh, el], 2.8, 2.6)
    lo = cv.strokem([el, wr], 2.6, 2.2)
    hand = cv.strokem([wr, hd], 2.2, 2.0)
    m = up | lo | hand
    cv.sep(m, 'O', on='hHjJpuvABbdSsT')
    cv.put(m, 'k')
    cv.put(cv.edge(m, 0, -1), 'K')
    cv.put(cv.edge(m, 0, 1) & ~hand, 'n')
    top = cv.linem([(sh[0] - 0.5, sh[1] - 1.2), (sh[0] + 0.5, sh[1] + 1.4)])
    cv.put(top & m, 'K')
    # 螺旋臂环：前臂上三道金斜线
    for t in (0.2, 0.45, 0.7):
        px, py = el[0] + (wr[0] - el[0]) * t, el[1] + (wr[1] - el[1]) * t
        cv.put(cv.linem([(px - 1.4, py - 0.6), (px + 1.4, py + 0.6)]) & m, 'G')
    cv.dots([(hd[0] - 1, hd[1])], 'K')


def _fan(cv, pivot, a0, a1, r0=4.0, r1=16.5):
    """大折扇（闪 III 立绘）：扇心在 pivot，从角度 a0 到 a1（度，y 向下）。
    由内向外：白色扇骨区（深蓝锯齿纹）→ 扇面（a0 一侧深蓝，渐变到 a1 一侧的薄荷绿，细深色扇骨线）
    → 深蓝镂空蕾丝宽边（浅色水滴孔）→ 扇贝形外缘。张角很小时画成合起的扇子。"""
    px, py = pivot
    dx, dy = cv.px - px, cv.py - py
    r = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 360
    lo, hi = a0 % 360, a1 % 360
    span = (hi - lo) % 360
    rel = (ang - lo) % 360
    t = rel / max(span, 1e-6)                          # 0 = a0 边，1 = a1 边
    sector = (rel <= span) & (r <= r1) & (r >= 0.8)
    n_sc = max(2, int(round(span / 11)))               # 扇贝个数
    ph = (t * n_sc) % 1
    sector &= ~((r > r1 - 0.7) & ((ph < 0.2) | (ph > 0.8)))
    cv.sep(sector, 'n', on='hHjJYuvABbdCFfkKnSsTmMNxGg')
    if span < 26:                                      # 合起的扇子：一根扇形长条
        cv.put(sector, 'R')
        cv.put(sector & (t > 0.5), 'Q')
        cv.put(sector & (r > r1 - 2.2), 'd')
        cv.put(sector & (r < r0 * 0.6), 'f')
        return
    lace = sector & (r > r1 - 2.6)
    face = sector & (r >= r0) & ~lace
    hub = sector & (r < r0)
    cv.put(face, 'r')
    cv.put(face & (t > 0.12), 'R')
    cv.put(face & (t > 0.38), 'Q')
    cv.put(face & (t > 0.64), 'P')
    # 扇心的白色扇骨区 + 深蓝锯齿
    inner = sector & (r < r0 + 2.2)
    cv.put(inner, 'f')
    cv.put(inner & (r > r0 - 0.5) & (np.floor(t * n_sc * 2) % 2 == 0), 'r')
    cv.put(hub & (r < 2.2), 'x')
    # 扇骨线
    n_rib = max(3, int(round(span / 18)))
    for k in range(1, n_rib):
        a = np.radians(lo + span * k / n_rib)
        cv.put(cv.linem([(px + np.cos(a) * (r0 + 2), py + np.sin(a) * (r0 + 2)),
                         (px + np.cos(a) * (r1 - 2.7), py + np.sin(a) * (r1 - 2.7))]) & face & ~cv.has('r'),
               'R' if k / n_rib > 0.5 else 'r')
    # 镂空蕾丝：深蓝底，每个扇贝一颗浅色水滴孔 + 外缘一圈浅色细边
    cv.put(lace, 'r')
    cv.put(lace & (r > r1 - 1.0), 'd')
    holes = lace & (np.abs(r - (r1 - 1.5)) < 0.6) & (np.abs(ph - 0.5) < 0.22)
    cv.put(holes, 'P')


# 魔杖（闪 II 起的杖）：金色杖身 + 叉形蓝水晶杖头（中间长刃深蓝，两侧青、最外黄绿）
STAFF_HEAD = [
    "V...A...V",
    "VC.AAB.CV",
    ".VCABbCV.",
    "..CABbC..",
    "...ABb...",
    "...GEG...",
    "....g....",
]


def _staff(cv, x, top, bottom):
    """竖直魔杖：x 为杖身列，top 为杖头顶端 y，bottom 为杖尾 y。返回杖尖坐标 (y, x)。"""
    cv.put(cv.polym([(x, top + 6), (x + 1, top + 6), (x + 1, bottom), (x, bottom)]), 'G')
    for y in range(int(top) + 9, int(bottom), 4):
        cv.dots([(x, y)], 'g')                      # 杖身上缠的螺旋纹
    cv.dots([(x, bottom - 1)], 'e')
    cv.patch(x - 4, top, STAFF_HEAD)
    return (int(top), int(x))


PAL['V'] = '#c4e878'   # 杖头最外侧的黄绿水晶


# ---------------------------------------------------------------- 姿势（关节坐标）
POSES = {
    'idle': dict(   # 官方闪 III 立绘：歪头、右手垂下持扇、左手叉腰、左腿在前
        head_top=(33, 3), chin=(29, 11), neck=(29.7, 15.3),
        shoulder_r=(24, 15), shoulder_l=(34.7, 15.3), elbow_r=(23, 24.6), elbow_l=(41.9, 17.8),
        wrist_r=(21, 33), wrist_l=(38, 25.4), hand_l=(34.8, 27.4),
        waist=(30.4, 24.6), belt_gem=(33, 30), crotch=(33.5, 40),
        hip_r=(31, 41), hip_l=(33.4, 40.5), knee_r=(28.6, 52.3), knee_l=(32, 50.6),
        ankle_r=(28.4, 66.2), ankle_l=(33.6, 67.2),
        fan=dict(pivot=(20, 35.5), a0=105, a1=208), sway=0.0, flow=0.0,
    ),
}
POSES['skill'] = dict(   # 深渊的苍之歌：闭眼高歌；左手握住高举离地的魔杖，右手把扇子扬向左上方
    POSES['idle'],
    shoulder_r=(24, 15), elbow_r=(19.5, 20), wrist_r=(15.5, 24),
    shoulder_l=(34.7, 15.3), elbow_l=(41.3, 20), wrist_l=(42.4, 12.2), hand_l=(42.6, 10.2),
    fan=dict(pivot=(15, 24.5), a0=200, a1=292, r1=15), sway=-1.0, flow=-1.5, sing=True,
    staff=dict(x=43, top=0, bottom=44),
)


# ---- 歌剧院一幕（scenes/vita.py）用的姿势。in-between 用 lerp_pose 在两个关键姿势之间插值。
POSES['sing'] = dict(   # 登台高歌：脸朝画面左上（观众席方向），右臂向外扬起、合起的扇子指向远处，左手按在胸口
    POSES['idle'], head='sing_l',
    shoulder_r=(24, 15), elbow_r=(18.5, 18.5), wrist_r=(13, 19.5),
    elbow_l=(37.6, 22.2), wrist_l=(34.2, 20.6), hand_l=(31.8, 18.8),
    fan=dict(pivot=(12.5, 20), a0=193, a1=201, r1=9), sway=0.5, flow=0.0,
)
POSES['sing_b'] = dict(   # 换气：手臂再抬高一点，头更仰
    POSES['sing'],
    elbow_r=(18.2, 17.4), wrist_r=(12.4, 16.8), hand_l=(32.2, 19.6),
    fan=dict(pivot=(12, 17.2), a0=203, a1=211, r1=9), sway=-0.5,
)
POSES['idle_smirk'] = dict(POSES['idle'], head='smirk')
POSES['raise_a'] = dict(   # 蓄力 1：魔杖在左手里现形（握在低处），右手把展开的扇子甩到身后
    POSES['idle'], head='tilt',
    elbow_r=(20.5, 22.5), wrist_r=(16.5, 28.5),
    elbow_l=(40.6, 21.2), wrist_l=(41.8, 17.2), hand_l=(42.5, 15.6),
    fan=dict(pivot=(16, 29.5), a0=112, a1=204, r1=14), sway=0.8, flow=0.6,
    staff=dict(x=43, top=7, bottom=51),
)
POSES['raise'] = dict(   # 蓄力 2：魔杖举高、闭眼吸气，扇子收到身侧
    POSES['raise_a'], head='sing',
    elbow_l=(41, 20.6), wrist_l=(42.3, 14.5), hand_l=(42.6, 12.6),
    elbow_r=(21.5, 23), wrist_r=(19, 29), fan=dict(pivot=(18.5, 30), a0=118, a1=190, r1=13),
    sway=1.2, flow=0.8, staff=dict(x=43, top=3, bottom=47),
)
POSES['cast'] = dict(   # 命中：魔杖顶到最高、扇子向左上扬开，长发与裙摆被冲击吹起
    POSES['skill'], head='sing',
    sway=-3.0, hair_lift=-3.0, flow=-3.0, lift=-2.0,
)
POSES['cast_b'] = dict(   # 余势：扇子继续向下扫
    POSES['cast'],
    elbow_r=(19, 21), wrist_r=(14, 25.5), fan=dict(pivot=(13.5, 26), a0=160, a1=262, r1=15),
    sway=-2.0, hair_lift=-2.0, flow=-2.0, lift=-1.5,
)
POSES['float'] = dict(   # 漂浮：足尖下垂、前腿微屈，扇子展开在身侧，魔杖竖在左手
    POSES['idle'], head='tilt',
    elbow_r=(19.5, 22), wrist_r=(15, 27), fan=dict(pivot=(14.5, 28), a0=122, a1=214, r1=14),
    elbow_l=(40.2, 21.4), wrist_l=(42, 17.6), hand_l=(42.6, 15.8),
    staff=dict(x=43, top=4, bottom=48),
    knee_l=(31.2, 49.4), ankle_l=(33.2, 60.6), toe_l=(0.2, 3.2),
    knee_r=(28.8, 52.2), ankle_r=(29.4, 64.8), toe_r=(0.8, 3.0),
    sway=-1.5, hair_lift=-3.0, flow=-1.0, lift=-3.0,
)
POSES['float_b'] = dict(   # 漂浮的呼吸帧
    POSES['float'],
    elbow_r=(19.2, 21.2), wrist_r=(14.6, 25.8), fan=dict(pivot=(14, 26.8), a0=126, a1=218, r1=14),
    sway=-0.5, hair_lift=-2.0, flow=0.0, lift=-2.0,
)

SCENE_POSES = ('sing', 'sing_b', 'idle_smirk', 'raise_a', 'raise', 'cast', 'cast_b', 'float', 'float_b')


def lerp_pose(a, b, t):
    """两个姿势（名字或关节 dict）之间的 in-between。数值逐项插值；非数值项（头部变体等）过半时切换。"""
    A = POSES[a] if isinstance(a, str) else a
    B = POSES[b] if isinstance(b, str) else b
    if A is None or B is None:
        return A if B is None else B
    t = float(np.clip(t, 0, 1))
    out = {}
    for k in set(A) | set(B):
        va, vb = A.get(k), B.get(k)
        if va is None or vb is None:
            out[k] = vb if (t >= 0.5 and vb is not None) or va is None else va
        elif isinstance(va, dict):
            out[k] = lerp_pose(va, vb, t)
        elif isinstance(va, tuple):
            out[k] = tuple(x + (y - x) * t for x, y in zip(va, vb))
        elif isinstance(va, (int, float)) and not isinstance(va, bool):
            out[k] = va + (vb - va) * t
        else:
            out[k] = vb if t >= 0.5 else va
    return out


def anchors(J):
    """某个姿势里各特效发射点的精灵坐标 (x, y)：杖尖、杖头宝石、扇心、扇尖、嘴、左手。"""
    J = POSES[J] if isinstance(J, str) else J
    out = {}
    if 'staff' in J:
        st = J['staff']
        out['staff_tip'] = (st['x'] + 0.5, st['top'] + 0.5)
        out['staff_orb'] = (st['x'] + 0.5, st['top'] + 3.5)
    fa = J['fan']
    mid = np.radians((fa['a0'] + fa['a1']) / 2)
    r1 = fa.get('r1', 16.5)
    out['fan_pivot'] = tuple(fa['pivot'])
    out['fan_tip'] = (fa['pivot'][0] + np.cos(mid) * r1, fa['pivot'][1] + np.sin(mid) * r1)
    cx, cy = J['chin']
    out['mouth'] = (cx + (0.5 if J.get('head', 'tilt').endswith('_l') else 1.5), cy - 1)
    out['hand_l'] = tuple(J['hand_l'])
    out['wrist_r'] = tuple(J['wrist_r'])
    return out


def _key(J):
    def q(v):
        if isinstance(v, dict):
            return tuple(sorted((k, q(x)) for k, x in v.items()))
        if isinstance(v, tuple):
            return tuple(round(x * 4) / 4 for x in v)
        if isinstance(v, float):
            return round(v * 4) / 4
        return v
    return tuple(sorted((k, q(v)) for k, v in J.items()))


_CACHE = {}


def render_joints(J):
    """按关节 dict 画全身（48x72），结果按量化后的关节缓存。"""
    k = _key(J)
    if k in _CACHE:
        return _CACHE[k]
    cv = Cv(72, 48, PAL)
    _hair_back(cv, J, J.get('sway', 0), J.get('hair_lift', 0))
    _leg(cv, J['hip_r'], J['knee_r'], J['ankle_r'], front=False, toe=J.get('toe_r', (1.8, 2.0)))
    _skirt(cv, J, J.get('flow', 0), J.get('lift', 0))
    _leg(cv, J['hip_l'], J['knee_l'], J['ankle_l'], front=True, toe=J.get('toe_l', (0.6, 2.4)))
    _torso(cv, J)
    _collar(cv, J)
    if 'staff' in J:
        st = J['staff']
        _staff(cv, int(round(st['x'])), int(round(st['top'])), int(round(st['bottom'])))
    _arm_l(cv, J)
    _head(cv, J, J.get('sing', False))
    _arm_r(cv, J)
    fa = J['fan']
    _fan(cv, fa['pivot'], fa['a0'], fa['a1'], r1=fa.get('r1', 16.5))
    _hand_r(cv, J)
    img = cv.rgba(OUT)
    if len(_CACHE) > 400:
        _CACHE.clear()
    _CACHE[k] = img
    return img


def body(pose='idle'):
    return render_joints(POSES.get(pose, POSES['idle'])).copy()


META = dict(
    key='vita',
    name_zh='薇塔·克洛提德',
    title_zh='苍之深渊',
    rank_zh='使徒 第二柱',
    skill_zh='深渊的苍之歌',
    quote_zh='来，聆听我的歌声吧',
    fx=dict(
        primary='#3163d4',
        secondary='#57cfc6',
        description=(
            '深渊的苍之歌（深淵の蒼き唄 / Aria of the Abyss）：薇塔闭眼高歌、魔杖高举、另一只手把大折扇扬起。'
            '脚下与杖头先后展开深蓝色魔法阵（外圈符文缓慢旋转），杖头水晶亮起青白光；'
            '随后蓝→青→黄绿渐变的光之音波（同心圆环 + 飘散的音符与蓝色羽毛，致敬使魔瑞鸟古里亚诺丝）'
            '一圈圈扩散，场景被深海般的苍蓝色吞没，敌方头上落下细碎的青色水晶碎片，'
            '最后以一道苍蓝光柱收尾。点缀色可用牡丹粉 #ffd9e3 与金 #f1ca62。'
        ),
    ),
    # 全身像里两只眼睛的像素 (y, x)：剪影镜头里点亮
    eyes_body={'idle': [(7, 30), (8, 32)], 'skill': [(8, 30), (9, 32)]},
    # skill 姿势魔杖杖尖 (y, x)：render.act_vita 在这里展开小魔法阵
    weapon_tip=(0, 43),
    # 以下是新增的参考坐标 (y, x)，供以后的镜头使用
    fan_pivot={'idle': (35, 20), 'skill': (24, 15)},
    # 各姿势的特效发射点（x, y），由 anchors() 计算：staff_tip / staff_orb / fan_pivot / fan_tip / mouth / hand_l
    anchors={k: anchors(k) for k in ('idle', 'skill') + SCENE_POSES},
    exprs=EXPRS,
)
LANDMARKS = {
    'portrait': {   # 闪 III 立绘头部按 0.13 缩放、不旋转（歪头保留）；与 data/landmarks/vita.json 的 bust 组同名
        'head_top': (58, 13), 'chin': (42, 52),
        'eye_near_out': (40, 28.5), 'eye_near_in': (47, 29.5), 'eye_near_top': (42.5, 27.5), 'eye_near_bot': (43, 31.5),
        'eye_far_in': (52.5, 36), 'eye_far_out': (58, 35.5), 'eye_far_top': (54, 34), 'eye_far_bot': (54.3, 37.5),
        'brow_near': (45.5, 22.8), 'nose': (51, 41), 'mouth': (45.5, 44),
        'cheek_far': (57.5, 39), 'jaw_far': (52, 47), 'jaw_near': (33, 45.4),
        'x_hair_top': (49, 4), 'x_bun_top': (26, 8), 'x_bun_left': (12.5, 23),
        'x_curl_tip_upper': (9, 37), 'x_curl_tip_lower': (11, 42.5),
        'x_hair_far_top': (64.5, 18.5), 'x_hair_far_edge': (68, 40),
        'x_part': (52.5, 15.5), 'x_hairline': (48, 21),
        'x_ear_near': (27, 30), 'x_earring_near': (26.5, 43), 'x_earring_far': (52, 48.5), 'x_mole': (56, 40),
        'x_collar_top_near': (29.5, 52), 'x_collar_tip_near': (21.5, 58.5), 'x_collar_tip_far': (64.5, 77),
        'x_shoulder_near': (14, 60),
    },
    'idle': {
        'head_top': (32.5, 3), 'chin': (29.5, 11), 'neck': (30, 15),
        'shoulder_r': (24, 15), 'shoulder_l': (35, 15), 'elbow_r': (23, 24), 'elbow_l': (41, 18),
        'wrist_r': (21, 33), 'wrist_l': (38, 25), 'waist': (30, 24), 'crotch': (33, 40),
        'knee_r': (28, 52), 'knee_l': (32, 50), 'ankle_r': (28, 66), 'ankle_l': (33, 67),
        'x_collar_tl': (26, 13), 'x_collar_tr': (34, 16), 'x_collar_bot': (31, 19),
        'x_hand_l': (35, 27), 'x_belt_gem': (33, 30),
        'x_fan_tl': (5, 30), 'x_fan_br': (15, 51),
        'x_hem_front': (27, 48), 'x_skirt_tip_r': (12, 63), 'x_skirt_tip_l': (37, 66), 'x_slit_frill_bot': (37, 56),
        'x_toe_r': (30, 68), 'x_toe_l': (34, 70),
        'x_bun': (25.5, 3.5), 'x_hair_tip_near': (13.5, 37.5), 'x_hair_tip_far': (44, 37), 'x_glove_top_r': (24, 18.5),
    },
}
