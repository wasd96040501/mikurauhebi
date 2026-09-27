"""钢之圣女 阿瑞安赫德（Arianrhod）像素立绘。

参考：
- 画像：碧之轨迹 Evolution 半身像（无头盔）refs/arianrhod/bust.png —— 脸朝画面左 3/4，
  白金长发、遮住额头的长刘海、一缕发丝斜穿两眼之间，青绿色眼睛、沉静的神情；
  护喉是多片象牙白甲（画像按原画配色，和全身图的银甲分开用 PPAL），身后是深色高领，两侧巨大的刃状肩甲，近侧胸前一条长辫（银色发箍）。
- 全身：闪之轨迹 III 立绘 refs/arianrhod/full.png —— 朝镜头迈步，右手（画面左）握巨型骑枪，
  枪尖斜指左下，枪尾从左肩后露出；酒红披风被风吹向画面左侧，长发随之飘起；
  银色全身甲、刃状大肩甲、金色腰带扣、尖角腰甲，下摆是两片金边酒红前摆（画面左的一片被风掀起）。

比例：全身图 72px ≈ 180cm，她 175cm → 头顶到脚底 70px，头高约 8px（官方立绘约 8.5 头身）。

画法：字符画布上用无抗锯齿的多边形 / 贝塞尔笔触 / 椭圆铺色块（每种材质 3~4 阶），
五官、护喉、肩甲等小部件用 ASCII 手点，最后外描边。
全身由可复用的部件拼装（披风、后发、骑枪、腿、前摆、躯干、手臂、肩甲、头），
每个部件只吃几个关节 / 锚点参数，姿势 = 一组参数（见 POSES）。
"""
import numpy as np

import pixelkit as pk

# ----------------------------------------------------------------- 画布工具


class Cv:
    def __init__(s, h, w, pal):
        s.h, s.w, s.pal = h, w, pal
        s.a = np.full((h, w), ' ', dtype='<U1')
        s.ox = s.oy = 0
        yy, xx = np.mgrid[0:h, 0:w]
        s.px, s.py = xx + .5, yy + .5

    # ---- 掩码
    def polym(s, pts):
        m = np.zeros((s.h, s.w), bool)
        pts = [(x + s.ox, y + s.oy) for x, y in pts]
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
        return ((s.px - cx - s.ox) / rx) ** 2 + ((s.py - cy - s.oy) / ry) ** 2 <= 1

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
        W = (w0 + (w1 - w0) * (ts - t0) / max(t1 - t0, 1e-9))[:, None] / 2
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
            xi, yi = int(np.floor(x + s.ox)), int(np.floor(y + s.oy))
            if 0 <= xi < s.w and 0 <= yi < s.h:
                m[yi, xi] = True
        return m

    def line(s, pts, c, n=None, **k):
        """1px 贝塞尔细线（按像素取整采样）。"""
        s.put(s.linem(pts, n), c, **k)

    def dots(s, pts, c, **k):
        m = np.zeros((s.h, s.w), bool)
        for x, y in pts:
            x, y = int(round(x + s.ox)), int(round(y + s.oy))
            if 0 <= x < s.w and 0 <= y < s.h:
                m[y, x] = True
        s.put(m, c, **k)

    def patch(s, x0, y0, rows, flip=False):
        """ASCII 小块：'.' 与 ' ' 透明。flip=True 时左右镜像。"""
        x0, y0 = int(round(x0 + s.ox)), int(round(y0 + s.oy))
        for dy, r in enumerate(rows):
            if flip:
                r = r[::-1]
            for dx, ch in enumerate(r):
                if ch not in '. ' and 0 <= y0 + dy < s.h and 0 <= x0 + dx < s.w:
                    s.a[y0 + dy, x0 + dx] = ch

    def shift(s, m, dx, dy):
        o = np.zeros_like(m)
        H, W = m.shape
        ys = slice(max(dy, 0), H + min(dy, 0))
        yd = slice(max(-dy, 0), H + min(-dy, 0))
        xs = slice(max(dx, 0), W + min(dx, 0))
        xd = slice(max(-dx, 0), W + min(-dx, 0))
        o[yd, xd] = m[ys, xs]
        return o

    def edge(s, m, dx, dy):
        """m 中沿 (dx,dy) 方向的最外一圈像素。"""
        return m & ~s.shift(m, -dx, -dy)

    def ring(s, m):
        g = m.copy()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            g |= s.shift(m, dx, dy)
        return g & ~m

    def sep(s, m, c='O', on=None):
        """在 m 外侧 1px、且已有颜色的地方描一圈分隔线（内描边）。"""
        r = s.ring(m) & (s.a != ' ')
        s.put(r, c, on=on)

    def rgba(s, outline_col):
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
OUT = '#1a0f1f'
PAL = {
    'O': OUT,
    # 皮肤
    'S': '#fde6d4', 's': '#eab8a2', 'q': '#b9786e',
    # 青绿眼
    'W': '#ffffff', 'I': '#8cf0e4', 'e': '#1fb0b4', 'i': '#0e5a68', 'L': '#2a1a26',
    # 白金发
    'Y': '#fffae0', 'H': '#f1dca2', 'h': '#cfab72', 'j': '#8c6a48',
    # 银甲
    'A': '#f6f7fb', 'M': '#c6cbd8', 'm': '#8b91a4', 'n': '#52586c',
    # 黑色内衬 / 护腿
    'K': '#2a2638', 'k': '#4a4662',
    # 金
    'G': '#f3c95c', 'g': '#a8772c',
    # 酒红：前摆 P/R/r，披风 R/r/d
    'P': '#cf4c72', 'R': '#962a4e', 'r': '#5e1a38', 'd': '#35102a',
    # 青色宝石 / 枪身嵌线
    'T': '#6ee6d6', 't': '#2a9c9c',
}


def _shade(cv, m, lit, mid, dark, lx=-1, ly=-1, k=1):
    """一块材质：先铺中间色，再按左上光源把朝光边描亮、背光边压暗。"""
    cv.put(m, mid)
    lm = np.zeros_like(m)
    dm = np.zeros_like(m)
    for i in range(1, k + 1):
        lm |= ~cv.shift(m, lx * i, ly * i)
        dm |= ~cv.shift(m, -lx * i, -ly * i)
    if dark:
        cv.put(m & dm, dark)
    if lit:
        cv.put(m & lm & ~dm, lit)


# ----------------------------------------------------------------- 画像（80x80）
# 底稿：碧之轨迹 Evolution 半身像 refs/arianrhod/bust.png（无头盔，脸朝画面左 3/4）。
# 按 tools/measure.py 的固定取景（双眼中心 + 下巴三个锚点，缩放 ≈0.202）把原画搬进 80x80，
# 剪影照原画描；头发按原画的发流用锥形笔触一簇簇画（亮 / 主 / 暗 / 深缝 4 阶，发梢压暗、头顶随头型的高光弧，
# 刘海分 4 簇尖梢，其中一缕斜穿两眼之间），发贴脸处与刘海下的额头压一阶暗、远侧颊与下颌两阶影；
# 甲按原画逐片平涂 + 1px 分缝线，护喉铜铆钉、银发箍、人字形三股辫。眼睛按表情另行盖章。
# 画像的甲按原画是象牙白（茶色 / 灰紫阴影），和全身图（闪III）的银甲不同，所以单独一套调色板 PPAL。
PPAL = {
    # 白金发（按官方图取色）：高光 / 主色 / 暗面 / 深 / 深缝
    'Y': '#fbf7c4', 'H': '#efe2b0', 'h': '#d6bc90', 'j': '#b0906c', 'J': '#75533d', 'o': '#4e3426',   # o = 头发外轮廓
    # 肤：亮 / 影 / 线
    'S': '#ffe6c3', 's': '#f2c9a4', 'q': '#d09474',
    # 青绿眼：眼白 / 虹膜亮 / 虹膜 / 虹膜暗 / 睫毛
    'W': '#f4f6f8', 'I': '#8af0e6', 'e': '#20b4b0', 'i': '#0b4a50', 'L': '#1a1216',
    # 象牙白甲：亮 / 次亮 / 茶影 / 深茶影 / 灰影 / 分缝线
    'A': '#fdf9ea', 'B': '#e6e0d8', 'M': '#c8b096', 'T': '#9c8870', 'N': '#a0928a', 'n': '#736862',
    # 辫子的银色发箍
    'V': '#c8cadc', 'v': '#8c90a8',
    # 深棕高领 / 披风
    'K': '#4e3c36', 'k': '#33231f', 'd': '#1d120e',
    # 铜铆钉
    'G': '#b08858', 'g': '#5a2e22',
    # 外轮廓（甲 / 高领）
    'O': '#2a1c18',
}
# 剪影最外一圈（画在剪影内侧，不外扩）按材质换成描边色
P_EDGE = {**{k: 'o' for k in 'YHhjJ'}, **{k: 'q' for k in 'Ss'}, **{k: 'O' for k in 'ABMTNnKkGgVv'}}
# 材质表（data/landmarks/arianrhod.json 的 _classes 引用这些名字）
P_SKIN = [PPAL[k] for k in 'Ssq']
P_HAIR = [PPAL[k] for k in 'YHhjJo']
P_EYES = [PPAL[k] for k in 'IeiL']
P_ARMOR = [PPAL[k] for k in 'ABMTNnVvW']   # 眼白 W 与白甲同色系，算进甲（否则白护喉会被量成眼睛）
P_DARK = [PPAL[k] for k in 'KkdGgO']

PORTRAIT_BASE = [
    '................................................................................',
    '................................................................................',
    '................................................................................',
    '................................................................................',
    '................................................................................',
    '................................................................................',
    '................................................................................',
    '................................................................................',
    '................................................................................',
    '....................................YYYYYYYYjYhHhH..............................',
    '.................................YYYYYYYYYYYhYYhhYhH............................',
    '...............................YYYYYYYYYYYYhYYHYHYYhHH..........................',
    '..............................HHYYYHHHYYYHhHHHHYYYYYYhH.........................',
    '.............................HYHYYjjjjjYYYHHHHHHYYYYYYhH........................',
    '............................HYHYYYjYYYjjYYYYYYYYHYYYYYYHH.......................',
    '...........................HYYYYYYHYYYYYYYYYYYYYYYHYYYYhHH......................',
    '..........................HYYYYYYYHHHYYYYYYYHHHHYHYHYYYYhH......................',
    '.........................HYYYYYYYYhHHYYYYYYYYHHHHHYhHYYYYhH.....................',
    '.........................YYYYYYYYhHHHYYYYYYYYhHHHHHYhHYYYhH.....................',
    '........................YYYYHHYYhhHHhHYYhYYYYYhHHHHYhhHYYYhH....................',
    '........................YYYHhHYYhhHHhHYYhYYYYYYhHHHYhhhHYYYh....................',
    '.......................YYYYhhHYYhhHHhHYYYhYYYYYhHHHHYhhHYYYhH...................',
    '.......................YYYYhhHYYhhHHhHYYYhYYYYYYhHHHYhhhHYYYh...................',
    '......................YYYYhhHYYYhhHHYhHYYhHYYYYYhHHHYhhhHYYYh...................',
    '......................YYYYhhHYYYhhHHYhHYYhhHYYYYYhHHYhhhhHYYhH..................',
    '......................HYYYhhHYYYhhHHYhHYYYhhHYYHHhHHYhhhhHYYYh..................',
    '......................HYYhHhHYYYhhHYhHhHYYhHhHHHHhHHhhhhhhHYYh..................',
    '.....................YYYYhHhHYYYhhhhhHhHYYhHHhHHHHhhhhhhhhHYYjj.................',
    '.....................YYYYhHYhHHHsshhhHhssYhHHHhHHHhhhhYhhHhHHjj.................',
    '.....................YYYYhHYhHHHsshhHHhssYYhHHhHHHhhhhYhhHhHHHj.................',
    '.....................HYYYhHYhHHHhhhhHHHssYYhHHHhHHhhhhYhhHjHHHj.................',
    '.....................HYYYhHYhHHhhhhhHHHssYYjHHHHjHjhhhYhhHjhHHj.................',
    '.....................HYYhHYYhHHSSSjSSSSssYYSSSSSSSjhhYYhhHjhHHjj................',
    '.....................HYYhHYYYjHSSSSSSSSSsHHSSSSSSSjhhYYhhHjhHHjj................',
    '.....................Y.HhHYYYjSSSSSSSSSSSHHSSSSSSSjhYYYjjjjjhHHj................',
    '.....................Y.HhHYYYjSSSSSSSSSSSHHSSSSSSSjHYYjjjjjjhHHj................',
    '.....................Y.HhHYYYjssSSSSSSSSSSHSSSSSSSjHYhhHYjjjhHHjj...............',
    '.....................H.HhHYYYjssSSSSSSSSSSHSSSSSSSjHYhhHYjjjhHHjj...............',
    '.....................HHHhHHHHjqsSSSSSSSSSSHsSSSSSSjHYhhHYjjjhjjjj...............',
    '.....................HHHhHHHjjqsSSSSSsSSSSSHSSSSSSjYYhhHYjjjhjjKjKK............K',
    '.....................HHHhHHHjjjsSSSSSsSSSSSHSSSSSsjYYhhHYjjjhjhKKKKKK.......KKKO',
    'K....................HHHhHHHjjjqSSSSSSSSSSSSSSSSssjYYhhHYjjjhjhKKKKKK.....KKKOOO',
    'OAAOK................HHHhHHHjjjqSSSSSSqsSSSSSSSSssjYYhhHYjjjhjKKKKKKK..KKKKOOAAA',
    'AAAAOAAAA..........KKHHHhjHHjjjqsSSSSSSSSSSSSSSsssjYYhhYjjjjhMKKKKKK..KKKOOAAAAA',
    'AAAAAAAAAAAAA...kkkkkkHHhjHHjjhjqSSSSSSSSSSSSSSsssjYYhhYjjjhKMKKKKKKKKKOOAAAAAAA',
    'AAAAAAAAAAAAAAAABkkkkkHHhjHHjjhjqqSSSSSSSSSSSSSsssjYYhhYjjJMMMKKKKKKKOOAAAAAAAAA',
    'AAAAAAAAAAAAAAAABkkkkkHHhjHHjjhjjqqSSqqqqqSSSSSssjHYYhhYjjJKMMMKKKKOOAAAAAAAAAAA',
    'AAAAAAAAAAAAAAAAABkkkkKHhjHHJJJjJjqqSSSssSSSSSSsjhHYYhhYjJJKKMMMMKOAAAAAAAAAAAAn',
    'AAAAAAAAAAAAAAAAABkkkkKHhjHHJJJjJJsqqSSSSSSSSqsqjhHHhhHYjJKKKMKKKOAAAAAAAAAAnnnM',
    'nnnAAAAAAAAAAAAAAABkkkKKhhjHJJJjhJJnqqqSSSqqqsnhhHHHhhHYjJKKKMKKOAAAAAAAAnnnMMMM',
    'MMnnnAAAAAAAAAAAAABkkkKOKhjJJJJjhJJnnssssssnnnnhhHHHhhHYJJKKKMOOAAAAAAnnnMMMMMMM',
    'MMMMnnnnMMMMMMAAAAOkKKKOOAAAnnnnnnnnnnnnnnnnnnhhhHHHhhhhJJKKKOnnAAnnnnMMMMMMMMMM',
    'TTMMMMMnnnnnnMMMMAABKKOnAAAAnnnNNNNNNNNNNNNNNNhhhHHHhhhhJKKKOAnnnnMMMMMMMMMnnnnn',
    'TTTTTTMMMMMMnnnnnnnBKKOnnAAAAAnnnNNNNNNNNNNnnnhhhHHHhhhhJKKOAAAnnnnnnnnnnnnAAAAA',
    'TTTTTTTTTMMMMMMMMMMOKKOAnnAAAAAAnnNNNNnnnnnNNNNhhHHHhhhhJKOAAAAAAAAABBBBBBAAAAAA',
    'TTTTTTTTTTTMMMMnnnnOKKOAAnnnAAAnnNnnnnNnnNNNNNNhhHHhhhhhjOAAAAABBBBBBBBBBBBBBBBB',
    'TTTTTTMMMMMnnnnABBBBOKKOAAAnnnnnnnBBBNNNnnNNNNNhnnnnnhhhhAAAABBBBBBBBBBBBBBBBBBB',
    'TMMMMMnnnnnBBBBBBBBBOKKOAAAABBnBBBBBBNNNNnNNNNNnvvvvvnhhhAAAABBBBBBBBBBBBBBBBBBB',
    'MnnnnnBBBBBBBBBBnnnOKKKKOAAABBnBBBBBBNNNNnnNNNNnvVVVVKOAAAAABBBBBBBBBBBBBBBBBBBB',
    'nABBBBBBBBBnnnnnMMOKKKKKOAAABBnBBBBBBNNNNNnnNNOKvVVVVKOnnnnnnnnnnnnnnnnnnnAAAAAA',
    'BBBBBBnnnnnMMnnnnnOKKKKKKOAABBnBBBBBBNNNNNnnnNOKvVVVVvnNNnnnnnnnnnnnnnnnnnnnnnnn',
    'BnnnnnnnnnnnnMMMMMOKKKKKKOAABBnBBBBBBNNNNNnNnnOKvVVVVVnNNnAAAAAAAAAAAAAAAAAMMMMM',
    'nnnnnnMMMMMMMMAAMOKKKKKKKKOABBnBBBBBBNNNNnNNNNOKvWWWWWKONnAAAAAAAAAAAAAAAAAAAAAM',
    'MMMMMMMAAAAAAAAAMOKKKKOOOOnnBBnBBBBBBNggNnNNNNOKjHHhhhjONnAAAAAAAAAAAAAAAAAAAAAA',
    'MMAAAAAAAAAAAAAAMOKKKKOBBGgnnnnBnnnnngGggNNNNNNjhYYhhHjOnAAAAAAAAAAAAAAAAAAAAAAM',
    'GgAAAAAAAAAAAAAAMOKKKKOBBGgnNNnnNNNNNgGggNNNNnnjHhhhHHYOMAAAAAAAAAAAAAAAAAAAAAMM',
    'ggMAAAAAAAAAAAAAMOKKKOBBBggnNNNNNNNNNNggNNnnnnnjYHHhYYjOMAAAAAAAAAAAAAAAAAAAAAMM',
    'MMMAAAAAAAAAAAAAOKKKKOBABBBnnnnnnnnnnnnnnnnnnnnjYYYhhhhjMAAAAAAAAAAAAAAAAAAAAMMM',
    'MMMAAAAAAAAAAAAMOKKKKOBAAAAnnnnnnnnnnnnnnnnnAAAOjhYhhhHjMAAAAAAAAAAAAAAAAAAAMMMM',
    'MMMAAAAAAAAAAAAMOKKKKOBAAAAAnnnnnnnnnnnnBnMMAAAOjHhhHHYYMAAAAAAAAAAAAAAAAAMMMMMB',
    'nnMMAAAAAAAAAAAMOKKKKOBAAAAAABnBBBBBBBNNMnMMAAAOjYHhYYhjOAAAAAAAAAAAAAAAMMMMMBBn',
    'NnnnAAAAAAAAAAAMOKKKKOBAAAAAABnNNNNNNNNNnMMMAAAOjYYHhhhjOAAAAAAAAAAAAAMMMMMBBBnN',
    'NNBnnMAAAAAAAAAMOKKKKOBBAAAAABnNNNNNNNNNnMMAAAAOjhYYhhHjOAAAAAAAAAAAMMMMMMBBnnNN',
    'NNNNnnMAAAAAAAAOKKKKKOBBBAAAABnNNNNNNNNnMMMAAAAOjHhhhHHYOAAAAAAAAAMMMMMMBBnnNNNN',
    'NNNNNnnnAAAAAAAOKKKKKOBBBBBAABnNNNNNNNnMMMMAAAAOKjHHhYYjOAAAAAAAMMMMMMBBnnNNNNNN',
    'NNNNNNNnnMAAAAAOKKKKKKOBBBBBBBnNNNNNNnAAAAAAAAAOKjYYhhhjKOAAAAAMMMMMBBnnNNNNNNNN',
    '.NNNNNNNnnnAAAAOKKKKKKOBBBBBBBOOOOOOnAAAAAAAAAAOKjhYhhhjKOAAAMMMMMBBnnNNNNNNNNNN',
    '.NNNNNNNNNnnnAAOKKKKKKOBBBBBBOKKKKKKOAAAAAAAAAAOKjHhhHHYKOMMMMMBBBnnNNNNNNNNNNNN',
    '.NNNNNNNNNNNnnAOKKKKKKOBBBBBOKKKKKKOAAAAAAAAAAAOKjYHhYYhjOMMMBBnnnNNNNNNNNNNNNNN',
    '.NNNNNNNNNNNNnnOKKKKKKOBBBBBOKKKKKKOAAAAAAAAAAAOKjYYhhhhjOBBBnnNNNNNNNNNNNNNNNNN',
]

# 眼睛章：(左上角 x, y, 行)。'.' 不动。远侧眼 = 画面左，近侧眼 = 画面右。
P_EYE = {
    None: [(29, 33, ['.LLLLLLL.',
                     'LLWiiiiL.',
                     '.WWeWieq.',
                     '..WIeIq..',
                     '...qqq...']),
           (43, 32, ['.......LL',
                     'LLLLLLLLL',
                     '.WWiiiiL.',
                     '.WWeWieL.',
                     '..WIeeIq.',
                     '...qqqq..'])],
    'stern': [(29, 33, ['.sssssss.',
                        'LLLLLLLL.',
                        '.WWeWieq.',
                        '..WIeIq..',
                        '...qqq...']),
              (43, 32, ['.........',
                        'sssssssss',
                        'LLLLLLLLL',
                        '.WWeWieL.',
                        '..WIeeIq.',
                        '...qqqq..'])],
    'closed': [(29, 33, ['.sssssss.',
                         'SSSSSSSS.',
                         'LLLLLLLq.',
                         '.qsSSsq..',
                         '.........']),
               (43, 32, ['.........',
                         'sssssssss',
                         'SSSSSSSSS',
                         '.LLLLLLLL',
                         '..qssssq.',
                         '.........'])],
}


def portrait(expr=None):
    """expr: None（沉静）/ 'stern'（凝视，上眼睑压低）/ 'closed'（闭眼）。"""
    expr = expr if expr in P_EYE else None
    cv = Cv(80, 80, PPAL)
    for y, r in enumerate(PORTRAIT_BASE):
        for x, ch in enumerate(r):
            if ch not in '. ':
                cv.a[y, x] = ch
    for x0, y0, rows in P_EYE[expr]:
        cv.patch(x0, y0, rows)
    # 剪影最外一圈描边（画布边缘不算）
    op = cv.a != ' '
    pad = np.pad(op, 1, constant_values=True)
    inner = op & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
    for k, c in P_EDGE.items():
        cv.a[inner & (cv.a == k)] = c
    img = np.zeros((80, 80, 4), np.float32)
    for ch, col in PPAL.items():
        m = cv.a == ch
        img[m, :3] = pk.hexc(col)
        img[m, 3] = 1
    return img


# ----------------------------------------------------------------- 全身（160x76，脚底在最后一行，身体居中 x≈80）
BW, BH = 160, 112
OY = 36            # 站姿坐标系（76 行，脚底 y≈75）在画布上的下移量

HEAD = [            # 头（含刘海），左上角 = 锚点；头顶 y+1，下巴 y+9
    "....jHHHj....",
    "...jHYYHHj...",
    "..jHYHHHHhj..",
    "..HYHHHHHHhj.",
    ".jHHHhHHhHhh.",
    ".hHHLSHHLLhh.",
    ".hHSeSSSeiSh.",
    ".hhSSSSSsshh.",
    ".hhhSSSSshhj.",
    ".jhh.SSs.hhj.",
    ".jh.......hj.",
]
HEAD_EYES = [(6, 4), (6, 8)]    # (行, 列)，相对 HEAD 锚点

TORSO = [           # 护喉 + 胸甲 + 腰带 + 尖角腰甲，左上角 = 锚点（颈根在 (8, 3)）
    "....nMAAAMMn.....",
    "...nAAMMMMMmn....",
    "...mMnnnnnnnmm...",
    "..nAAAMMMMMmmmn..",
    ".MAAMMKkKKMMMmmn.",
    "MAAAMKKGKKKMMmmmn",
    "MAAMMKGKGKKKMmmmn",
    "AAMMmKKgKKKKMMmmn",
    "AAMMmKKgKKKKmMmmn",
    "MAMmmKGgGKKKmmmmn",
    "MMmmKKKgKKKKmmmnn",
    ".MmmKKKgKKKKmmnn.",
    ".mmnKKGgGKKKnmnn.",
    "..mnKKKgKKKKnnn..",
    "..nMMmKKKKmMMmn..",
    "..nAMMMmmmMMMmn..",
    ".gGGgnKKKKnggGgn.",
    ".gGGgnKKKKngGGg..",
    "AAMMMMnKKnMMMMmmn",
    "AAAMMMMnnMMMMmmmn",
    "AAMMMMMAAMMMMmmmn",
    "MnnnnnnAAnnnnnnnn",
    "AAMMMMMAAMMMMmmmn",
    "AMMMMMAAAAMMMmmmn",
    "AMMMMMAAAMMMMmmn.",
    "nnGnnnAAAnnnGnnn.",
    ".MMMMMAAAMMMMmm..",
    "..MMMMAAAMMMmm...",
    "....MMAAAMMm.....",
    ".....MAAAMm......",
    "......AAMm.......",
    ".......AM........",
]


def _v(a):
    return np.asarray(a, float)


def _unit(ang):
    r = np.radians(ang)
    return np.array([np.cos(r), np.sin(r)])


def _lance(cv, grip, ang, front=66, back=40):
    """骑枪。grip = 握点，ang = 从握点指向枪尖的方向（度，图像坐标，0=右，90=下）。
    沿轴 t（握点=0，向枪尖为正）：枪尾饰(-back) · 黑色枪杆 · 笼状护手(1.5~15) · 金环+青宝石(15~18)
    · 金饰枪根(18~25) · 带青色凹槽的细长枪刃(25~front)，近枪尖有两根侧刺。返回 P(t, o)。"""
    g, d = _v(grip), _unit(ang)
    n = np.array([-d[1], d[0]])
    if n[1] > 0:            # 让 +o 永远朝上（受光面）
        n = -n
    P = lambda t, o=0: tuple(g + d * t + n * o)
    # 枪杆
    cv.stroke([P(-back), P(16)], 'K', 1.8)
    cv.line([P(-back + 1, 0.4), P(14, 0.4)], 'k', on='K')
    # 枪尾：金环 + 三片箭羽状银刃
    cv.stroke([P(-back + 3), P(-back + 4.5)], 'G', 2.6)
    for o in (1.6, -1.6):
        cv.poly([P(-back + 4, o * 0.4), P(-back - 1, o * 1.6), P(-back + 1, o * 0.3)], 'M' if o > 0 else 'm')
    cv.poly([P(-back + 4, 0.5), P(-back - 2.5, 0), P(-back + 4, -0.5)], 'A')
    # 笼状护手：前宽后窄的一圈银条
    C0, C1 = 1.5, 17
    cage = cv.polym([P(C0, 3.0), P(C1, 5.2), P(C1, -5.2), P(C0, -3.0)])
    cv.sep(cage, 'O')
    cv.put(cage, 'n')
    for o0, o1, c in ((2.4, 4.4, 'A'), (1.0, 1.8, 'A'), (-0.4, -0.6, 'M'), (-1.8, -2.8, 'm'), (-2.6, -4.6, 'm')):
        cv.put(cv.linem([P(C0 + 0.5, o0), P(C1 - 0.5, o1)]) & cage, c)
    cv.put(cv.strokem([P(C0 - 0.6, 0), P(C0 + 1.2, 0)], 6.4), 'm')
    rim = cv.strokem([P(C1 - 0.6, 0), P(C1 + 1.4, 0)], 11)
    cv.sep(rim, 'O')
    cv.put(rim, 'M')
    cv.put(rim & cv.strokem([P(C1 - 0.6, 3.5), P(C1 + 1.4, 3.5)], 3.4), 'A')
    cv.dots([P(C1 + 0.4, 0.5)], 'T')
    # 金环 + 青宝石
    cv.stroke([P(C1 + 1.4, 0), P(C1 + 3.6, 0)], 'G', 6)
    cv.line([P(C1 + 1.6, -2.5), P(C1 + 3.4, -2.5)], 'g')
    cv.dots([P(C1 + 2.4, 1.6), P(C1 + 2.4, -1.2)], 'T')
    # 金饰枪根
    R0, R1 = C1 + 3.5, C1 + 10
    root = cv.polym([P(R0, 2.6), P(R1, 1.8), P(R1, -1.8), P(R0, -2.6)])
    cv.put(root, 'M')
    cv.put(root & cv.strokem([P(R0, 1.8), P(R1, 1.3)], 1.2), 'A')
    cv.put(root & cv.strokem([P(R0, -1.9), P(R1, -1.4)], 1.2), 'm')
    cv.dots([P(R0 + 2, 0.6), P(R0 + 4.5, -0.6), P(R0 + 3, 1.6), P(R0 + 5.5, 1.2)], 'G')
    cv.dots([P(R1 - 0.5, 0)], 'T')
    # 枪刃：两侧银、中间青色凹槽，向枪尖收细
    L = front
    blade = cv.polym([P(R1, 1.9), P(L - 10, 1.2), P(L, 0), P(L - 10, -1.2), P(R1, -1.9)])
    cv.put(blade, 'M')
    cv.put(blade & cv.strokem([P(R1, 1.3), P(L - 6, 0.5)], 1.1), 'A')
    cv.put(blade & cv.strokem([P(R1, -1.4), P(L - 6, -0.6)], 1.1), 'm')
    cv.put(cv.linem([P(R1 + 1, 0.1), P(L - 8, 0.1)]) & blade, 'T')
    # 近枪尖的两根侧刺
    cv.poly([P(L - 13, 1.0), P(L - 5, 3.2), P(L - 11, 1.4)], 'A')
    cv.poly([P(L - 13, -1.0), P(L - 5, -3.2), P(L - 11, -1.4)], 'm')
    return P


def _arm(cv, sh, el, wr, hand, lit=True):
    """手臂：黑色内衬上臂 + 银色上臂甲 + 肘部金环与护肘 + 前臂甲 + 手甲。"""
    sh, el, wr, hand = _v(sh), _v(el), _v(wr), _v(hand)
    hi, mid, lo = ('A', 'M', 'm') if lit else ('M', 'm', 'n')
    up = cv.strokem([tuple(sh), tuple(el)], 3.4, 3.0)
    cv.sep(up, 'O')
    _shade(cv, up, hi, mid, lo, lx=-1, ly=0)
    fo = cv.strokem([tuple(el), tuple(wr)], 3.6, 2.8)
    cv.sep(fo, 'O')
    _shade(cv, fo, hi, mid, lo, lx=-1, ly=0)
    # 肘上金环
    k = el + (sh - el) * 0.22
    dd = (el - sh) / (np.linalg.norm(el - sh) + 1e-9)
    nn = np.array([-dd[1], dd[0]])
    cv.put(cv.strokem([tuple(k - nn * 1.9), tuple(k + nn * 1.9)], 1.4), 'G')
    cv.dots([tuple(k - nn * 1.5 + dd * 0.6)], 'g')
    # 护肘
    cv.ell(el[0] + 0.3, el[1] + 0.5, 1.6, 1.4, mid)
    cv.dots([tuple(el + np.array([-0.5, 0]))], hi)
    # 手甲
    hm = cv.ellm(hand[0], hand[1], 1.6, 1.5)
    cv.sep(hm, 'O')
    cv.put(hm, mid)
    cv.put(hm & cv.ellm(hand[0] - 0.6, hand[1] - 0.6, 1.0, 0.9), hi)


def _leg(cv, hip, knee, ankle, toe, lit=True):
    """腿：黑色护腿大腿 + 银护膝 + 银护胫 + 金色踝甲 + 尖头铁靴。"""
    hip, knee, ankle, toe = _v(hip), _v(knee), _v(ankle), _v(toe)
    hi, mid, lo = ('A', 'M', 'm') if lit else ('M', 'm', 'n')
    th = cv.strokem([tuple(hip), tuple(knee)], 4.2, 3.4)
    cv.sep(th, 'O')
    _shade(cv, th, 'k', 'K', 'K', lx=-1, ly=0)
    sh = cv.strokem([tuple(knee), tuple(ankle)], 3.6, 2.8)
    cv.sep(sh, 'O')
    _shade(cv, sh, hi, mid, lo, lx=-1, ly=0)
    kn = cv.ellm(knee[0], knee[1] + 0.3, 2.1, 1.8)
    cv.sep(kn, 'O')
    _shade(cv, kn, hi, mid, lo, lx=-1, ly=-1)
    cv.dots([tuple(knee + np.array([0, -1.2]))], 'G')
    # 踝甲（金）
    an = cv.strokem([tuple(ankle + (knee - ankle) * 0.12), tuple(ankle - (knee - ankle) * 0.1)], 3.6)
    cv.sep(an, 'O')
    cv.put(an, 'G')
    cv.put(an & ~cv.shift(an, 1, 0), 'g')
    # 靴
    ft = cv.strokem([tuple(ankle + (toe - ankle) * 0.25), tuple(toe)], 3.0, 1.6)
    cv.sep(ft, 'O')
    _shade(cv, ft, hi, mid, lo, lx=-1, ly=-1)


def _pauldron(cv, anchor, side, s=1.0, tilt=0.0):
    """刃状大肩甲。anchor = 肩关节；side=-1 朝画面左张开（角色右肩），+1 朝右。
    上刃斜向外上、侧刃水平外伸、圆肩盖 + 两层下甲片。tilt 为整体旋转（度）。"""
    a = _v(anchor)
    c, sn = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
    T = lambda x, y: tuple(a + s * np.array([c * x * -side - sn * y, sn * x * -side + c * y]))
    out = side < 0          # 画面左侧的肩甲朝光
    def blade(pts, lit_edge):
        m = cv.polym([T(*p) for p in pts])
        cv.sep(m, 'O')
        cv.put(m, 'A' if out else 'M')
        cv.put(m & ~cv.shift(m, 0, 1), 'm' if out else 'n')
        cv.put(cv.linem([T(*p) for p in lit_edge]) & m, 'A')
        return m
    # 下甲片（最底层）
    blade([(-6.5, 3), (3, 3.2), (2.5, 7.2), (-5.5, 6.6)], [(-6, 3.6), (2.5, 3.8)])
    # 侧刃（水平外伸的细刃）
    blade([(-2, -3), (-10.2, -2.3), (-6.5, 0.6), (-2, 1)], [(-3, -2.7), (-9.6, -2.3)])
    # 主甲片
    main = cv.polym([T(-8, -1.4), T(1.5, -3), T(3.8, 0.4), T(3.2, 3.2), T(-1, 4.4), T(-7.4, 3)])
    cv.sep(main, 'O')
    _shade(cv, main, 'A', 'M' if out else 'm', 'm' if out else 'n', lx=-1, ly=-1)
    cv.put(cv.linem([T(-6.5, 1.8), T(-1, 2.8), T(2.6, 1.6)]) & main, 'g')
    cv.dots([T(-2.5, 0.2)], 'G')
    # 上刃（斜向外上的长刃）
    blade([(2.2, -2.8), (-7.4, -9.2), (-5.2, -1.6)], [(1.4, -3), (-7, -8.8)])


def _panel(cv, pts, lit_edge=None, trim=None, flame=None, dark=False, lit=None):
    """一片酒红前摆 / 披风：R 底色，一侧 P 亮边、另一侧 r 暗边，金色镶边线，可选金色火焰纹。"""
    m = cv.polym(pts)
    cv.sep(m, 'r')
    if dark:
        _shade(cv, m, 'R', 'r', 'd', lx=-1, ly=-1)
    elif lit == 'P':
        _shade(cv, m, 'P', 'P', 'R', lx=-1, ly=-1)
    else:
        _shade(cv, m, 'P', 'R', 'r', lx=-1, ly=0)
    for tl in (trim or []):
        cv.put(cv.linem(tl) & m, 'G')
    if flame is not None:
        cv.patch(*flame, ["G.G.G", ".GGG.", "..G.."])
    return m


def _head(cv, x0, y0, expr=None):
    cv.patch(x0, y0, HEAD)
    if expr == 'stern':
        cv.patch(x0 + 3, y0 + 5, ["LL.LLL"])
    elif expr == 'closed':
        cv.patch(x0 + 4, y0 + 6, ["S...Ss"])


def _braid(cv, pts, n=7, r=(1.3, 1.1)):
    for t in np.linspace(0, 1, n):
        x, y = Cv.bez(pts, t)
        m = cv.ellm(x, y, *r)
        cv.sep(m, 'j')
        cv.put(m, 'H')
        cv.put(m & ~cv.ellm(x - .4, y - .4, r[0] - 0.2, r[1] - 0.2), 'h')


# 姿势参数：关节 / 部件锚点 / 布料与发丝外形。坐标都在「76 行」的站姿坐标系里（脚底 y≈75），
# 画到画布上时整体下移 OY（画布上方留 36 行给举高的骑枪）。新增动作帧 = 新增一组参数。
import copy as _copy

_IDLE = dict(   # 闪III 立绘：朝镜头迈步，右手握枪斜指左下，披风与长发被风吹向画面左
    head=(74, 4), torso=(72, 13),
    sh_r=(72.0, 17.5), el_r=(68.6, 26.2), wr_r=(63.8, 32.6), hand_r=(62.2, 34.2),
    sh_l=(87.8, 17.5), el_l=(88.4, 27.5), wr_l=(89.2, 35.0), hand_l=(89.5, 37.2),
    legs=[dict(hip=(78.2, 44), knee=(78.8, 57.4), ankle=(78.4, 68.6), toe=(78.4, 74.6), lit=True)],
    lance=dict(grip=(62.2, 34.2), ang=180 - 20.5, front=66, back=37.5),
    paul_r=(72.2, 17.2), paul_l=(88.0, 17.2),
    braid=[(83.5, 15), (80, 22), (74.5, 28.5)], braid_tip=[(74.5, 29), (73, 32.5)],
    collar=([(75, 15), (73.5, 10), (76.5, 12), (77, 15)], [(84, 15), (85.5, 10.5), (86.5, 11), (86, 15)]),
    cape=[[(76, 15), (66, 19), (52, 23), (34, 25.5), (20, 27.5), (11, 30), (19, 31.5), (30, 36),
           (41, 42), (50, 47), (60, 50), (72, 48), (78, 30)],
          [(72, 44), (62, 47), (52, 52), (46, 58), (56, 57), (64, 54), (74, 52)],
          [(88, 18), (94, 22), (97, 34), (98, 46), (92, 46), (88, 36)]],
    cape_folds=[('R', [(58, 25), (40, 29), (24, 29.5)]), ('d', [(62, 33), (48, 38), (36, 36)])],
    lining=[(94, 26), (98, 34), (103, 46), (103.5, 56), (99.5, 64), (98.5, 56), (97, 46), (95, 38)],
    hair=dict(poly=[(77, 7), (71, 11), (63, 14), (54, 15.5), (44, 17.5), (38, 20), (47, 19.5), (41, 23.5),
                    (50, 22), (46, 27), (56, 24), (54, 28.5), (62, 25), (68, 22.5), (74, 18), (77, 14)],
              dark=[[(74, 11), (64, 16), (52, 18.5)], [(74, 14), (64, 19.5), (54, 22)], [(72, 18), (63, 22.5)]],
              light=[[(75, 9), (66, 13), (56, 15.5)], [(70, 16), (60, 19)]]),
    tabard=[dict(pts=[(76, 40), (70, 46), (63, 53), (56, 60), (51, 67), (48, 71), (56, 68), (66, 64),
                      (74, 62), (78, 50)],
                 trim=[[(51, 69), (60, 65), (73, 60.5)], [(74.5, 42), (66, 51), (55, 62)]], lit='P',
                 flame=(59, 58)),
            dict(pts=[(82, 40), (88, 42), (92, 50), (95, 59), (90, 66), (86, 71), (83, 70), (81, 56)],
                 trim=[[(83.5, 45), (84, 56), (84.5, 68)], [(89, 44), (93.5, 58)]])],
    grip_r=[(61.2, 33.6), (63.2, 34.8)],
)

# 双腿站立（左腿也露出来）与垂下的前摆
_LEGS_STAND = [dict(hip=(82, 44), knee=(82.6, 57.5), ankle=(83, 68.6), toe=(83.6, 74.6), lit=False),
               dict(hip=(77.5, 44), knee=(77.4, 57.5), ankle=(77.2, 68.6), toe=(76.6, 74.6), lit=True)]
_LEGS_WIDE = [dict(hip=(82, 44), knee=(86, 57.5), ankle=(89, 68.6), toe=(91.5, 74.6), lit=False),
              dict(hip=(77, 44), knee=(72.5, 57), ankle=(69.5, 68.6), toe=(66, 74.6), lit=True)]
_TABARD_HANG = [dict(pts=[(76, 40), (72, 47), (68, 56), (64.5, 65), (62, 72), (69, 71), (76, 66), (78, 50)],
                     trim=[[(63, 71), (70, 69.5), (76, 65.5)], [(75.5, 42), (70.5, 53), (66, 64)]], lit='P',
                     flame=(67, 61)),
                _IDLE['tabard'][1]]


def _mk(**kw):
    d = _copy.deepcopy(_IDLE)
    d.update(_copy.deepcopy(kw))
    return d


def _shifted(P, dx, dy, keys=('head', 'torso', 'sh_r', 'sh_l', 'paul_r', 'paul_l')):
    """上半身整体平移（下蹲 / 前倾用）：头、躯干、肩、肩甲、领子、辫子、发、披风跟着动。"""
    P = _copy.deepcopy(P)
    mv = lambda p: (p[0] + dx, p[1] + dy)
    for k in keys:
        P[k] = mv(P[k])
    P['collar'] = tuple([mv(q) for q in c] for c in P['collar'])
    P['braid'] = [mv(q) for q in P['braid']]
    P['braid_tip'] = [mv(q) for q in P['braid_tip']]
    P['hair']['poly'] = [mv(q) for q in P['hair']['poly']]
    P['hair']['dark'] = [[mv(q) for q in l] for l in P['hair']['dark']]
    P['hair']['light'] = [[mv(q) for q in l] for l in P['hair']['light']]
    P['cape'] = [[mv(q) for q in c] for c in P['cape']]
    P['cape_folds'] = [(c, [mv(q) for q in l]) for c, l in P['cape_folds']]
    P['lining'] = [mv(q) for q in P['lining']]
    return P


BASE = {
    'idle': _IDLE,
    # 枪尖插在土里、右手扶着枪杆，闭目伫立（第 1 小节）
    'planted': _mk(expr='closed', legs=_LEGS_STAND, tabard=_TABARD_HANG,
                   sh_r=(72.0, 17.5), el_r=(67.8, 26), wr_r=(64.4, 33.4), hand_r=(63.6, 35.6),
                   lance=dict(grip=(63.6, 37), ang=90, front=66, back=37.5),
                   grip_r=[(62.6, 35.2), (64.6, 35.2)]),
    'planted_open': None,   # 同上但睁眼（下面补）
    # 双手握住枪杆，重心下沉（预备）
    'grip': _mk(legs=_LEGS_STAND, tabard=_TABARD_HANG,
                el_r=(67.8, 27), wr_r=(64.4, 33.4), hand_r=(63.6, 35.6),
                el_l=(84, 29), wr_l=(68.5, 30.5), hand_l=(64.4, 30.2),
                lance=dict(grip=(63.6, 37), ang=90, front=66, back=37.5),
                grip_r=[(62.6, 35.2), (64.6, 35.2)]),
    # 拔：枪升起一截，泥土崩开
    'pull1': _mk(legs=_LEGS_STAND, tabard=_TABARD_HANG,
                 el_r=(67.5, 24.5), wr_r=(65, 28.2), hand_r=(64, 29.6),
                 el_l=(84, 24), wr_l=(69, 23.5), hand_l=(65, 23),
                 lance=dict(grip=(64, 31), ang=92, front=66, back=37.5),
                 grip_r=[(63, 29.2), (65, 29.2)]),
    # 拔出：双手举过肩，枪尖离地斜指左下
    'pull2': _mk(legs=_LEGS_STAND, tabard=_TABARD_HANG,
                 el_r=(67.5, 11.5), wr_r=(66.6, 12.2), hand_r=(66, 12.4),
                 el_l=(82, 9.5), wr_l=(72, 6.6), hand_l=(68.8, 6.8),
                 lance=dict(grip=(66, 12.4), ang=115, front=66, back=37.5),
                 grip_r=[(65, 12), (67, 12.8)]),
    # 挥枪：枪尖向左上扫到水平的中途，右脚跨出
    'swing': _mk(legs=_LEGS_WIDE,
                 el_r=(67.4, 21.5), wr_r=(65.2, 20.6), hand_r=(64, 20.2),
                 el_l=(82.5, 22.5), wr_l=(74, 17.5), hand_l=(71, 16.2),
                 lance=dict(grip=(64, 20.2), ang=150, front=66, back=37.5),
                 grip_r=[(63, 19.6), (65, 20.6)]),
    # 端平：骑枪水平指向左，双手持枪，弓步待发
    'guard': _mk(legs=_LEGS_WIDE,
                 el_r=(70, 25), wr_r=(67.5, 27.5), hand_r=(66, 28),
                 el_l=(88, 25.5), wr_l=(82, 28.4), hand_l=(79, 28.2),
                 lance=dict(grip=(66, 28), ang=180, front=66, back=37.5),
                 grip_r=[(65, 27.2), (65, 28.8)]),
}
BASE['planted_open'] = dict(BASE['planted'], expr=None)
# 蓄力：再压低 4px、上身后撤，枪收到腰侧（枪尖将要迸出光）
_w = _shifted(BASE['guard'], 2, 4)
_w.update(expr='stern',
          el_r=(71, 29), wr_r=(72.5, 32), hand_r=(72, 33),
          el_l=(93, 28.5), wr_l=(87.5, 32.6), hand_l=(85, 33),
          lance=dict(grip=(72, 33), ang=180, front=66, back=37.5),
          grip_r=[(71, 32.2), (71, 33.8)],
          legs=[dict(hip=(84, 48), knee=(92, 59), ankle=(96, 68.8), toe=(99, 74.6), lit=False),
                dict(hip=(79, 48), knee=(70.5, 58), ankle=(67, 68.6), toe=(63, 74.6), lit=True)])
BASE['windup'] = _w
BASE['skill'] = dict(   # 圣技·大十字：压低重心弓步，骑枪水平向左刺出；披风、长发、前摆全部向右后方甩开
    head=(71, 11), torso=(70, 19), expr='stern',
    sh_r=(71.5, 23.5), el_r=(68.5, 28), wr_r=(67.2, 30.2), hand_r=(66, 30.3),
    sh_l=(86, 23.5), el_l=(91, 28.5), wr_l=(88.5, 31), hand_l=(86.8, 30.6),
    legs=[dict(hip=(82, 48), knee=(91, 59), ankle=(100, 69.5), toe=(105, 74.6), lit=False),
          dict(hip=(76, 48), knee=(68, 57), ankle=(66, 68.5), toe=(61, 74.6), lit=True)],
    lance=dict(grip=(66, 30.3), ang=180, front=66, back=37.5),
    paul_r=(70.5, 23.2), paul_l=(86.5, 23.2),
    braid=[(80.5, 22), (80, 31), (91, 34)], braid_tip=[(91, 34), (96, 35.5)],
    collar=([(73, 21), (71.5, 16), (74.5, 18), (75, 21)], [(82, 21), (83.5, 16.5), (84.5, 17), (84, 21)]),
    cape=[[(74, 21), (86, 20), (100, 17), (116, 18), (132, 22), (142, 27), (132, 31), (120, 37),
           (106, 42), (94, 45), (86, 36)],
          [(84, 46), (96, 48), (110, 50), (120, 54), (108, 57), (94, 55)]],
    cape_folds=[('R', [(100, 21), (118, 22), (134, 25)]), ('d', [(96, 32), (110, 32), (124, 30)])],
    lining=[(98, 36), (112, 37), (126, 34), (136, 30), (128, 36), (112, 40), (100, 40)],
    hair=dict(poly=[(79, 13), (86, 15), (96, 16), (106, 15), (114, 13), (110, 17), (118, 18), (108, 20),
                    (116, 23), (104, 22), (96, 22), (88, 21), (81, 19)],
              dark=[[(82, 17), (94, 19), (106, 19)], [(86, 20), (98, 21)]],
              light=[[(81, 14), (92, 16), (104, 16)]]),
    tabard=[dict(pts=[(80, 46), (86, 46), (96, 52), (106, 58), (112, 62), (104, 63), (94, 60), (84, 56)],
                 trim=[[(86, 55), (96, 59), (106, 61.5)], [(87, 47), (98, 53), (108, 59)]], lit='P',
                 flame=(94, 53)),
            dict(pts=[(76, 46), (82, 46), (88, 54), (94, 62), (88, 64), (82, 60), (76, 52)],
                 trim=[[(78, 50), (84, 57), (89, 62)]])],
    grip_r=[(65, 29.5), (65, 31)],
)
POSES = BASE     # 兼容旧名


def _flutter(P, k, amp):
    """布料 / 发丝随风抖动：离身体越远摆得越大。k = 帧号（4 帧一循环），amp = 幅度（像素）。"""
    if amp <= 0:
        return P
    P = _copy.deepcopy(P)
    ph = k * np.pi / 2
    cx = (P['sh_r'][0] + P['sh_l'][0]) / 2

    def wob(q, a=1.0):
        d = abs(q[0] - cx)
        w = min(1.0, max(0.0, (d - 6) / 40)) * a
        return (q[0] + 0.35 * amp * w * np.cos(ph + q[0] * 0.21),
                q[1] + amp * w * np.sin(ph + q[0] * 0.23))
    P['cape'] = [[wob(q) for q in c] for c in P['cape']]
    P['cape_folds'] = [(c, [wob(q) for q in l]) for c, l in P['cape_folds']]
    P['lining'] = [wob(q, 0.7) for q in P['lining']]
    P['hair']['poly'] = [wob(q, 0.8) for q in P['hair']['poly']]
    P['hair']['dark'] = [[wob(q, 0.8) for q in l] for l in P['hair']['dark']]
    P['hair']['light'] = [[wob(q, 0.8) for q in l] for l in P['hair']['light']]
    for tb in P['tabard']:
        tb['pts'] = [wob(q, 0.45) for q in tb['pts']]
        tb['trim'] = [[wob(q, 0.45) for q in l] for l in tb['trim']]
    return P


def pose_params(pose):
    """'guard' / 'guard:2'（第 2 帧抖动，默认幅度 1px）/ 'idle:1:3'（幅度 3px）。"""
    parts = str(pose).split(':')
    P = BASE.get(parts[0], BASE['idle'])
    if len(parts) > 1:
        amp = float(parts[2]) if len(parts) > 2 else 1.0
        P = _flutter(P, int(parts[1]), amp)
    return P


def tip(pose):
    """该姿势枪尖在画布上的坐标 (x, y)（与 body(pose) 同一坐标系）。"""
    L = pose_params(pose)['lance']
    d = _unit(L['ang'])
    return (L['grip'][0] + d[0] * L['front'], L['grip'][1] + d[1] * L['front'] + OY)


def hand(pose, side='r'):
    x, y = pose_params(pose)['hand_' + side]
    return (x, y + OY)


def _draw(cv, P):
    """按一组姿势参数，从后往前拼装全身。"""
    # ---- 披风（最后面）与银白内衬
    for pts in P['cape']:
        _panel(cv, pts, dark=True)
    for c, pts in P['cape_folds']:
        cv.put(cv.linem(pts) & cv.has('Rrd'), c)
    lining = cv.polym(P['lining'])
    cv.sep(lining, 'O')
    _shade(cv, lining, 'A', 'M', 'm', lx=-1, ly=0)
    # ---- 被风吹起的长发
    hd = P['hair']
    hm = cv.polym(hd['poly'])
    cv.put(hm, 'H')
    cv.put(hm & ~cv.shift(hm, 0, 1), 'h')
    for pts in hd['dark']:
        cv.put(cv.linem(pts) & hm, 'h')
    for pts in hd['light']:
        cv.put(cv.linem(pts) & hm, 'Y')
    # ---- 腿（后腿先画）
    for lg in P['legs']:
        _leg(cv, lg['hip'], lg['knee'], lg['ankle'], lg['toe'], lit=lg['lit'])
    # ---- 前摆
    for tb in P['tabard']:
        _panel(cv, tb['pts'], trim=tb['trim'], lit=tb.get('lit'))
        if 'flame' in tb:
            cv.patch(*tb['flame'], [".G..G..", "GG.GG.G", ".GG.GG.", "..G..G."])
    # ---- 骑枪（枪杆从身后穿过）
    L = P['lance']
    _lance(cv, L['grip'], L['ang'], L['front'], L['back'])
    # ---- 躯干
    cv.patch(*P['torso'], TORSO)
    # ---- 左臂（画面右，背光）
    _arm(cv, P['sh_l'], P['el_l'], P['wr_l'], P['hand_l'], lit=False)
    # ---- 胸前长辫
    _braid(cv, P['braid'], n=8)
    cv.stroke(P['braid_tip'], 'h', 1.6, 0.6)
    # ---- 右臂（画面左）握枪
    _arm(cv, P['sh_r'], P['el_r'], P['wr_r'], P['hand_r'])
    cv.put(cv.linem(P['grip_r']), 'm')
    # ---- 高领
    cv.poly(P['collar'][0], 'R')
    cv.poly(P['collar'][1], 'r')
    # ---- 肩甲
    _pauldron(cv, P['paul_r'], -1)
    _pauldron(cv, P['paul_l'], +1)
    # ---- 头
    _head(cv, *P['head'], P.get('expr'))


def body(pose='idle'):
    """全身像，160x112（宽x高），脚底在最后一行、身体居中（x≈80），以 anchor='bottom' 贴图。
    pose: idle（立绘站姿）/ planted / planted_open / grip / pull1 / pull2 / swing / guard / windup / skill，
    可加 ':k[:amp]' 取披风抖动帧（见 pose_params）。"""
    cv = Cv(BH, BW, PAL)
    cv.oy = OY
    _draw(cv, pose_params(pose))
    return cv.rgba(OUT)


def _eyes(pose):
    hx, hy = BASE[pose]['head']
    return [(hy + OY + r, hx + c) for r, c in HEAD_EYES]


META = dict(
    key='arianrhod',
    name_zh='雅里安洛德',
    title_zh='钢之圣女',
    rank_zh='使徒 第七柱',
    skill_zh='圣技·大十字',
    quote_zh='让我见识你们的觉悟',
    fx=dict(
        primary='#f3f5fa',
        secondary='#6ee6d6',
        description=(
            '圣技·大十字（聖技グランドクロス / Holy Grand Cross）：她压低身形、骑枪平举，'
            '先在枪尖凝聚青白色光点并绕枪身卷起螺旋气流（青色 #6ee6d6 细线），'
            '随即化作一道银白残影向左直线突进（身后拖出长条光迹与酒红披风残影）；'
            '命中点炸开一个巨大的银白色十字光（竖+横两道粗光柱，边缘带青色光晕与金色 #f3c95c 火花），'
            '十字停留片刻后向外碎裂成光屑。整体色调：银白为主、青色为辅、少量金色。'
        ),
    ),
    # 以下坐标均为 (行 y, 列 x)，全身图 160x76
    eyes_body={'idle': _eyes('idle'), 'skill': _eyes('skill')},
    weapon_tip=(30 + OY, 1),     # skill 姿势的枪尖（水平向左刺出）
    # 每个姿势的枪尖 / 右手（握枪手）坐标 (y, x)，特效从这里发出
    weapon_tips={k: tuple(round(float(v), 1) for v in tip(k)[::-1]) for k in BASE},
    hand_r={k: tuple(round(float(v), 1) for v in hand(k, 'r')[::-1]) for k in BASE},
    hand_l={k: tuple(round(float(v), 1) for v in hand(k, 'l')[::-1]) for k in BASE},
)


LANDMARKS = {
    # 像素下标（第 i 个像素 = i）。按 portrait() 实际画出的位置读
    'portrait': {
        'head_top': (40.5, 11.5), 'chin': (38.5, 51),
        'eye_near_in': (43, 33), 'eye_near_out': (50, 33), 'eye_near_top': (46, 33), 'eye_near_bot': (46, 35.5),
        'eye_far_in': (36, 34), 'eye_far_out': (30, 34), 'eye_far_top': (33, 34), 'eye_far_bot': (33, 36.5),
        'brow_near': (45, 30.7), 'nose': (38, 42), 'mouth': (40, 46),
        'cheek_far': (30, 37), 'jaw_far': (32.5, 47), 'jaw_near': (48.5, 48),
        'x_part': (36.1, 12.8), 'x_hair_far': (21, 32), 'x_hair_near': (64, 36), 'x_lock_tip': (42.8, 40.4),
        'x_ring_tl': (47.5, 57), 'x_ring_br': (53.1, 62.1),
        'x_rivet_c': (38.3, 64.5), 'x_rivet_l': (25.4, 64.8), 'x_rivet_far': (0.5, 65.5),
        'x_gorget_v': (29.5, 76.8), 'x_gorget_top': (38.5, 53.7),
        'x_blade_far': (15.3, 44.7), 'x_collar_near': (68, 40), 'x_blade_near': (61, 50.7),
    },
    # 像素下标（第 i 个像素 = i）。关节点与 POSES['idle'] 的连续坐标差 0.5。
    'idle': {
        'head_top': (79.5, 4.5), 'chin': (80, 13), 'neck': (79.5, 16),
        'shoulder_r': (71.5, 17), 'shoulder_l': (87.3, 17),
        'elbow_r': (68.1, 25.7), 'elbow_l': (87.9, 27), 'wrist_r': (63.3, 32.1), 'wrist_l': (88.7, 34.5),
        'waist': (79.5, 28.5), 'knee_r': (78.3, 56.9), 'ankle_r': (77.9, 68.1),
        'x_spike_r_up': (64.3, 7.5), 'x_spike_r_side': (61.5, 14.4),
        'x_spike_l_up': (94.9, 7.5), 'x_spike_l_side': (97.7, 14.4),
        'x_collar_tip_l': (85.5, 10), 'x_fauld_tip': (79.5, 43),
        'x_tabard_l_hem': (85.5, 70.5), 'x_tabard_l_edge': (94.5, 58.5), 'x_tabard_r_inner': (73.5, 61.5),
        'x_cape_tip_l': (99, 63.5), 'x_lance_butt': (99.3, 20), 'x_braid_end': (74, 28.5),
        'x_belt_r': (74.5, 29.5), 'x_belt_l': (84.5, 29.5),
    },
}
# 全身关键点是在站姿坐标系里读的，换到画布坐标（下移 OY）
LANDMARKS['idle'] = {k: (x, y + OY) for k, (x, y) in LANDMARKS['idle'].items()}
