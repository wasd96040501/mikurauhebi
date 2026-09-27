"""McBurn / 马克邦 —— 执行者 No.I「劫炎」。

照《闪之轨迹 II》立绘描的像素画（关键点对齐见 tools/measure.py、data/landmarks/mcburn.json）。
头像：头向画面左侧 3/4 转、带一点歪头的慵懒俯视；全身：3/4 朝左站立，远侧手藏在风衣里、
近侧（卷袖、黑绳缠臂）的手插在裤兜，风衣下摆向画面右后方大幅飘开。
识别点：青绿炸毛长发（刘海夹一缕粉、耳后长发束发尾转灰紫/粉，黑绳交叉扎住）、方框细眼镜、
暗红长风衣（黑色大翻领 + 银灰滚边，下摆橙色渐变 + 金色火纹）、敞开的白衬衫/深蓝马甲、
颈间几圈黑绳项圈、深紫长裤外侧青色交叉系带、酒红皮鞋。185cm 的瘦高个。
"""
import numpy as np
from PIL import Image, ImageDraw

import pixelkit as pk

PAL = {
    'O': '#1c0f18',                                                  # 描边
    'S': '#fff0d8', 's': '#fcd4b2', 'k': '#eb9e7e', 'K': '#b15a4c',  # 肤
    'A': '#a8e4d6', 'a': '#56b8a8', 'c': '#2f8078', 'C': '#1a4c4e',  # 发（青绿）
    'P': '#f68a92', 'p': '#c64e6e', 'm': '#7c2c4e',                  # 发（粉）
    'v': '#b09ab4', 'V': '#6c5678',                                  # 发尾（灰紫）
    'E': '#ec4a5e', 'e': '#c42646', 'f': '#8e1a3a', 'F': '#561232',  # 红风衣
    'H': '#ffbe78', 'h': '#f47868',                                  # 风衣下摆（橙渐变）
    'Y': '#ffe08a',                                                  # 金色火纹
    'L': '#5a4a5c', 'l': '#281c28',                                  # 黑翻领（银灰滚边用 t）
    'T': '#f6f0f4', 't': '#b4aec8',                                  # 白衬衫 / 滚边
    'N': '#34587c', 'n': '#1e2e46',                                  # 深蓝马甲
    'J': '#4e3e5c', 'j': '#2c2236', 'U': '#8c7cd2',                  # 深紫长裤 + 紫色缝线
    'Q': '#52c6c2', 'q': '#277478',                                  # 裤侧青色系带
    'B': '#7c2a46', 'b': '#44182a', 'Z': '#e8e2dc',                  # 皮鞋 / 鞋底
    'I': '#c45c98', 'i': '#5c1a42', 'W': '#ffffff',                  # 眼：瞳 / 瞳暗 / 眼白
    'G': '#e8eef8', 'g': '#f2c6b4',                                  # 眼镜：银框 / 镜片（透着肤色）
    'R': '#e02838',                                                  # 耳钉
    'D': '#7c7488',                                                  # 眼镜：深银框（头像）
    'X': '#ffd84a', 'x': '#ff7a2a',                                  # 魔人化：金瞳 / 火纹
    '1': '#ffd060', '2': '#e8401c', '3': '#8a1420', '4': '#2c060e',  # 魔人化火色（魔剑等仍在用）
    '5': '#300812', '6': '#64101e',                                  # 魔剑 Angbar 剑身
    '7': '#fffaf2', '8': '#e6dedc', '9': '#b8aab0', '0': '#6e5a66',  # 魔人化发色（白发：高光 / 基色 / 阴影 / 暗部）
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

    def m_ell(self, box):
        return self._mask(lambda d: d.ellipse(box, fill=1, outline=1))

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

    def ell(self, c, box, where=None):
        return self.fill(self.m_ell(box), c, where)

    def px(self, c, pts, where=None):
        for x, y in pts:
            if 0 <= x < self.w and 0 <= y < self.h:
                if where is None or KEYS[self.a[y, x]] in where:
                    self.a[y, x] = KID[c]

    def stamp(self, x0, y0, rows):
        """ASCII 贴片：'.'/' ' 跳过，'_' 擦成透明。"""
        for dy, r in enumerate(rows):
            for dx, ch in enumerate(r):
                x, y = x0 + dx, y0 + dy
                if ch in '. ' or not (0 <= x < self.w and 0 <= y < self.h):
                    continue
                self.a[y, x] = 0 if ch == '_' else KID[ch]

    def shade(self, base, dark, dy=1, dx=1, where=None, keep=None):
        """把 base 色区域中朝右下（背光）的边缘带换成 dark。"""
        m = self.is_(base)
        region = self.is_(keep or base)
        sh = np.zeros_like(region)
        sh[:self.h - dy, :self.w - dx] = region[dy:, dx:]
        edge = m & ~sh
        if where is not None:
            edge &= self.is_(where) if isinstance(where, str) else where
        self.a[edge] = KID[dark]
        return edge

    def light(self, base, lit, dy=1, dx=1, keep=None):
        """把 base 色区域中朝左上（受光）的边缘换成 lit。"""
        m = self.is_(base)
        region = self.is_(keep or base)
        up = np.zeros_like(region)
        up[dy:, dx:] = region[:self.h - dy, :self.w - dx]
        edge = m & ~up
        self.a[edge] = KID[lit]
        return edge

    def border(self, inner, outer_keys, c):
        """outer_keys 色块中与 inner 相邻（4 邻域）的一圈像素涂成 c（选择性内描边）。"""
        g = inner.copy()
        g[1:] |= inner[:-1]
        g[:-1] |= inner[1:]
        g[:, 1:] |= inner[:, :-1]
        g[:, :-1] |= inner[:, 1:]
        self.a[g & ~inner & self.is_(outer_keys)] = KID[c]

    def outline(self, c='O'):
        a = self.a > 0
        g = a.copy()
        g[1:] |= a[:-1]
        g[:-1] |= a[1:]
        g[:, 1:] |= a[:, :-1]
        g[:, :-1] |= a[:, 1:]
        self.a[g & ~a] = KID[c]

    def rgba(self):
        out = np.zeros((self.h, self.w, 4), np.float32)
        out[..., :3] = _RGB[self.a]
        out[..., 3] = (self.a > 0).astype(np.float32)
        return out


# ================================================================ 头像 80×80
# 底稿：官方《闪之轨迹 II》立绘头胸部（refs/mcburn/crop_sen2.png，full.png 的裁切）按固定取景
# （两眼中心 + 下巴三点，见 tools/measure.py fidelity）缩到 80×80，按材质分色块后手工清理
# （发束里的暗线按原画位置补回、细发梢补尖、胸口碎线抹平、翻领乱色归并）。
# 取景把原画约 20° 的歪头摆正：头骨顶 ≈ (42,13)，下巴尖 (36,53)。
# 近侧（画面右）眼 x40..49 / y29..31，远侧眼 x28..35 / y31..33；镜片在眼睛正下方（他从镜框上沿看人），
# 近侧镜片 x40..49 / y32..35，远侧 x26..34 / y35..37；鼻尖 (36,41)，嘴 (36..40,45)，耳 x55..60 / y30..43。
# 底稿里不含眼、眉、镜、嘴、耳廓线、耳钉、X 形绑发绳、项圈和翻领滚边——这些由 portrait() 按表情 / 部件画上。
PORTRAIT_BASE = [
    '....................................A............c..............................',
    '.....................................AA...........c.............................',
    '......................................AAA.........cc............................',
    '.......................................AAAAA......ccc......C....................',
    '........................................AAAAAAAA...ccc.....c....................',
    '........................................aaAAAAAAAACCccc.....a...................',
    '...............................CCcAAAAAAAAaAAAAAAAaaCcccc..ac...................',
    '................c................cccccAAAAAAAAAAAAAaacccc..ac...................',
    '.................c.............AAAAAAAAAAAAAAAAAAAAaaccccc.ac...................',
    '..................ccc......AAAAAAAAAAAAAAAAAAAAAAccAacccccCcC...................',
    '...................cccccccAAAAAAAAAAAAaccAAAAAAAAAccacccccCcC...................',
    '.....................cccccAAAAAAAAAAAacAAAAAAAAAAaaaccccccccC...................',
    '.......................cccccAAAAAAAAAAAAAAAAAAAcaaacCCcccccCCC..................',
    '..........................AAAAAAAAAAAAAAAAAAAAcccaaacccccccccca.................',
    '........................AAAAaaAAAAAAAAcAAAAAAcaaaacaaccccCCcccca................',
    '.......................ccccccAAAAAAAAcPAAAAAccccccccaccaaaAOCCccc...............',
    '..............P.....PPPccccaAAAAAAAAaPAAAAAccccccccccccaaaAAAaOCCC..............',
    '...............PPPPPPccccaaaAAAAAaAacPAAAaccCCCcccccccccCCAaaCaAAAaaaC..........',
    '..................PcccccaaaaAaAAaAacPAAAaccCCcccCaaaacccccCCCCOaaaC.............',
    '......................caaaaaaaaAcaccPAaacccCcssCCCCcccaccccCCCccccC.............',
    '......................aaaaaAaaacaacPaAaaccCkkssssscaaaaaaaCCcccccccC............',
    '.....................aaaaaAaaaccaccPaaaacckkssssssssCccaaaacccccccccc...........',
    '....................PaacaavcaaccacKPaaaccckkssssssssCcccaAAAAcccccccccccCCCC....',
    '...................PaacaavcaacCCacKavaacckksssssssssCCcaAaaAAOccccccccccCC......',
    '.................PPccccaavcacCCaCKPavacccCkksssssssssCCcaaAAAOaccccCC...........',
    '............C...PPccccaavcccCCCaCKPaaaccCCkksssssssssCCcCaAAAOaaOccaC...........',
    '.............CCCCCCcccaaccccCCCCCKPaaccckksssssssssssCCccCaOOOOOOCccaC..........',
    '.................CcCcaaPcccCcCCCCkPPcCCckssssssssssssscccCCCCaaaCOOAaaC.........',
    '...............CCCCCCaPcccCCcKKCCkPPcCCckkkksssssSSSssCCccCCCOOOOOAAAaCCcc......',
    '...................CaPPccCCccKkKKCPPcCckkkkKKKKKKsSSssCCccccccOOOcAAccccccc.....',
    '...................aaPcccCackkkKKKPPCCCkkkKKKKKKKsSSssCCCCCccccccccccccC........',
    '..................aaPcccCCaCkkkkKKPPCsCsssKKKKKKsSSSSsCCCkkkccAAAccccccCC.......',
    '.................acPcccCCCCCkkkkkkkPCssssssssssssssssCCCkkkkkccAAAAccccccccccc..',
    '...............CccPCcCCCCCCCkkkkkkkCkSSSSSSSSSSSSSssssCkKkkkkccccaaAAAcccccc....',
    '..............CccPCC.CCCCCCCkkkkkkkCkSSSSSSSSSSSSSssSsCkKkkkkCcccccccccccccccc..',
    '............CCCCCC..CC.CCCCCkkkkkksssSSSSSSSSSSSSSsSSSkkKkkKKCCccccCccccccaaaAA.',
    '...........CC...C...C..CCCCCkkkkkssssSSSSSSSSSSSSSsSSkkKKkkKKCCCccccC...........',
    '.......................CCCCCkkkkssssssSSSSSSSSSSSSSSSkkkkkKKcCcCCccccC..........',
    '......................CCCCCCkkkssssSsSSSSSSSSSSSSSSSSkkkkkKKcCcCCcccaAA.........',
    '......................CCCCCCCSSSSSSsSSSSSSSSSSSSSSSSkkkkkKkKcccCCCCCcaaac.......',
    '.....................CCCCCCCCSSSSSSSSSSSSSSSSSSSSSSSkkkkkkkccccCCCCcccccc.......',
    '......................CCC.CCCSSSSSSSSSSSSSSSSSSSSSSSkkkkkkCccccCcccCccccccc.....',
    '.....................CCC..CC.SSSSSSSSSSSSSSSSSSSSSSkkkkkkCcccccCccccC...........',
    '....................CC....C...SSSSSSSSSSSSSSSSSSSSSkkkKKCccccccCccccc...........',
    '..............................SSSSSSSSSSSSSSSSSSSSkkkKKCCccccccCcccccc..........',
    '...............................SSSSSSSSSSSSSSSSSSkkssKKCCccccccccccccccc........',
    '................................SSSSSSSSSSsssSSSSssKKKKCCccAcccccccAcc..........',
    '................................SSSSSsssssSSSSSssKKKKKKccccAAcccccccAAa.........',
    '.................................SsssssssssssssKKKKKkkkccccAAccccccccaAA........',
    '..................................sssssssssssKKKKKkkkkcccOAAAccccccCcccAAc......',
    '..................................sssssssssKKKKKkkkkkccccOAAOOcccccC..ccccCCC...',
    '................................lllssssssKKKKKkkkkkkkccccOOOcCcccccc............',
    '............................OlllllllsssKKKKKkkkkkkkkcCccOAAOcCccccac............',
    '.........................llvlllllllllsskkkkkkkkkkkkCCCccPAAOCcccccaAA...........',
    '......................LLllvlllllllLlllkkkkkkkkkkkOCCCCaaPAAOCccccccAA...........',
    '.........ff.........LLLlvllllllllLOOlkkkkkkkkkkkOVCCCCAOaAACCCCcCcccAA..........',
    '.........fffEEEEEffLLLLLLlllllllllnkkkkkkkkkkkkOVVCVCAAOOaOCCCCCCCCCCA..........',
    '.........ffffffFllLLLLLLLLLLlllllnkkkkkkkkkkkkkVVkVVTACOVVPCCCCCCCCCCCA.........',
    '........EEEEeeffFLLLLLLLLLLLLlllnkkkkkkkkkkkVVVkkvVTTvCvPPCCCCCCCCCCCCCC........',
    '........EeeeeeeLLLLLLLLLLLLLLlnnkkkkkkkkkkkkkkkkvVvvvvvvvPCVVCCCCCCCCC..CCC.....',
    '.......EEEeeeeLLLLLLLLLLLLLLlnnkkkkkkkkkkkkkkkkVVvVVvvPvvCVVVVVCCCCCCCCC........',
    '.......EFEeeeLLLLLLLLLLLLLLllnkkkkkkkkkkkkkkkkPVVVVvvvPvPVVVOVVVVLCCCC..........',
    '.......FFEeeLLLLLLLLLLLLLLllnkkkkkkksssssskkkPVVVVvvvPvvPVVVOVVVVvCCCC..........',
    '......eFEeeLLLLLLLLLLLllLllnkkkkkkksssssssskPVVVVVvvvPvVVVVCOVVVvvvCCCC.........',
    '......eEeeeLLLLLLLLllllLlllnkkkkkksssssssssPPVkkVvvvvvVPVVVCCVVVVvvvCCC.........',
    '.....eeEeeLLLLLLLlllllLLllnkkkkkkksssssssssPVkkkvvvVPvVPVVVOCCVVVvvvlCCC........',
    '.....EEeeLLLLLOOlllllLLllnkkkkkkkkssssssssPkkkkkvvVvPVVPVVVOOCVVVLvvllCCC.......',
    '.....EeeeLLLOlllllllLlLllnkkkkkkkkkssssssPkkkkkvvVVPVVVVVVVOllOVCOOLLLLLLC......',
    '....EEeeOOOllllllllllLllnkkkkkkkkkkkssssPmmkkkkvVVPVVVVVVVVOlllOCCOOOLLLLLC.....',
    '....eeeOlllllllllllllLlnskkkkkkkkkkkssssvmkkkkvVVVPVVVmVVVVOlllllOOCCCCCCC.C....',
    '....eeelllllllllllllLllnsskkkkkkkkkkkkkmvkkkkmvVVPVVVmmVVVVOllllOlOlOlCCcccCC...',
    '...feeOlllllllllllllllnsssskkkkkkkkkkkkmmkkkmmmmmmmmPVmVVVVOllllOlOllllOccccCC..',
    '...feelllOllllllllllllnssssskkkkkkkkkkkmmkkmmmmmPmmmPmmVVVVOllllOlOllllFOcccccCC',
    '..EeeOlOlllllllllllllnsssssssskkkkkkkkSPPPPmmmmPmmmmmmmmLLLOllllOlOlllllF.cccccc',
    '.EfeelllOllllllllllnnssssssssskkkkkkSSSPppmmmmPmmmmPmmmLLLmOllllOlOlllllFf...ccc',
    '.EfeOllllllllllllnnnnsssssssssskkkkkSSmpppmmmPmmmmPmmmmmmmmOllllOlOllllOFFf....c',
    'EfffOlllllllllllnnnnnssssssssssskkkSpppmpmmmPmmmmmPmmmmmmmmmllllllllllllFmmf...a',
    'EfffllllllllllllnnnnssssssssssssssppppmmmmmPmmmmmPmmmmmmmmmLllllllllLlllOmmm...a',
    'fffOlllllllllllnnnnnssssssssssssssSSSSpppPPmmmmmPPmmmmmmOlmmLlllllllLllllOmmE...',
    'fffOOOOOOOOOlllnnnnsssssssssssssssssSSSpPpmmmmmmPmmmmmmmllllmlllllllLllOOOOmmE..',
]

EAR = [  # 盖章左上角 (53, 30)：外缘 K、耳甲 s 受光、耳钉 R
    '....KKK.',
    '...KkkkK',
    '..KksKkK',
    '..kskKsK',
    '..ksKksK',
    '..ksKksK',
    '..kskKsK',
    '..ksKksK',
    '..kkKkK.',
    '..kkkKK.',
    '..kkkK..',
    '.kkkK...',
    '.kRK....',
    '.kK.....',
]

EYE_NEAR = {  # 盖章左上角 (40, 29)；i/I 换成魔人化的金瞳
    'bored': ['.OOOOOOOOO',
              'OWWWiiiWO.',
              '.kkWIIIk..'],
    'narrow': ['.OOOOOOOOO',
               'OOOWiiiOO.',
               '.kkWIIIk..'],
}
EYE_FAR = {  # 盖章左上角 (28, 31)
    'bored': ['OOOOOOO',
              '.WWiiiO',
              '.kWIIIk'],
    'narrow': ['OOOOOOO',
               'OOWiiiO',
               '.kWIIIk'],
}
MOUTHS = {  # 盖章左上角 (35, 43)：一道嘴线（左端最深）+ 下唇下一点阴影
    None: ['.......',
           '.......',
           '.OKKk..',
           '.......',
           '...k...'],
    'smirk': ['.......',
              '....KK.',
              '.OKK...',
              '.......',
              '...k...'],
    'grin': ['......O',
             'OOOOOO.',
             '.OWWWO.',
             '..OOO..',
             '..kkk..'],
    'demon': ['O.....O',
              'OOOOOOO',
              '.OWWWO.',
              '..OOO..',
              '..kkk..'],
}


def portrait(expr=None):
    """80×80 头像。expr: None（默认，慵懒半睁眼）/ 'smirk'（嘴角一挑、眼更眯）/ 'grin'（燃起来了：咧嘴、镜片反光）/
    'demon'（魔人化：头发转白、摘眼镜、金瞳、狞笑）。"""
    c = Cv(80, 80)
    c.stamp(0, 0, PORTRAIT_BASE)
    # 耳朵（耳廓外缘 + 内侧软骨线）与耳钉
    c.stamp(53, 30, EAR)
    # 黑绳交叉扎住的发束（X 形绑绳）：头顶一个、耳上两个并排、胸前发束两个
    for x, y in ((61, 16), (61, 25), (65, 25), (59, 50), (57, 55)):
        c.px('O', [(x - 1, y - 1), (x + 1, y - 1), (x, y), (x - 1, y + 1), (x + 1, y + 1)])
    # 黑翻领外缘一道银灰滚边
    c.line('t', [(21, 56), (16, 61), (12, 67), (10, 73), (8, 79)], where='lLOn')
    # 颈间黑绳项圈 + 垂到胸口的吊坠绳（不压在头发上）
    skin = 'SskK'
    c.line('O', [(36, 55), (40, 56), (44, 57), (47, 57)], where=skin)
    c.line('O', [(38, 57), (36, 63), (33, 71), (30, 79)], where=skin)
    demon = expr == 'demon'
    iris, dark = ('X', 'x') if demon else ('I', 'i')
    eyes = 'bored' if expr is None else 'narrow'
    # 眉（近侧；远侧眉被刘海盖住）、上眼睑的双眼皮线
    c.px('C', [(42, 25), (43, 25), (44, 24), (45, 24), (46, 24), (47, 24), (48, 24), (49, 24)], where='Ssk')
    c.px('k', [(43, 27), (44, 27), (45, 27), (46, 27), (47, 27), (48, 27)], where='Ss')
    c.stamp(40, 29, [r.replace('i', dark).replace('I', iris) for r in EYE_NEAR[eyes]])
    c.stamp(28, 31, [r.replace('i', dark).replace('I', iris) for r in EYE_FAR[eyes]])
    c.px('W', [(45, 30)])
    # 鼻：鼻梁背光一侧一道半影，鼻底两点暗影
    c.px('s', [(36, 38), (36, 39)], where='S')
    c.px('k', [(37, 40)])
    c.px('K', [(36, 41), (37, 41)])
    c.stamp(35, 43, MOUTHS.get(expr, MOUTHS[None]))
    if not demon:                  # 魔人化时摘掉眼镜
        # 细框眼镜：上框 1px 深银、框角亮银、下框只留一道透肤色的镜片边，镜片本身透明（眼睛完整露出）
        c.line('D', [(41, 32), (49, 32)])
        c.line('D', [(27, 35), (33, 35)])
        c.line('D', [(34, 35), (37, 33), (40, 32)])                  # 鼻梁架
        c.line('D', [(50, 32), (53, 31)])                            # 镜腿没入鬓发
        c.line('g', [(42, 35), (45, 35), (46, 34), (48, 34)], where='Ssk')
        c.line('g', [(28, 37), (33, 37)], where='Ssk')
        c.px('t', [(41, 32), (49, 32), (40, 33), (49, 33), (41, 34), (27, 35), (26, 36), (27, 37),
                   (33, 35), (34, 36), (39, 35)])                     # 框角 + 鼻托
        c.px('t', [(x, 37) for x in range(26, 34)], where='cCaA')      # 远侧下框压在头发上的一段
        c.px('W', [(47, 33)] if expr != 'grin' else [])                # 镜片反光 1px
    if expr == 'grin':             # 燃起来了：镜片一闪
        c.px('W', [(46, 34), (47, 33), (48, 33)])
        c.px('G', [(45, 34), (48, 32), (30, 36)])
    # 头发外缘选择性描边：只在背光一侧（右 / 下方挨着透明）把亮青换成暗青，左上受光的发梢保持亮
    op = c.a > 0
    rim = np.zeros_like(op)
    rim[:, :-1] |= ~op[:, 1:]
    rim[:-1] |= ~op[1:]
    c.a[rim & op & c.is_('Aa')] = KID['c']
    if demon:
        for k, v in DEMON_HAIR.items():
            c.a[c.a == KID[k]] = KID[v]
    return c.rgba()


# ================================================================ 全身 48×72
# 比例照 Sen II 立绘：头骨顶 y≈3、下巴 y≈10.5（头高 ≈7px，约 9 头身），锁骨 y13，腰带 y29，裆 y36，
# 膝 y50，踝 y68，鞋底贴最后一行。整个人（含炸毛）正好 72px 高。
# 画法：每个部件是一个函数，由 POSES 里的少量参数（髋/膝/踝位置、手臂姿势、下摆飘动量……）摆放，
# 下一轮做走路、冲刺、挥手放火等关键帧时只需要加新的参数组。

_BASE = dict(
    leg_r=((9.8, 36), (9.8, 50), (10.8, 69)),       # 远侧（画面左）腿：髋、膝、踝
    leg_l=((19, 37), (24, 50), (29, 69)),           # 近侧（画面右）腿斜跨出去
    shoe_l='front',                                 # 近侧鞋：front 朝镜头 / side 朝左（走路时）
    arm_far='coat',                                 # 远侧手：coat 藏在风衣里 / cast 张掌 / grip 握剑前伸 /
                                                    #   raise 举剑 / windup 横在胸前蓄力 / swing1 / swing2 / follow / shoulder
    arm_near='pocket',                              # 近侧手：pocket 插兜 / scratch 挠头
    flutter=0.0,                                    # 风衣下摆被热浪掀起的程度 0..1
    sway=0.0,                                       # 风衣下摆左右摆（走路），-1..1
    head=(9, 0), face='bored',                      # 脸：bored / yawn / grin / demon
    demon=False,                                    # 魔人化：头发转白、摘眼镜、金瞳
)


def _pose(**kw):
    d = dict(_BASE)
    d.update(kw)
    return d


# 走路（向画面左）：接地 - 过渡 - 接地 - 过渡，一步一拍
_HIP_R, _HIP_L = (11, 36), (16, 36.5)
POSES = {
    'idle': _pose(),
    'skill': _pose(leg_r=((9.8, 36), (9.6, 50), (10.4, 69)), arm_far='cast', flutter=1.0, face='demon', demon=True),
    'walk0': _pose(leg_r=(_HIP_R, (14.5, 52), (20, 68)), leg_l=(_HIP_L, (11, 52), (6, 69)), shoe_l='side', sway=-1.0),
    'walk1': _pose(leg_r=(_HIP_R, (8, 50), (10, 65)), leg_l=(_HIP_L, (15, 52), (15, 69)), shoe_l='side', sway=-0.3),
    'walk2': _pose(leg_r=(_HIP_R, (7.5, 52), (4, 69)), leg_l=(_HIP_L, (19, 52), (23, 68)), shoe_l='side', sway=1.0),
    'walk3': _pose(leg_r=(_HIP_R, (11, 52), (11, 69)), leg_l=(_HIP_L, (12.5, 50), (15, 65)), shoe_l='side', sway=0.3),
    'yawn': _pose(arm_near='scratch', face='yawn'),
    'reach': _pose(arm_far='cast', face='grin', flutter=0.3),
    'pull0': _pose(arm_far='grip', face='grin', flutter=0.5),
    'pull1': _pose(arm_far='raise', face='grin', flutter=0.7),
    'dwind': _pose(arm_far='windup', face='demon', demon=True, flutter=0.8,
                   leg_r=((9.8, 36), (8.5, 50), (7, 69)), leg_l=((19, 37), (25, 50), (31, 69))),
    'dswing1': _pose(arm_far='swing1', face='demon', demon=True, flutter=1.0,
                     leg_r=((9.8, 36), (8.5, 50), (7, 69)), leg_l=((19, 37), (25, 50), (31, 69))),
    'dswing2': _pose(arm_far='swing2', face='demon', demon=True, flutter=1.0,
                     leg_r=((9.8, 36), (8.5, 50), (7, 69)), leg_l=((19, 37), (25, 50), (31, 69))),
    'dfollow': _pose(arm_far='follow', face='demon', demon=True, flutter=0.8, sway=-0.6,
                     leg_r=((9.8, 36), (8.5, 50), (7, 69)), leg_l=((19, 37), (25, 50), (31, 69))),
    'dshoulder': _pose(arm_far='shoulder', face='demon', demon=True, flutter=0.4),
}

# 各姿势手里的剑：(握点 x, 握点 y, 剑身朝向角度°（0=画面右，顺时针为正）, 剑身透视长度系数, 剑在身体后面?)
BLADE = {
    'pull0': (3, 12, 115, 1.0, False),
    'pull1': (7, 3, -80, 1.0, False),
    'dwind': (23, 16, -8, 1.0, True),
    'dswing1': (11, 25, 150, 0.55, False),
    'dswing2': (2, 13, 183, 1.0, False),
    'dfollow': (6, 10, 235, 0.8, True),
    'dshoulder': (13, 14, -35, 1.0, True),
}
# 远侧手（掌心 / 拳）位置，火从这里出来
HAND_FAR = {'skill': (2, 11), 'reach': (2, 11), 'pull0': (3, 12), 'pull1': (7, 3), 'dwind': (23, 16),
            'dswing1': (11, 25), 'dswing2': (2, 13), 'dfollow': (6, 10), 'dshoulder': (13, 14)}

# 头（x 9..22, y 0..11）：3/4 朝左，远侧眼 x14、近侧眼 x16..17，眼镜一行在 y8
HEAD_BODY = [
    "......a..a....",
    ".....aAa.ac...",
    "...a.aAAaaac..",
    "....aAAaaaacc.",
    "..aaAAaaAaacC.",
    ".aaAaaSSSacaC.",
    ".capcSSOisacC.",
    "..ccaiSGtkcaC.",
    "..ccctkSskcaC.",
    "...cSSSKkkcaC.",
    "....CSSskkcaC.",
    "......sk..cac.",
]


# 脸的变体：替换 HEAD_BODY 的第 6..10 行
FACES = {
    'bored': None,
    'yawn': [".capcSSKKsacC.",
             "..ccaKSGtkcaC.",
             "..ccctkSskcaC.",
             "...cSSOOkkcaC.",
             "....CSSOkkcaC."],
    'grin': [".capcSSOisacC.",
             "..ccaiGGGkcaC.",
             "..ccctkSskcaC.",
             "...cSOWWOkcaC.",
             "....CSSskkcaC."],
    'demon': [".capcSSOXsacC.",
              "..ccaXSSSkcaC.",
              "..cccSkSskcaC.",
              "...cSOWWOkcaC.",
              "....CSSskkcaC."],
}
# 魔人化的发色：青绿 -> 白（场景再从发梢喷火）
DEMON_HAIR = {'A': '7', 'a': '8', 'c': '9', 'C': '0', 'v': '8', 'V': '9', 'P': '7', 'p': '8'}   # 魔人化：头发转白


def _leg(c, hip, knee, ankle, w, lace_side, shoe):
    """一条腿：深紫长裤（两段梯形）+ 一侧青色交叉系带 + 皮鞋。w=(髋宽, 膝宽, 踝宽)。"""
    pts = [np.array(p, float) for p in (hip, knee, ankle)]
    L, R = [], []
    for i, p in enumerate(pts):
        d = pts[min(i + 1, 2)] - pts[max(i - 1, 0)]
        n = np.array([d[1], -d[0]]) / np.hypot(*d)
        if n[0] < 0:
            n = -n
        L.append(p - n * w[i] / 2)
        R.append(p + n * w[i] / 2)
    poly = [tuple(v) for v in L + R[::-1]]
    m = c.poly('J', poly)
    # 背光：右半边
    c.fill(m & c.m_poly([tuple(v) for v in [(L[i] + R[i]) / 2 + np.array([0.6, 0]) for i in range(3)] + R[::-1]]), 'j')
    edge = L if lace_side == 'l' else R
    off = np.array([0.8, 0]) if lace_side == 'l' else np.array([-1.2, 0])
    lace = [tuple(v + off) for v in edge]
    c.line('q', lace, width=2, where='Jj')
    yy = np.mgrid[0:c.h, 0:c.w][0]
    c.fill(c.m_line(lace, width=2) & ((yy % 3) == 1) & c.is_('q'), 'Q')
    # 裤脚的紫色花纹
    ax, ay = ankle
    c.px('U', [(int(ax) - 1, int(ay) - 4), (int(ax), int(ay) - 3), (int(ax) + 1, int(ay) - 4)], where='Jj')
    c.stamp(int(round(ankle[0])) + shoe[1], int(round(ankle[1])) + shoe[2], shoe[0])


SHOE_SIDE = [          # 朝左的鞋（远侧脚）
    "...BpBBBBBb",
    ".BpBBBBBBBb",
    "ZZZZZZZZ.ZZ",
]
SHOE_FRONT = [         # 朝向镜头的鞋（近侧脚）
    "BpBBBb",
    "BBBBBb",
    "ZZZZZZ",
]


def _coat_tone(c, m, y_h, y_H, slope=0.0, x0=0):
    """风衣下摆的红→橙渐变：y 超过 y_h 转 h、超过 y_H 转 H（slope 让分界线斜着走）。"""
    yy, xx = np.mgrid[0:c.h, 0:c.w]
    yv = yy + slope * (xx - x0)
    c.fill(m & (yv >= y_h), 'h')
    c.fill(m & (yv >= y_H), 'H')


def _b_coat_back(c, P):
    """两腿之间露出的风衣后片（在腿后面）。"""
    fl, sw = P['flutter'], P['sway']
    m = c.poly('e', [(14, 34), (18, 36), (21, 46), (23 + sw, 54), (24 + fl + 2 * sw, 62 - fl), (19 + 2 * sw, 63 - fl),
                     (14 + sw, 62 - fl)])
    _coat_tone(c, m, 53, 60, 0.3, 14)
    c.line('f', [(17, 40), (18, 50), (19, 62)], where='ehH')
    c.line('Y', [(15, 58), (17, 55), (19, 57), (21, 61)], where='hH')


def _b_legs(c, P):
    _leg(c, *P['leg_r'], w=(5.5, 5, 4.5), lace_side='r', shoe=(SHOE_SIDE, -8, 0))
    shoe = (SHOE_FRONT, -2, 0) if P['shoe_l'] == 'front' else (SHOE_SIDE, -8, 0)
    _leg(c, *P['leg_l'], w=(5, 4.5, 4), lace_side='l', shoe=shoe)


def _b_coat_front(c, P):
    fl, sw = P['flutter'], P['sway']
    # 远侧（画面左）前片：从腋下斜着散开，下摆尖在 (1,64)
    m = c.poly('e', [(12, 14), (13, 22), (11, 30), (8, 36), (6.5, 44), (6, 52), (6 + sw, 58), (5.5 + 1.5 * sw, 61), (1 + 2 * sw, 65 - 2 * fl),
                     (0 + sw, 60 - 2 * fl), (1.5, 52), (3.5, 44), (6, 36), (8, 28), (9, 22), (10, 18)])
    c.line('f', [(9, 26), (5, 42), (3, 56)], where='e')
    c.line('F', [(10, 30), (6, 44), (5, 58)], where='e')
    _coat_tone(c, m, 50, 59, -0.4, 0)
    c.line('Y', [(2, 63), (3, 59), (5, 57), (6, 60)], where='hH')
    # 近侧前片 + 大幅向右后方飘开的下摆
    tail = [(46 + 2 * fl - 2 * abs(sw), 58 - 3 * fl + 2 * sw), (45 + 2 * fl - 2 * abs(sw), 60 - 3 * fl + 2 * sw),
            (40 + sw, 61 - 2 * fl + sw), (35 + sw, 61 - fl), (31 + sw, 62)]
    m = c.poly('e', [(19, 12), (24, 12), (27, 14), (28, 22), (28, 30), (29, 36), (31, 42), (35, 47), (40, 51),
                     (44, 55)] + tail + [(29, 58), (27, 52), (25, 45), (22, 37), (20, 29), (19, 20)])
    _coat_tone(c, m, 45, 52, -0.35, 26)
    c.line('f', [(21, 18), (23, 30), (26, 42), (29, 52), (31, 61)], where='ehH')     # 前片的深色压线
    c.line('F', [(22, 20), (24, 32)], where='e')
    c.line('f', [(30, 44), (36, 52), (42, 57)], where='ehH')
    c.line('Y', [(33, 56), (36, 53), (39, 55), (42, 54)], where='hH')
    c.line('Y', [(28, 50), (30, 47)], where='hH')
    c.shade('e', 'f', 1, 1, keep='efFhHY')
    c.light('e', 'E', 1, 1, keep='efFhHY')


def _b_torso(c, P):
    # 敞开的胸口 + 深蓝马甲 + 白衬衫边 + 两侧黑色大翻领；腰胯比肩膀偏左（懒散地斜靠着站）
    c.poly('s', [(14, 11), (19, 11), (18, 16), (17, 21), (16, 21), (15, 16)])
    c.px('k', [(18, 12), (18, 13), (17, 15), (16, 12)])
    c.poly('N', [(14, 18), (18, 17), (18, 22), (16, 30), (15, 36), (13, 36), (13, 30), (13, 22)])
    c.line('n', [(17, 20), (16, 28)])
    c.px('O', [(15, 23), (14, 26)], where='N')                 # 马甲交叉系绳
    c.poly('T', [(13, 16), (14, 16), (14, 20), (13, 26), (12, 29), (11, 29), (11, 27)])   # 远侧白衬衫
    c.poly('T', [(18, 17), (19, 16), (18, 24), (17, 29), (16, 29)])               # 近侧白衬衫
    c.px('t', [(11, 28), (12, 27), (17, 27), (17, 28), (16, 29)])
    c.poly('J', [(11, 30), (16, 30), (16, 36), (12, 36)])                          # 裤腰
    c.poly('N', [(9, 30), (12, 30), (12, 34), (10, 36)])                           # 马甲下摆（远侧）
    c.poly('N', [(14, 30), (16, 30), (16, 37), (15, 37)])                          # 马甲下摆（近侧）
    c.px('n', [(11, 33), (10, 35), (16, 33), (16, 34)])
    c.line('j', [(11, 29), (17, 29)])                                              # 腰带
    c.px('G', [(13, 29)])
    # 翻领：远侧一窄条、近侧宽大
    c.line('l', [(13, 12), (13, 20), (11, 29)])
    c.line('t', [(12, 13), (12, 20)])
    c.poly('l', [(19, 11), (23, 12), (21, 19), (19, 28), (18, 29), (18, 20)])
    c.line('L', [(22, 12), (20, 20), (19, 28)])
    # 项圈、吊坠
    c.line('O', [(15, 12), (18, 12)])
    c.px('O', [(16, 14), (16, 15)])
    # 胸前那束粉紫长发（从近侧耳后垂下）
    c.poly('v', [(19, 10), (21, 11), (21, 14), (20, 18), (19, 18), (19, 13)])
    c.px('P', [(20, 15), (19, 16), (20, 17), (19, 17)])
    c.px('V', [(21, 12), (21, 13), (20, 14)])
    c.px('c', [(20, 11)])


# 近侧小臂（x21..31, y22..31）：翻折的黑袖口 + 缠着黑绳的小臂斜插进裤兜
FOREARM = [
    ".....llllle",
    ".....lttttl",
    "......lllll",
    "........Ssk",
    ".......SOs.",
    ".....SsKs..",
    "....SsOk...",
    "..SsKsk....",
    ".kssk......",
    ".kk........",
]


def _b_arm_near(c, P):
    """近侧（画面右）手臂：红袖 → 翻折的黑袖口 → 黑绳缠绕的小臂 → 手插进裤兜（或抬起来挠头）。"""
    if P['arm_near'] == 'pocket':
        arm = c.poly('e', [(24, 13), (27, 13), (29, 16), (31, 22), (26, 22), (25, 19)])
        c.px('f', [(27, 16), (28, 18), (29, 20)], where='e')
        c.px('E', [(25, 14), (26, 14)], where='e')
        c.stamp(21, 22, FOREARM)
        for dy, r in enumerate(FOREARM):
            for dx, ch in enumerate(r):
                if ch != '.':
                    arm[22 + dy, 21 + dx] = True
        c.border(arm, 'eEfFhHNnTtlLJj', 'O')
    elif P['arm_near'] == 'scratch':
        arm = c.line('e', [(25, 14), (29, 9)], width=3)
        arm |= c.line('l', [(29, 8), (28, 6)], width=3)
        c.line('t', [(30, 8), (28, 5)])
        arm |= c.line('s', [(27, 5), (23, 3)], width=2)
        arm |= c.ell('s', (19, 1, 23, 4))
        c.px('O', [(26, 4), (24, 3)])
        c.px('k', [(20, 3), (21, 4)])
        c.px('E', [(26, 13), (27, 12)], where='e')
        c.border(arm, 'eEfFhHNnTtlLJjaAcC1234', 'O')


def _fist(c, x, y):
    """握拳（2x3 左右的肉色块 + 指节暗线）。"""
    m = c.ell('s', (x - 2, y - 1, x + 1, y + 2))
    c.px('k', [(x - 1, y + 1), (x, y + 1)])
    c.px('S', [(x - 1, y - 1)])
    return m


def _limb(c, pts, cuff=True):
    """远侧手臂：红袖（宽 3）→ 黑袖口 → 小臂（宽 2），pts = [肩, 肘, 腕]。"""
    sh, el, wr = [np.array(p, float) for p in pts]
    arm = c.line('e', [tuple(sh), tuple(el)], width=3)
    d = wr - el
    cf = el + d * 0.45
    arm |= c.line('e', [tuple(el), tuple(cf)], width=3)
    if cuff:
        arm |= c.line('l', [tuple(cf), tuple(cf + d * 0.2)], width=3)
    arm |= c.line('s', [tuple(cf + d * 0.2), tuple(wr)], width=2)
    return arm


def _b_arm_far(c, P):
    kind = P['arm_far']
    if kind == 'coat':
        return
    if kind == 'cast':
        # 远侧手臂从肩向画面左侧平平伸直，掌心朝前、五指张开成爪（劫炎从掌心喷出）
        arm = np.zeros((c.h, c.w), bool)
        arm |= c.line('e', [(12, 15), (8, 14)], width=3)
        arm |= c.line('l', [(7, 14), (6, 13)], width=3)
        arm |= c.line('s', [(5, 13), (4, 12)], width=2)
        arm |= c.ell('s', (1, 9, 4, 13))
        for pts in [[(2, 10), (0, 9)], [(2, 9), (1, 7)], [(3, 9), (3, 6)], [(4, 9), (5, 7)], [(4, 12), (6, 11)]]:
            arm |= c.line('s', pts)
        c.px('S', [(2, 10), (3, 10), (2, 11)])
        c.px('k', [(3, 12), (4, 12)])
        c.px('E', [(10, 14), (11, 15)])
        c.px('O', [(5, 13)])
    else:
        gx, gy = BLADE_POSE_GRIP[kind]
        arm = _limb(c, ARM_PATH[kind])
        arm |= _fist(c, gx, gy)
    c.border(arm, 'eEfFhHNnTtlLaAcCJj1234', 'O')


# 远侧手臂各姿势的 肩 -> 肘 -> 腕（腕之后接拳，拳心 = 剑的握点）
ARM_PATH = {
    'grip': [(12, 15), (8, 14), (5, 12)],
    'raise': [(12, 15), (9, 10), (8, 5)],
    'windup': [(12, 15), (17, 19), (21, 17)],
    'swing1': [(12, 15), (11, 20), (11, 23)],
    'swing2': [(12, 15), (8, 14), (4, 13)],
    'follow': [(12, 15), (9, 13), (7, 11)],
    'shoulder': [(12, 16), (9, 21), (12, 16)],
}
BLADE_POSE_GRIP = {'grip': (3, 12), 'raise': (7, 3), 'windup': (23, 16), 'swing1': (11, 25), 'swing2': (2, 13),
                   'follow': (6, 10), 'shoulder': (13, 14)}


def _b_head(c, P):
    x0, y0 = P['head']
    rows = list(HEAD_BODY)
    f = FACES.get(P['face'])
    if f:
        rows[6:11] = f
    c.stamp(x0, y0, rows)
    if P['demon']:
        c.px('x', [(14, 9), (18, 5)], where='Ss')


def body(pose='idle'):
    """全身 48×72。姿势见 POSES：idle / skill / walk0..3 / yawn / reach / pull0 / pull1 /
    dwind / dswing1 / dswing2 / dfollow / dshoulder。剑不画在全身图里，用 sword() + BLADE 另外贴。"""
    P = POSES.get(pose, POSES['idle'])
    c = Cv(48, 72)
    _b_coat_back(c, P)
    _b_legs(c, P)
    _b_coat_front(c, P)
    _b_torso(c, P)
    if P['arm_near'] != 'scratch':
        _b_arm_near(c, P)
    _b_head(c, P)
    if P['arm_near'] == 'scratch':
        _b_arm_near(c, P)
    _b_arm_far(c, P)
    if P['demon']:
        for k, v in DEMON_HAIR.items():
            c.a[c.a == KID[k]] = KID[v]
    c.outline()
    for edge in (c.a[0], c.a[:, 0], c.a[:, -1]):
        edge[edge > 0] = KID['O']
    return c.rgba()


# ================================================================ 魔剑 Angbar
def sword(angle=0.0, reveal=1.0, length=1.0, demon=True):
    """魔剑 Angbar（从火里抽出来的巨大魔剑，暗红黑剑身 + 熔岩色的锯齿刃与剑脊）。
    angle：剑尖方向（度，0=画面右，顺时针为正）；reveal：剑身抽出比例；length：透视缩短系数。
    返回 (RGBA, (gx, gy))：gx, gy 是握点在图里的位置。剑长 ≈ 46px（约 1.15 m，按全身图比例）。"""
    L = 46 * length
    Lb = 3 + (L - 3) * reveal
    R = int(L + 10)
    c = Cv(2 * R + 1, 2 * R + 1)
    th = np.radians(angle)
    u = np.array([np.cos(th), np.sin(th)])
    v = np.array([-u[1], u[0]])

    def P(a, b):
        p = R + a * u + b * v
        return (float(p[0]), float(p[1]))

    # 剑柄、护手（两只弯角）、柄头
    c.line('l', [P(-6, 0), P(1, 0)], width=2)
    c.px('f', [tuple(int(round(x)) for x in P(k, 0)) for k in (-4, -2, 0)])
    c.poly('E', [P(-8, -1.2), P(-6, -1.5), P(-6, 1.5), P(-8, 1.2)])
    c.poly('F', [P(1, -6), P(3, -4), P(3, 4), P(1, 6), P(0, 4), P(0, -4)])
    c.poly('6', [P(0, -6), P(-3, -8), P(1, -5)])
    c.poly('6', [P(0, 6), P(-3, 8), P(1, 5)])
    c.px('X', [tuple(int(round(x)) for x in P(1.5, 0))])
    if Lb > 4:
        # 剑身：宽 5 → 尖，一侧锯齿
        pts_top, pts_bot = [], []
        n = max(2, int(Lb / 4))
        for i in range(n + 1):
            a = 3 + (Lb - 3) * i / n
            k = (a - 3) / max(L - 3, 1)
            w = 3.6 * (1 - k ** 1.8) + 0.4
            tooth = 1.6 if (i % 2 == 1 and a < L - 6) else 0.0
            pts_top.append(P(a, -w - tooth))
            pts_bot.append(P(a, w))
        if reveal >= 0.999:
            pts_top.append(P(L + 1, 0))
        c.poly('5', pts_top + pts_bot[::-1])
        c.line('6', [P(3, 1.8), P(Lb - 2, 0.6)])
        c.line('6', [P(3, -1.6), P(Lb - 4, -0.8)])
        # 熔岩色剑脊与刃口
        spine_end = max(4, Lb - 5)
        c.line('x', [P(4, -0.3), P(spine_end, -0.2)], where='56')
        if demon:
            c.line('X', [P(5, -0.3), P(spine_end * 0.8, -0.2)], where='x')
        c.line('x', [P(4, 3.5), P(Lb - 3, 1.0)], where='56')
        c.shade('5', '4', 1, 1)
    c.outline()
    a = c.a > 0
    ys, xs = np.nonzero(a)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = c.rgba()[y0:y1, x0:x1]
    return img, (R - x0, R - y0)


META = dict(
    key='mcburn',
    name_zh='马克邦',
    title_zh='劫炎',
    rank_zh='执行者 No.I',
    skill_zh='亿兆灾劫',            # S 技「ジリオンハザード / Zillion Hazard」，简中官方译名未查到，此为意译
    quote_zh='来啊，让我燃起来吧。',
    fx=dict(
        primary='#ff5a1e',
        secondary='#ffd24a',
        description=(
            '劫炎·亿兆灾劫：火从 skill 姿势张开的手掌（weapon_tip）涌出，先在掌心聚成一颗不断膨胀的火球——'
            '外圈橙红（primary），内核金白（secondary），火舌里夹着几缕近黑的暗红「劫炎」(#3a0a18)；'
            '随后火球砸出，化作铺满画面的爆炎冲击波。蓄力时风衣下摆与发梢被热浪向上掀动，'
            '金色瞳孔（eyes_body）可加一圈辉光，地面可加一圈火纹。'
        ),
    ),
    # 全身图中两只眼（虹膜像素）的 (y, x)，远侧眼在前
    eyes_body={'idle': [(7, 14), (6, 17)], 'skill': [(7, 14), (6, 17)]},
    weapon_tip=(11, 2),              # skill 姿势掌心中心（火焰发射点），(y, x)
    hand_near=(31, 21),              # 近侧手插兜处 (y, x)
    blade=BLADE,                     # {姿势: (握点 x, 握点 y, 剑朝向°, 透视长度, 剑在身后?)}，全身图坐标 (x, y)
    hand_far=HAND_FAR,               # {姿势: 远侧手 (x, y)}，全身图坐标
    hair_fire=[(12, 1), (15, 0), (18, 0), (21, 2), (11, 4)],   # 魔人化时发梢喷火的位置 (x, y)
    exprs=(None, 'smirk', 'grin', 'demon'),
    poses=tuple(POSES),
)


# 像素图关键点（与 data/landmarks/mcburn.json 同名），从渲染结果上读出，(x, y)
LANDMARKS = {
    'portrait': {
        'head_top': (42, 13),            # 头骨顶（被头发盖住，按发量估）
        'chin': (36, 53),
        'eye_near_in': (40, 30), 'eye_near_out': (49, 29), 'eye_near_top': (45, 29.5), 'eye_near_bot': (45, 31.5),
        'eye_far_in': (35, 32), 'eye_far_out': (28.5, 31.5), 'eye_far_top': (32, 31.5), 'eye_far_bot': (32, 33.5),
        'brow_near': (46, 24),
        'nose': (36, 41),
        'mouth': (39, 45),
        'jaw_far': (31, 45),
        'jaw_near': (52, 45),
        'neck_near': (46, 60), 'neck_far': (35, 57),
        # 轮廓 / 饰物点（与官方图同名，见 data/landmarks/mcburn.json）
        'x_spike_top': (59, 3), 'x_spike_ul': (16, 7), 'x_spike_l': (14, 16), 'x_spike_ll': (12, 25),
        'x_lock_low': (11, 36), 'x_hair_r': (76, 50),
        'x_tie_top': (61, 16), 'x_tie_mid': (63, 25), 'x_tie_low1': (59, 50), 'x_tie_low2': (57, 55),
        'x_ear_top': (58, 30), 'x_earring': (55, 42),
        'x_lens_far_out': (26, 36), 'x_lens_near_out': (49, 32), 'x_nose_pad': (39, 35),
        'x_collar_tip': (9, 55),
    },
    'idle': {
        'head_top': (15, 3),             # 头骨顶（估）
        'chin': (16, 10.5),            # 脸最下一行的下缘
        'neck': (16, 13),
        'shoulder_r': (12, 13), 'shoulder_l': (27, 14),
        'elbow_l': (29, 24), 'wrist_l': (22, 30),
        'waist': (13, 29),               # 腰带扣
        'crotch': (14.5, 36),
        'knee_r': (10, 50), 'knee_l': (24, 50),
        'ankle_r': (11, 69), 'ankle_l': (29, 69),
        'x_coat_shoulder_r': (12, 12), 'x_coat_shoulder_l': (27, 13),
        'x_cuff_out': (31, 23),
        'x_coat_hem_r': (1, 64), 'x_coat_hem_r_in': (5.5, 61),
        'x_coat_mid_hem': (24, 62), 'x_coat_mid_hem_l': (14, 62),
        'x_tail_tip': (46, 58), 'x_tail_low': (31, 62),
        'x_vest_tip_r': (10, 36),
        'x_toe_r': (3, 71), 'x_toe_l': (29, 71),
    },
}
