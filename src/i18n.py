"""画面语言：PV_LANG=zh（默认）| en。日文原句不随语言变化，始终显示在原来的位置。

画面上的称呼（名字 / 称号 / 位阶）统一在 NAMES 里改，cast.py 会覆盖角色模块里写的值。
"""
import os

LANG = os.environ.get('PV_LANG', 'zh')
assert LANG in ('zh', 'en'), f'PV_LANG 只支持 zh / en，收到 {LANG!r}'


def T(zh, en):
    """按当前语言二选一。"""
    return en if LANG == 'en' else zh


# key: {lang: (名字, 称号, 位阶)}
NAMES = {   # 中文按云豹官方译名（肯帕雷拉是用户指定的旧译名，不改）；英文按 XSEED / NISA 官方英文版
    'campanella': {'zh': ('肯帕雷拉', '《小丑》', '执行者No.0'), 'en': ('Campanella', 'the Fool', 'Enforcer No. 0')},
    'mcburn': {'zh': ('马克邦', '《劫炎》', '执行者No.Ⅰ'), 'en': ('McBurn', 'the Almighty Conflagration', 'Enforcer No. I')},
    'leonhardt': {'zh': ('莱恩哈特', '《剑帝》', '执行者No.Ⅱ'), 'en': ('Leonhardt', 'the Bladelord', 'Enforcer No. II')},
    'arianrhod': {'zh': ('雅里安洛德', '《钢之圣女》', '使徒第七柱'), 'en': ('Arianrhod', 'the Steel Maiden', 'Seventh Anguis')},
    'bleublanc': {'zh': ('布卢布兰', '《怪盗绅士》', '执行者No.X'), 'en': ('Bleublanc', 'the Phantom Thief', 'Enforcer No. X')},
    'vita': {'zh': ('薇塔', '《苍之深渊》', '使徒第二柱'), 'en': ('Vita', 'the Azure Abyss', 'Second Anguis')},
    'grandmaster': {'zh': ('盟主', '盟主', '结社《噬身之蛇》'), 'en': ('Grandmaster', 'the Grandmaster', 'The Society of Ouroboros')},
}

TEXT = {
    'org': {'zh': '噬身之蛇', 'en': 'OUROBOROS'},
    'society': {'zh': '结社《噬身之蛇》', 'en': 'The Society of Ouroboros'},
}


def name(key):
    return NAMES[key][LANG]


def S(k):
    return TEXT[k][LANG]
