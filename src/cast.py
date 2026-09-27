"""角色表：加载 src/chars/<key>.py 的立绘、全身像和元数据，并缓存。

某个角色模块还没画好时用占位图顶上，方便先排镜头。
"""
import importlib
from functools import lru_cache

import numpy as np

from gfx import hexc
import i18n

ORDER = ['campanella', 'mcburn', 'leonhardt', 'arianrhod', 'bleublanc', 'vita']

DEFAULT_META = {
    'campanella': dict(name_zh='肯帕雷拉', title_zh='道化师', rank_zh='执行者 No.0', skill_zh='幻焰',
                       quote_zh='那么——好戏开场了。', fx=dict(primary='#ff8a20', secondary='#4ab0ff')),
    'mcburn': dict(name_zh='马克邦', title_zh='劫炎', rank_zh='执行者 No.I', skill_zh='焦土',
                   quote_zh='……无聊透顶。', fx=dict(primary='#ffb030', secondary='#301008')),
    'leonhardt': dict(name_zh='莱恩哈特', title_zh='剑帝', rank_zh='执行者 No.II', skill_zh='鬼炎斩',
                      quote_zh='来吧，拔剑。', fx=dict(primary='#ff5a1a', secondary='#fff0a0')),
    'arianrhod': dict(name_zh='雅里安洛德', title_zh='钢之圣女', rank_zh='使徒 第七柱', skill_zh='圣枪',
                      quote_zh='见识一下吧。', fx=dict(primary='#fff0c0', secondary='#e0a040')),
    'bleublanc': dict(name_zh='布卢布兰', title_zh='怪盗B', rank_zh='执行者 No.X', skill_zh='幻影舞台',
                      quote_zh='美，才是一切。', fx=dict(primary='#ff4a8a', secondary='#c8c0f0')),
    'vita': dict(name_zh='薇塔', title_zh='苍之深渊', rank_zh='使徒 第二柱', skill_zh='苍之咏叹',
                 quote_zh='来，听我歌唱。', fx=dict(primary='#4ab0ff', secondary='#c080ff')),
    'grandmaster': dict(name_zh='盟主', title_zh='盟主', rank_zh='身喰らう蛇', skill_zh='',
                        quote_zh='', fx=dict(primary='#ffe8a0', secondary='#c080ff')),
}
PLACEHOLDER_COL = {'campanella': '#c02848', 'mcburn': '#7a1a14', 'leonhardt': '#26305a', 'arianrhod': '#d8d8e8',
                   'bleublanc': '#f0f0f8', 'vita': '#2f5ad0', 'grandmaster': '#f2ecdc'}


def _placeholder(key, w, h):
    img = np.zeros((h, w, 4), np.float32)
    img[4:-2, 8:-8, :3] = hexc(PLACEHOLDER_COL[key])
    img[4:-2, 8:-8, 3] = 1
    return img


class Char:
    def __init__(self, key):
        self.key = key
        try:
            self.mod = importlib.import_module(f'chars.{key}')
        except Exception as e:  # 还没画好
            print(f'[cast] {key}: placeholder ({e.__class__.__name__}: {e})')
            self.mod = None
        meta = dict(DEFAULT_META[key])
        if self.mod is not None and hasattr(self.mod, 'META'):
            meta.update(self.mod.META)
        # 画面上的称呼统一以 i18n.NAMES 为准（按 PV_LANG 选语言），角色模块里写的不算
        meta['name'], meta['title'], meta['rank'] = i18n.name(key)
        meta['name_zh'], meta['title_zh'], meta['rank_zh'] = i18n.NAMES[key]['zh']
        self.meta = meta

    @lru_cache(None)
    def portrait(self, expr=None):
        if self.mod is None:
            return _placeholder(self.key, 80, 80)
        try:
            return self.mod.portrait(expr) if expr else self.mod.portrait()
        except TypeError:
            return self.mod.portrait()

    @lru_cache(None)
    def body(self, pose='idle'):
        if self.mod is None:
            return _placeholder(self.key, 48, 72)
        try:
            return self.mod.body(pose)
        except Exception:
            return self.mod.body('idle')

    def fx(self, k, default='#ffffff'):
        return hexc(self.meta.get('fx', {}).get(k, default))


@lru_cache(None)
def get(key):
    return Char(key)
