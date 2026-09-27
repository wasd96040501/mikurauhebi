"""v3 分镜与时间轴（带速度变化）。画面、配乐、音效共用这一份。

整部 PV 是肯帕雷拉（道化师 / 见证者）主持的一出戏：
  开场 —— 空荡的旧剧场，帷幕上浮出衔尾蛇纹章，他在幻焰中现身，响指开幕；
  每一幕 —— 他弹出一张扑克牌，牌面翻开就是下一位执行者 / 使徒的「场景」，演一段原作里的故事；
  幻焰计划 —— 克洛斯贝尔上空长出碧之大树，他站在塔顶「见证」；
  盟主 —— 玫瑰窗下的神座；
  总攻 —— 群像；
  谢幕 —— 回到剧场，众人剪影列队，他鞠躬、洒牌、退场，帷幕落下。

音乐：序章 90 BPM、D 小调、神秘（八音盒 + 竖琴 + 低音弦乐 + 人声 pad，无鼓）；
响指之后 144 BPM 进入 Boss 战（D 小调），每一幕换一种主奏音色；幻焰计划转 E 小调并半速，
盟主（管风琴 + 合唱）与总攻在 E 小调推到顶；谢幕回到 90 BPM、D 小调八音盒，结束在 D 大三和弦。

每一幕 6 小节 = 4 小节叙事 + 2 小节技能高潮/余韵。拍点约定（act 内的小节从 1 数）：
  1.0  场景建立（音乐：新一幕的主奏乐器进入）
  3.0  台词出现
  5.0  技能命中（音乐：全奏重音 / 鼓 fill 落点）
  6.x  余韵，最后两拍给转场（扑克牌）
每一幕的场景模块可以额外声明 CUES（音效）和 ACCENTS（希望音乐配合的重音/停顿），见 scenes/。
"""
from dataclasses import dataclass

FPS = 30


@dataclass(frozen=True)
class Sec:
    name: str
    bars: int
    bpm: float
    t0: float = 0.0

    @property
    def beat(self):
        return 60.0 / self.bpm

    @property
    def bar(self):
        return 4 * self.beat

    @property
    def dur(self):
        return self.bars * self.bar

    @property
    def t1(self):
        return self.t0 + self.dur

    def T(self, bar=1, beat=0.0):
        """本段第 bar 小节（从 1 数）第 beat 拍（从 0 数）的全局时刻（秒）。"""
        return self.t0 + (bar - 1) * self.bar + beat * self.beat


_PLAN = [
    ('prologue', 6, 90),       # 剧场、纹章、肯帕雷拉现身（神秘）
    ('title', 2, 144),         # 响指 -> 标题砸出，Boss 战开始
    ('mcburn', 6, 144),        # 煌魔城：插兜走进火海，从火里拔出魔剑，劫炎横扫
    ('leonhardt', 6, 144),     # 哈梅尔废村：墓前 -> 拔剑 -> 鬼炎斩
    ('arianrhod', 6, 144),     # 古战场的枪林：圣女持枪 -> 冲锋 -> 光之十字
    ('bleublanc', 6, 144),     # 月夜美术馆：玫瑰、分身、盗走宝石，留下卡片
    ('vita', 6, 144),          # 帝都歌剧院：Misty 登台歌唱 -> 舞台化作夜空，苍鸟掠月
    ('phantom', 4, 144),       # 幻焰计划：克洛斯贝尔与碧之大树，塔顶的见证者（E 小调，半速感）
    ('grandmaster', 4, 144),   # 盟主：玫瑰窗、光柱
    ('assault', 4, 144),       # 总攻群像
    ('slam', 2, 144),          # Logo 砸出
    ('epilogue', 3, 90),       # 谢幕：剧场、鞠躬、洒牌、落幕（回到 D 小调八音盒）
]

SECS = {}
_t = 0.0
for _n, _b, _bpm in _PLAN:
    SECS[_n] = Sec(_n, _b, _bpm, _t)
    _t += SECS[_n].dur
ORDER = [n for n, *_ in _PLAN]
DURATION = _t
FRAMES = int(round(DURATION * FPS))
ACTS = ['mcburn', 'leonhardt', 'arianrhod', 'bleublanc', 'vita']


def section_at(t):
    for n in ORDER:
        s = SECS[n]
        if t < s.t1 or n == ORDER[-1]:
            return s
    return SECS[ORDER[-1]]


def tempo_map():
    """[(全局拍位置, bpm)]，给 MIDI 写 set_tempo 用。拍位置按各段 bpm 累加。"""
    out, beats = [], 0.0
    for n in ORDER:
        s = SECS[n]
        out.append((beats, s.bpm, s.t0))
        beats += s.bars * 4
    return out, beats


if __name__ == '__main__':
    for n in ORDER:
        s = SECS[n]
        print(f'{n:<12}{s.bars:>3} bars @{s.bpm:>4}  {s.t0:7.2f} - {s.t1:7.2f}  ({s.dur:5.2f}s)')
    print(f'total {DURATION:.2f}s, {FRAMES} frames')
