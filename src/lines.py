"""画面上出现的全部台词：原作日文原句 + 中文译文。集中在这里改。

原句均出自游戏脚本（可在 trailsinthedatabase.com 按角色检索），译文按中文习惯意译。
"""
LINES = {
    # 空之轨迹 the 3rd / 碧之轨迹 等
    'campa_enter': ('呵呵……那么，开始吧。', 'ウフフ…それじゃあ、始めようかな。'),                  # 碧 e4810
    'campa_witness': ('这一次，我只负责见证而已。', '今回の僕の役割はあくまで『見届け役』なんだ。'),     # 空SC c1104
    'campa_exit': ('那么诸位，告辞了。', 'それでは皆様、ご機嫌よう。'),                               # 空SC c1104
    'mcburn': ('这世界是存是亡，说实话，与我无关。', 'この世界が続こうが終わろうが正直、俺には知ったことじゃない。'),  # 闪IV m5070
    'leonhardt': ('我有我自己选的路。', '俺は俺の、選んだ道がある。'),                                # 空SC 终章
    'arianrhod': ('好气魄。——那么，我来了。', '意気やよし。──それでは行きますよ。'),                  # 闪III m2010
    'bleublanc': ('所谓怪盗，便是美的信徒。', '怪盗とは、すなわち美の崇拝者。'),                        # 空SC c2400_1
    'vita': ('“故事”早已开始了。', '既に"物語"は始まっている。'),                                      # 闪II t4000
    'grandmaster': ('我只是“影”——向世界宣告时限的存在罢了。', '私は"影"──世界に刻限を告げるだけの存在です。'),  # 闪IV 终章
}

# 英文版台词（PV_LANG=en）。有官方英文版（XSEED / NISA）的用官方译文，没有的自译，见注释。
LINES_EN = {
    'campa_enter': 'Heh heh... Well then, shall we begin?',                               # 自译（碧 e4810 的官方英文只有后半句）
    'campa_witness': 'I\'m merely an "observer" on this occasion.',                      # 空 the 2nd 官方英文
    'campa_exit': 'Well then, I bid you all farewell.',                                   # 空 the 2nd 官方英文
    'mcburn': "Whether this world goes on or ends... honestly, it's no concern of mine.",  # 自译（闪IV 官方英文省略了前半句）
    'leonhardt': "I have my own path I've chosen for myself.",                             # 空SC XSEED 官方英文
    'arianrhod': 'A fine spirit. Now then, here I come.',                                  # 自译（闪III 官方英文后半句意译较远）
    'bleublanc': 'A phantom thief is, in essence, a worshiper of beauty.',                # 空SC XSEED 官方英文
    'vita': 'The "story" has already begun.',                                             # 闪III NISA 官方英文
    'grandmaster': 'I am but a "shadow." I exist solely to announce the appointed time to the world.',  # 闪IV NISA 官方英文
}


def get(key):
    """当前语言的 (译文, 日文原句, 中文原长度)。第三项用来让打字机节奏与中文版一致（打字音按中文排好）。"""
    import i18n
    zh, ja = LINES[key]
    return (LINES_EN[key] if i18n.LANG == 'en' else zh), ja, len(zh)
