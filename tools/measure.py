"""把"像不像"变成数字：官方图与像素图各标一组关键点，相似变换对齐后量误差。

思路
----
1. 关键点（landmark）用解剖学命名，官方图和像素图用同一套名字。
   脸（3/4 侧）用 near/far 区分离镜头近/远的一侧，全身用角色自己的左右（_l/_r），
   所以像素图左右翻转了也能对上（对齐时允许镜像）。
2. 对两组同名点做 Procrustes 相似对齐（平移 + 旋转 + 统一缩放 [+ 镜像]），
   残差除以"头高"（head_top→chin）归一化：
       face_err = RMS(残差) / 头高        ← 脸部比例误差，单位 = 头高的百分比
       body_err = RMS(残差) / 身高        ← 全身比例误差（head_top→脚踝中点）
3. 同时算一组可读的比例（眼睛在头高的哪个位置、眼宽、下巴长短、几头身……），
   逐项列出官方值 / 像素值 / 差值，看哪里偏。
4. compare 还会把官方图按对齐结果变换到像素图的坐标系、缩到同样分辨率，
   做成「官方(缩到同分辨率) | 像素图 | 叠影 | 剪影差」四联图，服装轮廓也能直接对照。
5. 剪影 IoU：关键点只量脸内部和关节，量不出"发量、领子、扇子的外形对不对"。
   所以把官方图的 alpha（没有 alpha 的图按四角背景色抠）用同一个相似变换
   搬到像素图坐标系，与像素图的 alpha 求交并比：
       iou = |官方剪影 ∩ 像素剪影| / |官方剪影 ∪ 像素剪影|     ← 越大越好，低于阈值算超标
   剪影差图里：红 = 像素图多出来的，蓝 = 官方有而像素图缺的。
   只在画框内比较（头像画框切掉的身体两边都不算）。

数据
----
- 官方图关键点：data/landmarks/<角色>.json  {"<参考图名>": {"点名": [x, y], ...}}
  参考图在 refs/<角色>/<参考图名>.png（tools/fetch_refs.py 下载）。
- 像素图关键点：角色模块里的 LANDMARKS = {'portrait': {...}, 'idle': {...}}。

命令
----
  python tools/measure.py grid refs/campanella/bust.png --crop 150 40 360 300 --step 10
      画带坐标刻度的放大网格图，用来读关键点坐标。
  python tools/measure.py sprite campanella portrait --step 5
      把像素图放大画网格（同上，读像素图坐标）。
  python tools/measure.py show campanella bust
      在官方图上画出已标的关键点，检查标得准不准。
  python tools/measure.py compare campanella bust portrait
      对齐 + 误差 + 三联图。误差超阈值时退出码为 1。
  python tools/measure.py report
      所有角色一张总表。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'preview' / 'measure'
LM_DIR = ROOT / 'data' / 'landmarks'

# ---------------------------------------------------------------- 关键点定义
FACE_KEYS = [
    'head_top',        # 头骨顶（不算呆毛/刺头，估头骨轮廓）
    'chin',            # 下巴尖
    'eye_near_in', 'eye_near_out',   # 近侧眼：内眼角 / 外眼角（上眼线两端）
    'eye_near_top', 'eye_near_bot',  # 近侧眼：眼眶开口最高点 / 最低点（虹膜中线上）
    'eye_far_in', 'eye_far_out',     # 远侧眼：内 / 外眼角
    'eye_far_top', 'eye_far_bot',    # 远侧眼：开口最高 / 最低点
    # 被头发 / 面具挡住的点可以不标，只用两边都有的点
    'brow_near',       # 近侧眉毛中点（被刘海挡住就估）
    'nose',            # 鼻尖
    'mouth',           # 嘴中点
    'cheek_far',       # 远侧脸颊轮廓，眼睛下缘高度
    'jaw_far',         # 远侧下颌轮廓，嘴的高度
    'jaw_near',        # 近侧下颌轮廓（下颌角附近，耳下）
    'neck_near', 'neck_far',         # 脖子两侧，与领口相接处
]
BODY_KEYS = [
    'head_top', 'chin', 'neck',                  # neck = 锁骨中点
    'shoulder_r', 'shoulder_l',                  # 角色自己的右 / 左肩峰
    'elbow_r', 'elbow_l', 'wrist_r', 'wrist_l',
    'waist',                                     # 腰带 / 最细处中点
    'crotch',
    'knee_r', 'knee_l', 'ankle_r', 'ankle_l',
]
# 服装 / 道具点可以随便加（名字以 x_ 开头，例如 x_coat_hem_r），两边都标了才参与计算。

# 阈值（归一化 RMS）
TOL = {'face': 0.045, 'body': 0.035}
# 剪影 IoU 下限（按肯帕雷拉标定：头像 0.873 / 全身 0.707，下限各留一点余量）
IOU_MIN = {'face': 0.80, 'body': 0.62}


# ---------------------------------------------------------------- 数据读写
def ref_path(char, ref):
    p = LM_DIR / f'{char}.json'
    if p.exists():
        ref = json.loads(p.read_text()).get(ref, {}).get('_image', ref)   # 同一张图可以有多组点
    for ext in ('.png', '.jpg', '.webp'):
        p = ROOT / 'refs' / char / (ref + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f'refs/{char}/{ref}.* 不存在，先跑 tools/fetch_refs.py')


def load_ref_lm(char, ref):
    p = LM_DIR / f'{char}.json'
    data = json.loads(p.read_text()) if p.exists() else {}
    if ref not in data:
        raise KeyError(f'{p} 里没有 "{ref}" 的关键点')
    return {k: np.array(v, float) for k, v in data[ref].items() if not k.startswith('_')}


def load_sprite(char, which):
    import cast
    c = cast.get(char)
    img = c.portrait() if which == 'portrait' else c.body(which)
    lm = getattr(c.mod, 'LANDMARKS', {}).get(which, {})
    return img, {k: np.array(v, float) for k, v in lm.items()}


def kind_of(lm):
    return 'body' if ('ankle_r' in lm or 'shoulder_r' in lm) else 'face'


# ---------------------------------------------------------------- 几何
def procrustes(src, dst, allow_reflect=True):
    """求 s,R,t 使 s*R@src + t ≈ dst（最小二乘）。返回 (变换函数, 参数)。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a)
    d = np.eye(2)
    if not allow_reflect and np.linalg.det(u @ vt) < 0:
        d[1, 1] = -1
    R = u @ d @ vt
    s = np.trace(np.diag(sig) @ d) / (a ** 2).sum()
    t = mu_d - s * R @ mu_s
    return (lambda p: (s * (R @ np.asarray(p, float).T)).T + t), (s, R, t)


def _mid(lm, a, b):
    return (lm[a] + lm[b]) / 2 if a in lm and b in lm else None


def _canon(lm):
    """把关键点转到"头朝上、近侧在 +x"的规范坐标，方便算比例。"""
    if 'head_top' not in lm or 'chin' not in lm:
        return lm
    up = lm['head_top'] - lm['chin']
    ang = np.arctan2(up[0], -up[1])          # 让 chin→head_top 指向 -y
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    out = {k: R @ (v - lm['chin']) for k, v in lm.items()}
    near = out.get('jaw_near', out.get('shoulder_r'))
    far = out.get('jaw_far', out.get('shoulder_l'))
    if near is not None and far is not None and near[0] < far[0]:
        out = {k: v * np.array([-1, 1]) for k, v in out.items()}
    return out


def ratios(lm):
    """一组可读的比例（都除以头高 H = head_top→chin）。"""
    c = _canon(lm)
    H = np.linalg.norm(lm['head_top'] - lm['chin'])
    r = {}
    y = lambda k: -c[k][1] / H     # 从下巴往上量，0=下巴，1=头顶
    x = lambda k: c[k][0] / H
    if kind_of(lm) == 'face':
        for side in ('near', 'far'):
            if f'eye_{side}_in' not in c or f'eye_{side}_out' not in c:
                continue
            e = _mid(c, f'eye_{side}_in', f'eye_{side}_out')
            r[f'{side}眼高度'] = -e[1] / H
            r[f'{side}眼宽'] = np.linalg.norm(c[f'eye_{side}_out'] - c[f'eye_{side}_in']) / H
            if f'eye_{side}_top' in c and f'eye_{side}_bot' in c:
                r[f'{side}眼开口高'] = np.linalg.norm(c[f'eye_{side}_top'] - c[f'eye_{side}_bot']) / H
        if 'eye_near_in' in c and 'eye_far_in' in c:
            r['两眼内角距'] = abs(x('eye_near_in') - x('eye_far_in'))
        for k, name in (('brow_near', '眉高'), ('nose', '鼻高'), ('mouth', '嘴高')):
            if k in c:
                r[name] = y(k)
        if 'nose' in c and 'mouth' in c:
            r['鼻-嘴水平差'] = x('mouth') - x('nose')
        if 'jaw_far' in c and 'jaw_near' in c:
            r['下颌宽'] = abs(x('jaw_near') - x('jaw_far'))
        if 'cheek_far' in c and 'mouth' in c:
            r['远颊到嘴'] = abs(x('mouth') - x('cheek_far'))
        if 'jaw_far' in c:
            r['远颌到下巴'] = abs(x('jaw_far'))
        if 'neck_near' in c and 'neck_far' in c:
            r['脖宽'] = abs(x('neck_near') - x('neck_far'))
    else:
        feet = _mid(c, 'ankle_r', 'ankle_l')
        if feet is not None:
            r['头身比'] = feet[1] / H + 1
        for k, name in (('neck', '锁骨'), ('waist', '腰'), ('crotch', '裆'), ('knee_r', '右膝')):
            if k in c:
                r[name + '(下巴以下,头高)'] = c[k][1] / H
        if 'shoulder_r' in c and 'shoulder_l' in c:
            r['肩宽(头高)'] = np.linalg.norm(c['shoulder_r'] - c['shoulder_l']) / H
    return r


def align(ref_lm, spr_lm):
    keys = [k for k in ref_lm if k in spr_lm]
    src = np.array([ref_lm[k] for k in keys])
    dst = np.array([spr_lm[k] for k in keys])
    f, (s, R, t) = procrustes(src, dst)
    res = np.linalg.norm(f(src) - dst, axis=1)
    if kind_of(spr_lm) == 'face':
        norm = np.linalg.norm(spr_lm['head_top'] - spr_lm['chin'])
    else:
        feet = _mid(spr_lm, 'ankle_r', 'ankle_l')
        norm = np.linalg.norm(spr_lm['head_top'] - feet) if feet is not None else \
            np.linalg.norm(spr_lm['head_top'] - spr_lm['chin']) * 7
    err = float(np.sqrt((res ** 2).mean()) / norm)
    return dict(keys=keys, f=f, s=s, R=R, t=t, res=res, norm=norm, err=err,
                reflect=bool(np.linalg.det(R) < 0))


# ---------------------------------------------------------------- 画图
def _font(size=11):
    for p in ('/System/Library/Fonts/Supplemental/Arial Unicode.ttf', '/System/Library/Fonts/PingFang.ttc'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert('RGBA')
    a = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(a, 'RGBA')


def checker(w, h, c0=(58, 58, 70), c1=(46, 46, 56), cell=8):
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((yy // cell + xx // cell) % 2).astype(bool)
    a = np.empty((h, w, 4), np.uint8)
    a[m], a[~m] = (*c0, 255), (*c1, 255)
    return Image.fromarray(a, 'RGBA')


def grid_img(im, crop=None, step=10, scale=None, lm=None):
    im = to_pil(im)
    if crop:
        im = im.crop(crop)
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    if scale is None:
        scale = max(1, int(900 / max(im.size)))
    big = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    pad = 34
    canvas = checker(big.width + pad, big.height + pad)
    canvas.alpha_composite(big, (pad, pad))
    d = ImageDraw.Draw(canvas)
    f = _font(10)
    for gx in range(0, im.width + 1):
        if (gx + ox) % step:
            continue
        X = pad + gx * scale
        major = (gx + ox) % (step * 5) == 0
        d.line([(X, pad), (X, canvas.height)], fill=(255, 255, 255, 110 if major else 45))
        d.text((X + 1, 2 + (12 if major else 0)), str(gx + ox), fill=(255, 230, 120, 255), font=f)
    for gy in range(0, im.height + 1):
        if (gy + oy) % step:
            continue
        Y = pad + gy * scale
        major = (gy + oy) % (step * 5) == 0
        d.line([(pad, Y), (canvas.width, Y)], fill=(255, 255, 255, 110 if major else 45))
        d.text((1, Y + 1), str(gy + oy), fill=(255, 230, 120, 255), font=f)
    if lm:
        for k, (x, y) in lm.items():
            X, Y = pad + (x - ox + 0.5) * scale, pad + (y - oy + 0.5) * scale
            d.ellipse([X - 3, Y - 3, X + 3, Y + 3], outline=(0, 255, 160, 255), width=2)
            d.text((X + 4, Y - 6), k, fill=(0, 255, 160, 255), font=f)
    return canvas


def ref_alpha(ref_im):
    """官方图 RGBA（float 0..1）。没有透明背景的图按四角背景色抠出剪影。"""
    ref = np.asarray(to_pil(ref_im), np.float32) / 255
    if ref[..., 3].min() > 0.98:
        corners = np.concatenate([ref[:8, :8, :3].reshape(-1, 3), ref[:8, -8:, :3].reshape(-1, 3),
                                  ref[-8:, :8, :3].reshape(-1, 3), ref[-8:, -8:, :3].reshape(-1, 3)])
        bg = np.median(corners, 0)
        dist = np.abs(ref[..., :3] - bg).max(-1)
        ref = ref.copy()
        ref[..., 3] = np.clip((dist - 0.06) / 0.08, 0, 1)
    return ref


def warp_ref(ref_im, A, h, w, ss=4):
    """官方图按对齐结果（官方坐标 -> 像素图坐标的相似变换）搬到像素图的 h x w 画框里，
    超采样 ss 倍再按 alpha 加权平均缩小。返回 float RGBA。"""
    s, R, t = A['s'], A['R'], A['t']
    inv = np.linalg.inv(s * R)
    yy, xx = np.mgrid[0:h * ss, 0:w * ss]
    P = np.stack([(xx + 0.5) / ss - 0.5, (yy + 0.5) / ss - 0.5], -1).reshape(-1, 2)
    Q = (inv @ (P - t).T).T
    ref = ref_alpha(ref_im)
    qx = np.clip(np.round(Q[:, 0]).astype(int), 0, ref.shape[1] - 1)
    qy = np.clip(np.round(Q[:, 1]).astype(int), 0, ref.shape[0] - 1)
    inside = (Q[:, 0] >= 0) & (Q[:, 0] < ref.shape[1]) & (Q[:, 1] >= 0) & (Q[:, 1] < ref.shape[0])
    samp = ref[qy, qx] * inside[:, None]
    samp = samp.reshape(h, ss, w, ss, 4)
    a = samp[..., 3:4]
    rgb = (samp[..., :3] * a).sum((1, 3)) / np.maximum(a.sum((1, 3)), 1e-6)
    alpha = a.mean((1, 3))
    return np.concatenate([rgb, alpha], -1)


def silhouette_iou(spr, warped):
    """像素图 alpha 与变换后官方剪影的交并比。返回 (iou, 像素多出的掩码, 像素缺的掩码)。"""
    ms = np.asarray(spr)[..., 3] > 0.5
    mr = warped[..., 3] >= 0.5
    union = (ms | mr).sum()
    iou = float((ms & mr).sum() / union) if union else 1.0
    return iou, ms & ~mr, mr & ~ms


def compare_img(ref_im, ref_lm, spr, spr_lm, A, scale=6, warped=None):
    """四联图：官方图（变换并缩到像素图分辨率）| 像素图 | 叠影 + 残差箭头 | 剪影差。"""
    h, w = spr.shape[:2]
    if warped is None:
        warped = warp_ref(ref_im, A, h, w)

    panels = [to_pil(warped), to_pil(spr)]
    ov = np.asarray(to_pil(spr), np.float32) / 255
    wa = warped.copy()
    mix = ov.copy()
    mix[..., :3] = ov[..., :3] * 0.55 + wa[..., :3] * 0.45 * (wa[..., 3:] > 0.1) + ov[..., :3] * 0.45 * (wa[..., 3:] <= 0.1)
    mix[..., 3] = np.maximum(ov[..., 3], wa[..., 3])
    panels.append(to_pil(mix))
    _, extra, miss = silhouette_iou(spr, warped)
    both = (np.asarray(spr)[..., 3] > 0.5) & ~extra
    sil = np.zeros((h, w, 4), np.float32)
    sil[both] = (0.62, 0.62, 0.66, 1)
    sil[extra] = (1.0, 0.25, 0.25, 1)
    sil[miss] = (0.25, 0.55, 1.0, 1)
    panels.append(to_pil(sil))
    W = (w * scale + 8) * len(panels) + 8
    canvas = checker(W, h * scale + 16)
    d = ImageDraw.Draw(canvas)
    for i, p in enumerate(panels):
        x0 = 8 + i * (w * scale + 8)
        canvas.alpha_composite(p.resize((w * scale, h * scale), Image.NEAREST), (x0, 8))
    x0 = 8 + 2 * (w * scale + 8)
    for k in A['keys']:
        pr = A['f'](ref_lm[k][None])[0]
        pp = spr_lm[k]
        P1 = (x0 + (pr[0] + .5) * scale, 8 + (pr[1] + .5) * scale)
        P2 = (x0 + (pp[0] + .5) * scale, 8 + (pp[1] + .5) * scale)
        d.line([P1, P2], fill=(255, 60, 60, 255), width=2)
        d.ellipse([P1[0] - 3, P1[1] - 3, P1[0] + 3, P1[1] + 3], outline=(80, 200, 255, 255), width=2)
        d.ellipse([P2[0] - 2, P2[1] - 2, P2[0] + 2, P2[1] + 2], fill=(0, 255, 140, 255))
    return canvas


# ---------------------------------------------------------------- 命令
def cmd_compare(char, ref, which, quiet=False):
    ref_lm = load_ref_lm(char, ref)
    spr, spr_lm = load_sprite(char, which)
    if not spr_lm:
        print(f'{char}.{which}: 模块里没有 LANDMARKS["{which}"]')
        return None
    A = align(ref_lm, spr_lm)
    kind = kind_of(spr_lm)
    rr, rs = ratios(ref_lm), ratios(spr_lm)
    ref_im = Image.open(ref_path(char, ref))
    warped = warp_ref(ref_im, A, *spr.shape[:2])
    iou = silhouette_iou(spr, warped)[0]
    ok_err = A['err'] <= TOL[kind]
    ok_iou = iou >= IOU_MIN[kind]
    ok = ok_err and ok_iou
    if not quiet:
        print(f'== {char}: 官方 {ref} vs 像素 {which}  [{kind}]  共 {len(A["keys"])} 点'
              f'{"  (镜像对齐)" if A["reflect"] else ""}')
        print(f'   归一化 RMS 误差 = {A["err"] * 100:.2f}%  (阈值 {TOL[kind] * 100:.1f}%)  {"OK" if ok_err else "超标"}')
        print(f'   剪影 IoU = {iou:.3f}  (下限 {IOU_MIN[kind]:.2f})  {"OK" if ok_iou else "超标"}')
        order = np.argsort(-A['res'])
        print('   最大残差: ' + ', '.join(f'{A["keys"][i]} {A["res"][i]:.1f}px' for i in order[:5]))
        print(f'   {"比例":<16}{"官方":>8}{"像素":>8}{"差":>8}')
        for k in rr:
            if k in rs:
                dlt = rs[k] - rr[k]
                flag = '  <--' if abs(dlt) > (0.04 if kind == 'face' else 0.25) else ''
                print(f'   {k:<16}{rr[k]:8.3f}{rs[k]:8.3f}{dlt:+8.3f}{flag}')
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f'{char}_{ref}_vs_{which}.png'
    compare_img(ref_im, ref_lm, spr, spr_lm, A, scale=6 if kind == 'face' else 5, warped=warped).save(p)
    if not quiet:
        print(f'   四联图: {p.relative_to(ROOT)}')
    return A['err'], ok, iou, ok_err, ok_iou


# ================================================================ 头像保真度（fidelity）
# compare 用全部关键点自由拟合相似变换，只能说明"五官之间的相对位置对不对"；
# 头发外形、歪头角度、领子 / 肩线的位置、构图都量不到。fidelity 改成：
#   1. 固定取景：只用 3 个稳定锚点（两眼中心 + 下巴尖）求一次相似变换（官方 -> 像素画框），之后所有量都在
#      这个坐标系里算，不再重新拟合。
#   2. 每个同名关键点（脸 + x_ 轮廓 / 服装点）的残差（像素），报 RMS 和最大值。
#   3. 剪影 IoU（同一个固定变换）。
#   4. 边缘倒角距离：官方图（高分辨率变换后找边、再池化到像素分辨率）与像素图的轮廓 / 色块边，双向平均距离（px）。
#   5. 材质区域 IoU：两边的像素都分成 skin / hair / eyes / cloth1 / cloth2 / other，
#      像素图按 JSON 里声明的「材质 -> 调色板颜色」查表；官方图用 Lab k-means（以像素图里出现的颜色为种子，
#      每个种子带着它的材质标签）聚类后取标签。skin、hair 两项最重要。
# 锚点：
#   - 官方图：JSON 该组点里的 "_anchors": {"eye_near": [x,y], "eye_far": [x,y], "chin": [x,y]} 优先；
#     否则眼睛中心 = eye_<side>_in/out/top/bot 的平均，chin = chin。被刘海挡住的眼睛在 _anchors 里估一个位置。
#   - 像素图：模块 LANDMARKS[which] 里的 a_eye_near / a_eye_far / a_chin 优先；否则同样由眼睛四点求中心；
#     像素图上也看不见的那只眼，用"全部关键点拟合的变换"把官方锚点映射过来代替（只在两边都缺时发生）。
# 材质表：JSON 顶层 "_classes": {"skin": [...], "hair": [...], "eyes": [...], "cloth1": [...], "cloth2": [...]}，
#   每项是 '#rrggbb'、'PAL.<键>'（模块 PAL 字典）或模块里的颜色列表名（如 'HAIR'）。没声明的颜色都算 other。
CLASSES = ('skin', 'hair', 'eyes', 'cloth1', 'cloth2', 'other')
CLASS_RGB = {'skin': (1.0, 0.72, 0.55), 'hair': (0.95, 0.85, 0.2), 'eyes': (0.3, 1.0, 0.4),
             'cloth1': (0.25, 0.5, 1.0), 'cloth2': (0.8, 0.35, 1.0), 'other': (0.45, 0.45, 0.5)}
# 阈值：按肯帕雷拉（用户认可的水准）标定，各留一点余量。见 report 里 campanella 那一行。
# campanella（2026-09 标定）：lm_rms 1.23 / lm_max 2.15 / n_x 10 / sil 0.905 / chamfer(内部) 1.33 / skin 0.78 / hair 0.86
FID_TOL = {
    'lm_rms': 1.8,       # 关键点残差 RMS（像素）<=
    'lm_max': 3.5,       # 关键点最大残差（像素）<=
    'n_x': 8,            # 参与计算的 x_ 轮廓 / 服装点个数 >=（点太少时 RMS 没有意义）
    'sil_iou': 0.86,     # 剪影 IoU >=
    'chamfer': 1.55,     # 剪影内部边缘的倒角距离（像素）<=；外轮廓不计（见 fidelity）
    'skin_iou': 0.72,    # 皮肤区域 IoU >=
    'hair_iou': 0.78,    # 头发区域 IoU >=
}


def _hex(c):
    c = c.lstrip('#')
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


def _srgb_to_lab(rgb):
    rgb = np.asarray(rgb, np.float32)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = lin @ M.T / np.array([0.9505, 1.0, 1.089], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _eye_center(lm, side):
    pts = [lm[f'eye_{side}_{k}'] for k in ('in', 'out', 'top', 'bot') if f'eye_{side}_{k}' in lm]
    return np.mean(pts, 0) if len(pts) >= 2 else None


def fid_anchors(char, ref, which):
    """返回 (官方锚点 dict, 像素锚点 dict, 全点拟合 A_full)。"""
    data = json.loads((LM_DIR / f'{char}.json').read_text())
    ref_set = data[ref]
    ref_lm = load_ref_lm(char, ref)
    spr, spr_lm = load_sprite(char, which)
    import cast
    raw = getattr(cast.get(char).mod, 'LANDMARKS', {}).get(which, {})
    A_full = align(ref_lm, {k: v for k, v in spr_lm.items() if not k.startswith('a_')})
    ra, sa = {}, {}
    for k in ('eye_near', 'eye_far', 'chin'):
        r = ref_set.get('_anchors', {}).get(k)
        ra[k] = np.array(r, float) if r is not None else (ref_lm.get('chin') if k == 'chin' else _eye_center(ref_lm, k[4:]))
        v = raw.get('a_' + k)
        sa[k] = np.array(v, float) if v is not None else (spr_lm.get('chin') if k == 'chin' else _eye_center(spr_lm, k[4:]))
        if sa[k] is None and ra[k] is not None:
            sa[k] = A_full['f'](ra[k][None])[0]
    return ra, sa, A_full


def _class_table(char, mod):
    data = json.loads((LM_DIR / f'{char}.json').read_text())
    decl = data.get('_classes', {})
    cols, labs = [], []
    for ci, cname in enumerate(CLASSES[:-1]):
        for e in decl.get(cname, []):
            if e.startswith('#'):
                vals = [e]
            elif e.startswith('PAL.'):
                vals = [getattr(mod, 'PAL')[e[4:]]]
            else:
                v = getattr(mod, e)
                vals = [v] if isinstance(v, str) else list(v)
            for h in vals:
                cols.append(_hex(h))
                labs.append(ci)
    return np.array(cols, np.float32).reshape(-1, 3), np.array(labs, int)


def label_sprite(spr, cols, labs):
    """像素图每个不透明像素 -> 材质编号（按声明的颜色查表，查不到 = other）。透明 = -1。"""
    h, w = spr.shape[:2]
    out = np.full((h, w), -1, int)
    op = spr[..., 3] > 0.5
    rgb = spr[..., :3][op]
    lab = np.full(len(rgb), len(CLASSES) - 1, int)
    if len(cols):
        d = np.abs(rgb[:, None, :] - cols[None]).max(-1)
        j = d.argmin(1)
        hit = d[np.arange(len(rgb)), j] < 0.012
        lab[hit] = labs[j[hit]]
    out[op] = lab
    return out


def _mode3(L):
    """3x3 众数滤波（去掉官方图分类里的孤立噪点）。"""
    from scipy import ndimage
    cnt = np.stack([ndimage.uniform_filter((L == c).astype(np.float32), 3) for c in range(-1, len(CLASSES))])
    return cnt.argmax(0) - 1


LAB_W = np.array([0.6, 1.0, 1.0], np.float32)   # 分材质时亮度权重打折：像素画的色阶和原画明暗不同，色相 / 彩度更能代表材质


def label_ref(warped, spr, spr_lab, iters=3):
    """官方图（已变换到像素分辨率）的材质：Lab k-means（亮度打折），种子 = 像素图里出现的每种颜色（带材质标签），
    迭代几轮让种子贴合原画的实际颜色，再做 3x3 众数滤波。"""
    h, w = warped.shape[:2]
    out = np.full((h, w), -1, int)
    op = warped[..., 3] >= 0.5
    sop = spr[..., 3] > 0.5
    seeds, idx = np.unique(np.round(spr[..., :3][sop] * 255).astype(int), axis=0, return_index=True)
    seed_lab = spr_lab[sop][idx]
    C = _srgb_to_lab(seeds / 255.0) * LAB_W
    X = _srgb_to_lab(np.clip(warped[..., :3][op], 0, 1)) * LAB_W
    for _ in range(iters):
        j = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for i in range(len(C)):
            m = j == i
            if m.any():
                C[i] = C[i] * 0.3 + X[m].mean(0) * 0.7
    j = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    out[op] = seed_lab[j]
    out = _mode3(out)
    out[~op] = -1
    return out


def _edges_lab(rgba, thr=35.0, bg=(0.5, 0.5, 0.52)):
    """RGBA -> 与 4 邻居的 Lab 色差超过 thr 的像素（先铺中灰底，剪影边缘也算边）。"""
    a = rgba[..., 3:4]
    rgb = rgba[..., :3] * a + np.array(bg, np.float32) * (1 - a)
    L = _srgb_to_lab(np.clip(rgb, 0, 1))
    e = np.zeros(L.shape[:2], bool)
    for dy, dx in ((0, 1), (1, 0)):
        d = np.linalg.norm(L[dy:, dx:] - L[:L.shape[0] - dy, :L.shape[1] - dx], axis=-1) > thr
        e[dy:, dx:] |= d
        e[:L.shape[0] - dy, :L.shape[1] - dx] |= d
    return e


def fidelity(char, ref=None, which='portrait', quiet=False, out_png=True):
    from scipy import ndimage
    import cast
    if ref is None:
        ref = next((r for r, wch in PAIRS.get(char, []) if wch == which), 'bust')
    ref_lm = load_ref_lm(char, ref)
    spr, spr_lm = load_sprite(char, which)
    spr = np.asarray(spr, np.float32)
    h, w = spr.shape[:2]
    ra, sa, A_full = fid_anchors(char, ref, which)
    keys = [k for k in ('eye_near', 'eye_far', 'chin') if ra[k] is not None and sa[k] is not None]
    src = np.array([ra[k] for k in keys])
    dst = np.array([sa[k] for k in keys])
    f, (sc, R, t) = procrustes(src, dst)
    A = dict(f=f, s=sc, R=R, t=t)
    # 2. 关键点残差（固定取景，不重拟合）
    common = [k for k in ref_lm if k in spr_lm]
    res = {k: float(np.linalg.norm(f(ref_lm[k][None])[0] - spr_lm[k])) for k in common}
    rv = np.array(list(res.values()))
    lm_rms, lm_max = float(np.sqrt((rv ** 2).mean())), float(rv.max())
    # 3. 剪影
    ref_im = Image.open(ref_path(char, ref))
    warped = warp_ref(ref_im, A, h, w)
    sil_iou = silhouette_iou(spr, warped)[0]
    # 4. 边缘倒角距离
    ss = 4
    A_hi = dict(s=sc * ss, R=R, t=t * ss + (ss - 1) / 2)      # 像素坐标 x -> 高分辨率 ss*x + (ss-1)/2
    hi = warp_ref(ref_im, A_hi, h * ss, w * ss, ss=2)
    e_hi = _edges_lab(hi)
    e_ref = e_hi.reshape(h, ss, w, ss).sum((1, 3)) >= 3     # 4x4 高分辨率块里至少 3 个边缘像素
    e_spr = _edges_lab(spr)
    # 只算剪影内部的线（发束、五官、衣褶）。外轮廓的位置已经由 sil_iou 衡量；
    # 算进来的话，浅色衣服在灰底上几乎没有边，像素画的外描边反而被当成「多出来的边」扣分。
    op_s, op_r = spr[..., 3] > 0.5, warped[..., 3] >= 0.5
    band = np.zeros_like(op_s)
    for m_ in (op_s, op_r):
        band |= ndimage.binary_dilation(m_, iterations=2) & ~ndimage.binary_erosion(m_, iterations=2)
    roi = (op_s | op_r) & ~band
    e_ref &= roi
    e_spr &= roi
    if e_ref.any() and e_spr.any():
        d_to_ref = ndimage.distance_transform_edt(~e_ref)
        d_to_spr = ndimage.distance_transform_edt(~e_spr)
        chamfer = float((d_to_ref[e_spr].mean() + d_to_spr[e_ref].mean()) / 2)
    else:
        chamfer = float('inf')
    # 5. 材质区域
    cols, labs = _class_table(char, cast.get(char).mod)
    sl = label_sprite(spr, cols, labs)
    rl = label_ref(warped, spr, sl)
    cls_iou = {}
    for ci, cname in enumerate(CLASSES):
        a_, b_ = sl == ci, rl == ci
        u = (a_ | b_).sum()
        cls_iou[cname] = float((a_ & b_).sum() / u) if u else float('nan')
    m = dict(lm_rms=lm_rms, lm_max=lm_max, n_x=sum(k.startswith('x_') for k in common), sil_iou=sil_iou, chamfer=chamfer,
             skin_iou=cls_iou['skin'], hair_iou=cls_iou['hair'])
    ok = {k: (m[k] <= FID_TOL[k]) if k in ('lm_rms', 'lm_max', 'chamfer') else (m[k] >= FID_TOL[k]) for k in FID_TOL}
    if not quiet:
        print(f'== {char} 头像保真度：官方 {ref} vs 像素 {which}（锚点 {"/".join(keys)}，缩放 {sc:.4f}'
              f'{"，镜像" if np.linalg.det(R) < 0 else ""}）')
        for k in FID_TOL:
            cmp_ = '<=' if k in ('lm_rms', 'lm_max', 'chamfer') else '>='
            v = f'{m[k]:8d}' if k == 'n_x' else f'{m[k]:8.3f}'
            print(f'   {k:<10}{v}   ({cmp_} {FID_TOL[k]})  {"OK" if ok[k] else "超标"}')
        print('   材质 IoU: ' + ', '.join(f'{c} {v:.2f}' for c, v in cls_iou.items()))
        order = sorted(res, key=lambda k: -res[k])
        print('   关键点残差(px): ' + ', '.join(f'{k} {res[k]:.1f}' for k in order))
    if out_png:
        p = OUT / f'{char}_fidelity.png'
        fidelity_img(warped, spr, sl, rl, e_ref, e_spr, ref_lm, spr_lm, f, common).save(p)
        if not quiet:
            print(f'   对照图: {p.relative_to(ROOT)}')
    return dict(metrics=m, ok=ok, all_ok=all(ok.values()), cls_iou=cls_iou, res=res, ref=ref)


def fidelity_img(warped, spr, sl, rl, e_ref, e_spr, ref_lm, spr_lm, f, keys, scale=5):
    """官方(固定取景) | 像素图 | 材质分歧热图 | 边缘叠图 + 关键点残差。"""
    h, w = spr.shape[:2]
    heat = np.zeros((h, w, 4), np.float32)
    any_ = (sl >= 0) | (rl >= 0)
    agree = any_ & (sl == rl)
    for ci, cname in enumerate(CLASSES):
        mm = agree & (sl == ci)
        heat[mm, :3] = np.array(CLASS_RGB[cname]) * 0.35
    dis = any_ & (sl != rl)
    heat[dis, :3] = np.array([1.0, 0.15, 0.15])
    heat[dis & (sl == 1) | dis & (rl == 1), :3] = np.array([1.0, 0.85, 0.1])      # 头发不一致
    heat[dis & ((sl == 0) | (rl == 0)), :3] = np.array([1.0, 0.45, 0.8])          # 皮肤不一致
    heat[any_, 3] = 1
    ed = np.zeros((h, w, 4), np.float32)
    ed[..., :3] = spr[..., :3] * 0.25
    ed[..., 3] = 1
    ed[e_ref & ~e_spr, :3] = (0.3, 0.6, 1.0)
    ed[e_spr & ~e_ref, :3] = (1.0, 0.3, 0.3)
    ed[e_spr & e_ref, :3] = (1.0, 1.0, 1.0)
    panels = [to_pil(warped), to_pil(spr), to_pil(heat), to_pil(ed)]
    W = (w * scale + 8) * len(panels) + 8
    canvas = checker(W, h * scale + 16)
    d = ImageDraw.Draw(canvas)
    for i, p in enumerate(panels):
        canvas.alpha_composite(p.resize((w * scale, h * scale), Image.NEAREST), (8 + i * (w * scale + 8), 8))
    for x0 in (8 + (w * scale + 8), 8 + 3 * (w * scale + 8)):
        for k in keys:
            pr = f(ref_lm[k][None])[0]
            pp = spr_lm[k]
            P1 = (x0 + (pr[0] + .5) * scale, 8 + (pr[1] + .5) * scale)
            P2 = (x0 + (pp[0] + .5) * scale, 8 + (pp[1] + .5) * scale)
            d.line([P1, P2], fill=(255, 60, 60, 255), width=2)
            d.ellipse([P1[0] - 3, P1[1] - 3, P1[0] + 3, P1[1] + 3], outline=(80, 200, 255, 255), width=2)
            d.ellipse([P2[0] - 2, P2[1] - 2, P2[0] + 2, P2[1] + 2], fill=(0, 255, 140, 255))
    return canvas


PAIRS = {  # 角色 -> [(官方参考图, 像素图)]
    'campanella': [('bust', 'portrait'), ('full', 'idle')],
    'mcburn': [('face', 'portrait'), ('full', 'idle')],     # face 这组点标在 full.png 上（_image）
    'leonhardt': [('bust', 'portrait'), ('full', 'idle')],
    'arianrhod': [('bust', 'portrait'), ('full', 'idle')],
    'bleublanc': [('bust', 'portrait'), ('full', 'idle')],
    'vita': [('bust', 'portrait'), ('full', 'idle')],     # bust 这组点标在 crop_sen3.png 上
    'grandmaster': [('full', 'idle')],
}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('grid')
    g.add_argument('image')
    g.add_argument('--crop', nargs=4, type=int)
    g.add_argument('--step', type=int, default=10)
    g.add_argument('--scale', type=int)
    g.add_argument('--out')
    sp = sub.add_parser('sprite')
    sp.add_argument('char')
    sp.add_argument('which')
    sp.add_argument('--step', type=int, default=5)
    sp.add_argument('--scale', type=int, default=10)
    sh = sub.add_parser('show')
    sh.add_argument('char')
    sh.add_argument('ref')
    sh.add_argument('--crop', nargs=4, type=int)
    sh.add_argument('--step', type=int, default=20)
    c = sub.add_parser('compare')
    c.add_argument('char')
    c.add_argument('ref')
    c.add_argument('which')
    fi = sub.add_parser('fidelity')
    fi.add_argument('char')
    fi.add_argument('--ref')
    fi.add_argument('--which', default='portrait')
    sub.add_parser('report')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.cmd == 'grid':
        im = Image.open(a.image)
        out = a.out or OUT / f'grid_{Path(a.image).parent.name}_{Path(a.image).stem}.png'
        grid_img(im, a.crop, a.step, a.scale).save(out)
        print(out)
    elif a.cmd == 'sprite':
        spr, lm = load_sprite(a.char, a.which)
        out = OUT / f'sprite_{a.char}_{a.which}.png'
        grid_img(spr, None, a.step, a.scale, lm).save(out)
        print(out)
    elif a.cmd == 'show':
        lm = load_ref_lm(a.char, a.ref)
        out = OUT / f'show_{a.char}_{a.ref}.png'
        grid_img(Image.open(ref_path(a.char, a.ref)), a.crop, a.step, None, lm).save(out)
        print(out)
    elif a.cmd == 'fidelity':
        r = fidelity(a.char, a.ref, a.which)
        sys.exit(0 if r['all_ok'] else 1)
    elif a.cmd == 'compare':
        r = cmd_compare(a.char, a.ref, a.which)
        sys.exit(0 if r and r[1] else 1)
    elif a.cmd == 'report':
        bad = 0
        print(f'{"角色":<12}{"对照":<22}{"误差":>8}{"阈值":>8}{"剪影IoU":>9}{"下限":>7}')
        for ch, pairs in PAIRS.items():
            for ref, which in pairs:
                try:
                    r = cmd_compare(ch, ref, which, quiet=True)
                except (KeyError, FileNotFoundError) as e:
                    print(f'{ch:<12}{ref + " vs " + which:<22}{"缺数据":>8}   ({e})')
                    bad += 1
                    continue
                if r is None:
                    print(f'{ch:<12}{ref + " vs " + which:<22}{"缺点":>8}')
                    bad += 1
                    continue
                kind = 'face' if which == 'portrait' else 'body'
                bad_what = [n for n, o in (('误差', r[3]), ('IoU', r[4])) if not o]
                print(f'{ch:<12}{ref + " vs " + which:<22}{r[0] * 100:7.2f}%{TOL[kind] * 100:7.1f}%'
                      f'{r[2]:9.3f}{IOU_MIN[kind]:7.2f}'
                      f'  {"OK" if r[1] else "超标(" + "/".join(bad_what) + ")"}')
                bad += not r[1]
        print()
        print('头像保真度（fidelity，固定取景：两眼中心 + 下巴三点定位，不再重拟合）')
        hdr = ''.join(f'{k:>10}' for k in FID_TOL)
        print(f'{"角色":<12}{"对照":<10}{hdr}')
        print(f'{"阈值":<12}{"":<10}' + ''.join(f'{("<=" if k in ("lm_rms", "lm_max", "chamfer") else ">=") + str(v):>10}'
                                             for k, v in FID_TOL.items()))
        for ch, pairs in PAIRS.items():
            for ref, which in pairs:
                if which != 'portrait':
                    continue
                try:
                    r = fidelity(ch, ref, which, quiet=True)
                except (KeyError, FileNotFoundError) as e:
                    print(f'{ch:<12}{ref:<10}缺数据 ({e})')
                    bad += 1
                    continue
                cells = ''.join((f'{r["metrics"][k]:9d}' if k == 'n_x' else f'{r["metrics"][k]:9.3f}')
                                + (' ' if r['ok'][k] else '*') for k in FID_TOL)
                print(f'{ch:<12}{ref:<10}{cells}  {"OK" if r["all_ok"] else "超标"}')
                bad += not r['all_ok']
        print('（* = 超标；对照图 preview/measure/<角色>_fidelity.png）')
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
