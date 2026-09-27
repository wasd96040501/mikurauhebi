"""下载官方参考图到 refs/（不入库，版权归 Falcom）。

来源：Kiseki 维基（kiseki.fandom.com）的 api.php，HTML 页面会 403，API 可用。
用法：python tools/fetch_refs.py
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'refs'
UA = {'User-Agent': 'Mozilla/5.0 (mikurauhebi fan PV reference fetch)'}

# 角色 -> [(本地名, 维基文件名)]；full = 官方立绘全身，bust = 半身/头像
REFS = {
    'campanella': [('full', 'Campanella_(Sen_III).png'), ('bust', 'Campanella_-_Bust_(SC_Evo).png'),
                   ('bust_ao', 'Campanella_-_Bust_(Ao).png'), ('face_ao', 'Campanella_-_Portrait_00_(Ao).png'),
                   ('full_ao', 'Campanella_(Ao).png'), ('scraft', 'Campanella_-_S-Craft_Full_(Sen_III).png')],
    'mcburn': [('full', 'McBurn_(Sen_II).png'), ('scraft', 'McBurn_-_S-Craft_(Sen_II).png'),
               ('concept', 'McBurn_-_Concept_Art_(Sen_II).jpg'), ('demon', 'McBurn_Blazing_Demon_(Hajimari).png')],
    'leonhardt': [('full', 'Loewe_full_artwork_(SC_EVO).png'), ('bust', 'Leonhardt_-_Bust_(SC_Evo).png'),
                  ('full_2nd', 'Leonhardt_(Sky_the_2nd).png'), ('face', 'Loewe_-_Portrait_0120_(SC).png'),
                  ('scraft', 'Leonhardt_-_S-Craft_(3rd_Evo).png')],
    'arianrhod': [('full', 'Arianrhod_-_Full_(Sen_III).png'), ('bust', 'Arianrhod_-_Bust_(Ao).png'),
                  ('face_ao', 'Arianrhod_-_Portrait_00_(Ao).png'), ('scraft', 'Arianrhod_-_S-Craft_Full_(Sen_III).png')],
    'bleublanc': [('full', 'Bleublanc_artwork_(Sen_II).png'), ('bust', 'Bleublanc_-_Bust_(SC_Evo).png'),
                  ('face', 'Bleublanc_-_Portrait_0230_(SC).png'), ('scraft', 'Bleublanc_-_S-Craft_Full_(Sen_II).png')],
    'vita': [('full', 'Vita_Clotilde_(Sen_III).png'), ('full_s2', 'Vita_Clotilde_(Sen_II).png'),
             ('scraft', 'Vita_Clotilde_-_S-Craft_(Sen_IV).png'), ('staff', 'Orbal_Staff_Vita_(Sen_Weapon).png'),
             ('small', 'Vita_Small_(Sen_IV).png'), ('intro', 'Vita_Clotilde_-_Introduction_(CS_III).png'),
             ('sketch1', 'Vita_Clotilde_-_Fine-tuning_Sketches_1_(Sen_III).png'),
             ('sketch2', 'Vita_Clotilde_-_Fine-tuning_Sketches_2_(Sen_III).png')],
    'grandmaster': [('full', 'Grandmaster_(Sen_IV).png'), ('kai', 'Grandmaster_(Kai).png'),
                    ('kv', 'Grandmaster_Key_Visual_(Hajimari).png')],
}

# 从已下载的官方大图里裁出头肩部分，给头像保真度度量用（data/landmarks/*.json 里的 _image 指向这些文件）。
# (角色, 输出名, 源图名, (左, 上, 右, 下))；候选图的完整来源记录见 refs/<角色>/SOURCES.txt
CROPS = [
    ('mcburn', 'crop_sen2', 'full', (400, 0, 1250, 850)),      # 闪II 立绘头肩（闪III/IV 沿用）
    ('vita', 'crop_sen3', 'full', (650, 0, 1450, 800)),        # 闪III 立绘头肩
]


def api(**params):
    params.setdefault('format', 'json')
    url = 'https://kiseki.fandom.com/api.php?' + urllib.parse.urlencode(params)
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))


def main():
    for key, items in REFS.items():
        (ROOT / key).mkdir(parents=True, exist_ok=True)
        for name, fname in items:
            info = api(action='query', titles='File:' + fname, prop='imageinfo', iiprop='url|size')
            page = next(iter(info['query']['pages'].values()))
            if 'imageinfo' not in page:
                print(f'  !! {key}/{name}: {fname} 不存在')
                continue
            ii = page['imageinfo'][0]
            ext = Path(ii['url'].split('/revision')[0]).suffix.lower() or '.png'
            dst = ROOT / key / f'{name}{ext}'
            if not dst.exists():
                data = urllib.request.urlopen(urllib.request.Request(ii['url'], headers=UA), timeout=120).read()
                dst.write_bytes(data)
                time.sleep(0.3)
            print(f'  {key}/{dst.name}  {ii["width"]}x{ii["height"]}')
    from PIL import Image
    for key, name, src, box in CROPS:
        dst = ROOT / key / f'{name}.png'
        if not dst.exists():
            Image.open(ROOT / key / f'{src}.png').crop(box).save(dst)
        print(f'  {key}/{dst.name}  crop {box} of {src}.png')


if __name__ == '__main__':
    main()
