"""把 preview/f_*.png 拼成若干张 2x2 联系表，方便检查。"""
import glob
import sys

from PIL import Image

fs = sorted(glob.glob('preview/f_*.png'))
per = int(sys.argv[1]) if len(sys.argv) > 1 else 4
for k in range(0, len(fs), per):
    ims = [Image.open(f) for f in fs[k:k + per]]
    w, h = ims[0].size
    cols = 2
    rows = (len(ims) + 1) // 2
    sheet = Image.new('RGB', (w * cols, h * rows))
    for j, im in enumerate(ims):
        sheet.paste(im, ((j % cols) * w, (j // cols) * h))
    sheet.save(f'preview/sheet{k // per}.png')
    print(f'preview/sheet{k // per}.png', [f[-9:-4] for f in fs[k:k + per]])
