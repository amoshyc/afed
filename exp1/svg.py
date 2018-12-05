from math import ceil
from io import BytesIO
from base64 import b64encode

import numpy as np
from PIL import Image
import svgwrite as sw


def g(elems):
    '''
    '''
    g = sw.container.Group()
    for elem in elems:
        if elem is not None:
            g.add(elem)
    return g


def pil(img):
    '''
    '''
    w, h = img.size
    buf = BytesIO()
    img.save(buf, 'png')
    b64 = b64encode(buf.getvalue()).decode()
    href = 'data:image/png;base64,' + b64
    elem = sw.image.Image(href, (0, 0), width=w, height=h)
    return elem


def img(img):
    '''
    '''
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    return pil(img)


def text(s, pos, color='orange', **extra):
    extra['fill'] = color
    return sw.text.Text(s, pos, **extra)


def save(elems, fname, size, per_row=-1, padding=2, pad_val=None):
    '''
    '''
    n_elem = len(elems)
    elems = [g.copy() for g in elems]
    imgH, imgW = size
    per_row = n_elem if per_row == -1 else per_row
    per_col = ceil(n_elem / per_row)
    gridW = per_row * imgW + (per_row - 1) * padding
    gridH = per_col * imgH + (per_col - 1) * padding

    svg = sw.Drawing(size=[gridW, gridH])
    if pad_val:
        svg.add(sw.shapes.Rect((0, 0), (gridW, gridH), fill=pad_val))
    for i in range(n_elem):
        c = (i % per_row) * (imgW + padding)
        r = (i // per_row) * (imgH + padding)
        elems[i].translate(c, r)
        svg.add(elems[i])

    with open(str(fname), 'w') as f:
        svg.write(f, pretty=True)


def to_png(src_path, dst_path, scale=2):
    '''
    '''
    import cairosvg
    pass


########################################

import torch

def bboxs(boxs, lbls, cfds, lw=1):
    '''
    Args:
        boxs: (FloatTensor) ccwh, sized [#obj, 4]
        lbls: (LongTensor) sized [#obj]
        cfds: (FloatTensor) sized [#obj]
    '''
    if boxs is None or lbls is None or cfds is None:
        return g([])

    boxs = boxs.numpy().tolist()
    lbls = lbls.numpy().tolist()
    cfds = cfds.numpy().tolist()
    colors = ['#8BC34A', '#F44336', '#1E88E5']

    def transform(i):
        x1, y1, x2, y2 = map(round, boxs[i])
        color = colors[lbls[i] - 1] # 0 is bg
        cfd_text = '{:.3f}'.format(cfds[i])
        return g([
            sw.shapes.Rect(**{
                'insert': (x1, y1),
                'size': (x2 - x1, y2 - y1),
                'stroke': color,
                'stroke_width': lw,
                'fill_opacity': 0.0
            }),
            sw.text.Text(cfd_text, **{
                'insert': (x1, y1 - 2),
                'font_size': 8,
                'font_weight': 'bold',
                'fill': color,
            }),
        ])

    vis_boxs = [transform(i) for i in range(len(boxs))]
    vis_text = [sw.text.Text('#obj: {}'.format(len(boxs)), **{
        'insert': (5, 10),
        'font_size': 10,
        'font_weight': 'bold',
        'fill': '#FF9800',
    })]

    return g(vis_boxs + vis_text)



##########################################
