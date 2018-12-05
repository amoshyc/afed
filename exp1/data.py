import xmltodict
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as tf

import svg
import util
from coder import coder


class Pixiv:
    def __init__(self, root_dir, img_size):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.xml_paths = sorted(list(self.root_dir.glob('pixiv*/*.xml')))
        self.img_paths = [p.with_suffix('.jpg') for p in self.xml_paths]
        self.img_size = img_size

    def __len__(self):
        return len(self.xml_paths)

    def __getitem__(self, idx):
        with self.xml_paths[idx].open() as f:
            ann = xmltodict.parse(f.read())['annotation']
        objs = ann['object']
        if not isinstance(objs, list):
            objs = [objs]
        boxs, lbls = [], []
        for obj in objs:
            lbl = obj['name']
            lbl = 1 if lbl == 'face' else lbl
            lbl = 2 if lbl == 're' else lbl
            lbl = 3 if lbl == 'le' else lbl
            lbls.append(lbl)
            box = obj['bndbox']
            x1, y1 = box['xmin'], box['ymin'],
            x2, y2 = box['xmax'], box['ymax']
            boxs.append([int(x1), int(y1), int(x2), int(y2)])

        boxs = torch.FloatTensor(boxs)
        lbls = torch.LongTensor(lbls)

        img = Image.open(self.img_paths[idx])
        srcW, srcH = img.size
        dstW, dstH = self.img_size

        img = img.convert('RGB').resize(self.img_size)
        boxs = boxs / torch.FloatTensor([srcW, srcH, srcW, srcH])
        boxs = boxs * torch.FloatTensor([dstW, dstH, dstW, dstH])
        boxs = util.xyxy2ccwh(boxs)

        loc, ctg = coder.encode(boxs, lbls)
        img = tf.to_tensor(img)

        return img, loc, ctg


if __name__ == '__main__':
    root_dir = '../raw/'
    img_size = (384, 384)
    dataset = Pixiv(root_dir, img_size)
    print(len(dataset))

    img, loc_true, ctg_true = dataset[-2]

    img = tf.to_pil_image(img)
    w, h = img.size

    loc_pred = loc_true
    ctg_pred = util.to_one_hot(ctg_true, coder.C)
    boxs, lbls, cfds = coder.decode(loc_pred, ctg_pred)
    boxs = util.ccwh2xyxy(boxs)
    print(boxs)
    print(lbls)
    print(cfds)

    vis = svg.g([
        svg.pil(img),
        svg.bboxs(boxs, lbls, cfds)
    ])
    svg.save([vis], './vis.svg', (h, w))
