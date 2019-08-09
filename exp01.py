import random
import xmltodict
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rect

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as tf
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.models import resnet18
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


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
            x1, y1 = box['xmin'], box['ymin']
            x2, y2 = box['xmax'], box['ymax']
            boxs.append([float(x1), float(y1), float(x2), float(y2)])

        boxs = torch.FloatTensor(boxs)
        lbls = torch.LongTensor(lbls)
        N = boxs.size(0)

        img = Image.open(self.img_paths[idx]).convert('RGB')
        img_id = torch.tensor([idx])
        srcW, srcH = img.size
        dstW, dstH = self.img_size
        img = img.resize(self.img_size)
        boxs = boxs / torch.FloatTensor([srcW, srcH, srcW, srcH])
        boxs = boxs * torch.FloatTensor([dstW, dstH, dstW, dstH])

        image = tf.to_tensor(img)
        target = {
            'boxes': boxs,
            'labels': lbls,
            'image_id': torch.tensor([idx]),
            'area': (boxs[:, 2] - boxs[:, 0]) * (boxs[:, 3] - boxs[:, 1]),
            'iscrowd': torch.zeros(N).long(),
        }

        return image, target

    @staticmethod
    def visualize_detection(ax, target):
        boxes = target['boxes']
        labels = target['labels']

        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = colors[labels % 20]
        colors = [mpl.colors.rgb2hex(c[:3]) for c in colors]

        for (x0, y0, x1, y1), color in zip(boxes, colors):
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
            ax.add_patch(Rect((x, y), w, h, ec=color, fc='none', lw=2))

        return ax


def get_model():
    backbone = resnet_fpn_backbone('resnet34', pretrained=True)
    rpn_anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model = FasterRCNN(
        backbone,
        num_classes=1 + 3,
        min_size=512,
        max_size=512,
        rpn_anchor_generator=rpn_anchor_generator,
    )

    return model


def run_train():
    pixiv = Pixiv('./dataset/', [512, 512])
    indices = list(range(len(pixiv)))
    train_set = Subset(pixiv, indices[: len(indices) * 4 // 5])
    valid_set = Subset(pixiv, indices[len(indices) * 4 // 5 :])
    visul_set = ConcatDataset(
        [
            Subset(train_set, random.sample(range(len(train_set)), k=25)),
            Subset(valid_set, random.sample(range(len(valid_set)), k=25)),
        ]
    )

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_set, 2, shuffle=True, num_workers=2, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_set, 2, shuffle=False, num_workers=2, collate_fn=collate_fn
    )
    visul_loader = DataLoader(
        visul_set, 2, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    device = 'cuda'
    model = get_model().to(device)
    parameter = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameter, lr=1e-4)

    log_dir = Path('./log/') / f'{datetime.now():%b.%d %H:%M:%S}'
    log_dir.mkdir(parents=True)
    history = {
        'train_loss': [],
        'train_loss_rpn_box_reg': [],
        'train_loss_objectness': [],
        'train_loss_box_reg': [],
        'train_loss_classifier': [],
        'valid_loss': [],
        'valid_loss_rpn_box_reg': [],
        'valid_loss_objectness': [],
        'valid_loss_box_reg': [],
        'valid_loss_classifier': [],
    }

    def train(epoch):
        model.train()
        metrics = defaultdict(list)
        for images, targets in tqdm(iter(train_loader), leave=True):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            for k, v in loss_dict.items():
                metrics[k].append(v.detach().item())
            metrics['loss'].append(losses.detach().item())
        for k, v in metrics.items():
            history[f'train_{k}'].append(sum(v) / len(v))

    @torch.no_grad()
    def valid(epoch):
        model.train()
        metrics = defaultdict(list)
        for images, targets in tqdm(iter(valid_loader), leave=True):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            for k, v in loss_dict.items():
                metrics[k].append(v.detach().item())
            metrics['loss'].append(losses.detach().item())
        for k, v in metrics.items():
            history[f'valid_{k}'].append(sum(v) / len(v))

    @torch.no_grad()
    def visul(epoch):
        model.eval()
        epoch_dir = log_dir / f'{epoch:03d}'
        epoch_dir.mkdir()
        idx = 0
        for images, targets in tqdm(iter(visul_loader), leave=True):
            predicts = model([image.to(device) for image in images])
            predicts = [{k: v.to('cpu') for k, v in p.items()} for p in predicts]
            for image, target, predict in zip(images, targets, predicts):
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(tf.to_pil_image(image))
                ax[1].imshow(tf.to_pil_image(image))
                pixiv.visualize_detection(ax[0], target)
                pixiv.visualize_detection(ax[1], predict)
                fig.savefig(epoch_dir / f'{idx:03d}.svg')
                plt.close()
                idx += 1

    def log(epoch):
        df = pd.DataFrame(history)
        df.to_csv(log_dir / 'metrics.csv')

        if epoch >= 1:
            fig, ax = plt.subplots(3, 2, figsize=(15, 15))
            keys = [
                'loss',
                'loss_rpn_box_reg',
                'loss_objectness',
                'loss_box_reg',
                'loss_classifier',
            ]
            for i, k in enumerate(keys):
                df[[f'train_{k}', f'valid_{k}']].plot(kind='line', ax=ax[divmod(i, 2)])
            fig.savefig(log_dir / 'metrics.svg')
            plt.close()

    for epoch in range(5):
        print(f'Epoch: {epoch:03d}')
        train(epoch)
        valid(epoch)
        visul(epoch)
        log(epoch)

        for k, v in history.items():
            print(f'{k}: {v[-1]:.2e}')
        print('-' * 10)


if __name__ == '__main__':
    run_train()
