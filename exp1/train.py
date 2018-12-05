import json
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib as mpl
mpl.use('SVG')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as tf
from torch.utils.data import Subset, ConcatDataset, DataLoader

import svg
import util
from data import Pixiv
from coder import coder
from model import Model, Loss


device = 'cuda'
model = Model().to(device)
criterion = Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

dataset = Pixiv('../raw/', coder.img_size)
pivot = len(dataset) * 9 // 10
train_set = Subset(dataset, range(0, pivot))
valid_set = Subset(dataset, range(pivot, len(dataset)))
visul_set = ConcatDataset([
    Subset(train_set, random.sample(range(len(train_set)), 50)),
    Subset(valid_set, random.sample(range(len(valid_set)), 50)),
])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_set, batch_size=32)
visul_loader = DataLoader(visul_set, batch_size=32)

log_dir = Path('./log/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
log_dir.mkdir(parents=True)


def train(pbar):
    model.train()
    metrics = {
        'loss': util.RunningAverage(),
        'loc_loss': util.RunningAverage(),
        'ctg_loss': util.RunningAverage()
    }
    for img_b, loc_true_b, ctg_true_b in iter(train_loader):
        img_b = img_b.to(device)
        loc_true_b = loc_true_b.to(device)
        ctg_true_b = ctg_true_b.to(device)

        optimizer.zero_grad()
        loc_pred_b, ctg_pred_b = model(img_b)
        loss, loc_loss, ctg_loss = \
            criterion(loc_pred_b, loc_true_b, ctg_pred_b, ctg_true_b)
        loss.backward()
        optimizer.step()

        metrics['loss'].update(loss.item())
        metrics['loc_loss'].update(loc_loss.item())
        metrics['ctg_loss'].update(ctg_loss.item())
        pbar.set_postfix(metrics)
        pbar.update(len(img_b))
    return metrics


def valid(pbar):
    model.eval()
    metrics = {
        'loss': util.RunningAverage(),
        'loc_loss': util.RunningAverage(),
        'ctg_loss': util.RunningAverage()
    }
    for img_b, loc_true_b, ctg_true_b in iter(valid_loader):
        img_b = img_b.to(device)
        loc_true_b = loc_true_b.to(device)
        ctg_true_b = ctg_true_b.to(device)

        loc_pred_b, ctg_pred_b = model(img_b)
        loss, loc_loss, ctg_loss = \
            criterion(loc_pred_b, loc_true_b, ctg_pred_b, ctg_true_b)

        metrics['loss'].update(loss.item())
        metrics['loc_loss'].update(loc_loss.item())
        metrics['ctg_loss'].update(ctg_loss.item())
        pbar.set_postfix(metrics)
        pbar.update(len(img_b))
    return {f'val_{k}':v for k, v in metrics.items()}


def visul(epoch, pbar):
    epoch_dir = log_dir / f'{epoch:03d}'
    epoch_dir.mkdir()

    for img_b, loc_true_b, ctg_true_b in iter(visul_loader):
        loc_pred_b, ctg_pred_b = model(img_b.to(device))
        loc_pred_b = loc_pred_b.cpu()
        ctg_pred_b = ctg_pred_b.cpu()

        for i in range(len(img_b)):
            img = tf.to_pil_image(img_b[i])
            w, h = img.size
            loc_pred, loc_true = loc_pred_b[i], loc_true_b[i]
            ctg_pred, ctg_true = ctg_pred_b[i], ctg_true_b[i]

            ctg_pred = torch.softmax(ctg_pred, dim=1)
            boxs_pred, lbls_pred, cfds_pred = coder.decode(loc_pred, ctg_pred)
            boxs_pred = util.ccwh2xyxy(boxs_pred)
            boxs_pred = util.clamp(boxs_pred, w, h)
            vis_pred = svg.g([
                svg.pil(img),
                svg.bboxs(boxs_pred, lbls_pred, cfds_pred)
            ])

            ctg_true = util.to_one_hot(ctg_true, coder.C)
            boxs_true, lbls_true, cfds_true = coder.decode(loc_true, ctg_true)
            boxs_true = util.ccwh2xyxy(boxs_true)
            boxs_true = util.clamp(boxs_true, w, h)
            vis_true = svg.g([
                svg.pil(img),
                svg.bboxs(boxs_true, lbls_true, cfds_true)
            ])

            vis_path = epoch_dir / f'{pbar.n:03d}.svg'
            svg.save([vis_true, vis_pred], vis_path, (h, w))
            pbar.update(1)


def log(epoch, train_metrics, valid_metrics):
    json_path = log_dir / 'log.json'
    if json_path.exists():
        df = pd.read_json(json_path)
    else:
        df = pd.DataFrame()

    metrics = {'epoch': epoch, **train_metrics, **valid_metrics}
    df = df.append(metrics, ignore_index=True)
    df = df.astype('str').astype('float')
    with json_path.open('w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2)

    fig, ax = plt.subplots(1, 3, dpi=100, figsize=(18, 5))
    df[['loss', 'val_loss']].plot(kind='line', ax=ax[0])
    df[['loc_loss', 'val_loc_loss']].plot(kind='line', ax=ax[1])
    df[['ctg_loss', 'val_ctg_loss']].plot(kind='line', ax=ax[2])
    fig.savefig(str(log_dir / 'loss.svg'))
    plt.close()

    if df['val_loss'].idxmin() == epoch:
        torch.save(model, log_dir / 'model.pth')


for epoch in range(50):
    print('Epoch', epoch)
    with tqdm(total=len(train_set), desc='  Train', ascii=True) as pbar:
        train_metrics = train(pbar)
    with torch.no_grad():
        with tqdm(total=len(valid_set), desc='  Valid', ascii=True) as pbar:
            valid_metrics = valid(pbar)
        with tqdm(total=len(visul_set), desc='  Visul', ascii=True) as pbar:
            visul(epoch, pbar)
        log(epoch, train_metrics, valid_metrics)

