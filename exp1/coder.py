import util
import torch
import numpy as np


class Coder:
    def __init__(self):
        self.img_size = (384, 384)
        self.C = 4
        self.fmps = [(12, 12)]
        self.areas = [32 * 32]
        self.ratios = [0.5, 1.0, 2]
        self.scales = [0.25, 0.5, 1, 2]
        self.A = len(self.ratios) * len(self.scales)
        self.anchors = self.get_anchors()

    def get_anchors(self):
        all_anchors = []

        for (fmpH, fmpW), area in zip(self.fmps, self.areas):
            # anchor sizes
            wh = []
            for ratio in self.ratios:
                w = np.sqrt(area / ratio)
                h = w * ratio
                for scale in self.scales:
                    ww = h * scale
                    hh = w * scale
                    wh.append([ww, hh])
            wh = torch.FloatTensor(wh)  # [#ratios * #scales, 2]
            wh = wh.view(1, 1, -1, 2)  # [1, 1, A, 2]
            wh = wh.expand(fmpH, fmpW, self.A, 2)

            # anchor centers
            strideH = self.img_size[0] / fmpH
            strideW = self.img_size[1] / fmpW
            xy = np.meshgrid(np.arange(fmpW), np.arange(fmpH))
            xy = np.stack(xy, axis=-1) + 0.5
            xy = xy * np.float32([strideW, strideH])
            xy = torch.from_numpy(xy).float()
            xy = xy.view(fmpH, fmpW, 1, 2)
            xy = xy.expand(fmpH, fmpW, self.A, 2)

            # anchor
            anchors = torch.cat([xy, wh], dim=3)  # [fmpH, fmpW, A, 4]
            anchors = anchors.view(-1, 4)  # [#anchor, 4]
            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=0)

    def encode(self, boxs, lbls):
        '''
        Args:
            boxs: (FloatTensor) ccwh, sized [#obj, 4]
            lbls: (LongTensor) sized [#obj]
        Return:
            loc: (FloatTensor) sized [#anchor, 4]
            ctg: (LongTensor) sized [#anchor #class + 1]
        '''
        n_obj, n_anchor = len(boxs), len(self.anchors)

        iou = util.iou(self.anchors, boxs) # [#anchor, #obj]
        # For each box, set the anchor with max iou as postive
        iou[iou.argmax(dim=0), torch.arange(n_obj)] = 1.0
        # For rest anchor, select the box with max iou
        iou_val, indices = iou.max(dim=1) # [#anchor], [#anchor]
        pos_mask = (iou_val > 0.3) # [#anchor]
        # Assign a box and a lbl to each anchor, ie anchor[i] <-> boxs[i], lbls[i]
        boxs = boxs[indices] # [#anchor, 4]
        lbls = lbls[indices] # [#anchor]

        loc = torch.zeros_like(self.anchors)
        loc[:, 0] = (boxs[:, 0] - self.anchors[:, 0]) / self.anchors[:, 2]
        loc[:, 1] = (boxs[:, 1] - self.anchors[:, 1]) / self.anchors[:, 3]
        loc[:, 2] = torch.log(boxs[:, 2] / self.anchors[:, 2])
        loc[:, 3] = torch.log(boxs[:, 3] / self.anchors[:, 3])

        ctg = torch.zeros(n_anchor).long() # background is zero
        ctg[pos_mask] = lbls[pos_mask]

        return loc, ctg

    def decode(self, loc, ctg, thresh=0.5):
        '''
        Args:
            loc: (FloatTensor) sized [#anchor, 4]
            ctg: (LongTensor) sized [#anchor, #category]
        Return:
            boxs: (FloatTensor or None) ccwh, sized [#pos, 4]
            lbls: (LongTensor or None) sized [#pos]
            cfds: (FloatTensor or None) sized [#pos]
        '''
        boxs = torch.zeros_like(self.anchors)
        boxs[:, 0] = loc[:, 0] * self.anchors[:, 2] + self.anchors[:, 0]
        boxs[:, 1] = loc[:, 1] * self.anchors[:, 3] + self.anchors[:, 1]
        boxs[:, 2] = torch.exp(loc[:, 2]) * self.anchors[:, 2]
        boxs[:, 3] = torch.exp(loc[:, 3]) * self.anchors[:, 3]

        cfds, lbls = ctg.max(dim=1)
        pos_mask = (lbls != 0) & (cfds >= thresh)
        if pos_mask.sum() == 0:
            return None, None, None
        boxs = boxs[pos_mask]
        lbls = lbls[pos_mask]
        cfds = cfds[pos_mask]

        return boxs, lbls, cfds


coder = Coder() # global var
