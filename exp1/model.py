import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from coder import coder

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(pretrained=True)
        self.pre = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.loc_head = self.make_head(512, coder.A * 4)
        self.ctg_head = self.make_head(512, coder.A * coder.C)

        del backbone

    def make_head(self, cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, 256, (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, cout, (1, 1)),
            nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        N = x.size(0)

        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        loc_pred_b = self.loc_head(x) # [N, A * 4, fmpH, fmpW]
        loc_pred_b = loc_pred_b.permute(0, 2, 3, 1) # [N, fmpH, fmpW, A * 4]
        loc_pred_b = loc_pred_b.contiguous().view(N, -1, 4)  # [N, #anchor, 4]

        ctg_pred_b = self.ctg_head(x) # [N, A * C, fmpH, fmpW]
        ctg_pred_b = ctg_pred_b.permute(0, 2, 3, 1)  # [N, fmpH, fmpW, A * C]
        ctg_pred_b = ctg_pred_b.contiguous().view(N, -1, coder.C)  # [N, #anchor, C]

        return loc_pred_b, ctg_pred_b


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc_ctiterion = nn.SmoothL1Loss()
        self.ctg_criterion = nn.CrossEntropyLoss()

    def forward(self, loc_pred_b, loc_true_b, ctg_pred_b, ctg_true_b):
        '''
        Args:
            loc_pred_b: (FloatTensor) sized [N, #anchor, 4]
            loc_true_b: (FloatTensor) sized [N, #anchor, 4]
            ctg_pred_b: (FloatTensor) sized [N, #anchor, C]
            ctg_true_b: (LongTensor) sized [N, #anchor]
        '''
        N = loc_pred_b.size(0)

        pos_mask_b = (ctg_true_b > 0) # [N, #anchor]
        loc_pred_b = loc_pred_b[pos_mask_b] # [#pos, 4]
        loc_true_b = loc_true_b[pos_mask_b] # [#pos, 4]

        ctg_pred_b = ctg_pred_b.view(-1, coder.C) # [N * #anchor, C]
        ctg_true_b = ctg_true_b.view(-1) # [N * #anchor]

        loc_loss = self.loc_ctiterion(loc_pred_b, loc_true_b) * 1
        ctg_loss = self.ctg_criterion(ctg_pred_b, ctg_true_b) * 1
        loss = loc_loss + ctg_loss

        return loss, loc_loss.detach(), ctg_loss.detach()


if __name__ == '__main__':
    device = 'cuda'
    model = Detector().to(device)
    img_b = torch.rand(32, 3, 384, 384).to(device)
    loc_pred_b, ctg_pred_b = model(img_b)

    print(len(coder.anchors))
    print(loc_pred_b.size())
    print(ctg_pred_b.size())
