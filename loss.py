#!/usr/bin/env python
import torch
import torch.nn as nn
from torchvision import models
from pytorch_msssim import  SSIM

class Percetual(nn.Module):
    def __init__(self):
        super(Percetual, self).__init__()
        self.select = ['3', '6', '8', '11']
        self.vgg = models.vgg16(pretrained=True).features.to('cuda:0').eval()

        for param in self.vgg.parameters():
            param.requires_grad = False
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.ssim_loss = SSIM(data_range=1, size_average=True, channel=1, spatial_dims=2)

    def forward(self, output, target):
        l1loss = self.l1(output, target)
        l2loss = self.l2(output, target)
        l3loss = 1- self.ssim_loss(output, target)
        threeoutput = torch.cat((output, output, output), 1)
        threetarget = torch.cat((target, target, target), 1)

        features = []
        truefeatures = []
        for name, layer in self.vgg._modules.items():
            threeoutput = layer(threeoutput)
            threetarget = layer(threetarget)
            if name in self.select:
                features.append(threeoutput)
                truefeatures.append(threetarget)

        perloss = 0
        weights = [0.5, 0.5, 0.5, 0.5]
        for i in range(len(features)):
            perloss += weights[i]*self.l2(features[i], truefeatures[i])
        loss =  perloss + 10*l1loss + 1*l2loss + 1*l3loss
        
        return loss