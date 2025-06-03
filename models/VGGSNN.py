# -*- coding: utf-8 -*-
"""
    @Author       : 保持理智
    @Version      : v1.0
    @Date         : 2025-05-20 09:16:18
    @Description  : Spiking VGGSNN
"""
import torch.nn as nn

from models.layers import tdBatchNorm2d, tdLayer, SpikingNeuron


class VGGSNN(nn.Module):
    """ Implementation of VGGSNN in 'Temporal efficient training of spiking neural network via gradient re-weighting (ICLR, 2022)' """
    def __init__(self, in_channels=2, encode_channels=64, T=10, output_size=10):
        super(VGGSNN, self).__init__()
        self.in_channels, self.encode_channels = in_channels, encode_channels
        self.T, self.output_size = T, output_size

        self.planes = [128, 256, 512]
        ## encode layer
        self.encode = nn.Sequential(
            tdLayer(
                nn.Conv2d(self.in_channels, self.encode_channels, kernel_size=3, stride=1, padding=1, bias=False),
                tdBatchNorm2d(self.encode_channels),
            ),
            SpikingNeuron(),
        )

        ## features layer
        self.features = nn.Sequential(
            # blocks 1
            tdLayer(
                nn.Conv2d(self.encode_channels, self.planes[0], kernel_size=3, stride=1, padding=1, bias=False),
                tdBatchNorm2d(self.planes[0]),
            ),
            SpikingNeuron(),
            tdLayer(nn.AvgPool2d(2, 2)),

            # blocks 2
            tdLayer(
                nn.Conv2d(self.planes[0], self.planes[1], kernel_size=3, stride=1, padding=1, bias=False),
                tdBatchNorm2d(self.planes[1]),
            ),
            SpikingNeuron(),
            tdLayer(
                nn.Conv2d(self.planes[1], self.planes[1], kernel_size=3, stride=1, padding=1, bias=False),
                tdBatchNorm2d(self.planes[1]),
            ),
            SpikingNeuron(),
            tdLayer(nn.AvgPool2d(2, 2)),

            # blocks 3
            tdLayer(
                nn.Conv2d(self.planes[1], self.planes[2], kernel_size=3, stride=1, padding=1, bias=False),
                tdBatchNorm2d(self.planes[2]),
            ),
            SpikingNeuron(),
            tdLayer(
                nn.Conv2d(self.planes[2], self.planes[2], kernel_size=3, stride=1, padding=1, bias=False),
                tdBatchNorm2d(self.planes[2]),
            ),
            SpikingNeuron(),
            tdLayer(nn.AvgPool2d(2, 2)),

            # blocks 4
            tdLayer(
                nn.Conv2d(self.planes[2], self.planes[2], kernel_size=3, stride=1, padding=1, bias=False),
                tdBatchNorm2d(self.planes[2]),
            ),
            SpikingNeuron(),
            tdLayer(
                nn.Conv2d(self.planes[2], self.planes[2], kernel_size=3, stride=1, padding=1, bias=False),
                tdBatchNorm2d(self.planes[2]),
            ),
            SpikingNeuron(),
            tdLayer(nn.AvgPool2d(2, 2)),
        )

        ## classification layer
        self.classifier = nn.Sequential(
            tdLayer(nn.Linear(self.planes[-1] * int(48/2/2/2/2)**2, self.output_size, bias=True)),
        )

        ## initialize_weights
        self._initialize_weights()

    def forward(self, inputs):
        if self.T != 10:
            raise 'The timesteps for CIFAR10-DVS must be 10'
        x = self.encode(inputs.float())
        x = self.features(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.classifier(x)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels}, encode_channels={self.encode_channels}'


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)