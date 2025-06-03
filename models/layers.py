# -*- coding: utf-8 -*-
"""
    @Author       : 保持理智
    @Version      : v1.0
    @Date         : 2025-05-20 09:16:18
    @Description  : Layers for Network Architecture
"""
import math
import torch
from torch import nn

from base_config import args
from neuron import LIFNode, PLIFNode


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x: torch.Tensor):
        x_shape = [x.shape[0], x.shape[1]]
        x_ = self.layer(x.flatten(0, 1).contiguous())
        x_shape.extend(x_.shape[1:])
        x_ = x_.view(x_shape)

        if self.bn is not None:
            x_, bn_param = self.bn(x_)
            return x_, bn_param
        else:
            return x_


class SpikingNeuron(nn.Module):
    def __init__(self, use_plif=args.use_plif):
        super().__init__()
        if use_plif:
            self.neuron = PLIFNode(args.init_a, args.Vth)
        else:
            self.neuron = LIFNode(args.tau, args.Vth)

    def forward(self, x: torch.Tensor, param: dict=None):
        if isinstance(x, tuple) and len(x) == 2:
            x, param = x[0], x[1]
        return self.neuron(x, param)


class tdBatchNorm2d(nn.BatchNorm2d):
    """ Implementation of tdBN in 'Going Deeper With Directly-Trained Larger Spiking Neural Networks (AAAI, 2021)' """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, alpha=1.0, Vth=0.5):
        super(tdBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.Vth = torch.tensor(Vth, dtype=torch.float)

    def forward(self, input: torch.Tensor):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 1, 3, 4]) # input: batch, T, channel, height, width
            var = input.var([0, 1, 3, 4], unbiased=False) # use biased var in train
            n = input.numel() / input.size(2)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        input = self.alpha * self.Vth * (input - mean[None, None, :, None, None]) / (torch.sqrt(var[None, None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, None, :, None, None] + self.bias[None, None, :, None, None]

        """ MPD-AGL: Theorem 1 """
        bn_param = {'tdBN_gammaAve': self.weight.mean([0]).clone().detach(), 'tdBN_betaAve': self.bias.mean([0]).clone().detach()}
        return input, bn_param

    def extra_repr(self):
        return f'num_features={self.num_features}, alpha={self.alpha}, threshold={self.Vth}'


class tdBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, alpha=1.0, Vth=0.5):
        super(tdBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.Vth = torch.tensor(Vth, dtype=torch.float)

    def forward(self, input: torch.Tensor):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 1]) # input: batch, T, channel
            var = input.var([0, 1], unbiased=False) # use biased var in train
            n = input.numel() / input.size(2)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var
        input = self.alpha * self.Vth * (input - mean[None, None, :]) / (torch.sqrt(var[None, None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, None, :] + self.bias[None, None, :]

        """ MPD-AGL: Theorem 1 """
        bn_param = {'tdBN_gammaAve': self.weight.mean([0]).clone().detach(), 'tdBN_betaAve': self.bias.mean([0]).clone().detach()}
        return input, bn_param

    def extra_repr(self):
        return f'num_features={self.num_features}, alpha={self.alpha}, threshold={self.Vth}'