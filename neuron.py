# -*- coding: utf-8 -*-
"""
    @Author       : 保持理智
    @Version      : v1.0
    @Date         : 2025-05-20 09:16:18
    @Description  : Spiking Neuron
"""
import math
import torch

from surrogate import Learnable_ActFun1, Learnable_ActFun2


class LIFNode(torch.nn.Module):
    def __init__(self, tau=0.2, Vth=0.5):
        super(LIFNode, self).__init__()
        self.tau = torch.tensor(tau)
        self.Vth = Vth
        self.act1 = Learnable_ActFun1.apply
        self.act2 = Learnable_ActFun2.apply

    def forward(self, x: torch.Tensor, param: dict=None):
        if param is not None:
            param['neuron_tau'], param['neuron_Vth'] = self.tau, self.Vth

        time_steps = x.size(1)
        mem = 0.0
        spike = torch.zeros(x.shape, device=x.device)
        for step in range(time_steps):
            mem = self.tau * mem + x[:, step, ...] ## computer membrane potentials
            if step == 0: ## firing spikes
                spike[:, step, ...] = self.act1(mem - self.Vth, param)
            else:
                spike[:, step, ...] = self.act2(mem - self.Vth, param)
            mem = mem * (1. - spike[:, step, ...]) ## hard reset
        return spike

    def extra_repr(self):
        return f'tau={self.tau}, Vth={self.Vth}'


class PLIFNode(torch.nn.Module):
    """ Implementation of PLIF in 'Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks (ICCV, 2021)' """
    def __init__(self, init_a=5.0, Vth=0.5):
        super(PLIFNode, self).__init__()
        init_w = - math.log(init_a - 1)
        self.w = torch.nn.Parameter(torch.tensor(init_w, dtype=torch.float))
        self.Vth = Vth
        self.act1 = Learnable_ActFun1.apply
        self.act2 = Learnable_ActFun2.apply

    def forward(self, x: torch.Tensor, param: dict=None):
        if param is not None:
            param['neuron_tau'], param['neuron_Vth'] = self.w.sigmoid().clone().detach(), self.Vth

        time_steps = x.size(1)
        mem = 0.0
        spike = torch.zeros(x.shape, device=x.device)
        for step in range(time_steps):
            mem = self.w.sigmoid() * mem + x[:, step, ...] ## computer membrane potentials
            if step == 0: ## firing spikes
                spike[:, step, ...] = self.act1(mem - self.Vth, param)
            else:
                spike[:, step, ...] = self.act2(mem - self.Vth, param)
            mem = mem * (1. - spike[:, step, ...]) ## hard reset
        return spike

    def extra_repr(self):
        return f'init_tau={self.w.sigmoid()}, Vth={self.Vth}'