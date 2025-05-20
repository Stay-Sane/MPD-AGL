# -*- coding: utf-8 -*-
"""
    @Author       : 保持理智
    @Version      : v1.0
    @Date         : 2025-5-20 11:17:44
    @Description  : Spiking Neuron
"""
import math
import torch

class Learnable_ActFun1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, param):
        ctx.save_for_backward(x)
        ctx.param = param
        # return (x > 0.).float()
        return x.gt(0.).float() ### todo

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.param is not None:
            width = 2.0 * ctx.param['tdBN_gammaAve'] * ctx.param['neuron_Vth'] #####
        else:
            width = 1.0
        s_x = 1.0 / width * (torch.abs(ctx.saved_tensors[0]) < (width / 2.0))
        grad_input = s_x * grad_output.clone()
        return grad_input, None


class Learnable_ActFun2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, param):
        ctx.save_for_backward(x)
        ctx.param = param
        # return (x > 0.).float()
        return x.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.param is not None:
            width = 2.0 * ctx.param['tdBN_gammaAve'] * ctx.param['neuron_Vth'] * torch.sqrt(1. + ctx.param['neuron_tau']**2) #####
        else:
            width = 1.0
        s_x = 1.0 / width * (torch.abs(ctx.saved_tensors[0]) < (width / 2.0))
        grad_input = s_x * grad_output.clone()
        return grad_input, None


class LIFNode(torch.nn.Module):
    def __init__(self, tau=0.2, Vth=0.5):
        super(LIFNode).__init__()
        self.tau = tau
        self.Vth = Vth
        self.act1 = Learnable_ActFun1.apply
        self.act2 = Learnable_ActFun2.apply

    def forward(self, x: torch.Tensor, param: dict=None):
        if isinstance(x, tuple) and len(x) == 2:
            x, param = x[0], x[1]
        if param is not None:
            param['neuron_tau'], param['neuron_Vth'] = self.tau, self.Vth

        time_steps = x.size(1)
        mem = 0.0
        spike = torch.zeros(x.shape, device=x.device)
        for step in range(time_steps):
            mem = self.tau * mem + x[:, step, ...]
            if step == 0:
                spike[:, step, ...] = self.act1(mem - self.Vth, param)
            else:
                spike[:, step, ...] = self.act2(mem - self.Vth, param)
            mem = mem * (1. - spike[:, step, ...])
        return spike

    def extra_repr(self):
        return f'tau={self.tau}, Vth={self.Vth}'


class PLIFNode(torch.nn.Module):
    """ Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks """
    def __init__(self, init_a=2.0, Vth=0.5):
        super(PLIFNode, self).__init__()
        init_w = - math.log(init_a - 1)
        self.w = torch.nn.Parameter(torch.tensor(init_w, dtype=torch.float))
        self.Vth = Vth
        self.act1 = Learnable_ActFun1.apply
        self.act2 = Learnable_ActFun2.apply

    def forward(self, x: torch.Tensor, param: dict=None):
        if isinstance(x, tuple) and len(x) == 2:
            x, param = x[0], x[1]
        if param is not None:
            param['neuron_tau'], param['neuron_Vth'] = self.w.sigmoid(), self.Vth

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