# -*- coding: utf-8 -*-
"""
    @Author       : 保持理智
    @Version      : v1.0
    @Date         : 2025-05-20 09:16:18
    @Description  : Surrogate Gradient
"""
import math
import torch

class Learnable_ActFun1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, param):
        ctx.save_for_backward(x)
        ctx.param = param
        return (x >= 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.param is not None:
            """ MPD-AGL: Theorem 2 """
            width = 2.0 * ctx.param['tdBN_gammaAve'] * ctx.param['neuron_Vth']
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
        return (x >= 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.param is not None:
            """ MPD-AGL: Theorem 2 """
            width = 2.0 * torch.sqrt(1. + ctx.param['neuron_tau']**2) * ctx.param['tdBN_gammaAve'] * ctx.param['neuron_Vth']
        else:
            width = 1.0
        s_x = 1.0 / width * (torch.abs(ctx.saved_tensors[0]) < (width / 2.0))
        grad_input = s_x * grad_output.clone()
        return grad_input, None