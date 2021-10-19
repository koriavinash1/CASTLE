import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as slin


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


trace_expm = TraceExpm.apply



class MaskedWts(nn.Module):
    def __init__(self, k, indims, outdims):
        super(MaskedWts, self).__init__()
        self.mask = 1
        pass

    def forward(self, x):
        return x*self.mask


class SharedNet(nn.Module):
    def __init__(self, infeatures, outfeatures):
        super(SharedNet, self).__init__()

        self.net = nn.ModuleList([])
        pass

    def forward(self, x):
        return self.net(x)


class RegHead(nn.Module):
    def __init__(self, infeatures, noutputs):
        super(RegHead, self).__init__()

        pass

    def forward(self, x):
        pass


