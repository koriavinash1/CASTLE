import torch
import torch.nn as nn
import torch.nn.functional as F
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



def getAllAdjWeights(layers):
    W = []
    for layer in layers:
        W.append(layer.weight)
    return torch.cat(W, dim = 0)

    

def getMask(idx, ninputs, nhidden):
    mask = torch.ones(ninputs, nhidden)
    mask[idx, :] = 0
    return mask



class MaskedWts(nn.Module):
    def __init__(self, indims, outdims):
        super(MaskedWts, self).__init__()
        self.indims = indims
        self.outdims = outdims

        self.weight = nn.Parameter(torch.Tensor(indims, outdims))  # define the trainable parameter
        self.bias = nn.Parameter(torch.Tensor(outdims))
        pass

    def forward(self, x, k):
        # in original implementation weights are overwritten 
        # maybe thats why indims set of weights were required

        self.mask = getMask(k, self.indims, self.outdims)
        feature = F.matmul(x, self.weight*self.mask) + self.bias
        feature = F.relu(feature)
        return feature


class MaskedInput(nn.Module):
    def __init__(self, indims, batchsize):
        super(MaskedWts, self).__init__()
        self.indims = batchsize
        self.outdims = indims
        pass

    def forward(self, x, k):
        self.mask = getMask(k, self.indims, self.outdims)
        return x*self.mask


class SharedNet(nn.Module):
    def __init__(self, infeatures, outfeatures):
        super(SharedNet, self).__init__()

        layers = []
        layers.append(nn.Linear(infeatures, outfeatures))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        pass

    def forward(self, x):
        return self.net(x)


class SharedNetMultiLayered(nn.Module):
    def __init__(self, infeatures, outfeatures):
        super(SharedNet, self).__init__()

        layers = []
        layers.append(nn.Linear(infeatures, 2*infeatures))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(2*infeatures, outfeatures))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        pass

    def forward(self, x):
        return self.net(x)


class RegHead(nn.Module):
    def __init__(self, infeatures, noutputs=1):
        super(RegHead, self).__init__()

        layers = []
        layers.append(nn.Linear(infeatures, noutputs))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        pass

    def forward(self, x):
        return self.net(x)


