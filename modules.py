import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg as slin


class TraceExpM(torch.autograd.Function):
    # https://github.com/xunzheng/notears/blob/ba61337bd0e5410c04cc708be57affc191a8c424/notears/trace_expm.py#L6
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        try:
            E = slin.expm(input.detach().numpy())
        except:
            E = slin.expm(input.detach().cpu().numpy())

        f = np.trace(E)
        E = torch.from_numpy(E).to(input.device)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors

        grad_input = grad_output * E.t()
        return grad_input


TraceExpM = TraceExpM.apply



def truncatedTraceExpM(Z, truncationOrder=2):
    d = Z.shape[0]
    dagL = d*1.0
    coff = 1.0

    Zin = torch.eye(d).to(Z.device)
    for i in range(truncationOrder):
        Zin = torch.matmul(Zin, Z)
        dagL += 1./coff * torch.trace(Zin)
        coff = coff*(i+1)

    return dagL - (d*1.0)



def getAllAdjWeights(layers):
    W = []
    for layer in layers:
        W.append(layer.weight.unsqueeze(0))
    return torch.cat(W, dim=0)



def getMask(idx, ninputs, nhidden):
    mask = torch.ones(ninputs, nhidden)
    mask[idx, :] = 0
    return mask


def weights_init_normal(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
    # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
    # m.bias.data should be 0
        m.bias.data.fill_(0)


class MaskedWts(nn.Module):
    def __init__(self, indims, outdims, device):
        super(MaskedWts, self).__init__()
        self.indims = indims
        self.outdims = outdims
        self.device = device
        self.weight = nn.Parameter(torch.Tensor(indims, outdims)) # define the trainable parameter
        self.bias = nn.Parameter(torch.Tensor(outdims))

        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bias, 0)
        pass

    def forward(self, x, k):
        # in original implementation weights are overwritten 
        # maybe thats why indims set of weights were required

        self.mask = getMask(k, self.indims, self.outdims).to(self.device)

        feature = torch.matmul(x, self.weight*self.mask) + self.bias
        feature = F.relu(feature)
        return feature


class MaskedInput(nn.Module):
    def __init__(self, indims, batchsize):
        super(MaskedWts, self).__init__()
        self.indims = batchsize
        self.outdims = indims
        self.apply(weights_init_normal)
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
        self.apply(weights_init_normal)
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
        self.apply(weights_init_normal)
        pass

    def forward(self, x):
        return self.net(x)


class FinalHead(nn.Module):
    def __init__(self, infeatures, noutputs=1):
        super(FinalHead, self).__init__()

        layers = []
        layers.append(nn.Linear(infeatures, noutputs))

        self.net = nn.Sequential(*layers)
        self.apply(weights_init_normal)
        pass

    def forward(self, x):
        return self.net(x)