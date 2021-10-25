import torch
import torch.nn as nn
from modules import (MaskedWts, 
                        SharedNet, 
                        SharedNetMultiLayered,
                        FinalHead,
                        getAllAdjWeights)
from losses import MainLoss



class TraditionalCASTLE(nn.Module):
    """
        Model as implemented in paper
        with (d+1) sub networks in input 
        and output layer.
    """
    def __init__(self, 
                    ntrain, 
                    inputdim, 
                    lambda_,
                    beta_,
                    nhidden=32,
                    noutputs=1,
                    nnodes=-1,
                    device='cuda:0',
                    weightThreshold=0.3):
        super(TraditionalCASTLE, self).__init__()

        self.ntrain = ntrain
        self.inputdim = inputdim
        self.lambda_ = lambda_
        self.beta_ = beta_
        self.weightThreshold = weightThreshold
        self.nhidden = nhidden
        self.noutputs = noutputs
        self.nnodes = self.inputdim if nnodes <= 0 else nnodes


        self.W0LayerSet = [MaskedWts(self.inputdim, self.nhidden, device).to(device) for _ in range(self.inputdim)]
        self.headSet = [FinalHead(self.nhidden, self.noutputs).to(device) for _ in range(self.inputdim)]

        self.shared = SharedNet(self.nhidden, nhidden)

        self.lossfn = MainLoss(self.lambda_, self.beta_, self.ntrain)

    def forward(self, x):
        out = []
        for i in range(self.inputdim):
            feature = self.W0LayerSet[i](x, i)
            feature = self.shared(feature)
            feature = self.headSet[i](feature)
            out.append(feature)

        return out[0], torch.cat(out, dim = 1)

    def loss(self, x, y, xrecon, pred):
        weights = getAllAdjWeights(self.W0LayerSet)
        # not sure about this step but, 
        # l116 in original implementation
        weight = torch.sqrt(torch.sum(torch.square(weights), dim=2))

        return self.lossfn(pred, y, x, xrecon, weight)
        

    def predict(self, x):
        return self.headSet[0](\
                        self.shared(\
                        self.W0LayerSet[0](x, 0)))



class CustomCASTLE(nn.Module):
    """
    Idea here is to share the first layer as well
    and mask weights dynamically and have a one
    single weight matrix in input layer
    """

    def __init__(self, 
                    ntrain, 
                    inputdim, 
                    lambda_,
                    beta_,
                    nhidden=32,
                    noutputs=1,
                    nnodes=-1,
                    device = 'cuda:0',
                    weightThreshold=0.3):
        super(CustomCASTLE, self).__init__()

        self.ntrain = ntrain
        self.inputdim = inputdim
        self.lambda_ = lambda_
        self.beta_ = beta_
        self.weightThreshold = weightThreshold
        self.nhidden = nhidden
        self.noutputs = noutputs
        self.nnodes = self.inputdim if nnodes <= 0 else nnodes

        self.W0Layer = MaskedWts(self.inputdim, self.inputdim, device).to(device)
        self.shared = SharedNet(self.inputdim, nhidden)

        self.headSet = [FinalHead(self.nhidden, self.noutputs).to(device) for _ in range(self.inputdim)]

        self.lossfn = MainLoss(self.lambda_, self.beta_, self.ntrain)


    def forward(self, x):
        out = []
        for i in range(self.inputdim):
            out.append(self.headSet[i](\
                            self.shared(\
                                self.W0Layer(x, i))))

        return out[0], torch.cat(out, dim = 1)

    def loss(self, x, y, xrecon, pred):
        weight = getAllAdjWeights([self.W0Layer])[0]
        return self.lossfn(pred, y, x, xrecon, weight)

    def predict(self, x):
        return self.headSet[0](\
                        self.shared(\
                        self.W0Layer(x, 0)))