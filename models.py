import torch
import torch.nn as nn
from modules import (MaskedWts, 
						SharedNet, 
						SharedNetMultiLayered,
						FinalHead,
						getAllAdjWeights)
from losses import MainLoss



class TraditionalCASTLE(nn.Module):
	def __init__(self, 
					ntrain, 
					inputdim, 
					lambda_,
					beta_,
					nhidden=32,
					noutputs=1,
					nnodes=-1,
					weightThreshold=0.3):
		super(CASTLE, self).__init__()

		self.ntrain = ntrain
		self.inputdim = inputdim
		self.lambda_ = lambda_
		self.beta_ = beta_
		self.weightThreshold = weightThreshold
		self.nhidden = nhidden
		self.noutputs = noutputs
		self.nnodes = self.inputdim if nnodes <= 0 else nnodes


		self.W0LayerSet = [MaskedWts(self.inputdim, self.nhidden) for _ in range(self.inputdim)]
		self.headSet = [FinalHead(self.nhidden, self.noutputs) for _ in range(self.inputdim)]

		self.shared = SharedNet(self.nhidden, nhidden)

		self.loss = MainLoss(self.lambda_, self.beta_, self.ntrain)

	def forward(self, x):
		out = []
		for i in range(self.inputdim):
			out.append(self.headSet[i](\
							self.shared(\
								self.W0LayerSet[i](x, i))))

		return out[0], torch.cat(out, dim = 1)

	def loss(self, x, y, xrecon, pred):
		weights = getAllAdjWeights(self.W0LayerSet)
		# not sure about this step but, 
		# l116 in original implementation
		weights = torch.sqrt(torch.sum(torch.square(weights), \
								dim=1, keepdim=True))

		weight = torch.cat(weights, dim=1)
		print (weight.shape)

		return self.loss(pred, y, x, xrecon, weight)
		

	def predict(self, x):
		return self.headSet[0](\
							self.shared(\
								self.W0LayerSet[0](x, 0)))



class CustomCASTLE(nn.Module):
	def __init__(self, 
					ntrain, 
					inputdim, 
					lambda_,
					beta_,
					nhidden=32,
					noutputs=1,
					nnodes=-1,
					weightThreshold=0.3):
		super(CASTLE, self).__init__()

		self.ntrain = ntrain
		self.inputdim = inputdim
		self.lambda_ = lambda_
		self.beta_ = beta_
		self.weightThreshold = weightThreshold
		self.nhidden = nhidden
		self.noutputs = noutputs
		self.nnodes = self.inputdim if nnodes <= 0 else nnodes

		self.W0Layer = MaskedWts(self.inputdim, self.nhidden)
		self.shared = SharedNet(self.nhidden, nhidden)

		self.headSet = [FinalHead(self.nhidden, self.noutputs) for _ in range(self.inputdim)]



	def forward(self, x):
		out = []
		for i in range(self.inputdim):
			out.append(self.headSet[i](\
							self.shared(\
								self.W0Layer(x, i))))

		return out[0], torch.cat(out, dim = 1)

	def loss(self, x, y):
		weight = getAllAdjWeights([self.W0Layer])[0]
		return self.loss(pred, y, x, xrecon, weight)

	def predict(self, x):
		return self.headSet[0](\
							self.shared(\
								self.W0Layer(x, 0)))