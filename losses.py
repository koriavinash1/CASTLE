import torch
import torch.nn as nn
from modules import TraceExpM, truncatedTraceExpM
import numpy as np


def reconstructionLoss(X, Xpred, N):
	residual = X - Xpred
	return 1./N * torch.norm(residual)**2


def RwFunction(AdjMatrix):
	d = AdjMatrix.shape[0]
	AdjMatrix = AdjMatrix.view(d, -1, d)  # [j, m1, i]
	AdjMatrix = torch.sum(AdjMatrix * AdjMatrix, dim=1).t()  # [i, j]
	# return (TraceExpM(AdjMatrix) - d)**2
	return (truncatedTraceExpM(AdjMatrix) - d)**2

def L1(AdjMatrix):
	return torch.norm(AdjMatrix, 1)


def RDAGLoss(X, Xpred, AdjMatrix, beta, N):
	Lrecon = reconstructionLoss(X, Xpred, N)
	Rw = RwFunction(AdjMatrix)
	l1 = L1(AdjMatrix)
	loss = Lrecon + Rw + beta*l1
	# print ("Lrecon: {}, Trace: {}, L1: {}".format(Lrecon, Rw, l1))
	return loss


class MainLoss(nn.Module):
	def __init__(self, lambda_, beta_, N):
		super(MainLoss, self).__init__()
		self.lambda_ = lambda_
		self.beta_ = beta_
		self.N = N

		self.supervised = nn.MSELoss()

	def forward(self, pred, target, X, Xrecon, AdjMatrix):
		dagloss = RDAGLoss(X, 
							Xrecon, 
							AdjMatrix, 
							self.beta_, 
							self.N)
		loss = self.supervised(pred, target)
		# print ("Supervised: {}".format(loss))

		return loss + self.lambda_*dagloss
