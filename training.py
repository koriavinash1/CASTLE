import torch
import random
import numpy as np
from models import TraditionalCASTLE, CustomCASTLE

from sklearn.model_selection import KFold  
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error
from sklearn.dataset import load_boston
import torch.optim as optim

def getData():
	scaler = StandardScaler()
	bostonData = load_boston()
	X = bostonData['data']
	y = bostonData['target']

	data = np.concatenate([y[..., None], X], axis=1)
	data = scaler.fit_transform(data)
	return data


data = getData()
kfSplits = KFold(n_splits = 3, random_state = 1)
fold = 0
REG_castle = []

w_threshold = 0.3
beta_ = 5
lambda_ = 1
batchsize = 32
nepochs = 100


for train_idx, val_idx in kf.split(data):
	fold += 1
	print("fold = ", fold)
    
    # data parameters
    X_train = data[train_idx]
    y_train = data[train_idx][:,0][..., None]

    X_val = data[val_idx]
    y_val = data[val_idx][:,0]

    # model and optimizer

    castle = TraditionalCASTLE(ntrain=X_train.shape[0], 
					inputdim = X_train.shape[1], 
					lambda_ = lambda_,
					beta_ = beta_,
					w_threshold = w_threshold)
    castle = castle.to(device)

    optimizer = optim.Adam (castle.parameters(), 
                            lr=0.001, 
                            betas=(0.9, 0.999), 
                            eps=1e-05, 
                            weight_decay=1e-5)

    for epoch in nepochs:
    	trainloss = []

    	castle.train()
    	for step in range(X_train.shape[0]//batchsize + 1):
    		idxs = random.sample(range(X_train.shape[0], self.batchsize))

    		Xbatch = X_train[idxs]
    		Ybatch = y_train[idxs]

    		# torchify
    		Xbatch = torch.tensor(Xbatch).to(device)
    		Ybatch = torch.tensor(Ybatch).to(device)

    		# forward pass
    		pred, Xrecon = castle(Xbatch)
    		loss = castle.loss(Xbatch, Ybatch, Xrecon, pred)

    		# for logs
    		trainloss.append(loss)
    		

    		# backward pass
    		optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        trainloss = torch.mean(trainloss).detach().cpu().numpy()

        # validation run
        valpreds = []

        castle.eval()
    	for step in range(X_valid.shape[0]//batchsize + 1):

    		Xbatch = X_valid[step*batchsize:(step + 1)*batchsize]

    		# torchify
    		Xbatch = torch.tensor(Xbatch).to(device)

    		# forward pass
    		preds, _ = castle(Xbatch)

    		# for logs
    		valpreds.extend(preds.detach().cpu().numpy())

    	valloss = mean_squared_error(valpreds, y_val)


    	print ("TrainLoss: {}, ValLoss: {}".format(trainloss, valloss))