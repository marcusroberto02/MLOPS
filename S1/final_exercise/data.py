import torch
import numpy as np

def mnist():
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784) 
    root = "C:/Users/marcu/Google Drev/DTU/02476_MLOPS/dtu_mlops/data/corruptmnist"
    Xtrain = np.load(root + "/train_0.npz")["images"]
    Ytrain = np.load(root + "/train_0.npz")["labels"]
    for i in range(1,5):
        Xtrain = np.concatenate((Xtrain,np.load(root + "/train_{}.npz".format(i))["images"]),axis=0)
        Ytrain = np.concatenate((Ytrain,np.load(root + "/train_{}.npz".format(i))["labels"]),axis=0)

    Xtest = np.load(root + "/test.npz")["images"]
    Ytest = np.load(root + "/test.npz")["labels"]

    trainset = torch.utils.data.TensorDataset(torch.from_numpy(Xtrain).float(),torch.from_numpy(Ytrain).long())
    testset = torch.utils.data.TensorDataset(torch.from_numpy(Xtest).float(),torch.from_numpy(Ytest).long())

    return trainset, testset
