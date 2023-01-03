import torch

def mnist():
    Xtrain = torch.load("/data/processed/Xtrain.pt")
    Ytrain = torch.load("/data/processed/Ytrain.pt")
    Xtest = torch.load("/data/processed/Xtest.pt")
    Ytest = torch.load("/data/processed/Ytest.pt")

    trainset = torch.utils.data.TensorDataset(Xtrain,Ytrain)
    testset = torch.utils.data.TensorDataset(Xtest,Ytest)

    return trainset, testset