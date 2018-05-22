import torch
import numpy as np
import torch.nn.functional as F


dtype = torch.float
device = torch.device("cpu")

def EDGE(X,Y,U=torch.tensor([20], dtype=dtype, requires_grad=False)):
    device = torch.device("cpu")
    dtype = torch.float
    r=torch.tensor([1], device=device, dtype=dtype, requires_grad=False)
    I = X.sum() + Y.sum() +U
    #ctx.save_for_backward(I)
    return I

if __name__ == "__main__":

    device = torch.device("cpu")
    dtype = torch.float
	# Independent Datasets
    X = torch.tensor(np.random.rand(3,3), device=device, dtype=dtype, requires_grad=True)
    Y = torch.tensor(np.random.rand(3,3), device=device, dtype=dtype, requires_grad=True)

    I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
    I.backward()



