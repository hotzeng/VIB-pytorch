import torch
import numpy as np
import torch.nn.functional as F


dtype = torch.float
device = torch.device("cpu")

#class EDGE(torch.nn.Module):
#    def __init__(self):
#        super(EDGE, self).__init__()
#
#
#    def forward(self, X,Y,U=torch.tensor([20], dtype=dtype, requires_grad=False)):
#        device = torch.device("cpu")
#        dtype = torch.float
#        #r=torch.tensor([1], device=device, dtype=dtype, requires_grad=False)
#        I1 = torch.normal(torch.tensor([0.0]), torch.tensor([1.0]))
#        I2 = torch.normal(torch.tensor([0.0]), torch.tensor([1.0]))
#        I1 = I1*Y + X 
#        I2 = I2*Y + X
#        #I = I*Y + X
#        #I = X.sum() + Y.sum() +U
#        #ctx.save_for_backward(I)
#        return (I1, I2)

if __name__ == "__main__":

    np.random.seed(1)
    device = torch.device("cpu")
    dtype = torch.float
	# Independent Datasets
    X = torch.tensor(np.random.rand(1), device=device, dtype=dtype, requires_grad=True)
    Y = torch.tensor(np.random.rand(1), device=device, dtype=dtype, requires_grad=True)

    for e in range(10):
        print("epoch: ", e)
        #edge = EDGE()
        ####################
        i1 = torch.normal(torch.tensor([0.0]), torch.tensor([1.0]))
        i2 = torch.normal(torch.tensor([0.0]), torch.tensor([1.0]))
        X.requires_grad_(True)
        Y.requires_grad_(True)
        I1 = i1*Y + X 
        I2 = i2*Y + X


        ####################
        #I1, I2 = edge(X,Y) # Estimated Mutual Information between X and Y using EDGE method
        I = torch.cat((I1, I2))
        mean = torch.mean(I)
        std = torch.std(I)
        mean_loss = torch.abs(mean - torch.tensor([1.0]))
        std_loss = torch.abs(std - torch.tensor([0.5]))
        loss = mean_loss + std_loss
        loss.backward()
        #X.grad.zero_()
        #Y.grad.zero_()

        print("X.grad:", X.grad)
        print("Y.grad:", Y.grad)
        #with torch.no_grad():
        X = X - 0.001*X.grad
        Y = Y - 0.001*Y.grad
        print("X:", X)
        print("Y:", Y)





