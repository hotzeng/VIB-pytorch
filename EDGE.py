# EDGE Estimator for Shannon Mutual Information
# I = EDGE(X,Y): estimate of mutual information between X snd Y
# X is N * d_x and Y is N * d_Y data sets
# U is optional upper bound for the pointwise mutual information: I = EDGE(X,Y, U)

# Version: Automatic epsilon for all dimensions, Ensemble Estimation with Optimized wights, 
#          Repeat Estimation for different random epsilons

import numpy as np
import math
import pdb
import cvxpy as cvx # Need to install CVXPY package 
import torch
import random



# Return the normalizing factor of epsilon regarding STD and some randomness
# Used in: EDGE_single_run
def gen_eps(X,Y):
    
    X = X.data.numpy()
    Y = Y.data.numpy()
	# Parameter: range of random epsilon coefficient:
    eps_l, eps_u = 0.7, 1.8
	
    dim_X , dim_Y  = X.shape[1], Y.shape[1]

    std_X = np.array([np.std(X[:,[i]]) for i in range(dim_X)])
    std_Y = np.array([np.std(Y[:,[i]]) for i in range(dim_Y)])

	# random coeffs 
    cf_X = np.random.rand(1,dim_X)*(eps_u - eps_l) + eps_l
    cf_Y = np.random.rand(1,dim_Y)*(eps_u - eps_l) + eps_l

	# Random Shifts
    b_X = 10.0*np.random.rand(1,dim_X)*std_X
    b_Y = 10.0*np.random.rand(1,dim_Y)*std_Y

	# Random espilons
    eps_X = std_X * cf_X
    eps_Y = std_Y * cf_Y

    return (eps_X,eps_Y,b_X,b_Y)
	

# Compute the number of collisions in each bucket
# Used in: find_interval, Compute_MI 
class input_2_collision(torch.autograd.Function):
   # def __init__(self, X,Y,t_m,eps_X,eps_Y,b_X,b_Y,Ni_max):
   #     super(compute_collision, self).__init__()
   #     X = X
   #     Y = Y
   #     t_m = t_m
   #     eps_X = eps_X
   #     eps_Y = eps_Y
   #     b_X = b_X
   #     b_Y = b_Y
   #     Ni_max = Ni_max

    @staticmethod
    def forward(ctx, X,Y,t_m,eps_X,eps_Y,b_X,b_Y,Ni_max, Ni_min):
        #print("t_m:", t_m)
        CX_X, CY_Y, CXY_XY = {}, {}, {}
        alpha = 1 # this 1 should be an args parameter

    	# Num of Samples
        #print(X)
        shape_X = list(X.size())
        shape_Y = list(Y.size())
        N = shape_X[0]
    
    	# Find dimensions
        dim_X , dim_Y  = shape_X[1], shape_Y[1]
        dim = dim_X + dim_Y
    
    	# normalize epsilons
        eps_X = eps_X * 1.0*t_m.double() / pow(N,1.0/(2*dim))
        eps_Y = eps_Y * 1.0*t_m.double() / pow(N,1.0/(2*dim))
    
    	# Compute X_tilde and Y_tilde, the mappings of X and Y using H_1
        #print("X:", X)
        #print("b_X:", b_X)
        #print("eps_X:", eps_X)
        X_t, Y_t = 1.0*X.add(b_X.float()).div(eps_X.float()), 1.0*Y.add(b_Y.float()).div(eps_Y.float())
        #print(X_t)
        X_t, Y_t= X_t.gt(0).float() * X_t.floor() + X_t.lt(0).float() * X_t.ceil(), Y_t.gt(0).float() * Y_t.floor() + Y_t.lt(0).float() * Y_t.ceil()  	
        #print(X_t.shape)
        #print(X_t)

    	# Hash vectors as dictionaries
        CX, CY, CXY = {}, {}, {} 
        # question: why all the X_t and Y_t are 0???:? 

    	# Computing Collisions
        for i in range(N):
    
    		# Convert to list
            X_l, Y_l = tuple(X_t[i].tolist()), tuple(Y_t[i].tolist())
    		
    		# X collisionsZZ  
            if X_l in CX:
                CX[X_l] += 1
                #ctx.CX_X[X_l].append(i)
            else: 
                CX[X_l] = 1 ## question: why here the value is kept 1?
                #ctx.CX_X[X_l] = [i]
   
    		# Y collisions
            if Y_l in CY:
                CY[Y_l] += 1
                #ctx.CY_Y[Y_l].append(i)                
            else: 
                CY[Y_l] = 1
                #ctx.CY_Y[Y_l] = [i]    

    		# XY collisions
            if (X_l,Y_l) in CXY:
                CXY[(X_l,Y_l)] += 1
                #ctx.CXY_XY[(X_l,Y_l)].append(i)
            else: 
                CXY[(X_l,Y_l)] = 1
                #ctx.CXY_XY[(X_l,Y_l)] = [i]

        #CX_keys = list(CX.keys)
        #CY_keys = list(CY.keys)
        #CXY_keys = list(CXY.keys)
        #X_keys_num = len(CX_keys)
        #Y_keys_num = len(list(CY_keys))
        #XY_keys_num = len(list(CXY_keys))
        #for i in range(N):
        #    for j in range(X_keys_num):
        #        if 




        # convert the three dictionaries to tensors
        #CX_keys, CX_num, CY_keys, CY_num, CXY_X_keys, CXY_Y_keys, CXY_num = [], [], [], [], [], [], []
        #for X_l in CX_X.keys():
        #    CX_keys.append(X_l)
        #    CX_num.append(ctx.CX_X[X_l])

        #for Y_l in CY_Y.keys():
        #    CY_keys.append(Y_l)
        #    CY_num.append(ctx.CY_Y[Y_l])

        #for XY in CXY_XY.keys():
        #    CXY_X_keys.append(XY[0])
        #    CXY_Y_keys.append(XY[1])
        #    CXY_num.append(ctx.CXY_XY[XY])

       # CX_keys = torch.tensor(CX_keys, device=device, dtype=dtype, requires_grad=False)
       # CY_keys = torch.tensor(CY_keys, device=device, dtype=dtype, requires_grad=False)
       # #CXY_keys = torch.tensor(CXY_keys, device=device, dtype=dtype, requires_grad=False)
       # CX_num = torch.tensor(CX_num, device=device, dtype=dtype, requires_grad=False)
       # CY_num = torch.tensor(CY_num, device=device, dtype=dtype, requires_grad=False)
       # CXY_num = torch.tensor(CXY_num, device=device, dtype=dtype, requires_grad=False)


        f = torch.tensor( max(CXY.values())/(Ni_max*pow(N,2.0/4)), requires_grad=False)
    
        mini = torch.tensor(Ni_min*pow(N,2.0/4), requires_grad=False)
       
    	# Computing Mutual Information Function
        Ni = []
        Mj = []
        Nij = []
        X_keys = []
        Y_keys = []
    
        # change the following lines to eliminate the dictionary
        for e in CXY.keys():
    		#Ni, Mj, Nij = CX[e[0]], CY[e[1]], CXY[e]
            X_keys.append(e[0])
            Y_keys.append(e[1])
            Ni.append(CX[e[0]])
            Mj.append(CY[e[1]])
            Nij.append(CXY[e])

        #print(X_keys)
        #print(Y_keys)
        Ni = torch.tensor(Ni, requires_grad=True)
        Mj = torch.tensor(Mj, requires_grad=True)
        Nij = torch.tensor(Nij, requires_grad=True)
        X_keys=tuple(X_keys)
        Y_keys=tuple(Y_keys)
        #print(keys)
        X_keys = torch.tensor(X_keys, requires_grad=False)
        Y_keys = torch.tensor(Y_keys, requires_grad=False)
        #print(keys)

        X_t = torch.tensor(X_t, device=device, dtype=dtype, requires_grad=False)
        Y_t = torch.tensor(Y_t, device=device, dtype=dtype, requires_grad=False)
        ctx.save_for_backward(X, Y, t_m, eps_X, eps_Y, b_X, b_Y, Ni_max, Ni_min, X_keys, Y_keys, Ni, Mj, Nij, X_t, Y_t)

        return Ni, Mj, Nij, mini, X_keys, Y_keys, f


    # the grad_CX, grad_CY and grad_CXY should all be dictionaries
    @staticmethod
    def backward(ctx, grad_Ni, grad_Mj, grad_Nij, grad_mini, grad_X_keys, grad_Y_keys, grad_f):

        X,Y,t_m,eps_X,eps_Y,b_X,b_Y,Ni_max, Ni_min, X_keys, Y_keys, Ni, Mj, Nij, X_t, Y_t = ctx.saved_tensors
        #print(X_keys)
        grad_CX, grad_CY, grad_CXY = {}, {}, {}
        CX, CY, CXY = {}, {}, {}
        CX_X, CY_Y, CXY_XY = {}, {}, {}        

        i = 0
        # Attention: x_key and y_key have repeated elements!
        for x_key, y_key in zip(X_keys, Y_keys):
            x_key = tuple(x_key.data.numpy())
            y_key = tuple(y_key.data.numpy())
            grad_CX[x_key] = 0
            grad_CY[y_key] = 0
            grad_CXY[(x_key, y_key)] = grad_Nij[i]
            CX[x_key] = Ni[i]
            CY[y_key] = Mj[i]
            CXY[(x_key, y_key)] = Nij[i]
            i = i + 1

        i = 0
        #print("X_keys:", X_keys)
        #print("grad_CX_keys:", grad_CX.keys())
        #print("grad_CX:", grad_CX)
        #print(Y_keys)
        for x_key, y_key in zip(X_keys, Y_keys):
            x_key = tuple(x_key.data.numpy())
            y_key = tuple(y_key.data.numpy())
            #print("x_key:", x_key)
            #print("y_key:", y_key)
            #print("grad_CX_keys:", grad_CX.keys())
            #print(grad_CX[x_key])
            grad_CX[x_key] += grad_Ni[i]
            grad_CY[y_key] += grad_Mj[i]
            i = i + 1

        #print(grad_CX)
        N = X_t.shape[0]
        for i in range(N):
    
    		# Convert to list
            X_l, Y_l = tuple(X_t[i].tolist()), tuple(Y_t[i].tolist())
    		
    		# X collisionsZZ  
            if X_l in CX_X:
                CX_X[X_l].append(i)
            else: 
                CX_X[X_l] = [i]
   
    		# Y collisions
            if Y_l in CY_Y:
                CY_Y[Y_l].append(i)                
            else: 
                CY_Y[Y_l] = [i]    

    		# XY collisions
            if (X_l,Y_l) in CXY_XY:
                CXY_XY[(X_l,Y_l)].append(i)
            else: 
                CXY_XY[(X_l,Y_l)] = [i]



        X_D1, X_D2 = X.shape
        Y_D1, Y_D2 = Y.shape
  
        dtype = torch.float
        device = torch.device("cpu")
        # Get the keys
        # optimize further
        CX_keys = tuple(grad_CX.keys())
        #print(CX_keys)
        CY_keys = tuple(grad_CY.keys())
        CXY_keys = tuple(grad_CXY.keys())
        # Initialize the gradients for the keys
        grad_X_l = {}
        grad_Y_l = {}
        grad_XY_l = {}
        # Initialize the gradients for the inputs, X and Y
        grad_X = torch.zeros(X.shape, dtype = dtype, device = device)
        grad_Y = torch.zeros(Y.shape, dtype = dtype, device = device)
        # Calculate the gradients w.r.t X_l, Y_l, XY_l
        for X_l in CX_keys:
            grad_X_l[X_l] = grad_CX[X_l] / CX[X_l]

        for Y_l in CY_keys:
            grad_Y_l[Y_l] = grad_CY[Y_l] / CY[Y_l]

        for XY_l in CXY_keys:
            grad_X_l[XY_l[0]] = 0.5 * grad_CXY[XY_l] / CXY[XY_l]
            grad_Y_l[XY_l[1]] = 0.5 * grad_CXY[XY_l] / CXY[XY_l]
            
        for X_l in CX_keys:
            #print(X_l)
            N_X_i = len(CX_X[X_l])
            X_D1 -= N_X_i
            for i in CX_X[X_l]:
                grad_X[i] = torch.ones(X_D2).float() * (grad_X_l[X_l].float() / (N_X_i * eps_X.float())).float()

        for Y_l in CY_keys:
            N_Y_i = len(CY_Y[Y_l])
            Y_D1 -= N_Y_i            
            for i in CY_Y[Y_l]:
                grad_Y[i] = torch.ones(Y_D2).float() * (grad_Y_l[Y_l].float() / (N_Y_i * eps_Y.float())).float()

        assert X_D1 == 0, "Dimension of grad_X have some problem"
        assert Y_D1 == 0, "Dimension of grad_Y have some problem"
        zero_grad = torch.tensor([0], device=device, dtype=dtype,requires_grad=False)
        return (grad_X, grad_Y, zero_grad, zero_grad, zero_grad, zero_grad, zero_grad, zero_grad, zero_grad)


# Used in: EDGE_single_run
def find_interval(X,Y, eps_X,eps_Y,b_X,b_Y,t_l, t_u, Ni_max = 1):

    #X = X.data.numpy()
    #Y = Y.data.numpy()

	# Num of Samples
    N = X.shape[0]

	# parameter: C_balance: Minimum ratio of number of distinct hashing (L_XY) with respect to max collision  
    C_balance_l, C_balance_u = 0.7 , 1.5

	# Find the appropriate interval
    f_l, f_u = 0, 3.0
    err = 0
    Ni_min = torch.tensor([0], requires_grad=False)
    while  (f_l < C_balance_l)  or ( C_balance_u < f_u): 
        # If cannot find the right interval make error
        err +=1
        if err > 200:
            raise ValueError('Error: Correct interval cannot be found. Try modifying t_l and t_u', t_l,t_u)  
    
        t_m = (t_u+t_l)/2
        
        t_m = torch.tensor(t_m, requires_grad=False)
        eps_X = torch.tensor(eps_X, requires_grad=False)
        eps_Y = torch.tensor(eps_Y, requires_grad=False)
        b_X = torch.tensor(b_X, requires_grad=False)
        b_Y = torch.tensor(b_Y, requires_grad=False)
        Ni_max = torch.tensor(Ni_max, requires_grad=False)
        (Ni, Mj, Nij, mini, X_keys, Y_keys, f_m) = input_2_collision.apply(X,Y,t_m,eps_X,eps_Y,b_X,b_Y,Ni_max,Ni_min )
 
        not_in_interval = (f_l < C_balance_l) or (C_balance_u < f_u) 
        if f_m < 1 and not_in_interval:
            t_l=t_m
            f_l=f_m
        elif f_m > 1 and not_in_interval:
            t_u=t_m
            f_u=f_m

    return (t_l,t_u)


# Compute mutual information function given epsilons and radom shifts
# Used in: EDGE_single_run

#class hash_2_array(torch.autograd.Function):
#    # the input parameters are all from the compute_collision function
#    def forward(ctx, f_m, CX, CY, CXY, N, Ni_max, Ni_min):    
#    	# Parameter: Lower bound for Ni and Mj for being counted:
#        N = N
#        Ni_min = Ni_min
#        Ni_max = Ni_max
#        mini = Ni_min * pow(N,2.0/4)
#       
#    	# Computing Mutual Information Function
#        I = 0
#        N_c = 0
#        Ni = []
#        Mj = []
#        Nij = []
#        keys = []
#    
#        # change the following lines to eliminate the dictionary
#        for e in CXY.keys():
#    		#Ni, Mj, Nij = CX[e[0]], CY[e[1]], CXY[e]
#            keys.append(e)
#            Ni.append(CX[e[0]])
#            Mj.append(CY[e[1]])
#            Nij.append(CXY[e])
#
#        ctx.save_for_backward(keys)
#        return Ni, Mj, Nij, mini
#
#    def backward(self, grad_Ni, grad_Mj, grad_Nij): # return three hashes of the gradients w.r.t CX, CY, CXY
#        keys = ctx.saved_tensors
#        grad_CX, grad_CY, grad_CXY = {}, {}, {}
#        i = 0
#        for e in keys:
#            grad_CX[e] = grad_Ni[i]
#            grad_CY[e] = grad_Mj[i]
#            grad_CXY[e] = grad_Nij[i]
#            i = i + 1
#
#        return grad_CX, grad_CY, grad_CXY 
        
# Attention! Ni and Mj have repeated elements!
def collision_2_MI(Ni, Mj, Nij, N, mini, U=20):
    # All the inputs are tensors
    I = ((Nij.float()*N/(Ni.float()*Mj.float())).clamp(max=U).clamp(min=1.0/U).float().log2() * (Nij.float().gt(mini).float()*Nij.float())).sum()
    N_c = Nij.sum()

    I = I.div(N_c.float())
    
    return I.view(1) 

def Compute_MI(X,Y,U,t,eps_X,eps_Y,b_X,b_Y,Ni_max,Ni_min):
    N = X.shape[0]
    Ni, Mj, Nij, mini, X_keys, Y_keys, f_m = input_2_collision.apply(X,Y,t,eps_X,eps_Y,b_X,b_Y,Ni_max,Ni_min)
    I = collision_2_MI(Ni, Mj, Nij, N, mini)
    return I


# A single run of Mutual Information estimation using EDGE
def EDGE_single_run(X, Y, U=20):
	# parameter: Ni_min * sqrt(N) < N_i, M_i < Ni_max * sqrt(N)
    #print(X)
    device = torch.device("cpu")
    dtype = torch.float

    Ni_min = 0.2
    Ni_max = 1.0 
    
    # Find dimensions
    dim_X , dim_Y  = X.shape[1], Y.shape[1]
    dim = dim_X + dim_Y
    
    # Number of terms in ensemble method
    L = dim+1
    
    # Num of Samples
    N = X.shape[0]
    t_l,t_u = 0.1, 40
    
    # Use less number of samples for learning the interval
    N_t=1000
    if N > N_t:
        X_test,Y_test=X[:N_t],Y[:N_t]
    else:
        X_test,Y_test=X,Y
    
    # get eps and b:
    (eps_X,eps_Y,b_X,b_Y) = gen_eps(X_test,Y_test)
    
    # Find the appropriate Interval
    (t_l,t_u) = find_interval(X, Y, eps_X, eps_Y, b_X, b_Y, t_l, t_u, Ni_max)
    
    # Find a range of indices
    l = t_u - t_l
    c = 1.0*l / L
    
    T_U = np.linspace(t_l, t_u, L)
    
    if L**(1.0/(2*dim)) < 1.0*t_u/t_l:
        T = t_l* torch.tensor(np.array(range(1,L+1))**(1.0/(2*dim))).float()
    else:
        T = torch.tensor(np.linspace(t_l, t_u, L), requires_grad=False)
    
    # Vector of weights
    W = compute_weights(L, dim, T, N)
    
    T = torch.tensor(T, requires_grad=False)
    # optimize further, maybe these arrays do not need to be converted to tensors
    eps_X = torch.tensor(eps_X, requires_grad=False)
    eps_Y = torch.tensor(eps_Y, requires_grad=False)
    b_X = torch.tensor(b_X, requires_grad=False)
    b_Y = torch.tensor(b_Y, requires_grad=False)
    Ni_max = torch.tensor(Ni_max, requires_grad=False)
    Ni_min = torch.tensor(Ni_min, requires_grad=False)
    U = torch.tensor([20], requires_grad=False)

    #I_list = [Compute_MI.apply(X,Y,U,T[i],eps_X,eps_Y,b_X,b_Y,Ni_max,Ni_min) for i in range(L)]
    #MV = torch.tensor(I_list , device=device, dtype=dtype, requires_grad=True )
    MV = Compute_MI(X,Y,U,T[0],eps_X,eps_Y,b_X,b_Y,Ni_max,Ni_min)
    for i in range(1,L):
        MV = torch.cat((MV,Compute_MI(X,Y,U,T[i],eps_X,eps_Y,b_X,b_Y,Ni_max,Ni_min)))

	# Ensemble MI
    W = np.squeeze(W)
    WT = torch.tensor(W.T, device=torch.device("cpu"), dtype=torch.float, requires_grad=False)
    I = MV.dot(WT)
	
    return I
   


##### Linear Program for Ensemble Estimation ####
# Used in: EDGE_single_run
def compute_weights(L, d, T, N):
	
	# Correct T
    T = torch.Tensor.numpy(T)
    T = 1.0*T/T[0]

	# Create optimization variables.
    cvx_eps = cvx.Variable()
    cvx_w = cvx.Variable(L)

	# Create constraints:
    constraints = [cvx.sum_entries(cvx_w)==1, cvx.pnorm(cvx_w, 2)- cvx_eps/2 < 0 ]
    for i in range(1,d+1):
        Tp = ((1.0*T/N)**(1.0*i/(2*d)))
        cvx_mult = cvx_w.T * Tp
        constraints.append(cvx.sum_entries(cvx_mult) - cvx_eps*2 < 0)
	
	# Form objective.
    obj = cvx.Minimize(cvx_eps)

	# Form and solve problem.
    prob = cvx.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.

    sol = np.array(cvx_w.value)

    return sol.T

#########
# EDGE Estimator of Mutual Information
def EDGE(X,Y,U=torch.tensor([20], requires_grad=False)):
   # r = 5	# Repeat for LSH with different random parameters and Take mean: By increasing this parameter you get more accurate estimate
   # EDGE_single = []
   # #print(X)
   # for i in range(r):
   #     EDGE_single.append(EDGE_single_run())

   # I_SUM = EDGE_single[0].apply(X,Y,U)
   # for i in range(r):
   #     I_SUM.add_(EDGE_single[i].apply(X,Y,U))

   # I_SUM.sub_(EDGE_single[0].apply(X,Y,U))

    ## the following is just used for debug
    device = torch.device("cpu")
    dtype = torch.float
    #r=torch.tensor([1], device=device, dtype=dtype, requires_grad=False)
    I_SUM = EDGE_single_run(X,Y,U)
    #I_SUM = X.sum()
    #I = I_SUM.div(r)
    return I_SUM

   # @staticmethod
   # def backward(ctx, grad_I):
   #     I, = ctx.saved_tensors
   #     I.backward(grad_I)
   #     # delete further
   #     #one = torch.tensor([1], device=device, dtype=dtype, requires_grad=False)
   #     return X.grad, Y.grad


####################################
####################################


if __name__ == "__main__":
	
    np.random.seed(1)
    device = torch.device("cpu")
    dtype = torch.float
    one = torch.tensor([1], device=device, dtype=dtype, requires_grad=False)
	# Independent Datasets
    X = torch.tensor(np.random.rand(5,3), device=device, dtype=dtype, requires_grad=True)
    Y = torch.tensor(np.random.rand(5,2), device=device, dtype=dtype, requires_grad=True)

    I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
    I.backward(one)
    print("X gradients:", X.grad)
    print("Y gradients:", Y.grad)
    print ('Independent Datasets: ', I)

	# Dependent Datasets
  #  X_array = np.random.rand(1000,2)
  #  X = torch.tensor(X_array, device=device, dtype=dtype, requires_grad=True)
  #  Y = torch.tensor(X_array + np.random.rand(1000,2), device=device, dtype=dtype, requires_grad=True)

  #  I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
  #  print ('Dependent Datasets: ', I)

  #  # Stronger dependency between datasets
  #  X_array = np.random.rand(1000,2)
  #  X = torch.tensor(X_array, device=device, dtype=dtype, requires_grad=True)
  #  Y = torch.tensor(X_array + np.random.rand(1000,2)/4, device=device, dtype=dtype, requires_grad=True)

  #  I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
  #  print ('Stronger dependency between datasets: ',I)

  #  # Large size independent datasets
  #  X_array = np.random.rand(5000,40)
  #  X = torch.tensor(X_array, device=device, dtype=dtype, requires_grad=True)
  #  Y = torch.tensor(X_array**2 + np.random.rand(5000,40)/2, device=device, dtype=dtype, requires_grad=True)

  #  I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
  #  print ('Large size independent datasets: ', I)
