# EDGE Estimator for Shannon Mutual Information
# (I,G) = EDGE(X,Y): 
# X is N * d_x and Y is N * d_Y data sets
# I is the estimation of mutual information between X snd Y
# G is N by d_X gradient matrix (gradient with respect to X)
# U is optional upper bound for the pointwise mutual information: (I,G) = EDGE(X,Y, U)

# Version 1.3: Automatic epsilon for all dimensions, Ensemble Estimation with Optimized wights, 
#          Repeat Estimation for different random epsilons and b's, compute gradient

import numpy as np
import math
import pdb
import cvxpy as cvx # Need to install CVXPY package 
import torch
from time import gmtime, strftime
from datetime import datetime
from multiprocessing import Pool
from functools import partial


def gen_eps(X,Y):

    print("     Begin gen_eps:")
    strftime("      %Y-%m-%d %H:%M:%S", gmtime())

    # Parameter: range of random epsilon coefficient:
    eps_l, eps_u = 0.7, 1.8
    
    # Num of Samples
    N = X.shape[0]
    
    # Find dimensions
    dim_X , dim_Y  = X.shape[1], Y.shape[1]
    
    std_X = np.array([np.std(list(set(np.squeeze( X[:,[i]] ).tolist()))) for i in range(dim_X)])
    std_Y = np.array([np.std(list(set(np.squeeze( Y[:,[i]] ).tolist()))) for i in range(dim_Y)])
    
    # random coeffs, changed by yuzeng 
    Tx=np.random.rand(1,dim_X)
    cf_X = Tx[0]*(eps_u - eps_l) + eps_l
    Ty=np.random.rand(1,dim_Y)
    cf_Y = Ty[0]*(eps_u - eps_l) + eps_l

    # Random espilons
    eps_X = std_X * cf_X
    eps_Y = std_Y * cf_Y

    # Random Shifts, changed by yueng
    Tx = np.random.rand(1,dim_X)
    b_X = 1.0*Tx[0]*eps_X
    Ty = np.random.rand(1,dim_Y)
    b_Y = 1.0*Ty[0]*eps_Y
    
    return (eps_X,eps_Y,b_X,b_Y)

# Define H1 (LSH)
def H1(X,b,eps):
    # Compute X_tilde, the mappings of X using H_1 (floor function)
    #if abs(eps).min() == 0:
    #    raise ValueError('Error: The norm of eps is 0', eps) 
 
    nonzero = eps > 0
    X_te = 1.0*(X[nonzero]+b[nonzero])/eps[nonzero]
    X_t = np.floor(X_te)
    R = tuple(X_t.tolist())
    return R

# Compuate Hashing: Compute the number of collisions in each bucket
def Hash(X,Y,t_m,eps_X,eps_Y,b_X,b_Y,Ni_max):

    print("	Begin Hash:")
    strftime("	    %Y-%m-%d %H:%M:%S", gmtime())
    # Num of Samples
    N = X.shape[0]
    
    # Find dimensions
    dim_X , dim_Y  = X.shape[1], Y.shape[1]
    dim = dim_X + dim_Y
    
    # Hash vectors as dictionaries
    CX, CY, CXY = {}, {}, {} 
    
    # Computing Collisions
    for i in range(N):
    
        # Compute H_1 hashing of X_i and Y_i: Convert to tuple (vectors cannot be taken as keys in dict)
        X_l, Y_l = H1(X[i],b_X,eps_X), H1(Y[i],b_Y,eps_Y)
        		
        # X collisions: compute H_2 
        if X_l in CX:
            CX[X_l].append(i)
        
        else: 
            CX[X_l] = [i]
        	
        # Y collisions: compute H_2
        if Y_l in CY:
            CY[Y_l].append(i)
        else: 
            CY[Y_l] = [i]
        # XY collisions
        if (X_l,Y_l) in CXY:
            CXY[(X_l,Y_l)].append(i)
        else: 
            CXY[(X_l,Y_l)] = [i]
    # Use f as measure of quality of hashing H1
    N_temp = [len(S) for S in CXY.values()]
    f = max(N_temp)/(Ni_max*pow(N,2.0/4))
    
    return (f, CX, CY, CXY, N_temp)


def find_interval(X,Y, eps_X_temp,eps_Y_temp,b_X_temp,b_Y_temp,t_l, t_u, Ni_max = 1):

    print("	Begin find_interval:")
    strftime("	    %Y-%m-%d %H:%M:%S", gmtime())
    # Num of Samples
    N = X.shape[0]
    
    # Find dimensions
    dim_X , dim_Y  = X.shape[1], Y.shape[1]
    dim = dim_X + dim_Y
    
    # parameter: C_balance: Minimum ratio of number of distinct hashing (L_XY) with respect to max collision  
    C_balance_l, C_balance_u = 0.63 , 1.65
    
    # Find the appropriate interval
    f_l, f_u = 0, 3.0
    err = 0

    # store the series of f_m, changed by yuzeng 
    f_m_list = []
    t_m_list = []
    while  (f_l < C_balance_l)  or ( C_balance_u < f_u): 
        # If cannot find the right interval make error
        err +=1
        if err > 90:
            a = 0
        if err > 100:
            raise ValueError('Error: Correct interval cannot be found. Try modifying t_l and t_u', t_l,t_u) 
        t_m = (t_u+t_l)/2
        t_m_list.append(t_m)
        # Normalize epsilons
        eps_X = eps_X_temp * 1.0*t_m / pow(N,1.0/(2*dim))
        eps_Y = eps_Y_temp * 1.0*t_m / pow(N,1.0/(2*dim))
        
        b_X = b_X_temp * 1.0*t_m / pow(N,1.0/(2*dim))
        b_Y = b_Y_temp * 1.0*t_m / pow(N,1.0/(2*dim))

        (f_m, CX, CY, CXY, N_temp) = Hash(X,Y, t_m,eps_X,eps_Y,b_X,b_Y,Ni_max)
        f_m_list.append(f_m)
        
        not_in_interval = (f_l < C_balance_l) or (C_balance_u < f_u) 
        if f_m < 1 and not_in_interval:
            t_l=t_m
            f_l=f_m
        elif f_m > 1 and not_in_interval:
            t_u=t_m
            f_u=f_m
    
    return (t_l,t_u)

# Compute mutual information and gradient given epsilons and radom shifts
def Compute_MI(X,Y,U,t,eps_X,eps_Y,b_X,b_Y,Ni_min,Ni_max):

    print("Begin Compute_MI:")
    strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # parameter: choose random r_grad form each bucket for grad computation
    r_grad = 1

    # Num of Samples and dim
    N = X.shape[0]
    d = X.shape[1]

    # Find dimensions
    dim_X , dim_Y  = X.shape[1], Y.shape[1]
    
    # Parameter: Lower bound for Ni and Mj for being counted:
    mini = Ni_min * pow(N,2.0/4)
    
    (f_m, CX, CY, CXY, N_temp) = Hash(X,Y,t,eps_X,eps_Y,b_X,b_Y,Ni_max)
    
    # Computing Mutual Information Function
    I = 0
    N_c=0
    
    for e in CXY.keys():
        Ni, Mj, Nij = len(CX[e[0]]), len(CY[e[1]]), len(CXY[e])
    
        #if mini<Ni and mini<Mj: 
        if mini<Nij:
            I += Nij* math.log(max(min(1.0*Nij*N/(Ni*Mj), U),1.0/U),2)
            N_c+=Nij
    # Compute MI
    I = 1.0* I / N_c

    print("I completed! Begin Grad_mat:")
    strftime("%Y-%m-%d %H:%M:%S", gmtime())
    
   	## Compute gradient matrix (N by d)

    Grad_mat= np.zeros((N,d))
    
    # Compute X_tilde and Y_tilde, the mappings of X and Y using H_1 (floor function)
    X_non_zero = eps_X>0
    Y_non_zero = eps_Y>0
    
    X_tf,Y_tf=np.zeros([N,dim_X]),np.zeros([N,dim_Y])
    X_t,Y_t=np.zeros([N,dim_X]),np.zeros([N,dim_Y])
    
    X_tf, Y_tf = 1.0*(X[:, X_non_zero]+b_X[X_non_zero])/eps_X[X_non_zero], 1.0*(Y[:,Y_non_zero]+b_Y[Y_non_zero])/eps_Y[Y_non_zero]
    X_t, Y_t= np.floor(X_tf), np.floor(Y_tf)
    
    
    # initialize bakward and forward buckets
    dw_b,dw_f = np.zeros(d),np.zeros(d)
    
    # initialize current, forward and backward Delta functions
    grad_Ni, grad_Ni_b_vec, grad_Ni_f_vec = np.zeros(d),np.zeros(d), np.zeros(d)
    
    t_count=0
    
    # Compute gradient for the bucket representatives
    for r in CX.values():
    	# pick the index of representatives
        i = r[0]
        # compute the gradiant with respect to X_i
        Grad_vec = np.zeros((1,d))
        
        # for random points in each bucket r
        q = np.random.choice(r, min(r_grad,len(r)), replace=False) 
        
        for i in q:
            t_count=t_count+1
            X_l, Y_l = H1(X[i],b_X,eps_X), H1(Y[i],b_Y,eps_Y)
            
            Ni = len(CX[X_l])
            Mj = len(CY[Y_l])
            Nij = len(CXY[(X_l,Y_l)])		
            
            # All current and neighbor bucket colllisions: 
            #      backward and forward buckets Ni and Nij
            direct = np.zeros(d)
                
            Rd = np.asarray(range(d))
            for z in Rd[X_non_zero]:
            
                # current bucket gradient
                grad_Ni=Delta(X_tf[i,z]-X_t[i,z]) - Delta(X_tf[i,z]-X_t[i,z]-1)
                Grad_vec += 1.0/math.log(2)*grad_Ni*(1.0/Nij-1.0/Ni)
            
                # Neighbors:
                direct[z] = 1
                # Backward buckets in each dimension
                X_temp = X_t[i]-direct
                X_l = tuple(X_temp.tolist())
                grad_Ni_b_vec[z] = -Delta(X_tf[i,z]-X_t[i,z])
                
                # compute dw_b
                if (X_l,Y_l) in CXY:
                    Ni_b= len(CX[X_l])
                    Nij_b =len(CXY[(X_l,Y_l)])
                    dw_b[z]=1.0/math.log(2)*(1.0/Nij_b-1.0/Ni_b)
                
                else:
                    if X_l in CX:
                        Ni_b= len(CX[X_l])
                    else: 
                        Ni_b=0		
                    dw_b[z] = math.log(1.0*N/(Mj*(Ni_b+1)))
            
                # Forward buckets in each dimension
                X_temp = X_t[i]+direct
                X_l = tuple(X_temp.tolist())
                grad_Ni_f_vec[z] = Delta(X_tf[i,z]-X_t[i,z]-1)
                
                # compute dw_f
                if (X_l,Y_l) in CXY:
                    Ni_f= len(CX[X_l])
                    Nij_f =len(CXY[(X_l,Y_l)])
                    dw_f[z]=1.0/math.log(2)*(1.0/Nij_f-1.0/Ni_f)
                
                else:
                    if X_l in CX:
                        Ni_f= len(CX[X_l])
                    else: 
                        Ni_f=0		
                    dw_f[z] = math.log(1.0*N/(Mj*(Ni_f+1)))
            
                direct[z] = 0
            
            # backward bucket gradient
            back_grad_vec = grad_Ni_b_vec*dw_b
            Grad_vec += back_grad_vec
            
            # forward bucket gradient
            forward_grad_vec = grad_Ni_f_vec*dw_f
            Grad_vec += forward_grad_vec
    
    	    # set all of the gradients corresponding to the nodes in bucket r
            Grad_mat[r]=1.0*Grad_vec/N
    
    print("Grad_mat completed!")
    strftime("%Y-%m-%d %H:%M:%S", gmtime())

    return (I,Grad_mat)

# Delta function used for estimation of gradient
def Delta(x):
    # parameter r: as increases, creates sharper function 
    r=4 #(r=4 creats hat func with bandwidth=eps/2)
    return r*max(1-r*abs(x),0)


# A single run of Mutual Information estimation using EDGE
def EDGE_single_run(X,Y,U=20):

	# Num of Samples and dim
    N = X.shape[0]
    d = X.shape[1]
    
    # parameter: Ni_min * sqrt(N) < N_i, M_i < Ni_max * sqrt(N)
    Ni_min = 0.2
    Ni_max = 1.0 
    
    # Find dimensions
    dim_X , dim_Y  = X.shape[1], Y.shape[1]
    dim = dim_X + dim_Y
    
    # Number of terms in ensemble method
    #L = dim+1
    L=min(4, dim+1)
    
    # Num of Samples, increase t_u, changed by yuzeng
    N = X.shape[0]
    t_l,t_u = 0.1, 2000
    
    # Use less number of samples for learning the interval
    N_t=1000
    if N>N_t:
        X_test,Y_test=X[:N_t],Y[:N_t]
    else:
        X_test,Y_test=X,Y
    
    # get eps and b:
    (eps_X_temp,eps_Y_temp,b_X_temp,b_Y_temp) = gen_eps(X_test,Y_test)
    
    # Find the appropriate Interval
    (t_l,t_u) = find_interval(X_test,Y_test, eps_X_temp,eps_Y_temp,b_X_temp, b_Y_temp,t_l, t_u, Ni_max)
    
    # Find a range of indices
    l = t_u - t_l
    c = 1.0*l / L
    
    T_U = np.linspace(t_l, t_u, L)
    
    if L**(1.0/(2*dim)) < 1.0*t_u/t_l:
        T = t_l* np.array(range(1,L+1))**(1.0/(2*dim))
    else:
        T = np.linspace(t_l, t_u, L)
    
    # Vector of weights
    W = compute_weights(L, dim, T, N)
    
    # Vector of MI and its grad
    I_vec, Grad_vec = np.zeros(L), np.zeros((L,N,d))
    for i in range(L):
    
    	# Normalize epsilons
        eps_X = eps_X_temp * 1.0*T[i] / pow(N,1.0/(2*dim))
        eps_Y = eps_Y_temp * 1.0*T[i] / pow(N,1.0/(2*dim))

        b_X = b_X_temp * 1.0*T[i] / pow(N,1.0/(2*dim))
        b_Y = b_Y_temp * 1.0*T[i] / pow(N,1.0/(2*dim))
    
        (I_vec[i], Grad_vec[i,:,:]) = Compute_MI(X,Y,U,T[i],eps_X,eps_Y,b_X,b_Y,Ni_min,Ni_max)
    
    # Ensemble MI
    I = np.dot(I_vec,W.T)
    Grad_mat= np.array([[np.dot(Grad_vec[:,i,j],W.T) for j in range(d)] for i in range(N)])
    Grad_mat=np.reshape(Grad_mat,(N,d))
    
    return (I,Grad_mat)

##### Linear Program for Ensemble Estimation ####
def compute_weights(L, d, T, N):
	
    print("     Begin compute_weights:")
    strftime("      %Y-%m-%d %H:%M:%S", gmtime())
    # Correct T
    T = 1.0*T/T[0]
    
    # Create optimization variables.
    cvx_eps = cvx.Variable()
    cvx_w = cvx.Variable(L)
    
    # Create constraints:
    constraints = [cvx.sum_entries(cvx_w)==1, cvx.pnorm(cvx_w, 2)- cvx_eps/2 < 0 ]
    for i in range(1,L):
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
def EDGE_top(X,Y,U=20):

    device = torch.device("cpu")
    dtype = torch.float

    # adjust the dimensions of X and Y, by yuzeng
    # if the variable has only one dimension, transpose it
    if len(X.shape) == 1:
        X = X.reshape(-1,1)

    if len(Y.shape) == 1:
        Y = Y.reshape(-1,1)
	
    # Num of Samples and dim
    N = X.shape[0]
    d = X.shape[1]

	# Repeat for LSH with different random parameters and Take mean: 
	#	By increasing r you get more accurate estimate
    r = 5
    I_vec, Grad_vec = np.zeros(r), np.zeros((r,N,d))

    # Use multiprocess
    pool = Pool(r)
    func = partial(EDGE_single_run, X, Y)
    U_list = [U] * r
    result = pool.map(func, U_list)
    pool.close()
    pool.join()

    for i in range(r):
            print("Begin run %:", i)
            strftime("%Y-%m-%d %H:%M:%S", gmtime())
            (I_vec[i], Grad_vec_temp) = EDGE_single_run(X,Y,U)
            Grad_vec[i,:,:]=Grad_vec_temp

    I = np.mean(I_vec)
    Grad_mat= np.mean(Grad_vec,0)
    return (I,Grad_mat)





class EDGE(torch.autograd.Function):


    #def __init__(self, size_X):
    #    self.Grad_mat = np.zeros(size_X)
    #    self.Grad_Y = None


    # Return the normalizing factor of epsilon regarding STD and some randomness

    @staticmethod    
    def forward(ctx, X, Y):

        print("Begin forward: ")
        str(datetime.now())

        device = torch.device("cpu")
        dtype = torch.float

        X = X.cpu().data.numpy()
        Y = Y.cpu().data.numpy()


        # The following part should follow all the subfunctions
        I, Grad_mat = EDGE_top(X,Y)
        #I = torch.tensor(I, device=torch.device("cuda"), dtype=dtype, requires_grad=False)
        #Grad_mat = torch.tensor(Grad_mat, device=torch.device("cuda"), dtype=dtype, requires_grad=False)
        size = list(Y.shape)
        Grad_Y = torch.zeros(size, device=torch.device("cpu"), dtype=dtype, requires_grad=False)
        #self.Grad_mat = Grad_mat
        ctx.save_for_backward(Grad_mat, Grad_Y)
        return I

    @staticmethod    
    def backward(ctx, grad_I=1):

        print("Begin backward:")
        strftime("%Y-%m-%d %H:%M:%S", gmtime())
        grad_X, grad_Y = ctx.saved_tensors
        return (grad_mat, grad_Y)



####################################
####################################


if __name__ == "__main__":
	

    device = torch.device("cpu")
    dtype = torch.float
    
    # Independent Datasets
    np.random.seed(1)
    X = np.random.rand(1000,2)
    Y = np.random.rand(1000,2)
    
    X = torch.tensor(X, device=device, dtype=dtype, requires_grad=True)
    Y = torch.tensor(Y, device=device, dtype=dtype, requires_grad=False)
    
    
    I = EDGE.apply(X,Y) # Estimated Mutual Information between X and Y using EDGE method
    I.backward()
    G = X.grad
    print ('Independent Datasets: ', I, G)
    
    # Dependent Datasets
    X = np.random.rand(1000,2)
    Y = X + np.random.rand(1000,2)
    
    X = torch.tensor(X, device=device, dtype=dtype, requires_grad=True)
    Y = torch.tensor(Y, device=device, dtype=dtype, requires_grad=False)
    
    I = EDGE.apply(X,Y) # Estimated Mutual Information between X and Y using EDGE method
    print ('Dependent Datasets: ', I)
    
    # Stronger dependency between datasets
    X = np.random.rand(1000,2)
    Y = X + np.random.rand(1000,2)/4
    
    X = torch.tensor(X, device=device, dtype=dtype, requires_grad=True)
    Y = torch.tensor(Y, device=device, dtype=dtype, requires_grad=False)
    
    I = EDGE.apply(X,Y) # Estimated Mutual Information between X and Y using EDGE method
    print ('Stronger dependency between datasets: ',I)
    
    # Large size independent datasets
    
    #X = np.random.rand(5000,40)
    #Y = X**2 + np.random.rand(5000,40)/2
    
    #I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
    #print ('Large size independent datasets: ', I)



