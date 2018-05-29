# EDGE Estimator for Shannon Mutual Information
# (I,G) = EDGE(X,Y): 
# X is N * d_x and Y is N * d_Y data sets
# I is the estimation of mutual information between X snd Y
# G is N by d_X gradient matrix (gradient with respect to X)
# U is optional upper bound for the pointwise mutual information: (I,G) = EDGE(X,Y, U)

# Version 1.5: 
#             *1.1 Automatic epsilon for all dimensions,
#			  *1.2 Ensemble Estimation with Optimized wights (L ristricted to 4)
#             *1.3 Repeat Estimation for different random epsilons and b's
#			  *1.4 Compute gradient
#			  	*1.45 Only small subset of points in 
#				        each bucket are used for computing gradient 
#			  *1.5: use a only portion of samples for find_interval
#			  *1.6: b_X and b_Y are updated in find_interval
#			  *1.7: (To be done) Avoid devision by zero epsilon in computin H1 and Gradient
#			  *1.8: control num of X and Y buckets separately
#			  *1.8: Update b_X even in find_interval
#			  *1.9: Define a random wieght matrix in H1 to reduce dimension
#			  *2.0: Turn computing gradient a separate function, 1D vectors works well
#			  * (To be done) STD only needs to be computed for unique samples
#			  * (To be done) turn computing gradient to parallel 

import numpy as np
import math
import pdb
import cvxpy as cvx # Need to install CVXPY package 
import time

# Return the normalizing factor of epsilon regarding STD and some randomness
def gen_eps(X,Y,W,V):

	# Parameter: range of random epsilon coefficient:
	eps_l, eps_u = 0.7, 1.8
	
	# Num of Samples
	N = X.shape[0]

	# Find dimensions
	dim_X , dim_Y  = X.shape[1], Y.shape[1]
	d_X_shrink, d_Y_shrink=W.shape[1], V.shape[1]

	std_X = np.array([np.std(np.dot(X,W)[:,[i]]) for i in range(d_X_shrink)])
	std_Y = np.array([np.std(np.dot(Y,V)[:,[i]]) for i in range(d_Y_shrink)])

	# random coeffs 
	Tx=np.random.rand(1,d_X_shrink)
	cf_X = Tx[0]*(eps_u - eps_l) + eps_l
	Ty=np.random.rand(1,d_Y_shrink)
	cf_Y = Ty[0]*(eps_u - eps_l) + eps_l

	# Random espilons
	eps_X = std_X * cf_X
	eps_Y = std_Y * cf_Y

	# Random Shifts
	Tx = np.random.rand(1,d_X_shrink)
	b_X = Tx[0]*eps_X
	Ty = np.random.rand(1,d_Y_shrink)
	b_Y = Ty[0]*eps_Y

	return (eps_X,eps_Y,b_X,b_Y)

# Define H1 (LSH)
def H1(X,W,b,eps):
	# Compute X_tilde, the mappings of X using H_1 (floor function)
	# if not scalar
	d_X = X.shape[0]
	X=X.reshape(1,d_X)
	#print(X.shape,W.shape)
	#print(np.dot(X,W))
	if d_X > 1:
		#print('XW',np.squeeze(np.dot(X,W)))
		#print('eps',eps)
		#print('b',b)
		eps_NZ=np.flatnonzero(eps)
		#print('non_z_eps',eps_NZ)
		X_te = 1.0*(np.squeeze(np.dot(X,W))[eps_NZ]+b[eps_NZ])/eps[eps_NZ]
	elif eps>0:
		X_te = 1.0*(X+b)/eps
	else:
		X_te=X

	X_t = np.floor(X_te)
	if d_X>1: 
		R = tuple(X_t.tolist())
	else: R=np.asscalar(np.squeeze(X_t))
	return R

# Compuate Hashing: Compute the number of collisions in each bucket
def Hash(X,Y,W,V,eps_X,eps_Y,b_X,b_Y,Ni_max):

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
		#print('W,V',W.shape,V.shape)
		X_l, Y_l = H1(X[i],W,b_X,eps_X), H1(Y[i],V,b_Y,eps_Y)
		#print(X_l,Y_l)
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
	f = np.max(N_temp)/(Ni_max*pow(N,2.0/4))

	#print('max',np.max(N_X_temp),np.max(N_Y_temp))
	return (f, CX, CY, CXY)


def find_interval(X,Y, W,V,eps_X_temp,eps_Y_temp,b_X_temp,b_Y_temp,t_l, t_u, Ni_max = 1):

	# Num of Samples
	N = X.shape[0]

	# Find dimensions
	dim_X  = X.shape[1]
	dim_Y = Y.shape[1]
	dim=dim_X+dim_Y
	# parameter: C_balance: Minimum ratio of number of distinct hashing (L_XY) with respect to max collision  
	C_balance_l, C_balance_u = 0.7 , 1.5

	# Find the appropriate interval
	f_l, f_u = 0, 3.0
	err = 0
	while  (f_l < C_balance_l)  or ( C_balance_u < f_u): 
		# If cannot find the right interval make error
		err +=1
		if err > 20:
			print('Warning: Mutual information estimate may not be accurate')
			break
			#raise ValueError('Error: Correct interval cannot be found. Try modifying t_l and t_u', t_l,t_u)  

		t_m = (t_u+t_l)/2

		# Normalize epsilons

		eps_Y = eps_Y_temp * 1.0*t_m / pow(N,1.0/(2*dim))
		b_Y = b_Y_temp * 1.0*t_m / pow(N,1.0/(2*dim))
		eps_X = eps_X_temp * 1.0*t_m / pow(N,1.0/(2*dim))
		b_X = b_X_temp * 1.0*t_m / pow(N,1.0/(2*dim))
			
		#Z,eps_Z,b_Z=np.array([1]),np.array([1]),np.array([1])

		(f_m, CX, CY, CXY) = Hash(X,Y,W,V,eps_X,eps_Y,b_X,b_Y,Ni_max)
		#print('find_interval',f_m, len(CX.keys()),len(CY.keys()),len(CXY.keys()) )

		not_in_interval = (f_l < C_balance_l) or (C_balance_u < f_u) 
		if f_m < 1 and not_in_interval:
			t_l=t_m
			f_l=f_m
		elif f_m > 1 and not_in_interval:
			t_u=t_m
			f_u=f_m

	return (t_l,t_u)

# Compute mutual information and gradient given epsilons and radom shifts
def Compute_MI(X,Y,U,W,V,eps_X,eps_Y,b_X,b_Y,Ni_min,Ni_max):
	
	# Num of Samples and dim
	N = X.shape[0]

	# Find dimensions
	dim_X , dim_Y  = X.shape[1], Y.shape[1]
	d=dim_X
	dim=dim_X+dim_Y

	# Parameter: Lower bound for Ni and Mj for being counted:
	mini = Ni_min * pow(N,2.0/4)
	min_product = 0.5*N

	(f_m, CX, CY, CXY) = Hash(X,Y,W,V,eps_X,eps_Y,b_X,b_Y,Ni_max)

####
	#print('(f_m, CX, CY, CXY)',f_m, len(CX.keys()),len(CY.keys()),len(CXY.keys()))
	#print('CX',CX)
	#print(CX.values())
	#print('CX',[len(W) for W in CX.values()])
	#print('CY',[len(W) for W in CY.values()])
	#print('CXY',[len(W) for W in CXY.values()])
	# Computing Mutual Information Function
	I = 0
	N_c=0

	for e in CXY.keys():
		Ni, Mj, Nij = len(CX[e[0]]), len(CY[e[1]]), len(CXY[e])
####
		#print('Ni, Mj, Nij', Ni,Mj,Nij, 'PI:', 1.0*Nij*N/(Ni*Mj))

		#if mini<Ni and mini<Mj: 
		if Nij>1.0*mini:
			I += Nij* math.log(max(min(1.0*Nij*N/(Ni*Mj), U),1.0/U),2)
			N_c+=Nij
	# Compute MI
	if N_c==0:
		N_c=1
	I = 1.0* I / N_c
	#print('N_c',N_c)
	#I = 1.0* I / N
	#print('I,Nc',I,N_c)

	Grad_mat = Compute_Gradient(X,Y,U,W,V,eps_X,eps_Y,b_X,b_Y,CX,CY,CXY)
	#print(Grad_mat)
	return (I,Grad_mat)

####################################
## Compute gradient matrix (N by d)
def Compute_Gradient(X,Y,U,W,V,eps_X,eps_Y,b_X,b_Y,CX,CY,CXY):

	# parameter: choose random r_grad form each bucket for grad computation
	r_grad = 3

	# Num of Samples and dim
	N = X.shape[0]
	# Find dimensions
	dim_X , dim_Y  = X.shape[1], Y.shape[1]
	dim=dim_X+dim_Y
	d_shrink= W.shape[1]

	Grad_mat= np.zeros((N,dim_X))

	# Compute X_tilde and Y_tilde, the mappings of X and Y using H_1: X_te and X_tf are N by d_sh
	X_t = 1.0*(np.dot(X,W)+b_X)/eps_X
	X_tf = np.floor(X_t)

	# initialize bakward and forward buckets
	dw_b,dw_f = 0,0

	# initialize current, forward and backward Delta functions
	grad_Ni, grad_Ni_b_vec, grad_Ni_f_vec = np.zeros(dim_X),np.zeros([d_shrink,dim_X]), np.zeros([d_shrink,dim_X])

	t_count=0

	# Compute gradient for the bucket representatives
	for r in CX.values():
		if len(r) <= 5: 
			continue 
		# pick the index of representatives
		i = r[0]
		# compute the gradiant with respect to X_i
		Grad_vec = np.zeros((1,dim_X))

		# for random points in each bucket r
		q = np.random.choice(r, min(r_grad,len(r)), replace=False) 

		for i in q:
			t_count=t_count+1
			X_l, Y_l = H1(X[i],W,b_X,eps_X), H1(Y[i],V,b_Y,eps_Y)
			
			Ni = len(CX[X_l])
			Mj = len(CY[Y_l])
			Nij = len(CXY[(X_l,Y_l)])		

			# All current and neighbor bucket colllisions: 
			#      backward and forward buckets Ni and Nij
			direct = np.zeros(d_shrink)
			
			# current bucket gradient
			Dl=(Delta(X_t[i,:]-X_tf[i,:])-Delta(X_t[i,:]-X_tf[i,:]-1))
			grad_Ni = np.matmul(Dl.reshape(1,d_shrink),W.T)
			Grad_vec += Ni/len(q)*grad_Ni*1.0/math.log(2)*(1.0/Nij-1.0/Ni)
			Grad_vec += Ni/len(q)*grad_Ni*1.0/Nij *math.log(max(min(1.0*Nij*N/(Ni*Mj), U),1.0/U),2) 
			#print('i ',i,' Grad_vec', Grad_vec)
			
			# copmute d_sh by d_X matrices dw_b, dw_f, grad_Ni_b_vec and grad_Ni_f_vec 
			for z in range(d_shrink):
				# Neighbors:
				direct[z] = 1
				# Backward buckets in each dimension
				X_temp = X_tf[i]-direct
				X_l = tuple(X_temp.tolist())
				Dl_b = -Delta(X_t[i,:]-X_tf[i,:])
				grad_Ni_b_vec[z,:] += np.matmul(Dl_b.reshape(1,d_shrink),W.T).reshape(dim_X,)
				
				# compute dw_b
				if (X_l,Y_l) in CXY:
					Ni_b= len(CX[X_l])
					Nij_b =len(CXY[(X_l,Y_l)])
					Mj_b = len(CY[Y_l])
					dw_b= 1.0/math.log(2)*(1.0/Nij_b-1.0/Ni_b)
					dw_b+=1.0/Nij_b *math.log(max(min(1.0*Nij_b*N/(Ni_b*Mj_b), U),1.0/U),2) 
				else:
					if X_l in CX:
						Ni_b= len(CX[X_l])
					else: 
						Ni_b=0		
					dw_b = math.log(min(1.0*N/(Mj*(Ni_b+1)),U))
				grad_Ni_b_vec[z,:] = grad_Ni_b_vec[z,:] * dw_b
				# Forward buckets in each dimension
				X_temp = X_tf[i]+direct
				X_l = tuple(X_temp.tolist())
				Dl_f=Delta(X_t[i,:]-X_tf[i,:]-1)
				grad_Ni_f_vec[z,:] =  np.matmul(Dl_f.reshape(1,d_shrink),W.T).reshape(dim_X,)

				# compute dw_f
				if (X_l,Y_l) in CXY:
					Ni_f= len(CX[X_l])
					Nij_f =len(CXY[(X_l,Y_l)])
					Mj_f = len(CY[Y_l])
					dw_f=1.0/math.log(2)*(1.0/Nij_f-1.0/Ni_f)
					dw_f+=1.0/Nij_f *math.log(max(min(1.0*Nij_f*N/(Ni_f*Mj_f), U),1.0/U),2) 
				else:
					if X_l in CX:
						Ni_f= len(CX[X_l])
					else: 
						Ni_f=0		
					dw_f = math.log(min(1.0*N/(Mj*(Ni_f+1)),U))
					grad_Ni_f_vec[z,:]=dw_f*grad_Ni_f_vec[z,:]
				direct[z] = 0
			# backward and forward bucket gradient
			#print('back_for_ward',np.sum(grad_Ni_b_vec+ grad_Ni_f_vec, axis=0))
			Grad_vec += np.sum(grad_Ni_b_vec+ grad_Ni_f_vec, axis=0).reshape(1,dim_X)

		# set all of the gradients corresponding to the nodes in bucket r
		Grad_mat[r,:]=1.0*Grad_vec/N
		#print('Grad_mat[r,:]',Grad_mat[r,:])
	return Grad_mat
# Delta function used for estimation of gradient
def Delta(x):
	# parameter r: as increases, creates sharper function 
	r=4 #(r=4 creats hat func with bandwidth=eps/2)
	s=r*np.maximum(1-r*np.absolute(x),0)
	return s 


# A single run of Mutual Information estimation using EDGE
def EDGE_single_run(X,Y,U=20):

	# Num of Samples and dim
	N = X.shape[0]
	d = X.shape[1]

	# parameter: Ni_min * sqrt(N) < N_i, M_i < Ni_max * sqrt(N)
	Ni_min = 0.2
	Ni_max = 4.0 

	# Find dimensions
	dim_X , dim_Y  = X.shape[1], Y.shape[1]
	dim = dim_X + dim_Y
	
	# Number of terms in ensemble method
	#L = dim+1
	L=min(4,dim+1)

	# Num of Samples
	N = X.shape[0]
	t_l,t_u = 0.1, 200
	
	# Use less number of samples for learning the interval
	N_t=1000
	if N>N_t:
		X_test,Y_test=X[:N_t],Y[:N_t]
	else:
		X_test,Y_test=X,Y


	# Generate random transformation matrices W and V
	#d_X_shrink=d_Y_shrink=2
	d_X_shrink=min(dim_X,math.floor(0.5*math.log(N,2)))
	W= np.random.rand(dim_X,d_X_shrink)
	d_Y_shrink=min(dim_Y,math.floor(0.5*math.log(N,2)))
	V= np.random.rand(dim_Y,d_Y_shrink)

	#print('W,V',W.shape,V.shape)
	#time.sleep(2)

	# get eps and b:
	(eps_X_temp,eps_Y_temp,b_X_temp,b_Y_temp) = gen_eps(X_test,Y_test,W,V)
	#print('gen_eps',eps_X_temp,eps_Y_temp,b_X_temp,b_Y_temp)
	# Find the appropriate Intervals
	(t_l,t_u) = find_interval(X_test,Y_test,W,V,eps_X_temp,eps_Y_temp,b_X_temp,b_Y_temp,t_l, t_u, Ni_max)
	
	T = np.linspace(t_l, t_u, L)

	# Vector of weights
	Opt_W = compute_weights(L, dim, T, N)

	# Vector of MI and its grad
	I_vec, Grad_vec = np.zeros(L), np.zeros((L,N,d))
	
	for i in range(L):
		# Normalize epsilons
		#eps_X = eps_X_temp * 1.0*T_X[i] / pow(N,1.0/(2*dim))
		#eps_Y = eps_Y_temp * 1.0*T_Y[i] / pow(N,1.0/(2*dim))
		eps_X = eps_X_temp * 1.0*T[i]
		eps_Y = eps_Y_temp * 1.0*T[i]

		#b_X = b_X_temp * 1.0* T_X[i] / pow(N,1.0/(2*dim))
		#b_Y = b_Y_temp * 1.0* T_Y[i] / pow(N,1.0/(2*dim))

		b_X = b_X_temp * 1.0* T[i]
		b_Y = b_Y_temp * 1.0* T[i] 

		(I_vec[i], Grad_vec[i,:,:]) = Compute_MI(X,Y,U,W,V,eps_X,eps_Y,b_X,b_Y,Ni_min,Ni_max)
####
		#print('I_vec', I_vec[i])

	# Ensemble MI
	I = np.dot(I_vec,Opt_W.T)
	Grad_mat= np.array([[np.dot(Grad_vec[:,i,j],Opt_W.T) for j in range(d)] for i in range(N)])
	Grad_mat=np.reshape(Grad_mat,(N,d))

	return (I,Grad_mat)

##### Linear Program for Ensemble Estimation ####
def compute_weights(L, d, T, N):
	
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
def EDGE(X,Y,U=20):
	# Num of Samples and dim
	N = X.shape[0]
	d = X.shape[1]

	# Repeat for LSH with different random parameters and Take mean: 
	#	By increasing r you get more accurate estimate
	r = 2
	I_vec, Grad_vec = np.zeros(r), np.zeros((r,N,d))
	for i in range(r):
		(I_vec[i], Grad_vec_temp) = EDGE_single_run(X,Y,U)
		Grad_vec[i,:,:]=Grad_vec_temp

	I = np.mean(I_vec)
	Grad_mat= np.mean(Grad_vec,0)
	return (I,Grad_mat)

####################################
####################################

if __name__ == "__main__":


	# Discrete dataset
	X=np.zeros((200,1))
	Y=np.zeros((200,1))

	X[100:200,0]=1
	Y[0:100,0]=np.random.binomial(1,0,(100,1)).reshape(100,)
	Y[100:200,0]=np.random.binomial(1,1,(100,1)).reshape(100,)

	#print(X)
	#print(Y)

	(I,G) = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
	print ('Independent Datasets: ', I,G)


	# Independent Large Datasets
	X = np.random.rand(1000,50)
	Y = np.random.rand(1000,50)

	(I,G) = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
	print ('Independent Datasets: ', I,G)

	# Independent Datasets: Gaussian
	mean=[0, 0]
	cov=[[50, 0],[0,50]]
	X = np.random.multivariate_normal(mean,cov,1000)
	Y = np.random.multivariate_normal(mean,cov,1000)

	(I,G) = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
	print ('Normal Distribution: ', I,G)

	# Dependent Datasets
	X = np.random.rand(1000,2)
	Y = X + np.random.rand(1000,2)

	I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
	print ('Dependent Datasets: ', I)

	# Stronger dependency between datasets
	X = np.random.rand(1000,50)
	Y = X + np.random.rand(1000,50)/16

	I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
	print ('Stronger dependency between datasets: ',I)

	# Large size independent datasets
	#X = np.random.rand(5000,40)
	#Y = X**2 + np.random.rand(5000,40)/2

	#I = EDGE(X,Y) # Estimated Mutual Information between X and Y using EDGE method
	#print ('Large size independent datasets: ', I)


