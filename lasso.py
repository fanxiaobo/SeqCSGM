"""
 add by fxb, In order to achieve the goal of using lasso for sequential estimation, we implement weighted lasso
In sequence estimation, update the weight to match the support from the previous moment
reference:
 Regularized Modified BPDN for Noisy Sparse Reconstruction with Partial Erroneous Support and Signal Value Knowledge,
 Modified-CS: Modifying Compressive Sensing for Problems with Partially Known Support
"""
import numpy as np
from scipy import sparse
import osqp
import numpy as np
from scipy import sparse

def solve_weighted_lasso(A, y, lasso_weights, pos, lmbd):
    ## min ||y-Ax||+lambda*wi|xi|
    m, n = A.shape
    w = lasso_weights[pos]  # Ensure that w is a vector of length n
    
    # Quadratic term P = A.T @ A
    P = A.T @ A
    P = 0.5 * (P + P.T)  # Forced to be symmetry
    P = sparse.csc_matrix(P)
    
    # linear term q = -A.T @ y
    q = -A.T @ y
    
    # Constructing Sparse Constraint Matrix G
    def build_G_sparse(n, w):
        eye_n = sparse.eye(n, format="csc")
        zero_nn = sparse.csc_matrix((n, n))
        diag_w = sparse.diags(w, format="csc")
        
        G_x_ub = sparse.hstack([eye_n, zero_nn], format="csc")  # x <= 1
        G_x_lb = sparse.hstack([-eye_n, zero_nn], format="csc") # x >= 0
        G_t_ub = sparse.hstack([diag_w, -eye_n], format="csc")  # t >= w_i x_i
        G_t_lb = sparse.hstack([-diag_w, -eye_n], format="csc") # t >= -w_i x_i
        return sparse.vstack([G_x_ub, G_x_lb, G_t_ub, G_t_lb], format="csc")
    
    G = build_G_sparse(n, w)
    
    
    h = np.hstack([np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)])
    
    # OSQP model setting
    model = osqp.OSQP()
    model.setup(
        P=sparse.block_diag([P, sparse.csc_matrix((n, n))], format="csc"),
        q=np.hstack([q, lmbd * np.ones(n)]),  
        A=G,
        l=-np.inf * np.ones(4*n),
        u=h,
        verbose=False,
        eps_abs=1e-3,         # 1e-8
        eps_rel=1e-3,
        max_iter=50000,    # 50000
        polish=True
    )
    
    # solve
    result = model.solve()
    if result.info.status != 'solved':
        raise RuntimeError(f"failure: {result.info.status}")
    
    return result.x[:n]  



def wavelet_basis(path='../models/wavelet_basis.npy'):
    W_ = np.load(path)
    # W_ initially has shape (4096,64,64), i.e. 4096 64x64 images
    # reshape this into 4096x4096, where each row is an image
    # take transpose to make columns images
    W_ = W_.reshape((4096, 4096))
    W = np.zeros((12288, 12288))
    W[0::3, 0::3] = W_
    W[1::3, 1::3] = W_
    W[2::3, 2::3] = W_
    return W

def lasso_wavelet_estimator(lmbd=0.1, th=0.4):  #pylint: disable = W0613
    """LASSO with Wavelet"""
    def estimator(A_val, y_batch_val, lasso_weights):
        x_hat_batch = []
        batch_size = y_batch_val.shape[0]
        W = wavelet_basis()
        WA = np.dot(W, A_val.T)
        for j in range(batch_size):
            y_val = y_batch_val[j]
            z_hat = solve_weighted_lasso(WA.T, y_val, lasso_weights, j, lmbd)
            x_hat = np.dot(z_hat, W)
            x_hat_max = np.abs(x_hat).max()
            x_hat = x_hat / (1.0 * x_hat_max)
            x_hat_batch.append(x_hat)
            lasso_weights[j]  = (x_hat < th).astype(int)  # update lasso_weights
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch, lasso_weights
    return estimator

def lasso_estimator_mmnist(lmbd=0.1, th=0.4): 
    """LASSO estimator"""
    def estimator(A_val, y_batch_val, lasso_weights):
        batch_size = y_batch_val.shape[0]
        print(f"batch_size:{batch_size}")
        x_hat_batch = []       
        for i in range(batch_size):
            y_val = y_batch_val[i]
            x_hat = solve_weighted_lasso(A_val, y_val, lasso_weights, i, lmbd)    # add
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
            lasso_weights[i]  = (x_hat < th).astype(int)  # update lasso_weights

        x_hat_batch = np.asarray(x_hat_batch) 

        # print("x_hat_batch shape") 
        print(x_hat_batch.shape)         
        return x_hat_batch, lasso_weights
    return estimator