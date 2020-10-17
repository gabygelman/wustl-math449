#%%
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.sparse import *
Inf = np.inf

# %%
def getMatrix(n=10, isDiagDom=True):
    '''
    Return an nxn matrix which is the discretization matrix
    from finite difference for a diffusion problem.
    To visualize A: use plt.spy(A.toarray())
    '''
    # Representation of sparse matrix and right-hand side
    assert n >= 2
    n -= 1
    diagonal  = np.zeros(n+1)
    lower     = np.zeros(n)
    upper     = np.zeros(n)

    # Precompute sparse matrix
    if isDiagDom:
        diagonal[:] = 2 + 1/n**2
    else:
        diagonal[:] = 2
    lower[:] = -1  #1
    upper[:] = -1  #1
    # Insert boundary conditions
    # diagonal[0] = 1
    # upper[0] = 0
    # diagonal[n] = 1
    # lower[-1] = 0

    
    A = diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(n+1, n+1),
        format='csr')

    return A

def getAuxMatrix(A):
    '''
    return D, L, U matrices for Jacobi or Gauss-Seidel
    D: array
    L, U: matrices
    '''
    # m = A.shape[0]
    D = csr_matrix.diagonal(A)
    L = -tril(A, k=-1)
    U = -triu(A, k=1)
    return D, L, U

def jacobiIteration(A, b, x0=None, tol=1e-13, numIter=100):
    '''
    Jacobi iteraiton:
    A: nxn matrix
    b: (n,) vector
    x0: initial guess
    numIter: total number of iteration
    tol: algorithm stops if ||x^{k+1} - x^{k}|| < tol
    return: x
    x: solution array such that x[i] = i-th iterate
    '''
    n = A.shape[0]
    x = np.zeros((numIter+1, n))
    if x0 is not None:
        x[0] = x0
    D, L, U = getAuxMatrix(A)
    for k in range(numIter):
        x[k+1] = ((L+U)@x[k])/D + b/D
        if norm(x[k+1] - x[k]) < tol:
            break
    
    return x[:k+1]

A = getMatrix(n=5, isDiagDom=False)
A = A/2
print(A.toarray())
x_true = np.random.randn(5)
b = A@x_true
x = jacobiIteration(A, b)

def plotConvergence(x_true, x, k=2, scale='log', rate=True):
    '''
    Plot the error ||x[i] - x_true||_k against i or the rate of the convergence
    x_true: (n,) vector
    x: (m, n) matrix, x[i] is the i-th iterate, 
    x[0] = x0 is the initial guess
    m: total number of iterations
    scale: 'log' or 'linear'
    k: 1,2,..., of Inf
    rate: bool (optional) If true, plot the rate of the convergence
    ||x[i+1] - x_true||_k/||x[i] - x_true||_k against
    '''
    numIter = x.shape[0]
    error = np.zeros(numIter)
    for i in range(numIter):
        error[i] = norm(x_true - x[i], k)
    if (scale is 'log') and (not rate):
        plt.semilogy(error)
    elif (scale is 'linear') and (not rate):
        plt.plot(error)
    elif (scale is 'log') and rate:
        plt.plot(np.log(error[1:]/error[:-1]))
    elif (scale is 'linear') and rate:
        plt.plot(error[1:]/error[:-1])

plotConvergence(x_true, x, k=2, scale='log', rate=True)


# %%
