import math
import numpy.linalg
import numpy as np

def shrink(X, tau):
    """
    Apply the shrinkage operator the the elements of X.
    Returns V such that V[i,j] = max(abs(X[i,j]) - tau,0).
    """
    V = np.copy(X).reshape(X.size)
    for i in range(V.size):
        V[i] = math.copysign(max(abs(V[i]) - tau, 0), V[i])
        if V[i] == -0:
            V[i] = 0
    return V.reshape(X.shape)

def frobeniusNorm(X):
    """
    Evaluate the Frobenius norm of X
    Returns sqrt(sum_i sum_j X[i,j] ^ 2)
    """
    accum = 0
    V = np.reshape(X,X.size)
    for i in range(V.size):
        accum += abs(V[i] ** 2)
    return math.sqrt(accum)

def L1Norm(X):
    """
    Evaluate the L1 norm of X
    Returns the max over the sum of each column of X
    """
    return max(np.sum(X,axis=0))


def converged(Y,W,X,E):
    """
    A simple test of convergence based on accuracy of matrix reconstruction
    from sparse and low rank parts
    """
    error = frobeniusNorm(Y - np.dot(W,X) - E) / frobeniusNorm(Y)

    print("error =", error)
    return error <= 5*10e-5

def run(X_list,Y_list):
    """
    """

    Y = Y_list[0]
    X = X_list[0]

    L = np.zeros(Y.shape)
    W = np.zeros([3,3])
    E = np.zeros(X.shape)

    mu = (Y.shape[0] * Y.shape[1]) / (4.0 * L1Norm(Y))
    lamb = max(Y.shape) ** -0.5

    print(mu)
    i = 1
    while not converged(Y,W,X,E):

        Y = Y_list[i]
        X = X_list[i]

        tmp = Y - E + L*(mu**-1)

        # print(tmp)

        W = np.dot(np.dot(tmp,np.transpose(X)),np.linalg.inv(np.dot(X,np.transpose(X))))
        W = -W
        #print(np.dot(Y,np.linalg.det(X)))

        #print(W)

        E = shrink(Y-np.dot(W,X)  + (mu**-1) * L, (mu)*lamb)

        #mu = max(mu * 0.98,mu*0.1)
        #print(mu)
        L = L - 1 * (Y - np.dot(W,X) - E)

        #print(mu)

        #i = (i+1) % X_list.__len__()

    return W,E

def GaussieNoisy(image, sigma):

    row, col = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy

if __name__ == '__main__':

    x_list = []
    y_list = []

    W = np.random.randint(0,10,size=[3,3]) / 255.0

    for i in range(10):
        x = np.random.randint(0, 255, size=[3, 1000])/255.0
        x_list.append(x)
        E = GaussieNoisy(x,0.1)

        y = np.dot(W,x) + E

        y_list.append(y)

    W_res, E_res = run(x_list,y_list)

    print(W)
    print(W_res)
    print(E_res)
