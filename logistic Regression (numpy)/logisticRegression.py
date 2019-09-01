import numpy as np

m = 1000
w = np.random.rand(100)
x = np.random.rand(100)
y = np.random.rand(100)


def sigmoid(x):
    expon = 1+np.exp(-x)
    return 1/expon


def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    loss = np.sum(np.abs(y-yhat))
    return loss


def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """
    loss = np.sum(np.dot(y-yhat, y-yhat))
    return loss


for iter in range(m):
    # NO way to get rid of this for loop
    z = np.dot(w.T, x)+b
    a = sigmoid(z)
    dz = a-y
    dw = 1/m*x*dz.T
    db = 1/m*np.sum(dz)
    b = 1/m*(np.sum(dz))
