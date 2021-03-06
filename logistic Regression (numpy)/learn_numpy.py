import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size
    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s


def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    Arguments:
    x -- A scalar or numpy array
    Return:
    ds -- Your computed gradient.
    """
    s = 1/(1+np.exp(-x))
    ds = s*(1-s)
    return ds


def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2]), 1)
    return v


def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x/x_norm
    return x


def softmax(x):
    """Calculates the softmax for each row of the input x.
    Your code should work for a row vector and also for matrices of shape (n, m).
    Argument:
    x -- A numpy matrix of shape (n,m)
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)
    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum
    return s


v = np.random.rand(100)
u = np.exp(v)
log = np.log(v)
absol = np.abs(v)
maxi = np.maximum(v, 0)
# print(v)
# print(u)
# print(log)
# print(absol)
# print(maxi)

x = np.array([1, 2, 3])
sigmoid(x)

x = np.array([1, 2, 3])
print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

image = np.array([[[0.67826139,  0.29380381],
                   [0.90714982,  0.52835647],
                   [0.4215251,  0.45017551]],

                  [[0.92814219,  0.96677647],
                   [0.85304703,  0.52351845],
                   [0.19981397,  0.27417313]],

                  [[0.60659855,  0.00533165],
                   [0.10820313,  0.49978937],
                   [0.34144279,  0.94630077]]])
print('image2vector(image) = \n' + str(image2vector(image)))

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0, 0]])
print("softmax(x) = " + str(softmax(x)))
