import numpy as np 

m=1000
w=np.random.rand(100)
# x=
# y=

def sigmoid(x):
    expon=1+np.exp(-x)
    return 1/expon

for iter in range(m):
    ### NO way to get rid of this for loop 
    z=np.dot(w.T,x)+b 
    a=sigmoid(z)
    dz=a-y
    dw=1/m*x*dz.T
    db=1/m*np.sum(dz)
    b=1/m*(np.sum(dz))