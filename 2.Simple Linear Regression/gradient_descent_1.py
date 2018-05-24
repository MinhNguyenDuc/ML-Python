import numpy as np
import math
import matplotlib.pyplot as plt

def grad(x):
    return 2*x + 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(learning_rate, x0):
    x = [x0]
    for i in range(100):
        x_new = x[-1] - learning_rate*grad(x[-1])
        if(abs(grad(x_new)))  < 1e-3:
            break
        x.append(x_new)
    return (x, i)


def myGD2(learning_rate, x):
    for i in range(100):
        x = x - learning_rate*grad(x)
        if(abs(grad(x)) < 1e-3):
            break
    return (x, i)

(x1, i1) = myGD1(.1, -5)
(x2, i2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), i1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), i2))

(x3, i3) = myGD2(.1, -5)
(x4, i4) = myGD2(.1, 5)
print('Solution x1 = %f, cost = %f obtained after %d iterations'%(x3, cost(x3) ,i3))
print('Solution x2 = %f, cost = %f obtained after %d iterations'%(x4, cost(x4),i4))


# Solution x1 = -1.110667, cost = -3.246394, obtained after 11 iterations
# Solution x2 = -1.110341, cost = -3.246394, obtained after 29 iterations

