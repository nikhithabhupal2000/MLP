import numpy 
import math
def sigmoid(value):
    return 1/(1+numpy.exp(-value))


def tanh(value):
    return (1-numpy.exp(-2*value))/(1+numpy.exp(-2*value))


def relu(value):
    return max(0,value)

print(sigmoid(0))
print(tanh(0))
print(relu(3))
