import numpy 
import math

def sigmoid(value):
    return 1 / (1 + numpy.exp(- value))

def tanh(value):
    return tanh(value)

def relu(value):
    return numpy.maximum(value,0)

