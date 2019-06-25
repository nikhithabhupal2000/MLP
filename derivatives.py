import sys
import numpy

def sigmoid_derivative(self, value):
    return self.sigmoid(value) * (1 - self.sigmoid(value))
    
def tanh_derivative(self, value):
    return  1 - (self.tanh(value) ** 2)
    
<<<<<<< HEAD
def relu_derivative(self, value):     
     return max(sign(value), 0)
    
=======
def relu_derivative(self, value):
    if value >= 0 :
        return 1
    else :
        return 0
    
>>>>>>> 295c2ee8fb202d03605ca42b0d7736ad9e496035
