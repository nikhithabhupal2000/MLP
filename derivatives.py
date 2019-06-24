def sigmoid_derivative(self, value):
    return self.sigmoid(value) * (1 - self.sigmoid(value))
    
def tanh_derivative(self, value):
    return  1 - self.tanh(value) * self.tanh(value)
    
def relu_derivative(self, value):     
     return max(sign(value), 0)
    