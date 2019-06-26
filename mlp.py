import numpy 

class Mlp:
    def __init__(self , feature_matrix , output_matrix , number_of_neurons_in_each_layer , activation_function = "relu" , loss_function = "mean_square_error"):
      
        self.feature_matrix = feature_matrix
        self.output_matrix = output_matrix 
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_layer
        self.activation_function = activation_function
        self.number_of_hidden_layers = len(number_of_neurons_in_each_layer) - 2
        self.weights = []
        self.biases = []
        self.prediction = []
        self.loss_function = loss_function
        self.left_error = []
        self.learning_rate = 0.1
        for i in range(self.number_of_hidden_layers + 1):
            temp_weights = numpy.matrix(numpy.random.rand(number_of_neurons_in_each_layer [ i ],number_of_neurons_in_each_layer [ i + 1]))
            temp_bias = numpy.matrix(numpy.random.rand( 1 , number_of_neurons_in_each_layer [ i + 1 ]))
            self.weights.append(temp_weights)
            self.biases.append(temp_bias)
    def sigmoid(self,value):
        return 1/(1+numpy.exp(-value))


    def tanh(self,value):
        return (1-numpy.exp(-2*value))/(1+numpy.exp(-2*value))


    def relu(self,value):
        return numpy.maximum(value,0)
    

    def sigmoid_derivative(self, value):
       # return value * (1-value)
        return numpy.asmatrix(numpy.asarray(value) * numpy.asarray(1 - value))
    
    def tanh_derivative(self, value):
        return  1 - (self.tanh(value) ** 2)
    
    def relu_derivative(self, value):
        if value >= 0 :
            return 1
        else:
            return 0


    def forward_propagation(self):
        self.prediction = []
        self.prediction.append(self.feature_matrix)
        for i in range(self.number_of_hidden_layers + 1):
             self.prediction.append(self.next_layer_input(self.prediction[i],self.weights[i],self.biases[i]))

    def next_layer_input(self,input_matrix,weight_matrix,bias_matrix):
        output_before_activation= numpy.add( numpy.matmul(input_matrix ,weight_matrix) , (bias_matrix))
        next_input = self.apply_activation_function(output_before_activation)
        return next_input


    def derivative(self,predicted_output):
        if self.activation_function.lower() == "sigmoid":
            return self.sigmoid_derivative(predicted_output)
        elif self.activation_function.lower() == "tanh":
            return self.tanh_derivative(predicted_output)
        else:
            return self.relu_derivative(predicted_output)

    def apply_activation_function(self,output_before_activation):
        if self.activation_function.lower() == "sigmoid":
            return self.sigmoid(output_before_activation)
        elif self.activation_function.lower() == "tanh":
            return self.tanh(output_before_activation)
        else:
            return self.relu(output_before_activation)


    def absolute_error(self,prediction , expected):
        #print(numpy.subtract(prediction , expected))
        return numpy.subtract(prediction , expected)
	

    def mean_square_error(self, prediction , expected):
        return numpy.square(numpy.subtract(prediction , expected))/2

    def apply_loss_function(self):
        if self.loss_function == "absolute_error":
        
            return self.absolute_error(self.output_matrix, self.prediction[-1])
        else :
            
            return self.mean_square_error(self.output_matrix, self.prediction[-1])
        
    def propagate_error(self , i):
        return numpy.matmul(self.left_error[-1],numpy.transpose(self.weights[self.number_of_hidden_layers-i]))
        

    def backward_propagation(self):
        self.left_error = []
        self.left_error.append(numpy.asmatrix(numpy.asarray(self.derivative(self.prediction[-1]))*numpy.asarray( self.apply_loss_function())))
        
        
        for i in range(self.number_of_hidden_layers ):        
            self.left_error.append(numpy.asmatrix(numpy.asarray(self.derivative(self.prediction[self.number_of_hidden_layers - i ]) )* numpy.asarray( self.propagate_error(i))))
        
    
    def update_weights(self):
        for i in range(self.number_of_hidden_layers  , -1 , -1):
            self.weights[i] = self.weights[i] + numpy.matmul(numpy.transpose(self.prediction[i]) , self.left_error[self.number_of_hidden_layers - i])*self.learning_rate 

    def update_biases(self):
        for i in range(self.number_of_hidden_layers , -1 , -1):
            self.biases[i] = numpy.add(self.biases[i] , self.left_error[self.number_of_hidden_layers -i])*self.learning_rate 

    def update(self):
        self.update_weights()
        self.update_biases()

X=numpy.matrix([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
y=numpy.matrix([[1],[0],[1]])
f3 = [4,3,1]
ob = Mlp(X,y,f3, "sigmoid" , "absolute_error")
for i in range(100000):
    ob.forward_propagation()
    ob.backward_propagation()
    ob.update()
"""
print("input_matrix")
print(ob.feature_matrix)
print("output_matrix")
print(ob.output_matrix)
print("weights")
print(ob.weights)
print("biases")
print(ob.biases)
print("predictions")
print(ob.prediction)
print("left_error")
print(ob.left_error)"""
print(ob.prediction[-1])
    


        




        






