import numpy 

class Mlp:
    def __init__(self , feature_matrix , output_matrix , number_of_neurons_in_each_layer , activation_function = "relu"):
    
        self.feature_matrix = feature_matrix
        self.output_matrix = output_matrix 
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_layer
        self.activation_function = activation_function
        self.number_of_hidden_layers = len(number_of_neurons_in_each_layer) - 2
        self.weights = []
        self.biases = []
        self.prediction = []
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
        return self.sigmoid(value) * (1 - self.sigmoid(value))
    
    def tanh_derivative(self, value):
        return  1 - (self.tanh(value) ** 2)
    
    def relu_derivative(self, value):
        if value >= 0 :
            return 1
        else:
            return 0


    def forward_propagation(self):
        self.prediction.append(self.feature_matrix)
        for i in range(self.number_of_hidden_layers + 1):
             self.prediction.append(self.next_layer_input(self.prediction[i],self.weights[i],self.biases[i]))

    def next_layer_input(self,input_matrix,weight_matrix,bias_matrix):
        output_before_activation= numpy.add( numpy.dot(input_matrix ,weight_matrix) , (bias_matrix))
        next_input = self.apply_activation_function(output_before_activation)
        return next_input


    def derivative(self,error):
        if self.activation_function.lower() == "sigmoid":
            return self.sigmoid_derivative(error)
        elif self.activation_function.lower() == "tanh":
            return self.tanh_derivative(error)
        else:
            return self.relu_derivative(error)

    def apply_activation_function(self,output_before_activation):
        if self.activation_function.lower() == "sigmoid":
            return self.sigmoid(output_before_activation)
        elif self.activation_function.lower() == "tanh":
            return self.tanh(output_before_activation)
        else:
            return self.relu(output_before_activation)
	


    def mean_square_error(self, prediction , expected):
        return numpy.square(numpy.subtract(prediction , expected))/2

    def update(self ,error , nth_layer):
        self.weights[nth_layer] = self.weights[nth_layer] - self.learning_rate * self.derivative(error)

    def backward_propagation(self):
        for i in range(self.number_of_hidden_layers + 1):
            error = self.mean_square_error(self.prediction[self.number_of_hidden_layers + 1 - i ] , self.prediction[self.number_of_hidden_layers  - i])
            self.update(error , self.number_of_hidden_layers + 1 - i)
        for i in self.weights:
            print(i)


f1 = numpy.array([1 , 2, 3])
f2 = numpy.array([1])
f3 = [3 , 2 , 1]
ob = Mlp(f1 , f2 , f3 , "sigmoid")
ob.forward_propagation()
ob.backward_propagation()
 
    




        


        




        



