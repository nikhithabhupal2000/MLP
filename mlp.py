import numpy 

class mlp:
    def __init__(self , feature_matrix , output_matrix , number_of_neurons_in_each_layer , activation_function = "relu"):

        self.feature_matrix = feature_matrix
        self.output_matrix = output_matrix 
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_layer
        self.activation_function = activation_function
        self.number_of_hidden_layers = len(number_of_neurons_in_each_layer) - 2
        self.weights = []
        self.biases = []
        for i in range(self.number_of_hidden_layers + 1):
            temp_weights = numpy.matrix(numpy.random.rand(number_of_neurons_in_each_layer [ i ],number_of_neurons_in_each_layer [ i + 1]))
            temp_bias = numpy.matrix(numpy.random.rand( 1 , number_of_neurons_in_each_layer [ i + 1 ]))
            self.weights.append(temp_weights)
            self.biases.append(temp_bias)
        for i in self.weights:
            print(i)
        for i in self.biases:
            print(i)
f1 = numpy.array([1 , 2, 3])
f2 = numpy.array([1])
f3 = [3 , 2 , 1]
ob = mlp(f1 , f2 , f3)
 


