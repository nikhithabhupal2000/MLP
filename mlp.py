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
        return max(0,value)
    

    def forward_propagation(self):
        input_matrix = self.feature_matrix
        for i in range(self.number_of_hidden_layers):
             input_matrix = self.next_layer_input(input_matrix,self.weights[i],self.biases[i])
        return  input_matrix


    def next_layer_input(self,input_matrix,weight_matrix,bias_matrix):
        output_before_activation= numpy.add( numpy.dot(input_matrix ,weight_matrix) , (bias_matrix))
        next_input = self.apply_activation_function(output_before_activation)
        return next_input


    def apply_activation_function(self,output_before_activation):
        if self.activation_function.lower() == "sigmoid":
            return self.sigmoid(output_before_activation)
        elif self.activation_function.lower() == "tanh":
            return self.tanh(output_before_activation)
        else:
            return self.relu(output_before_activation)
f1 = numpy.array([1 , 2, 3])
f2 = numpy.array([1])
f3 = [3 , 2 , 1]
ob = Mlp(f1 , f2 , f3 , "sigmoid")
print(ob.forward_propagation())

 
    




        


        




        


