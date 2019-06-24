import numpy
class Mlp:
    
    def forward_propagation(self):
        for i in range(self.number_of_hidden_layers):
            input_matrix = self.next_layer_input(input_matrix,self.weight_list[i],self.bias_list[i])
        return  input_matrix  

    
    def next_layer_input(self,input_matrix,weight_matrix_,bias_matrix):
        output_before_activation= numpy.add( numpy.dot(self.input_matrix ,weight_matrix) , (bias_matrix))
        next_input = self.apply_activation_function(output_before_activation)
	return next_input


    def apply_activation_function(self,output_before_activation):
        if self.activation_function.lower() == "sigmoid":
		return sigmoid(output_before_activation)
	elif self.activation_function.lower() == "tanh":
		return tanh(output_before_activation)
	else:
		return relu(output_before_activation)

    




        


        




        
