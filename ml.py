import numpy
class Mlp:
    
    def forward_propagation(self):
        for i in range(len(weight_list)):
            self.input_matrix = self.output_of_next_layer(self.input_matrix,self.weight_list[i],self.bias_list[i])
        return  self.input_matrix  

    
    def output_of_next_layer(self,input_matrix,weight_matrix_,bias_matrix):
        output_before_transformation = numpy.dot(self.input_matrix ,self.weight_matrix) + bias_matrix
        output_next = self.activation_function(output_before_transformation)


    def activation_function(self,output_before_trans):
        return 1/(1+pow(e,self.output_before_trans))
    




        


        




        
