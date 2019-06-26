import numpy
class lib_functions:
    def sigmoid(value):
        return 1/(1+numpy.exp(-value))

    def tanh(value):
        return (1-numpy.exp(-2*value))/(1+numpy.exp(-2*value))

    def relu(value):
        return numpy.maximum(value,0)

    def sigmoid_derivative(value):
        return numpy.asmatrix(numpy.asarray(value) * numpy.asarray(1 - value))

    def tanh_derivative(value):
        return  1 - numpy.asmatrix(numpy.asarray(value) * numpy.asarray(value))
    

    def relu_derivative(value):
        value[value <= 0] = 0
        value[value > 0] = 1
        return value


class Mlp:
    def __init__(self , feature_matrix , output_matrix , number_of_neurons_in_each_layer , testing_samples ,activation_function = "relu" , loss_function = "mean_square_error"):
      
        self.feature_matrix = feature_matrix
        self.output_matrix = output_matrix 
        self.number_of_neurons_in_each_layer = number_of_neurons_in_each_layer
        self.activation_function = activation_function
        self.number_of_hidden_layers = len(number_of_neurons_in_each_layer) - 2
        self.testing_samples = testing_samples
        self.weights = []
        self.biases = []
        self.prediction = []
        self.loss_function = loss_function
        self.left_error = []
        self.solution = []
        self.learning_rate = 0.1
        for i in range(self.number_of_hidden_layers + 1):
            temp_weights = numpy.matrix(numpy.random.rand(number_of_neurons_in_each_layer [ i ],number_of_neurons_in_each_layer [ i + 1]))
            temp_bias = numpy.matrix(numpy.random.rand( 1 , number_of_neurons_in_each_layer [ i + 1 ]))
            self.weights.append(temp_weights)
            self.biases.append(temp_bias)

    def forward_propagation(self):
        self.prediction = []
        self.prediction.append(self.feature_matrix)
        for i in range(self.number_of_hidden_layers + 1):
             self.prediction.append(self.next_layer_input(self.prediction[i],self.weights[i],self.biases[i]))
    def predict(self):
        self.solution . append(self.testing_samples)
        for i in range(self.number_of_hidden_layers + 1):
            self.solution.append(self.next_layer_input(self.solution[i] , self.weights[i] , self.biases[i]))

        print(self.solution[-1])


    def next_layer_input(self,input_matrix,weight_matrix,bias_matrix):
        output_before_activation= numpy.add( numpy.matmul(input_matrix ,weight_matrix) , (bias_matrix))
        next_input = self.apply_activation_function(output_before_activation)
        return next_input


    def derivative(self,predicted_output):
        if self.activation_function.lower() == "sigmoid":
            return lib_functions.sigmoid_derivative(predicted_output)
        elif self.activation_function.lower() == "tanh":
            return lib_functions.tanh_derivative(predicted_output)
        else:
            return lib_functions.relu_derivative(predicted_output)

    def apply_activation_function(self,output_before_activation):
        if self.activation_function.lower() == "sigmoid":
            return lib_functions.sigmoid(output_before_activation)
        elif self.activation_function.lower() == "tanh":
            return lib_functions.tanh(output_before_activation)
        else:
            return lib_functions.relu(output_before_activation)


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

<<<<<<< HEAD
            
 


X=numpy.matrix([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
y=numpy.matrix([[1],[0],[1]])
f3 = [4,3,1]
test = numpy.matrix([[0,0,1,1],[1,1,0,0],[1,1,1,1]])
print(test)
ob = Mlp(X,y,f3,test, "sigmoid" , "absolute_error")
for i in range(100000):
=======
X=numpy.matrix([[1,3,1,2]])
y=numpy.matrix([[1]])
f3 = [4,3,1]
ob = Mlp(X,y,f3, "tanh" , "absolute_error")
for i in range(1000):
>>>>>>> e09c8fc5ea8faaf3aa46090f360d273004fe2eb3
    ob.forward_propagation()
    ob.backward_propagation()
    ob.update()
print(ob.prediction[-1])
<<<<<<< HEAD
ob.predict()
#test = numpy.matrix([[1,0,0,0],[0,0,1,1],[1,1,1,1],[1,0,1,0]])




        




        





=======
   
>>>>>>> e09c8fc5ea8faaf3aa46090f360d273004fe2eb3


