import numpy
import pandas


class Activation_functions:
    def sigmoid(value):
        return 1 / (1 + numpy.exp(-value))

    def tanh(value):
        return (1 - numpy.exp(-2 * value)) / (1 + numpy.exp(-2 * value))

    def relu(value):
        return numpy.maximum(value, 0)

    def sigmoid_derivative(value):
        return numpy.asmatrix(numpy.asarray(value) * numpy.asarray(1 - value))

    def tanh_derivative(value):
        return  1 - numpy.asmatrix(numpy.asarray(value) * numpy.asarray(value))
    
    def relu_derivative(value):
        value[value <= 0] = 0
        value[value > 0] = 1
        return value






class Mlp:
    def __init__(self, feature_matrix, output_matrix, number_of_neurons_in_each_layer, testing_samples , weights, activation_function = "relu", loss_function = "mean_square_error"):  
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
        self.solution.append(self.testing_samples)
        self.learning_rate = 0.1
        self.nums_output = []
        for i in range(self.number_of_hidden_layers + 1):
            temp_bias = numpy.matrix(numpy.random.rand( 1 , number_of_neurons_in_each_layer [ i + 1]))
            self.biases.append(temp_bias)

        if weights != []:
            self.weights = weights
        else:
            for i in range(self.number_of_hidden_layers + 1):
                temp_weights = numpy.matrix(numpy.random.rand(number_of_neurons_in_each_layer [ i ],number_of_neurons_in_each_layer[ i + 1]))
                self.weights.append(temp_weights)
        




    def forward_propagation(self):
        self.prediction = []
        self.prediction.append(self.feature_matrix)
        for i in range(self.number_of_hidden_layers + 1):
             self.prediction.append(self.next_layer_prediction(self.prediction[i], self.weights[i], self.biases[i]))



    def predict(self):
        for i in range(self.number_of_hidden_layers + 1):
            self.solution.append(self.next_layer_prediction(self.solution[i], self.weights[i], self.biases[i]))
        #print(self.solution[-1])
        for i in range(len(self.solution)):
            index_max = numpy.unravel_index(numpy.argmax(self.solution[i], axis = None), self.solution[i].shape)[1]
            self.nums_output.append(index_max)
        print(self.nums_output)   



    def next_layer_prediction(self, input_matrix, weight_matrix, bias_matrix):
        prediction_before_activation = numpy.add(numpy.matmul(input_matrix, weight_matrix), (bias_matrix))
        return  self.apply_activation_function(prediction_before_activation)
        



    def derivative(self, predicted_output):
        if self.activation_function.lower() == "sigmoid":
            return Activation_functions.sigmoid_derivative(predicted_output)
        elif self.activation_function.lower() == "tanh":
            return Activation_functions.tanh_derivative(predicted_output)
        else:
            return Activation_functions.relu_derivative(predicted_output)




    def apply_activation_function(self, prediction_before_activation):
        if self.activation_function.lower() == "sigmoid":
            return Activation_functions.sigmoid(prediction_before_activation)
        elif self.activation_function.lower() == "tanh":
            return Activation_functions.tanh(prediction_before_activation)
        else:
            return Activation_functions.relu(prediction_before_activation)




    def absolute_error(self, prediction, expected):
        return numpy.subtract(prediction, expected)
	

    def mean_square_error(self, prediction, expected):
        return numpy.square(numpy.subtract(prediction, expected)) / 2



    def apply_loss_function(self):
        if self.loss_function == "absolute_error":
            return self.absolute_error(self.output_matrix, self.prediction[-1])
        else :
            
            return self.mean_square_error(self.output_matrix, self.prediction[-1])



    def propagate_error(self, i):
        return numpy.matmul(self.left_error[-1], numpy.transpose(self.weights[self.number_of_hidden_layers - i]))
        


    def backward_propagation(self):
        self.left_error = []
        self.left_error.append(numpy.asmatrix(numpy.asarray(self.derivative(self.prediction[-1])) * numpy.asarray(self.apply_loss_function())))
        for i in range(self.number_of_hidden_layers):        
            self.left_error.append(numpy.asmatrix(numpy.asarray(self.derivative(self.prediction[self.number_of_hidden_layers - i])) * numpy.asarray(self.propagate_error(i))))
        
    


    def update_weights(self):
        for i in range(self.number_of_hidden_layers, -1, -1):
            self.weights[i] = self.weights[i] + numpy.matmul(numpy.transpose(self.prediction[i]), self.left_error[self.number_of_hidden_layers - i]) * self.learning_rate 



    def update_biases(self):
        for i in range(self.number_of_hidden_layers, -1, -1):
            self.biases[i] = numpy.add(self.biases[i], self.left_error[self.number_of_hidden_layers - i]) * self.learning_rate 




    def update(self):
        self.update_weights()
        self.update_biases()



def make_output_matrix(array):
    matrixl = []
    for i in range(len(array)):
        tmp_matrix = [0,0,0,0,0,0,0,0,0,0]
        tmp_matrix[array[i]] = 1
        matrixl.append(tmp_matrix)
    return numpy.matrix(matrixl)


train_data_object = pandas.read_csv("fashion-mnist_train.csv")
input_data = train_data_object.drop('label', axis = 1)
input_data = input_data.head()
features_matrix = input_data.values
actual_output = make_output_matrix(train_data_object['label'].head(5))
perceptrons_in_each_layer = [features_matrix.shape[1], 2, 10]
test_data_object = pandas.read_csv("fashion-mnist_test.csv")
testing_sample = (test_data_object.drop( 'label' ,axis = 1))
testing_sample = testing_sample.values
ob = Mlp(features_matrix, actual_output, perceptrons_in_each_layer, testing_sample, [], "sigmoid" , "absolute_error")



for i in range(1000):
    ob.forward_propagation()
    ob.backward_propagation()
    ob.update()


ob.predict()






        






