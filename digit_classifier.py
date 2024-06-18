# Model Network that Computes whether a digit is an 8 digit number or not
# Input 16x16 region image Output: Confidence Score: How confident the network is in the number being classified as an 8


# Problem Taxonomy - Decision Problem.
# "Procedure" / Framework -> factor 1: is there a loop? factor 2: are there 2 loops?
# Why is it reasonable to even have a weights and bias framework?



# Question 1 - If I wanted to create a model that classified if a number was an 8, how would I go about that?
# - I am building this "thing" that does some sort of task. 

# - need for subproblems / subcomponents as a way for simplifying a task. 
    #     --> Why are "factors" important?: difficulty to observe a solution at its whole, only hope is to divide it into manageable subproblems, 
    # that one can solve. (I think about inputs, as really just being solved problems because one observes it already, and now has to be built into a more complex solution)
    # things that one must consider.


# Emphasis on Certain Factors that are relevant to answering the question.

# problems: - problems associated when activaiton function yields a number to 0, sometimes there are like dying neurons, that dont change weightx


import numpy as np
from math import tanh 
    

# TODO: Figure Out How I can tune weights that are not attatched
# Created a mask that allows custom connections and is computed 
class Layer():
    
    def __init__(self, num_incoming_input = 4, num_nodes = 4): 
        self._weights = np.zeros((num_nodes, num_incoming_input))
        self.initialize_weights()

        self._weights_mask = np.ones((num_nodes, num_incoming_input))
        self._biases = np.ones((num_nodes, 1)) * 0.01
        self.activation = np.tanh
        self._shape = (num_nodes, num_incoming_input)


        # Stores most recently used linear combination
        self._linear_combination_cache = -1

    def initialize_weights(self):
        n_in = self._weights.shape[1]
        n_out = self._weights.shape[0]
        std_dev = np.sqrt(2.0 / (n_in + n_out))
        self._weights = np.random.normal(0, std_dev, size=self._weights.shape)


    def set_weights(self, weights):

        self._weights = weights

    def get_weights(self): 
        return self._weights
    def get_weights_mask(self): 
        return self._weights_mask
    def get_shape(self):
        return self._shape
    def get_linear_combination(self):
        return self._linear_combination_cache
    def set_connections(self, mask): 
        self._weights_mask = mask

    

    def forward(self, X):
        new_weights = self._weights * self._weights_mask
        self._linear_combination_cache = (new_weights @ X) + self._biases
        return self.activation(self._linear_combination_cache)
    

# Create a fully connected neural network
class Network():

    def __init__(self, layer_nodes, masks = None):
        self._layers = []
        self.input_size = layer_nodes[0]
        self.output_size = layer_nodes[len(layer_nodes) - 1]


        # stores the recent activations from the recent forward() computation, for gradient purposes.
        self._activation_cache = []
        self._linear_combination_cache = [] 
        self._weight_gradient = []
        self._bias_gradient = []
        self._y_linear_combination_gradient_cache = []

        self._learning_rate = 0.001

        for i in range(1, len(layer_nodes)):
            num_weight_nodes = layer_nodes[i]
            num_input_nodes = layer_nodes[i - 1]
            self._layers.append(Layer(num_input_nodes, num_weight_nodes))

        if masks is None:
            for layer in self._layers: 
                layer.set_connections(np.ones(layer.get_shape()))

    def forward(self, X):
        self._activation_cache = [X]

        for layer in self._layers:
            X = layer.forward(X)
            self._activation_cache.append(X)
        return X; 


    def mse_cost_function(self, y_true):
  
        mse = np.mean((y_true - self._activation_cache[-1]) ** 2)
        return mse
    

    def tanh_derivative(self, input):
        return 1 - np.tanh(input)**2

    def compute_gradient(self, y_true):

        
        for i in range(len(self._layers) - 1, -1, -1):
            #print(f"computing gradient for layer {i}")
            last_layer = self._layers[i]
            y_pred = self._activation_cache[i + 1] # +1 omre activaiton nodes than layers


            #print(y_pred.shape)
            # Change in the cost function in reference to the cost, change to if statement on the last layer. 
            if i == (len(self._layers) - 1):
                y_pred_gradient = 2 / self.output_size * (y_pred - y_true)
                ##print(f"cost function gradient {y_pred_gradient}")
                
                #print(f"h {y_pred_gradient.shape}")
            else: 
                
                size = transposed_activations.shape[0]
                lc_gradient_mask = np.tile(self._y_linear_combination_gradient_cache, (1, size))
                ##print(f"{i} gradient mask: {lc_gradient_mask}")
        
                y_pred_gradient = self._layers[i + 1].get_weights() * lc_gradient_mask# 0 represents the most recently inserted element
                # TODO: investigate change in this configuration. 
                # we shoudl be using the weights themselves as the rate of change



                y_pred_gradient = np.sum(y_pred_gradient, axis=0)

            
                y_pred_gradient = y_pred_gradient.reshape(1, -1) # add a dimension for processing ()

                
                y_pred_gradient = np.transpose(y_pred_gradient)

                ##print(f"{i} y_pred_gradient: {y_pred_gradient}")

                

           
            

            # Activations are what the network knows, but only the layer knows the linear combinations + weights
            y_linear_combination_gradient = self.tanh_derivative(last_layer.get_linear_combination()) * y_pred_gradient

            #print(f"{i} dtanh / dlc gradient {y_linear_combination_gradient.shape}")
            ##print(f"{i}last layer lc {last_layer.get_linear_combination()}")

            

            self._y_linear_combination_gradient_cache = y_linear_combination_gradient
        
            transposed_activations = np.transpose(self._activation_cache[i])  # layer i has input in activation cache i and output cache i + 1


            output_size = len(self._activation_cache[i + 1]) 
            # really want to represent the number of outputs in this layer, as each layer, will have weights whose gradient is the activation
            #print(f"{i} input size: {transposed_activations.shape} output size {output_size}")



            weight_matrix_local_gradient = np.tile(transposed_activations, (output_size, 1)) * last_layer.get_weights_mask()


            # given the current layer transposed_activations, first index finds the number of inputs to the network for us to extned.
            input_size = transposed_activations.shape[1] # weird thing done with the shape why is it that i use 0 and 1?


            # size from linear combination seems to be obtained as the number of outputs; when definnig size, we actually want to
            # define the number of inputs we have to the sequence as a mask. (where could you find the number of inputs?)




            # sometimes i take teh shape of 1, and the shape of 0
            # when should i use the shape 


            #print(f"{i} {y_linear_combination_gradient.shape} weight_matrix local gradient{weight_matrix_local_gradient.shape}")

            #print(f"y_linear combination shape: {y_linear_combination_gradient.shape}, input_size: {input_size}")
            previous_gradient_mask = np.tile(y_linear_combination_gradient, (1, input_size))
            weight_matrix_cost_gradient = weight_matrix_local_gradient * previous_gradient_mask

            #print(f"{i} weight matrix cost: {weight_matrix_cost_gradient.shape}")


        



            bias_matrix_cost_gradient = y_linear_combination_gradient
            self._weight_gradient.insert(0, weight_matrix_cost_gradient)
            self._bias_gradient.insert(0, bias_matrix_cost_gradient)
        
        
        
        # find values of the activation functions from the previous layer, construct a weight matrix, with columns coresponding to specific
        # elementwise multiply the mask
        
       ##print(self._weight_gradient)
        return (self._weight_gradient, self._bias_gradient)

        
    def step(self):
        # update the weights and values.

        

        for layer, weight_gradient, bias_gradient  in zip(self._layers, self._weight_gradient, self._bias_gradient): # use a zip function to align the gradients
        # TODO: create an update function or interface for the layer to have an interface.

            layer._weights -= weight_gradient * self._learning_rate
            layer._biases -= bias_gradient * self._learning_rate


            #print(f"layer: {layer.get_weights()} update gradient: {weight_gradient}")

        return 0
    
    def train_1_data(self, X, y):
        self.forward(X)
        self.compute_gradient(y)
        self.step()






    

        
        


    # review how do I change my process for last time to learn for next time?
    # mse = 1/n (y_true - tanh(activation))^2






    

   
        

    

