# -*- coding: utf-8 -*-
import numpy as np

inputs = np.array([[-1, -1],[-1, 1],[1, -1],[1, 1]])
standard_output = np.array([[0],[1],[1],[0]])   

t = 1
lr = 10.0
input_Layer, hidden_Layer, output_Layer = 2,2,1

hidden_weights = np.array([[1.0, -1.0], [-1.0, 1.0]])
hidden_bias = np.array([[1.0, 1.0]])
output_weights = np.array([[1.0], [1.0]])
output_bias = np.array([[1.0]])
#hidden_weights = np.random.uniform(size=(input_Layer,hidden_Layer))
# hidden_bias =np.random.uniform(size=(1,hidden_Layer))
# output_weights = np.random.uniform(size=(hidden_Layer,output_Layer))
# output_bias = np.random.uniform(size=(1,output_Layer))

# print("\nInitial hidden weights: ")
# print(hidden_weights)
# print("\nInitial hidden biases: ")
# print(hidden_bias)
# print("\nInitial output weights: ")
# print(output_weights)
# print("\nInitial output biases: ")
# print(output_bias)

def sigmoid (x):   
    return 1/(1 + np.exp(-x))

def sigmoid_derivative (x):
    return x * (1 - x)

for i in range(t):
	#Forward 
    hidden_net = np.dot(inputs,hidden_weights)
    hidden_net -= hidden_bias
    # print(hidden_net)##net
    hidden_layer_output = sigmoid(hidden_net)
    # print(hidden_layer_output)##H
    
    output = np.dot(hidden_layer_output,output_weights)
    # print(output)
    output -= output_bias
    # print(output)##net5
    predicted_output = sigmoid(output)
    # print(predicted_output)##Y

 	#Backward
    delta = standard_output - predicted_output
    d_predicted_output = delta * sigmoid_derivative(predicted_output)
    # print(d_predicted_output)##delta5
    
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    # print(d_hidden_layer)##delta3 & 4
    
 	#adjust parameters
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr 
    # print(*output_weights )
    output_bias -= np.sum(d_predicted_output * lr) 
    print(*output_bias)
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    # print(*hidden_weights)
    hidden_bias -= (np.sum(d_hidden_layer) * lr)/ 2
    # print(*hidden_bias)
    
# print(predicted_output)
# print("\nOutput hidden weights: ")
# print(hidden_weights)
# print("\nOutput hidden biases: ")
# print(hidden_bias)
# print("\nOutput output weights: ")
# print(output_weights)
# print("\nOutput output biases: ")
# print(output_bias)

