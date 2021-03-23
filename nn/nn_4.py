# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np

inputs = np.array([[-1, -1],[-1, 1],[1, -1],[1, 1]])
standard_output = np.array([[0],[1],[1],[0]])   

t = 1
lr = 10.0
input_Layer, hidden_Layer, output_Layer = 2,2,1

w1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
b1 = np.array([[1.0, 1.0]])
w2 = np.array([[1.0], [1.0]])
b2 = np.array([[1.0]])
#w1 = np.random.uniform(size=(input_Layer,hidden_Layer))
# b1 =np.random.uniform(size=(1,hidden_Layer))
# w2 = np.random.uniform(size=(hidden_Layer,output_Layer))
# b2 = np.random.uniform(size=(1,output_Layer))

# print("\nInitial hidden weights: ")
# print(w1)
# print("\nInitial hidden biases: ")
# print(b1)
# print("\nInitial output weights: ")
# print(w2)
# print("\nInitial output biases: ")
# print(b2)

def sigmoid (x):   
    return 1/(1 + np.exp(-x))

def sigmoid_derivative (x):
    return x * (1 - x)

for i in range(t):
	#Forward 
    hidden_net = np.dot(inputs,w1)
    hidden_net -= b1
    # print(hidden_net)##net
    hidden_layer_output = sigmoid(hidden_net)
    # print(hidden_layer_output)##H
    
    output = np.dot(hidden_layer_output,w2)
    # print(output)
    output -= b2
    # print(output)##net5
    predicted_output = sigmoid(output)
    # print(predicted_output)##Y

 	#Backward
    delta = standard_output - predicted_output
    d_predicted_output = delta * sigmoid_derivative(predicted_output)
    # print(d_predicted_output)##delta5
    
    error_hidden_layer = d_predicted_output.dot(w2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    # print(d_hidden_layer)##delta3 & 4
    
 	#adjust parameters
    w2 += hidden_layer_output.T.dot(d_predicted_output) * lr 
    # print(*w2 )
    b2 -= np.sum(d_predicted_output * lr) 
    print(*b2)
    w1 += inputs.T.dot(d_hidden_layer) * lr
    # print(*w1)
    b1 -= (np.sum(d_hidden_layer) * lr)/ 2
    # print(*b1)
    
# print(predicted_output)
# print("\nOutput hidden weights: ")
# print(w1)
# print("\nOutput hidden biases: ")
# print(b1)
# print("\nOutput output weights: ")
# print(w2)
# print("\nOutput output biases: ")
# print(b2)


