import numpy as np

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
standard_output = np.array([[0],[1],[1],[0]])   

epochs = 1
lr = 0.5
input_Layer, hidden_Layer, output_Layer = 2,2,1

hidden_weights = np.random.uniform(size=(input_Layer,hidden_Layer))
hidden_bias = np.random.uniform(size=(1,hidden_Layer))
output_weights = np.random.uniform(size=(hidden_Layer,output_Layer))
output_bias = np.random.uniform(size=(1,output_Layer))

def sigmoid (x):   
    return 1/(1 + np.exp(-x))

def sigmoid_derivative (x):
    return x * (1 - x)

hidden_net = np.zeros([4, 2])###input * hidden_weight
hidden_layer_output = np.zeros([4, 2])
output = np.zeros([4, 1])
predict_output = np.zeros([4, 1])
for t in range(epochs):
    ###forward
    for j in range(2):
        hidden_net[:, j] -= hidden_bias[0, j]
        for i in range(4):
            if i==0 or i==1:
                hidden_net[i, j] += inputs[i, j] * hidden_weights[i, j]
            else:
                hidden_net[i, j] += inputs[i, j] * hidden_weights[i-2, j]
               
        hidden_layer_output = sigmoid(hidden_net)      
    
    for k in range(1):
        output[k, :] = -output_bias[0, 0]
        for i in range(4):
            if i==0 or i==1:
                predict_output[i, k] += hidden_layer_output[i, k] * output_weights[i, k]
            else:
                predict_output[i, k] += hidden_layer_output[i, k] * output_weights[i-2, k]
               
        hidden_layer_output = sigmoid(hidden_net)     
        ###hidden layer n = 2
        #第一次隨機
        # n = 2
        # theta = np.zeros([n])
        # net = np.zeros([n])
        # H = np.zeros([n])
        # for j in range(n):
        #     theta[j]
        #     net[j] = theta[j] = -(np.random.randn())
        #     ###input layer m = 2
        #     m = 2
        #     for i in range(m):
        #         v = np.zeros([i, j])
        #         net[j] = net[j] + v[i, j] * d1 + v[i, j] * d2
        #     H[j] = sigmoid(net[j])
        #     ##output layer p = 1
        #     p = 1
        #     Y = np.zeros([p])
        # for k in range(p):
        #     net[k] = -theta[k]
        #     for j in range(n):
        #         w = np.zeros([j, k])
        #         net[k] = net + w[j][k] * H[j]
        #     Y[k] = sigmoid(net[k])
        # ###backward
        # delta = np.zeros([p])
        # delta_h = np.zeros([n])
        # lr = 0.10
        # for k in range(p):
        #     delta[k] = Y[k] * (1 - Y[k]) * (label[p])
        # for j in range(n):
        #     delta_h[n] = 0
        #     for k in range(p):
        #         delta_h[j] = delta_h[j] + w[j, k] * lr * delta[k]
        # delta_h = H[j] * (1 - H[j]) * dleta_h[j]
        # big_theta = np.zeros([p])
        # big_theta_h = np.zeros([n])
        # ##adjust
        # for k in range(p):
        #     theta[k] = theta[k] - lr * delta[k]
        #     for j in range(n):
        #         w[j, k] = w[j, k] + lr * delta[k] + H[j]
            
        # for j in range(n):
        #     big_theta_h[j] = big_theta_h[j] - lr * delta_h[j]
        #     for i in range(m):
        #         v[i, j] = v[i, j] + lr * delta_h[j] * d1[i] + lr * delta_h[j] * d2[i]
                
# nn(2, data)