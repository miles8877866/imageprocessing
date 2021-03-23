import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [-1, 1],
    [1, -1],
    [1, 1],
    [-1, -1]
])
y = np.array([
    [1],
    [1],
    [0],
    [0]
])

m = 2
n = 2
p = 1

lr = 0.1 # 0.001, 0.01 <- Magic values
max_iter = 1 # 5000 <- Magic value
# The model needs to be over fit to make predictions. Which 
W1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
W2 = np.array([[1.0, 1.0]])
B1 = np.array([[1.0], [1.0]])
B2 = np.array([[1.0]])

def sigmoid(z, derv=False):
    if derv: return z * (1 - z)
    return 1 / (1 + np.exp(-z))

def forward(x, predict=False):
    a1 = x.reshape(x.shape[0], 1) # Getting the training example as a column vector.

    z2 = W1.dot(a1) + B1 # 2x2 * 2x1 + 2x1 = 2x1
    a2 = sigmoid(z2) # 2x1

    z3 = W2.dot(a2) + B2 # 1x2 * 2x1 + 1x1 = 1x1
    a3 = sigmoid(z3)

    if predict: return a3
    return (a1, a2, a3)

dW1 = 0
dW2 = 0

dB1 = 0
dB2 = 0

cost = np.zeros((max_iter, 1))
for i in range(max_iter):
    # c = 0

    dW1 = 0
    dW2 = 0

    dB1 = 0
    dB2 = 0
    for j in range(4):

        # Forward Prop.
        a0 = X[j].reshape(X[j].shape[0], 1) # 2x1
        # print(*a0)
        #乘上hidden_weight再加上hidden_bias
        #z1:hidden_net
        z1 = W1.dot(a0) - B1 # 2x2 * 2x1 + 2x1 = 2x1
        # print(*z1)
        # 求neroun的net(3,4)值，做sigmoid
        a1 = sigmoid(z1) # 2x1
        # print(*a1)
        
        # 作第二層，到output,net5
        z2 = W2.dot(a1) - B2 # 1x2 * 2x1 + 1x1 = 1x1
        # print(*z2)
        # Y5:predict output
        a2 = sigmoid(z2) # 1x1 #predict
        # print(*a2)
        # Backward
        # delta3&4
        dz2 = y[j] - a2 # 1x1(T - Y)
        #
        print(dz2)
        dW2 += dz2 * a1.T # 1x1 .* 1x2 = 1x2
        # print(*dW2)
        # delta5:Y * (1-Y) is Y's sigmoid-derv
        dz1 = np.multiply( dz2, sigmoid(a2, derv=True)) # (2x1 * 1x1) .* 2x1 = 2x1
        print(*dz1)
        dW1 += dz1.dot(a0.T) # 2x1 * 1x2 = 2x2
        # print(*dW1)
        dB1 += dz1 # 2x1
        dB2 += dz2 # 1x1
        # print(*dB1)
        # print(*dB2)
        # c = c + (-(y[j] * np.log(a2)) - ((1 - y[j]) * np.log(1 - a2)))
        # print(*c)
    W1 = W1 - lr * (dW1 / m) 
    # print(*W1)
    # + ( (reg_param / m) * W1)
    W2 = W2 - lr * (dW2 / m) 
    # print(*W2)
    # + ( (reg_param / m) * W2)
    
    #求最後的B1和B2
    B1 = B1 - lr * (dB1 / m)
    B2 = B2 - lr * (dB2 / m)
    # cost[i] = (c / m) + ( 
    #     (reg_param / (2 * m)) * 
    #     (
    #         np.sum(np.power(W1, 2)) + 
    #         np.sum(np.power(W2, 2))
    #     )
    # )
    # print(cost[i])


# for x in X:
#     print("\n")
#     print(x)
#     print(forward(x, predict=True))

# plt.plot(range(max_iter), cost)
# plt.xlabel("Iterations")
# plt.ylabel("Cost")
# plt.show()