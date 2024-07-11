import numpy as np
import matplotlib.pyplot as plt

from tools import generate_linear, generate_XOR_easy, sigmoid, derivative_sigmoid

def initialize_weights(input_size, hidden_size1, hidden_size2, output_size):
    weights = {
        'W1': np.random.randn(input_size, hidden_size1),
        'b1': np.zeros((1, hidden_size1)),
        'W2': np.random.randn(hidden_size1, hidden_size2),
        'b2': np.zeros((1, hidden_size2)),
        'W3': np.random.randn(hidden_size2, output_size),
        'b3': np.zeros((1, output_size))
    }
    return weights

def forward_propagation(X, weights):
    Z1 = np.dot(X, weights['W1']) + weights['b1']
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, weights['W2']) + weights['b2']
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, weights['W3']) + weights['b3']
    A3 = sigmoid(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def back_propagation(X, y, weights, Z1, A1, Z2, A2, Z3, A3, learning_rate):
    m = y.shape[0]
    
    dZ3 = A3 - y
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m
    
    dA2 = np.dot(dZ3, weights['W3'].T)
    dZ2 = dA2 * derivative_sigmoid(A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, weights['W2'].T)
    dZ1 = dA1 * derivative_sigmoid(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    weights['W3'] -= learning_rate * dW3
    weights['b3'] -= learning_rate * db3
    weights['W2'] -= learning_rate * dW2
    weights['b2'] -= learning_rate * db2
    weights['W1'] -= learning_rate * dW1
    weights['b1'] -= learning_rate * db1

def train(X, y, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs):
    weights = initialize_weights(input_size, hidden_size1, hidden_size2, output_size)
    
    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, weights)
        loss = compute_loss(y, A3)
        back_propagation(X, y, weights, Z1, A1, Z2, A2, Z3, A3, learning_rate)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return weights

def predict(X, weights):
    _, _, _, _, _, A3 = forward_propagation(X, weights)
    return A3

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] < 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

# 線性數據
X_linear, y_linear = generate_linear()
input_size = 2
hidden_size1 = 4
hidden_size2 = 4
output_size = 1
learning_rate = 0.1
epochs = 10

# 訓練模型
weights_linear = train(X_linear, y_linear, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs)

# 預測結果
pred_y_linear = predict(X_linear, weights_linear)

# 可視化結果
show_result(X_linear, y_linear, pred_y_linear)

# # XOR 數據
# X_xor, y_xor = generate_XOR_easy()

# # 訓練模型
# weights_xor = train(X_xor, y_xor, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs)

# # 預測結果
# pred_y_xor = predict(X_xor, weights_xor)

# # 可視化結果
# show_result(X_xor, y_xor, pred_y_xor)