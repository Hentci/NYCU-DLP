import numpy as np
import matplotlib.pyplot as plt

from tools import generate_linear, generate_XOR_easy, sigmoid, derivative_sigmoid, plot_loss_curve

def initialize_weights(input_size, hidden_size1, hidden_size2, output_size, method='normal'):
    if method == 'normal':
        weight_initializer = lambda x, y: np.random.randn(x, y)
    elif method == 'uniform':
        weight_initializer = lambda x, y: np.random.uniform(-1.0, 1.0, (x, y))
    elif method == 'xavier':
        weight_initializer = lambda x, y: np.random.randn(x, y) * np.sqrt(1 / x)

    weights = {
        'W1': weight_initializer(input_size, hidden_size1),
        'b1': np.zeros((1, hidden_size1)),
        'W2': weight_initializer(hidden_size1, hidden_size2),
        'b2': np.zeros((1, hidden_size2)),
        'W3': weight_initializer(hidden_size2, output_size),
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

# cost(loss) function: MSE
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# .T: transpose
def back_propagation(X, y, weights, Z1, A1, Z2, A2, Z3, A3, learning_rate):
    m = y.shape[0] # number of samples
    
    # compute gradients
    dZ3 = (A3 - y) * 2 * derivative_sigmoid(A3)
    dW3 = np.dot(A2.T, dZ3) / m  # / m 取平均值
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m # 沿著 row 求和，並保持維度
    
    dA2 = np.dot(dZ3, weights['W3'].T)
    dZ2 = dA2 * derivative_sigmoid(A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, weights['W2'].T)
    dZ1 = dA1 * derivative_sigmoid(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    # update weights
    weights['W3'] -= learning_rate * dW3
    weights['b3'] -= learning_rate * db3
    weights['W2'] -= learning_rate * dW2
    weights['b2'] -= learning_rate * db2
    weights['W1'] -= learning_rate * dW1
    weights['b1'] -= learning_rate * db1



# Existing functions for forward_propagation, compute_loss, back_propagation remain unchanged

def train(X, y, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs, init_method):
    weights = initialize_weights(input_size, hidden_size1, hidden_size2, output_size, method=init_method)
    loss_history = []
    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, weights)
        loss = compute_loss(y, A3)
        loss_history.append(loss)
        back_propagation(X, y, weights, Z1, A1, Z2, A2, Z3, A3, learning_rate)
        if epoch % 5000 == 0:
            print(f'Epoch {epoch}, Loss: {loss}, Method: {init_method}')
    return loss_history

# Main execution block to compare different initializations
methods = ['normal', 'uniform', 'xavier']
loss_histories = []

X_linear, y_linear = generate_linear()
input_size = 2
hidden_size1 = 4
hidden_size2 = 4
output_size = 1
learning_rate = 0.1
epochs = 10000

for method in methods:
    print(f"Training with {method} initialization")
    loss_histories.append(train(X_linear, y_linear, input_size, hidden_size1, hidden_size2, output_size, 0.1, 10000, method))

# Plotting loss curves for different initialization methods
plt.figure(figsize=(10, 5))
for i, losses in enumerate(loss_histories):
    plt.plot(losses, label=f'{methods[i]} initialization')
plt.title('Loss Curves for Different Weight Initializations')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()