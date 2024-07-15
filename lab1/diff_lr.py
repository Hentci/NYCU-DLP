import numpy as np
import matplotlib.pyplot as plt

from tools import generate_linear, generate_XOR_easy, sigmoid, derivative_sigmoid, plot_loss_curve

def initialize_weights(input_size, hidden_size1, hidden_size2, output_size):
    weights = {
        'W1': np.random.randn(input_size, hidden_size1), # shape: (input_size, hidden_size1)
        'b1': np.zeros((1, hidden_size1)), # shape: (1, hidden_size1)
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

def train(X, y, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs):
    weights = initialize_weights(input_size, hidden_size1, hidden_size2, output_size)
    loss_history = []

    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, weights)
        loss = compute_loss(y, A3)
        loss_history.append(loss)
        back_propagation(X, y, weights, Z1, A1, Z2, A2, Z3, A3, learning_rate)

        if epoch % 5000 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return weights, loss_history

def predict(X, weights):
    _, _, _, _, _, A3 = forward_propagation(X, weights)
    return A3

def show_result(x, y, pred_y, title):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title(title, fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] < 0.5: # sigmoid output, 0.5 is the threshold
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

# 顯示測試結果
def show_testing_result(X, y, pred_y, loss, accuracy):
    for i in range(X.shape[0]):
        print(f"Iter{i+1} | Ground truth: {y[i][0]} | prediction: {pred_y[i][0]:.5f} |")
    print(f"loss={loss:.10f} accuracy={accuracy:.2f}%")

def plot_multiple_loss_curves(loss_histories, learning_rates, title):
    plt.figure(figsize=(10, 5))
    for i, loss_history in enumerate(loss_histories):
        plt.plot(loss_history, label=f'lr={learning_rates[i]}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# 線性數據
X_linear, y_linear = generate_linear()
input_size = 2
hidden_size1 = 4
hidden_size2 = 4
output_size = 1
learning_rates = [1, 0.1, 0.00001]
epochs = 100000

loss_histories_linear = []

# 針對不同的learning rate進行訓練
for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    weights_linear, loss_history_linear = train(X_linear, y_linear, input_size, hidden_size1, hidden_size2, output_size, lr, epochs)
    loss_histories_linear.append(loss_history_linear)
    pred_y_linear = predict(X_linear, weights_linear)
    loss_linear = compute_loss(y_linear, pred_y_linear)
    accuracy_linear = np.mean((pred_y_linear > 0.5) == y_linear) * 100
    print(f"Final loss: {loss_linear}, Accuracy: {accuracy_linear}%")
    show_result(X_linear, y_linear, pred_y_linear, title=f'Predict result (lr={lr})')

plot_multiple_loss_curves(loss_histories_linear, learning_rates, title="Loss Curves for Linear Data")

# XOR 數據
X_xor, y_xor = generate_XOR_easy()

loss_histories_xor = []

# 針對不同的learning rate進行訓練
for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    weights_xor, loss_history_xor = train(X_xor, y_xor, input_size, hidden_size1, hidden_size2, output_size, lr, epochs)
    loss_histories_xor.append(loss_history_xor)
    pred_y_xor = predict(X_xor, weights_xor)
    loss_xor = compute_loss(y_xor, pred_y_xor)
    accuracy_xor = np.mean((pred_y_xor > 0.5) == y_xor) * 100
    print(f"Final loss: {loss_xor}, Accuracy: {accuracy_xor}%")
    show_result(X_xor, y_xor, pred_y_xor, title=f'Predict result (lr={lr})')

plot_multiple_loss_curves(loss_histories_xor, learning_rates, title="Loss Curves for XOR Data")