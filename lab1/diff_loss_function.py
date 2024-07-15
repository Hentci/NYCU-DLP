import numpy as np
import matplotlib.pyplot as plt

from tools import generate_linear, generate_XOR_easy, sigmoid, derivative_sigmoid, plot_loss_curve

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

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

def back_propagation(X, y, weights, Z1, A1, Z2, A2, Z3, A3, learning_rate, loss_type='mse'):
    m = y.shape[0]
    if loss_type == 'cross_entropy':
        dA3 = -(y / A3) + (1 - y) / (1 - A3)
        dZ3 = dA3 * derivative_sigmoid(A3)
    else:  # mse
        dZ3 = (A3 - y) * 2 * derivative_sigmoid(A3)
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
    return weights

def calculate_accuracy(y_true, y_pred):
    y_pred_class = (y_pred > 0.5).astype(int)
    return np.mean(y_true == y_pred_class)

def train_and_compare(X, y, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs):
    weights_initial = initialize_weights(input_size, hidden_size1, hidden_size2, output_size)
    for loss_type in ['mse', 'cross_entropy']:
        weights = {k: v.copy() for k, v in weights_initial.items()}
        loss_history = []
        for epoch in range(epochs):
            Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, weights)
            loss = cross_entropy_loss(y, A3) if loss_type == 'cross_entropy' else mse_loss(y, A3)
            loss_history.append(loss)
            weights = back_propagation(X, y, weights, Z1, A1, Z2, A2, Z3, A3, learning_rate, loss_type)
            if epoch % 5000 == 0:
                print(f'Epoch {epoch}, Loss ({loss_type}): {loss}')
        plot_loss_curve(loss_history, title=f"Loss Curve using {loss_type}")
        accuracy = calculate_accuracy(y, A3)
        print(f'Final accuracy using {loss_type}: {accuracy * 100:.2f}%')

# Example usage
X_linear, y_linear = generate_linear()
train_and_compare(X_linear, y_linear, 2, 4, 4, 1, 0.1, 50000)