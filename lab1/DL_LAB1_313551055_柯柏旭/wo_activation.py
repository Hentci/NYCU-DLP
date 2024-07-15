import numpy as np
import matplotlib.pyplot as plt

from tools import generate_linear, generate_XOR_easy, sigmoid, derivative_sigmoid, plot_loss_curve

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

def forward_propagation(X, weights, use_activation=True):
    Z1 = np.dot(X, weights['W1']) + weights['b1']
    A1 = sigmoid(Z1) if use_activation else Z1
    Z2 = np.dot(A1, weights['W2']) + weights['b2']
    A2 = sigmoid(Z2) if use_activation else Z2
    Z3 = np.dot(A2, weights['W3']) + weights['b3']
    A3 = sigmoid(Z3) if use_activation else Z3
    return Z1, A1, Z2, A2, Z3, A3

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def back_propagation(X, y, weights, Z1, A1, Z2, A2, Z3, A3, learning_rate, use_activation=True):
    m = y.shape[0]
    
    dZ3 = (A3 - y) * 2
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m
    
    dA2 = np.dot(dZ3, weights['W3'].T)
    dZ2 = dA2 * derivative_sigmoid(A2) if use_activation else dA2
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, weights['W2'].T)
    dZ1 = dA1 * derivative_sigmoid(A1) if use_activation else dA1
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    weights['W3'] -= learning_rate * dW3
    weights['b3'] -= learning_rate * db3
    weights['W2'] -= learning_rate * dW2
    weights['b2'] -= learning_rate * db2
    weights['W1'] -= learning_rate * dW1
    weights['b1'] -= learning_rate * db1

def train(X, y, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs, use_activation=True):
    weights = initialize_weights(input_size, hidden_size1, hidden_size2, output_size)
    loss_history = []

    for epoch in range(epochs):
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, weights, use_activation)
        loss = compute_loss(y, A3)
        loss_history.append(loss)
        back_propagation(X, y, weights, Z1, A1, Z2, A2, Z3, A3, learning_rate, use_activation)

        if epoch % 5000 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    return weights, loss_history

def predict(X, weights, use_activation=True):
    _, _, _, _, _, A3 = forward_propagation(X, weights, use_activation)
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
        if pred_y[i] < 0.5: # output, 0.5 is the threshold
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

def show_testing_result(X, y, pred_y, loss, accuracy):
    for i in range(X.shape[0]):
        print(f"Iter{i+1} | Ground truth: {y[i][0]} | prediction: {pred_y[i][0]:.5f} |")
    print(f"loss={loss:.10f} accuracy={accuracy:.2f}%")

def plot_multiple_loss_curves(loss_histories, labels, title):
    plt.figure(figsize=(10, 5))
    for i, loss_history in enumerate(loss_histories):
        plt.plot(loss_history, label=labels[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


X_linear, y_linear = generate_linear()
X_xor, y_xor = generate_XOR_easy()
input_size = 2
hidden_size1 = 4
hidden_size2 = 4
output_size = 1
learning_rate = 0.1
epochs = 100000


def train_and_compare(X, y, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs, title):
    loss_histories = []
    labels = []

    for use_activation in [True, False]:
        label = 'With Activation' if use_activation else 'Without Activation'
        print(f"Training {label} for {title} Data")
        weights, loss_history = train(X, y, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs, use_activation)
        loss_histories.append(loss_history)
        labels.append(label)

        pred_y = predict(X, weights, use_activation)
        loss = compute_loss(y, pred_y)
        accuracy = np.mean((pred_y > 0.5) == y) * 100
        print(f"Final loss: {loss}, Accuracy: {accuracy}%")
        show_result(X, y, pred_y, title=f'{label}')

    plot_multiple_loss_curves(loss_histories, labels, title=f'Loss Curves for {title} Data')


train_and_compare(X_linear, y_linear, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs, title="Linear")


train_and_compare(X_xor, y_xor, input_size, hidden_size1, hidden_size2, output_size, learning_rate, epochs, title="XOR")