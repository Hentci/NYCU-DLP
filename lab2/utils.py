# script for drawing figures, and more if needed

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_loss_curve(loss_history, title='Loss Curve'):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=cmap)
    plt.title(title)
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Model loaded from {path}")
    else:
        print(f"No model found at {path}")

def calculate_accuracy(y_true, y_pred):
    correct = (y_pred == y_true).sum().item()
    total = y_true.size(0)
    return correct / total

# Test functions
if __name__ == '__main__':
    # Example loss history
    loss_history = np.random.rand(50)
    plot_loss_curve(loss_history)

    # Example confusion matrix
    y_true = [0, 1, 2, 2, 0]
    y_pred = [0, 0, 2, 2, 1]
    classes = ['Class 0', 'Class 1', 'Class 2']
    plot_confusion_matrix(y_true, y_pred, classes, normalize=True)

    # Example save and load model
    from model.SCCNet import SCCNet
    model = SCCNet(numClasses=4, timeSample=125, Nu=22, C=22, Nc=22, Nt=1, dropoutRate=0.5)
    save_model(model, 'sccnet_model.pth')
    load_model(model, 'sccnet_model.pth')