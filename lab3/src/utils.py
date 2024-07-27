import numpy as np
import torch
import matplotlib.pyplot as plt

def dice_score(pred_mask, gt_mask):
    # 添加一個很小的數，防止除以0的錯誤
    smooth = 1e-5
    
    # 將預測的掩碼通過Sigmoid函數轉換為概率值
    pred_mask = torch.sigmoid(pred_mask)
    
    # 將概率值轉換為二值掩碼，大於0.5的部分設為1，否則設為0
    pred_mask = (pred_mask > 0.5).float()
    
    # 計算預測掩碼和真實掩碼的交集（乘積）並在寬和高兩個維度上求和
    intersection = (pred_mask * gt_mask).sum(dim=(2, 3))
    
    # 分別計算預測掩碼和真實掩碼的總和（面積）並在寬和高兩個維度上求和
    union = pred_mask.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
    
    # 計算Dice Score，公式為 (2 * 交集 + 平滑項) / (預測掩碼面積 + 真實掩碼面積 + 平滑項)
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # 返回整個批次的平均Dice Score
    return dice.mean()

def plot_sample(image, mask, pred_mask=None):
    """
    Plot a sample image with its ground truth mask and optionally the predicted mask.
    
    Args:
    image (np.ndarray or torch.Tensor): Input image
    mask (np.ndarray or torch.Tensor): Ground truth mask
    pred_mask (np.ndarray or torch.Tensor, optional): Predicted mask
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().squeeze()
    if pred_mask is not None and isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy().squeeze()
    
    fig, ax = plt.subplots(1, 3 if pred_mask is not None else 2, figsize=(12, 4))
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis("off")

    if pred_mask is not None:
        ax[2].imshow(pred_mask, cmap="gray")
        ax[2].set_title("Predicted Mask")
        ax[2].axis("off")

    plt.tight_layout()
    plt.show()

def plot_loss(train_losses, val_losses, save_path='loss_plot.png'):
    """
    Plot the training and validation loss.

    Args:
    train_losses (list of float): Training losses
    val_losses (list of float): Validation losses
    save_path (str): Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def train_loss(train_losses, val_losses, save_path='loss_plot.png'):
    """
    Save the training and validation loss plot at the end of training.

    Args:
    train_losses (list of float): Training losses
    val_losses (list of float): Validation losses
    save_path (str): Path to save the plot
    """
    plot_loss(train_losses, val_losses, save_path)