import numpy as np
import torch
import matplotlib.pyplot as plt

import torch

def dice_score(pred_mask, gt_mask):
    smooth = 1e-5
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    
    intersection = (pred_mask * gt_mask).sum(dim=(2, 3))
    union = pred_mask.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()  # 取 batch 的平均

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

def compute_metrics(pred_mask, gt_mask):
    """
    Compute various segmentation metrics.
    
    Args:
    pred_mask (torch.Tensor or np.ndarray): Predicted binary mask
    gt_mask (torch.Tensor or np.ndarray): Ground truth binary mask
    
    Returns:
    dict: Dictionary with computed metrics
    """
    dice = dice_score(pred_mask, gt_mask)
    metrics = {
        "dice_score": dice
    }
    return metrics