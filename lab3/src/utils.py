import numpy as np
import torch
import matplotlib.pyplot as plt

def dice_score(pred_mask, gt_mask, smooth=1e-6):
    """
    Calculate the Dice score, which is a measure of overlap between two samples.
    
    Args:
    pred_mask (torch.Tensor or np.ndarray): Predicted binary mask
    gt_mask (torch.Tensor or np.ndarray): Ground truth binary mask
    smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
    float: Dice score
    """
    pred_mask = pred_mask.float() if isinstance(pred_mask, torch.Tensor) else torch.tensor(pred_mask, dtype=torch.float32)
    gt_mask = gt_mask.float() if isinstance(gt_mask, torch.Tensor) else torch.tensor(gt_mask, dtype=torch.float32)
    
    intersection = (pred_mask * gt_mask).sum()
    dice = (2. * intersection + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)
    return dice.item()

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