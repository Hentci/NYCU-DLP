import os
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from models.unet import UNet
from models.resnet34_unet import Res34_UNet
from oxford_pet import SimpleOxfordPetDataset
from utils import dice_score, train_loss
from PIL import Image

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Data transformations
    training_transform = transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(img)),  # Convert numpy array to PIL image
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Validation and test transformations
    test_transform = transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(img)),  # Convert numpy array to PIL image
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load datasets
    train_dataset = SimpleOxfordPetDataset(root=args.data_path, mode='train', transform=training_transform)
    val_dataset = SimpleOxfordPetDataset(root=args.data_path, mode='valid', transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    if args.model_type == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
        save_path = '../saved_models/unet_3_best_model.pth'
    else:
        model = Res34_UNet(in_channels=3, out_channels=1).to(device)
        save_path = '../saved_models/res34_best_model.pth'
    
    # Use a combination of BCEWithLogitsLoss and Dice Loss
    bce_loss = nn.BCEWithLogitsLoss()
    def dice_loss(pred, target, smooth=1.):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)
        return 1 - dice.mean()
    
    def combined_loss(pred, target):
        return bce_loss(pred, target) + dice_loss(pred, target)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss_epoch = 0.0
        for batch in train_loader:
            images = batch['image'].float().to(device)
            masks = batch['mask'].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item() * images.size(0)
        
        train_loss_epoch /= len(train_loader.dataset)
        train_losses.append(train_loss_epoch)
        
        # Validation loop
        model.eval()
        val_loss_epoch = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].float().to(device)
                masks = batch['mask'].float().to(device)
                
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_loss_epoch += loss.item() * images.size(0)
                
                val_dice += dice_score(outputs, masks).item() * images.size(0)
        
        val_loss_epoch /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        val_losses.append(val_loss_epoch)
        
        # Print epoch information including learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, Val Dice: {val_dice:.4f}, Learning Rate: {current_lr:.6f}')
        
        scheduler.step(val_loss_epoch)
        
        # Save the model if it has the best validation loss so far
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model with val loss: {val_loss_epoch:.4f}')
    
    # Plot and save the loss graph
    train_loss(train_losses, val_losses, save_path='unet_loss_plot.png')

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=7e-4, help='learning rate')
    parser.add_argument('--model_type', type=str, default='unet', help='unet or res34')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)