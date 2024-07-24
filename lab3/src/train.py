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
from utils import dice_score

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load datasets
    train_dataset = SimpleOxfordPetDataset(root=args.data_path, mode='train', transform=transform)
    val_dataset = SimpleOxfordPetDataset(root=args.data_path, mode='valid', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    
    'Choose model'
    if args.model_type == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
        save_path = '../saved_models/unet_best_model.pth'
    else:
        model = Res34_UNet(in_channels=3, out_channels=1).to(device)
        save_path = '../saved_models/res34_best_model.pth'
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch['image'].float().to(device)
            masks = batch['mask'].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].float().to(device)
                masks = batch['mask'].float().to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                val_dice += dice_score(outputs, masks).item() * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        
        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model with val loss: {val_loss:.4f}')

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model_type', type=str, default='unet', help='unet or res34')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)