import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from torch.utils.checkpoint import checkpoint

# TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.args = args
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"], batch_size=args.batch_size).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()
        # self.scaler = GradScaler()
        self.start_epoch = args.start_from_epoch  # Initialize start_epoch from args

    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return optimizer, scheduler

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(tqdm(train_loader, ncols=140)):
            images = data.to(device=self.args.device)
            logits, z_indices = self.model(images)
            # print('================')

            # 計算 cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))

            # 梯度累積
            loss = loss / self.args.accum_grad
            loss.backward()

            if (batch_idx + 1) % self.args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()

            running_loss += loss.item() * self.args.accum_grad
            

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Train Epoch: {epoch} Loss: {epoch_loss:.6f}")
        return epoch_loss

    def eval_one_epoch(self, val_loader, epoch):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader, ncols=120)):
                images = data.to(device=self.args.device)
                logits, z_indices = self.model(images)

                # 計算 cross-entropy loss
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))

                running_loss += loss.item()

        epoch_loss = running_loss / len(val_loader.dataset)
        print(f"Val Epoch: {epoch} Loss: {epoch_loss:.6f}")
        return epoch_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    # TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:1", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Partial data used for training (default: 1.0)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    # you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=10, help='Save CKPT per ** epochs(default: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Starting epoch number.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Checkpoint interval.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()
    
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    for epoch in range(train_transformer.start_epoch + 1, args.epochs + 1):
        train_loss = train_transformer.train_one_epoch(train_loader, epoch)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch)
        
        train_transformer.scheduler.step()

        # Clear the cache after each epoch to avoid memory overflow
        # torch.cuda.empty_cache()

        # if epoch % args.save_per_epoch == 0:
        #     checkpoint_path = os.path.join("transformer_checkpoints", f"epoch_{epoch}.pt")
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': train_transformer.model.state_dict(),
        #         'optimizer_state_dict': train_transformer.optim.state_dict(),
        #         'scheduler_state_dict': train_transformer.scheduler.state_dict(),
        #         'train_loss': train_loss,
        #         'val_loss': val_loss
        #     }, checkpoint_path)
        #     print(f"Checkpoint saved at {checkpoint_path}")
        
        if epoch % args.save_per_epoch == 0:
            transformer_checkpoint_path = os.path.join("transformer_checkpoints", f"epoch_{epoch}.pt")
            torch.save(train_transformer.model.transformer.state_dict(), transformer_checkpoint_path)
            print(f"Transformer weights saved at {transformer_checkpoint_path}")