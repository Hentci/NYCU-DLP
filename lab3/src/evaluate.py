import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from models.unet import UNet  # 假設你的UNet模型實現文件名為unet.py
from models.resnet34_unet import Res34_UNet
from oxford_pet import SimpleOxfordPetDataset  # 假設你的數據集實現文件名為oxford_pet.py
from utils import dice_score, plot_sample  # 假設你的工具函數文件名為utils.py

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the UNet model on validation dataset')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weights')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--output_path', type=str, default='evaluation', help='path to save evaluation results')
    parser.add_argument('--model_type', type=str, default='unet', help='unet or res34')

    return parser.parse_args()

def load_model(model_path, device, type):
    if type == 'unet':
        model = UNet(in_channels=3, out_channels=1)  # 修改根據你的UNet定義
    else:
        model = Res34_UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate(model, dataloader, device):
    dice_scores = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].float().to(device)
            masks = batch['mask'].float().to(device)
            outputs = model(images)
            preds = outputs > 0.5
            dice = dice_score(preds, masks)
            dice_scores.append(dice.item())
    return np.mean(dice_scores)

if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加載數據集
    dataset = SimpleOxfordPetDataset(root=args.data_path, mode='test')  
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 加載模型
    model = load_model(args.model, device, args.model_type)

    # 進行評估
    mean_dice_score = evaluate(model, dataloader, device)

    print(f"Mean Dice Score on validation dataset: {mean_dice_score:.4f}")

    # 保存評估結果
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Mean Dice Score: {mean_dice_score:.4f}\n")