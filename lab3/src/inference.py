import argparse
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from models.unet import UNet  # 假設你的UNet模型實現文件名為unet.py
from oxford_pet import SimpleOxfordPetDataset  # 假設你的數據集實現文件名為oxford_pet.py
from utils import dice_score, plot_sample  # 假設你的工具函數文件名為utils.py

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weights')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--output_path', type=str, default='predictions', help='path to save the predicted masks')

    return parser.parse_args()

def load_model(model_path, device):
    model = UNet(n_channels=3, n_classes=1)  # 修改根據你的UNet定義
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, dataloader, device):
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            outputs = model(images)
            preds.append(outputs.cpu().numpy())
    return np.vstack(preds)

def save_predictions(predictions, filenames, output_path):
    os.makedirs(output_path, exist_ok=True)
    for pred, filename in zip(predictions, filenames):
        pred_mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255  # 二值化
        pred_image = Image.fromarray(pred_mask)
        pred_image.save(os.path.join(output_path, f"{filename}.png"))

if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加載數據集
    dataset = SimpleOxfordPetDataset(root=args.data_path, mode='test')  # 假設test模式加載未見數據
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 加載模型
    model = load_model(args.model, device)

    # 進行推理
    predictions = predict(model, dataloader, device)

    # 保存預測結果
    save_predictions(predictions, dataset.filenames, args.output_path)

    print(f"Predictions saved to {args.output_path}")