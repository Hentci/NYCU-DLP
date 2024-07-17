import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.SCCNet import SCCNet, SCCNet_v2  # 確保 SCCNet 模型定義在 SCCNet.py 中
from Dataloader import MIBCI2aDataset  # 確保數據加載器定義在 dataloader.py 中
from utils import plot_loss_curve  # 從 utils.py 中導入 plot_loss_curve 函數

def train_sccnet_LOSO(num_epochs, batch_size, learning_rate, dropout_rate, model_save_path):
    # 設置設備（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加載數據集
    train_dataset = MIBCI2aDataset(mode='LOSO_train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 創建模型
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, Nc=20, dropoutRate=dropout_rate).to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1)  # 添加學習率調度器，每 20 個 epoch 乘以 0.1

    # 訓練模型
    model.train()
    loss_history = []  # 用於存儲每個 epoch 的損失
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度歸零
            optimizer.zero_grad()

            # 前向傳播
            outputs = model(inputs)

            # 計算損失
            loss = criterion(outputs, labels)

            # 反向傳播
            loss.backward()

            # 更新權重
            optimizer.step()

            # 累加損失
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)  # 記錄當前 epoch 的平均損失

        # 打印每個 epoch 的損失和學習率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Learning Rate: {current_lr}')

        # 更新學習率
        scheduler.step()

    # 保存模型權重
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # 繪製損失曲線
    plot_loss_curve(loss_history, title='Training Loss Curve')

if __name__ == '__main__':
    num_epochs = 300
    batch_size = 64
    learning_rate = 0.001
    dropout_rate = 0.5
    model_save_path = './sccnet_LOSO_model.pth'

    train_sccnet_LOSO(num_epochs, batch_size, learning_rate, dropout_rate, model_save_path)