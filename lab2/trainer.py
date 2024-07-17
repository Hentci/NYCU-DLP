import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.SCCNet import SCCNet  # 確保 SCCNet 模型定義在 SCCNet.py 中
from Dataloader import MIBCI2aDataset  # 確保數據加載器定義在 dataloader.py 中

def train_sccnet(num_epochs, batch_size, learning_rate, dropout_rate, model_save_path):
    # 設置設備（GPU 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加載數據集
    train_dataset = MIBCI2aDataset(mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 創建模型
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, Nc=20, dropoutRate=dropout_rate).to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練模型
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs.shape, labels.shape)  # 調試輸出形狀

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

            if i % 10 == 9:  # 每10個batch打印一次
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    # 保存模型權重
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == '__main__':
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001
    dropout_rate = 0.5
    model_save_path = './sccnet_model.pth'

    train_sccnet(num_epochs, batch_size, learning_rate, dropout_rate, model_save_path)