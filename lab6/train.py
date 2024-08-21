import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler
from model import MultiLabelConditionedUnet  # 確保這個是你在 model.py 中定義的模型類
from dataloader import get_dataloader  # 確保這個是你在 dataloader.py 中定義的數據加載器函數

# 設置設備
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 創建噪聲調度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

# 設定數據加載器
train_json = './train.json'
dataset_path = '/home/hentci/code/iclevr'
object_mapping = {"gray cube": 0, "red cube": 1, "blue cube": 2, "green cube": 3, 
                  "brown cube": 4, "purple cube": 5, "cyan cube": 6, "yellow cube": 7, 
                  "gray sphere": 8, "red sphere": 9, "blue sphere": 10, "green sphere": 11, 
                  "brown sphere": 12, "purple sphere": 13, "cyan sphere": 14, "yellow sphere": 15, 
                  "gray cylinder": 16, "red cylinder": 17, "blue cylinder": 18, "green cylinder": 19, 
                  "brown cylinder": 20, "purple cylinder": 21, "cyan cylinder": 22, "yellow cylinder": 23}

train_dataloader = get_dataloader(train_json, dataset_path, object_mapping, batch_size=32, shuffle=True)

# 設定訓練迭代次數
n_epochs = 10

# 創建模型並將其移動到設備上
net = MultiLabelConditionedUnet(num_classes=24, class_emb_size=4).to(device)

# 定義損失函數
loss_fn = nn.MSELoss()

# 定義優化器
opt = torch.optim.Adam(net.parameters(), lr=5e-4)

# 保留損失值以便稍後查看
losses = []

# 訓練迴圈
for epoch in range(n_epochs):
    running_loss = 0.0
    for x, y in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        
        # 獲取數據並準備添加噪聲的版本
        x = x.to(device) * 2 - 1  # 將數據移動到 GPU 並映射到 (-1, 1) 區間
        y = [label.to(device) for label in y]  # 將每個標籤張量移動到 GPU
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)  # 將不同長度的標籤序列填充為相同長度

        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # 獲得模型的預測
        pred = net(noisy_x, timesteps, y)  # 注意我們傳入了標籤 y

        # 計算損失
        loss = loss_fn(pred, noise)  # 輸出與噪聲的接近程度

        # 反向傳播並更新參數
        opt.zero_grad()
        loss.backward()
        opt.step()

        # 儲存損失值以便稍後查看
        losses.append(loss.item())
        running_loss += loss.item()

        # 更新進度條中的當前損失
        tqdm.write(f"Batch loss: {loss.item():.4f}")

    # 計算並打印該 epoch 的平均損失
    avg_loss = running_loss / len(train_dataloader)
    print(f'Finished epoch {epoch+1}. Average loss: {avg_loss:.6f}')

    # 每個 epoch 結束後保存模型
    torch.save(net.state_dict(), f'./64x64_saved_models/model_epoch_{epoch+1}.pth')
    print(f'Model saved after epoch {epoch+1}')

# 保存損失曲線到文件
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.savefig('training_loss_curve.png')
print('Loss curve saved as training_loss_curve.png')