import os
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler
from model import MultiLabelConditionedUnet
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import random

# Assuming evaluation_model is defined as in the provided code
from evaluator import evaluation_model

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 檢查並創建資料夾
os.makedirs('generated_images', exist_ok=True)
os.makedirs('denoise_process', exist_ok=True)

# 加載已訓練的模型
net = MultiLabelConditionedUnet(num_classes=24, class_emb_size=4).to(device)
net.load_state_dict(torch.load('/home/hentci/code/NYCU-DLP/lab6/DL_lab6_313551055_柯柏旭.pth'))
net.eval()

# 創建噪聲調度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

# 讀取 test.json
with open('./test.json', 'r') as f:
    test_data = json.load(f)

# 定義 object_mapping
object_mapping = {"gray cube": 0, "red cube": 1, "blue cube": 2, "green cube": 3, 
                  "brown cube": 4, "purple cube": 5, "cyan cube": 6, "yellow cube": 7, 
                  "gray sphere": 8, "red sphere": 9, "blue sphere": 10, "green sphere": 11, 
                  "brown sphere": 12, "purple sphere": 13, "cyan sphere": 14, "yellow sphere": 15, 
                  "gray cylinder": 16, "red cylinder": 17, "blue cylinder": 18, "green cylinder": 19, 
                  "brown cylinder": 20, "purple cylinder": 21, "cyan cylinder": 22, "yellow cylinder": 23}

# 準備生成樣本
generated_images = []

# 創建保存去噪過程的文件夾
os.makedirs('denoise_process', exist_ok=True)

for idx, labels in enumerate(tqdm(test_data, desc="Generating samples")):
    print(labels)
    
    # 將標籤轉換為 one-hot 編碼
    y = torch.zeros((1, len(object_mapping)), device=device)  # 初始化 one-hot 向量
    for label in labels:
        label_idx = object_mapping[label]
        y[0, label_idx] = 1
    
    print(y)
    
    # 準備隨機噪聲作為起點
    x = torch.randn(1, 3, 64, 64).to(device)  # 根據你的圖片尺寸設定

    denoise_images = []  # 儲存去噪過程中的中間結果

    # 生成迴圈
    for i, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = net(x, t, y)  # 傳入 one-hot 編碼的標籤 y

        # 更新樣本
        x = noise_scheduler.step(residual, t, x).prev_sample

        # 每隔一定的步驟保存中間結果
        if i % 100 == 0 or i == len(noise_scheduler.timesteps) - 1:
            denoise_images.append(x.clone())

    # 將生成的圖像轉換為PIL格式並保存
    generated_image = ((x + 1) / 2).clamp(0, 1).cpu().squeeze().permute(1, 2, 0).numpy() * 255
    generated_image = Image.fromarray(generated_image.astype('uint8'))
    generated_images.append(generated_image)

    # 保存生成的最終圖片
    generated_image.save(f'generated_images/sample_{idx}.png')

    # 保存去噪過程圖像
    denoise_images = [(img + 1) / 2 for img in denoise_images]  # 從 [-1, 1] 映射回 [0, 1]
    denoise_images = torch.cat(denoise_images, dim=0)
    grid = make_grid(denoise_images, nrow=len(denoise_images))
    grid = grid.permute(1, 2, 0).cpu().numpy() * 255
    grid_image = Image.fromarray(grid.astype('uint8'))
    grid_image.save(f'denoise_process/denoise_process_{idx}.png')
    
    print('save image: ', idx)


print("Image generation and denoise process saving completed.")