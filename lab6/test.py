import os
import torch
import json
import numpy as np
import random
from tqdm import tqdm
from diffusers import DDPMScheduler
from model import MultiLabelConditionedUnet
from torchvision.utils import make_grid
from PIL import Image

# Assuming evaluation_model is defined as in the provided code
from evaluator import evaluation_model

# 設置隨機種子
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

# 加載已訓練的模型
net = MultiLabelConditionedUnet(num_classes=24, class_emb_size=4).to(device)
net.load_state_dict(torch.load('/home/hentci/code/NYCU-DLP/lab6/64x64_saved_models/model_epoch_20.pth'))
net.eval()

# 創建噪聲調度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

# 讀取 test.json
with open('./new_test.json', 'r') as f:
    test_data = json.load(f)

# 定義 object_mapping
object_mapping = {"gray cube": 0, "red cube": 1, "blue cube": 2, "green cube": 3, 
                  "brown cube": 4, "purple cube": 5, "cyan cube": 6, "yellow cube": 7, 
                  "gray sphere": 8, "red sphere": 9, "blue sphere": 10, "green sphere": 11, 
                  "brown sphere": 12, "purple sphere": 13, "cyan sphere": 14, "yellow sphere": 15, 
                  "gray cylinder": 16, "red cylinder": 17, "blue cylinder": 18, "green cylinder": 19, 
                  "brown cylinder": 20, "purple cylinder": 21, "cyan cylinder": 22, "yellow cylinder": 23}

# 準備生成樣本
batch_size = 32

# 將標籤轉換為 one-hot 編碼
y = torch.zeros((batch_size, len(object_mapping)), device=device)  # 初始化 one-hot 向量
for idx, labels in enumerate(test_data):
    for label in labels:
        label_idx = object_mapping[label]
        y[idx, label_idx] = 1

# 準備隨機噪聲作為起點
x = torch.randn(batch_size, 3, 64, 64).to(device)  # 根據你的圖片尺寸設定

denoise_steps = 10  # 去噪過程中保存的步驟數量
denoise_images = torch.zeros((denoise_steps + 1, batch_size, 3, 64, 64)).to(device)
denoise_images[0] = x

# 生成迴圈
cnt = 1
for i, t in enumerate(tqdm(noise_scheduler.timesteps)):
    with torch.no_grad():
        residual = net(x, t, y)  # 傳入 one-hot 編碼的標籤 y

    # 更新樣本
    x = noise_scheduler.step(residual, t, x).prev_sample

    if i % (len(noise_scheduler.timesteps) // denoise_steps) == 0 and cnt <= denoise_steps:
        denoise_images[cnt] = x
        cnt += 1

# 將最終生成的圖像轉換為 [0, 1] 區間並保存
result_images = ((x + 1) / 2).clamp(0, 1)

# # 保存生成的最終圖片
# for idx in range(batch_size):
#     generated_image = result_images[idx].cpu().permute(1, 2, 0).numpy() * 255
#     generated_image = Image.fromarray(generated_image.astype('uint8'))
#     generated_image.save(f'generated_images/sample_{idx}.png')

# # 保存去噪過程圖像
# denoise_images = (denoise_images / 2 + 0.5).clamp(0, 1)
# for idx in range(batch_size):
#     grid = make_grid(denoise_images[:, idx], nrow=denoise_steps + 1)
#     grid_image = grid.permute(1, 2, 0).cpu().numpy() * 255
#     grid_image = Image.fromarray(grid_image.astype('uint8'))
#     grid_image.save(f'denoise_process/denoise_process_{idx}.png')

# 評估生成的圖像
eval_model = evaluation_model()
# accuracy = eval_model.compute_acc(x.clip(-1, 1), y)
accuracy = eval_model.eval(x.clip(-1, 1), y)
print(f"Accuracy of the generated images: {accuracy * 100:.2f}%")

print("Image generation, denoise process saving, and evaluation completed.")