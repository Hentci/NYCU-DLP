import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler
from model import MultiLabelConditionedUnet  # 確保這個是你在 model.py 中定義的模型類
import torchvision.transforms as transforms
from PIL import Image

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加載已訓練的模型
net = MultiLabelConditionedUnet(num_classes=24, class_emb_size=4).to(device)
net.load_state_dict(torch.load('/home/hentci/code/NYCU-DLP/lab6/64x64_saved_models/model_epoch_5.pth'))
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

for labels in tqdm(test_data, desc="Generating samples"):
    print(labels)
    
    # 將標籤轉換為索引
    label_indices = [object_mapping[label] for label in labels]
    print(label_indices)
    
    y = torch.tensor(label_indices).unsqueeze(0).to(device)  # 添加 batch 維度

    # 準備隨機噪聲作為起點
    x = torch.randn(1, 3, 64, 64).to(device)  # 根據你的圖片尺寸設定

    # 生成迴圈
    for i, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = net(x, t, y)  # 注意我們傳入了標籤 y

        # 更新樣本
        x = noise_scheduler.step(residual, t, x).prev_sample

    # 將生成的圖像轉換為PIL格式並保存
    generated_image = ((x + 1) / 2).clamp(0, 1).cpu().squeeze().permute(1, 2, 0).numpy() * 255
    generated_image = Image.fromarray(generated_image.astype('uint8'))
    generated_images.append(generated_image)

    # 你可以將每張生成的圖片保存在指定的目錄下
    generated_image.save(f'generated_images/sample_{test_data.index(labels)}.png')
    
    print('save image: ',test_data.index(labels))

print("Image generation completed.")