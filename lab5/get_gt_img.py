import os
import pandas as pd
import numpy as np
from PIL import Image

# 讀取 CSV 檔案
csv_file = './faster-pytorch-fid/test_gt.csv'
data = pd.read_csv(csv_file, header=None)
num_images = data.shape[0]

# 創建資料夾來存儲圖片
output_dir = './ground_truth'
os.makedirs(output_dir, exist_ok=True)

# 將每一列轉換為圖片並儲存
for i in range(num_images):
    row = data.iloc[i].values
    
    # 將數據重塑為 64x64x3 的圖片格式
    img = row.reshape(64, 64, 3).astype(np.uint8)
    
    # 將 NumPy 陣列轉換為圖片
    img = Image.fromarray(img)
    if i == 0:
        continue
    
    # 儲存圖片
    img.save(os.path.join(output_dir, f'image_{i - 1:04d}.png'))

print(f"已經將 {num_images} 張圖片儲存到資料夾 '{output_dir}' 中。")