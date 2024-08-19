# Lab5 MaskGIT for Image Inpainting - Experiment Score

#### 313551055 柯柏旭

## Part 1: Prove your code implementation is correct

### 1. Show iterative decoding

#### 固定的設定 (image_000.png, total_iter = 25, sweet_spot = 20)

![image-20240819133710854](/Users/hentci/Library/Application Support/typora-user-images/image-20240819133710854.png) ![image-20240819133725773](/Users/hentci/Library/Application Support/typora-user-images/image-20240819133725773.png)

- cosine (FID:  44.242456643580766)

(a) Mask in latent domain

<img src="/Users/hentci/Library/Application Support/typora-user-images/image-20240819141615284.png" alt="image-20240819141615284" style="zoom:150%;" />

(b) Predict image

![image-20240819141637690](/Users/hentci/Library/Application Support/typora-user-images/image-20240819141637690.png)

- linear (FID:  44.29381567836265)

(a) Mask in latent domain

<img src="/Users/hentci/Library/Application Support/typora-user-images/image-20240819141238991.png" alt="image-20240819141238991" style="zoom:150%;" />

(b) Predict image

![image-20240819141224831](/Users/hentci/Library/Application Support/typora-user-images/image-20240819141224831.png)

- square (FID:  43.10640596267979)

(a) Mask in latent domain

<img src="/Users/hentci/Library/Application Support/typora-user-images/image-20240819141949968.png" alt="image-20240819141949968" style="zoom:150%;" />

(b) Predict image

![image-20240819142001257](/Users/hentci/Library/Application Support/typora-user-images/image-20240819142001257.png)

## Part2: The Best FID Score

### Screenshot 

FID:  38.06985274794948

![image-20240819143433935](/Users/hentci/Library/Application Support/typora-user-images/image-20240819143433935.png)

###  Masked Images v.s MaskGIT Inpainting Results v.s Ground Truth

首先，透過以下程式將`gt.csv`轉換成圖片:

```python
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
```

按照作業要求，生成以下比較圖片: 

First row: Masked images

Second row: MaskGIT Inpainting Results

Third row: Ground Truth

![image-20240819145937341](/Users/hentci/Library/Application Support/typora-user-images/image-20240819145937341.png)

### The setting about training strategy, mask scheduling parameters, and so on

train 100 epochs，選取 valid loss 最小的 model

- Training hyperparameters

```python
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--save-per-epoch', type=int, default=5, help='Save CKPT per ** epochs(default: 1)')
parser.add_argument('--start-from-epoch', type=int, default=0, help='Starting epoch number.')
parser.add_argument('--ckpt-interval', type=int, default=0, help='Checkpoint interval.')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
```

- Inference hyperparameters

```python
#MVTM parameter
parser.add_argument('--sweet-spot', type=int, default=2, help='sweet spot: the best step in total iteration')
parser.add_argument('--total-iter', type=int, default=10, help='total step for mask scheduling')
parser.add_argument('--mask-func', type=str, default='square', help='mask scheduling function')
```

- loss curve

![image-20240819144112764](/Users/hentci/Library/Application Support/typora-user-images/image-20240819144112764.png)

