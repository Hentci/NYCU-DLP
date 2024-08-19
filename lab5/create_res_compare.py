import os
import matplotlib.pyplot as plt
from PIL import Image

# 定義圖片路徑
masked_images_dir = '/home/hentci/code/lab5_dataset/masked_image/'
inpainting_results_dir = './test_results/'
ground_truth_dir = './ground_truth/'

# 定義要處理的圖片數量
num_images = 6

# 創建圖像佈局
fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))

# 讀取並顯示 Masked Images
for i in range(num_images):
    img_path = os.path.join(masked_images_dir, f'image_{i:03d}.png')
    img = Image.open(img_path)
    axes[0, i].imshow(img)
    axes[0, i].axis('off')

# 讀取並顯示 MaskGIT Inpainting Results
for i in range(num_images):
    img_path = os.path.join(inpainting_results_dir, f'image_{i:03d}.png')
    img = Image.open(img_path)
    axes[1, i].imshow(img)
    axes[1, i].axis('off')

# 讀取並顯示 Ground Truth
for i in range(num_images):
    img_path = os.path.join(ground_truth_dir, f'image_{i:03d}.png')
    img = Image.open(img_path)
    axes[2, i].imshow(img)
    axes[2, i].axis('off')

# 標記行
axes[0, 0].set_ylabel('Masked Images', fontsize=12)
axes[1, 0].set_ylabel('MaskGIT Inpainting Results', fontsize=12)
axes[2, 0].set_ylabel('Ground Truth', fontsize=12)

# 顯示並保存圖像
plt.tight_layout()
plt.savefig('comparison_image.png')
plt.show()