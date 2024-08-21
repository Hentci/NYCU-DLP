import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class CLEVRDataset(Dataset):
    def __init__(self, json_file, root_dir, object_mapping, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 讀取 json 文件
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # 獲取所有圖片名稱
        self.image_names = list(self.data.keys())

        # 將物件名稱轉換為索引
        self.object_mapping = object_mapping

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 獲取圖片名稱和其對應的標籤
        img_name = self.image_names[idx]
        label_names = self.data[img_name]

        # 讀取圖片
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 如果有transform，進行圖片轉換
        if self.transform:
            image = self.transform(image)

        # 將物件名稱轉換為索引
        labels = [self.object_mapping[label] for label in label_names]

        return image, torch.tensor(labels)

def custom_collate_fn(batch):
    images, labels = zip(*batch)

    # 将图片堆叠为一个张量
    images = torch.stack(images, 0)

    # 保持labels为不同大小的列表
    labels = list(labels)

    return images, labels

def get_dataloader(json_file, root_dir, object_mapping, batch_size=32, num_workers=4, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CLEVRDataset(json_file=json_file, root_dir=root_dir, object_mapping=object_mapping, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=custom_collate_fn)

    return dataloader

# 使用範例
object_mapping = {"gray cube": 0, "red cube": 1, "blue cube": 2, "green cube": 3, 
                  "brown cube": 4, "purple cube": 5, "cyan cube": 6, "yellow cube": 7, 
                  "gray sphere": 8, "red sphere": 9, "blue sphere": 10, "green sphere": 11, 
                  "brown sphere": 12, "purple sphere": 13, "cyan sphere": 14, "yellow sphere": 15, 
                  "gray cylinder": 16, "red cylinder": 17, "blue cylinder": 18, "green cylinder": 19, 
                  "brown cylinder": 20, "purple cylinder": 21, "cyan cylinder": 22, "yellow cylinder": 23}

train_json = './train.json'
dataset_path = '/home/hentci/code/iclevr'
train_loader = get_dataloader(train_json, dataset_path, object_mapping, batch_size=32)

# 測試 DataLoader
for images, labels in train_loader:
    print(images.shape)  
    print(labels)  # Labels 會是大小不同的 tensor 列表
    break