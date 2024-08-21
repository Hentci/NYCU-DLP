import torch
import torch.nn as nn
from diffusers import UNet2DModel

class MultiLabelConditionedUnet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=4):
        super().__init__()

        # 使用卷積來處理 one-hot 編碼向量，使其適應影像維度
        self.fc = nn.Linear(num_classes, class_emb_size * 64 * 64)  # 全連接層將 one-hot 編碼展開到整個影像維度

        self.conv1 = nn.Conv2d(class_emb_size, class_emb_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(class_emb_size, class_emb_size, kernel_size=3, padding=1)

        # Self.model 是一個 UNet，並包含額外的輸入通道以接受條件信息
        self.model = UNet2DModel(
            sample_size=64,           # 目標影像解析度 (64x64)
            in_channels=3 + class_emb_size, # 額外的輸入通道用於 class conditioning
            out_channels=3,            # 輸出通道數
            layers_per_block=2,        # 每個 UNet 區塊使用的 ResNet 層數
            block_out_channels=(32, 64, 64), 
            down_block_types=( 
                "DownBlock2D",        # 普通的 ResNet 下採樣區塊
                "AttnDownBlock2D",    # 帶空間自注意力的 ResNet 下採樣區塊
                "AttnDownBlock2D",
            ), 
            up_block_types=(
                "AttnUpBlock2D",      # 帶空間自注意力的 ResNet 上採樣區塊
                "AttnUpBlock2D",
                "UpBlock2D",          # 普通的 ResNet 上採樣區塊
            ),
        )

    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape

        # 使用全連接層將 one-hot 向量展開到 64x64 的影像大小
        class_cond = self.fc(class_labels)  # Shape: (bs, class_emb_size * 64 * 64)
        class_cond = class_cond.view(bs, -1, 64, 64)  # Reshape 為影像形狀

        # 使用卷積處理嵌入以捕捉更複雜的特徵
        class_cond = torch.relu(self.conv1(class_cond))
        class_cond = torch.relu(self.conv2(class_cond))

        # 拼接影像與條件信息
        net_input = torch.cat((x, class_cond), 1)  # (bs, 3 + class_emb_size, 64, 64)

        # 通過 UNet 輸出預測結果
        return self.model(net_input, t).sample  # (bs, 3, 64, 64)

# 測試模型
model = MultiLabelConditionedUnet(num_classes=24, class_emb_size=4)