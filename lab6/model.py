import torch
import torch.nn as nn
from diffusers import UNet2DModel


class MultiLabelConditionedUnet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=4):
        super().__init__()
        
        self.label_embedding = nn.Embedding(num_classes, class_emb_size)

        # # 使用全連接層將 one-hot 編碼向量轉換為 class embedding
        # self.class_embedding = nn.Sequential(
        #     nn.Linear(num_classes, class_emb_size * 64 * 64),  # 將 one-hot 編碼展開到整個影像維度
        #     nn.ReLU()
        # )

        # Self.model 是一個 UNet，並包含額外的輸入通道以接受條件信息
        self.model = UNet2DModel(
            sample_size=64,           # 目標影像解析度 (64x64)
            in_channels=3 + num_classes, # 額外的輸入通道用於 class conditioning
            out_channels=3,            # 輸出通道數
            time_embedding_type="positional",
            layers_per_block=2,        # 每個 UNet 區塊使用的 ResNet 層數
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape

        # 使用全連接層將 one-hot 向量展開到 64x64 的影像大小
        class_cond = class_labels.view(bs, class_labels.shape[1], 1, 1).expand(bs, class_labels.shape[1], w, h)  # Shape: (bs, class_emb_size * 64 * 64)
        # class_cond = class_cond.view(bs, -1, 64, 64)  # Reshape 為影像形狀

        # 拼接影像與條件信息
        net_input = torch.cat((x, class_cond), 1)  # (bs, 3 + class_emb_size, 64, 64)

        # 通過 UNet 輸出預測結果
        return self.model(net_input, t).sample  # (bs, 3, 64, 64)

# 測試模型
model = MultiLabelConditionedUnet(num_classes=24, class_emb_size=4)