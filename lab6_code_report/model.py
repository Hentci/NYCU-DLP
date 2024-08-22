import torch
import torch.nn as nn
from diffusers import UNet2DModel


class MultiLabelConditionedUnet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=4):
        super().__init__()

        # 定義一個嵌入層，將類別標籤映射為大小為 class_emb_size 的嵌入向量
        # 這裡的 num_classes 是類別的總數（24個），class_emb_size 是嵌入向量的維度
        self.label_embedding = nn.Embedding(num_classes, class_emb_size)
        
        # # 使用全連接層將 one-hot 編碼向量轉換為 class embedding
        # self.class_embedding = nn.Sequential(
        #     nn.Linear(num_classes, class_emb_size * 64 * 64),  # 將 one-hot 編碼展開到整個影像維度
        #     nn.ReLU()
        # )

        # 定義UNet模型，這裡的UNet用於圖像生成，接受額外的類別條件信息
        self.model = UNet2DModel(
            sample_size=64,           # 設定輸入影像的解析度為64x64
            in_channels=3 + num_classes,  # 設定輸入通道數，3個RGB通道加上one-hot類別條件信息的通道數
            out_channels=3,            # 設定輸出通道數，這裡輸出為RGB圖像，所以是3個通道
            time_embedding_type="positional",  # 使用位置嵌入來處理時間步長
            layers_per_block=2,        # 每個UNet區塊使用2層ResNet
            block_out_channels=(128, 128, 256, 256, 512, 512),  # 定義每個UNet區塊的輸出通道數
            down_block_types=(
                "DownBlock2D",  # 定義下采樣區塊類型，這裡是普通的ResNet下采樣
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # 添加帶有空間自注意力的ResNet下采樣區塊
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # 定義上采樣區塊類型，這裡是普通的ResNet上采樣
                "AttnUpBlock2D",  # 添加帶有空間自注意力的ResNet上采樣區塊
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        # x 的形狀：批次大小 (bs), 通道數 (ch), 寬度 (w), 高度 (h)
        bs, ch, w, h = x.shape

        # 將one-hot類別標籤轉換為與影像相同大小的張量，用於條件輸入
        # class_labels的形狀為(bs, num_classes)，這裡先調整維度再擴展成(bs, num_classes, w, h)
        class_cond = class_labels.view(bs, class_labels.shape[1], 1, 1).expand(bs, class_labels.shape[1], w, h)

        # 將影像x與條件信息class_cond在通道維度上拼接
        # 拼接後的net_input形狀為(bs, 3 + num_classes, 64, 64)
        net_input = torch.cat((x, class_cond), 1)

        # 將拼接後的輸入傳入UNet模型進行前向傳播，並返回生成的影像
        return self.model(net_input, t).sample  # 返回形狀為(bs, 3, 64, 64)的輸出影像

# 測試模型
model = MultiLabelConditionedUnet(num_classes=24, class_emb_size=4)