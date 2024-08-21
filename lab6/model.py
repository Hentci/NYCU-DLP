import torch
import torch.nn as nn
from diffusers import UNet2DModel

class MultiLabelConditionedUnet(nn.Module):
    def __init__(self, num_classes=24, class_emb_size=4):
        super().__init__()
        
        # Embedding layer to map class labels to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embeddings)
        self.model = UNet2DModel(
            sample_size=64,           # the target image resolution, matching your data (64x64)
            in_channels=3 + class_emb_size, # Additional input channels for class conditioning
            out_channels=3,            # the number of output channels
            layers_per_block=2,        # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64), 
            down_block_types=( 
                "DownBlock2D",        # a regular ResNet downsampling block
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ), 
            up_block_types=(
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "AttnUpBlock2D",
                "UpBlock2D",          # a regular ResNet upsampling block
            ),
        )

    def forward(self, x, t, class_labels):
        # Shape of x:
        bs, ch, w, h = x.shape
        
        # class conditioning in right shape to add as additional input channels
        class_cond = self.class_emb(class_labels) # Map to embedding dimension

        # Sum the class embeddings along the label dimension
        # class_cond = class_cond.sum(dim=1)
        
        # Option 2: Average the embeddings along the label dimension
        class_cond = class_cond.mean(dim=1)

        # Reshape and expand the class_cond to match the image dimensions
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        
        # x is shape (bs, 3, 64, 64) and class_cond is now (bs, class_emb_size, 64, 64)
        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_cond), 1) # (bs, 3 + class_emb_size, 64, 64)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample # (bs, 3, 64, 64)

# 測試模型
model = MultiLabelConditionedUnet(num_classes=24, class_emb_size=4)