import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels):
        super(ResNet34Encoder, self).__init__()
        self.in_channels = in_channels

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(64, 64, 3)
        self.layer2 = self.make_layer(64, 128, 4, stride=2)
        self.layer3 = self.make_layer(128, 256, 6, stride=2)
        self.layer4 = self.make_layer(256, 512, 3, stride=2)

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.initial(x)  # [batch, 64, H/4, W/4]
        x2 = self.layer1(x1)  # [batch, 64, H/4, W/4]
        x3 = self.layer2(x2)  # [batch, 128, H/8, W/8]
        x4 = self.layer3(x3)  # [batch, 256, H/16, W/16]
        x5 = self.layer4(x4)  # [batch, 512, H/32, W/32]
        return x1, x2, x3, x4, x5

class Res34_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res34_UNet, self).__init__()

        # ResNet34 Encoder
        self.encoder = ResNet34Encoder(in_channels)

        # Expansive path (Decoder)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Final layer
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

        # Additional upsampling to match the input size
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        enc1, enc2, enc3, enc4, enc5 = self.encoder(x)

        # print(f'enc1: {enc1.size()}')
        # print(f'enc2: {enc2.size()}')
        # print(f'enc3: {enc3.size()}')
        # print(f'enc4: {enc4.size()}')
        # print(f'enc5: {enc5.size()}')

        # Decoder
        dec4 = self.upconv4(enc5)
        dec4 = self.center_crop_and_concat(dec4, enc4)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.center_crop_and_concat(dec3, enc3)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.center_crop_and_concat(dec2, enc2)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.center_crop_and_concat(dec1, enc1)
        dec1 = self.dec1(dec1)

        final_output = self.conv_last(dec1)

        # Match the output size to the input size
        final_output = F.interpolate(final_output, size=x.size()[2:], mode='bilinear', align_corners=False)

        return final_output

    def center_crop_and_concat(self, upsampled, bypass):
        _, _, H, W = upsampled.size()
        _, _, H_b, W_b = bypass.size()
        
        # Resize bypass if necessary
        if H_b != H or W_b != W:
            bypass = F.interpolate(bypass, size=(H, W), mode='bilinear', align_corners=False)
        
        return torch.cat((upsampled, bypass), dim=1)

if __name__ == "__main__":
    # Test the implementation
    model = Res34_UNet(in_channels=3, out_channels=1)  # RGB input, binary output
    x = torch.randn(1, 3, 256, 256)  # Example input
    output = model(x)
    print(output.shape)  # Should print torch.Size([1, 1, 256, 256])