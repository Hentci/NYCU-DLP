import torch
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet, self).__init__()
        
        # 使用预训练的 ResNet34 作为编码器
        resnet = models.resnet34(pretrained=True)
        
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = nn.Sequential(resnet.layer1)
        self.encoder3 = nn.Sequential(resnet.layer2)
        self.encoder4 = nn.Sequential(resnet.layer3)
        self.encoder5 = nn.Sequential(resnet.layer4)
        
        # 解码器部分
        self.upconv5 = self._upconv(resnet.layer4[-1].conv1.out_channels, 256)
        self.decoder5 = self._decoder_block(resnet.layer4[-1].conv1.out_channels + 256, 256)
        
        self.upconv4 = self._upconv(256, 128)
        self.decoder4 = self._decoder_block(256 + 128, 128)
        
        self.upconv3 = self._upconv(128, 64)
        self.decoder3 = self._decoder_block(128 + 64, 64)
        
        self.upconv2 = self._upconv(64, 64)
        self.decoder2 = self._decoder_block(64 + 64, 64)
        
        self.upconv1 = self._upconv(64, 32)
        self.decoder1 = self._decoder_block(64 + 32, 32)
        
        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # 编码器部分
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        # 解码器部分
        dec5 = self.upconv5(enc5)
        dec5 = torch.cat([dec5, enc4], dim=1)
        dec5 = self.decoder5(dec5)
        
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)
        
        final = self.final_conv(dec1)
        return final
    
    def _upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

# 測試用例
if __name__ == "__main__":
    model = UNet(n_classes=1)
    print(model)
    x = torch.randn((1, 3, 256, 256))
    output = model(x)
    print(output.shape)