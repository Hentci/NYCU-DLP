import torch
import torch.nn as nn
import torch.nn.functional as F

class SquareLayer(nn.Module):
    def __init__(self):
        super(SquareLayer, self).__init__()

    def forward(self, x):
        return x ** 2


# LOSO60.07 and SD61.02
# class SCCNet(nn.Module):
#     def __init__(self, numClasses=4, timeSample=438, Nu=44, Nc=22, dropoutRate=0.5):
#         super(SCCNet, self).__init__()

#         # First convolutional block: Spatial component analysis
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(Nc, 16), stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(Nu)
#         # self.square1 = SquareLayer()
#         # self.dropout1 = nn.Dropout(dropoutRate)

#         # Second convolutional block: Spatio-temporal filtering
#         self.conv2 = nn.Conv2d(in_channels=Nu, out_channels=20, kernel_size=(1, 12), stride=1, padding=(0, 11))
#         self.bn2 = nn.BatchNorm2d(20)
#         self.square2 = SquareLayer()
#         self.dropout2 = nn.Dropout(dropoutRate)

#         # Pooling layer: Temporal smoothing
#         self.pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))

#         # Fully connected layer
#         pooled_time = (timeSample - 62) // 12 + 1
#         self.fc = nn.Linear(in_features=20 * pooled_time, out_features=numClasses)

#     def forward(self, x):
#         # Add a channel dimension to the input (加上 batch dimension)
#         x = x.unsqueeze(1)  
#         # print(f'After unsqueeze: {x.shape}')  # 調試輸出形狀

#         # First convolutional block
#         x = self.conv1(x)
#         x = self.bn1(x)
#         # x = self.square1(x)
#         # x = self.dropout1(x)

#         # Second convolutional block
#         x = self.conv2(x)
#         # print(f'After conv2: {x.shape}')  # 調試輸出形狀
#         x = self.bn2(x)
#         x = self.square2(x)
#         x = self.dropout2(x)

#         # Pooling layer
#         x = self.pool(x)
#         # print(f'After pool: {x.shape}')  # 調試輸出形狀

#         # Flatten the tensor
#         x = x.view(x.size(0), -1)
#         # print(f'After flatten: {x.shape}')  # 調試輸出形狀

#         # Fully connected layer
#         x = self.fc(x)
#         # print(f'After fc: {x.shape}')  # 調試輸出形狀

#         return x


# FT acc = 80.56
# class SCCNet(nn.Module):
#     def __init__(self, numClasses=4, timeSample=438, Nu=44, Nc=22, dropoutRate=0.5):
#         super(SCCNet, self).__init__()

#         # First convolutional block: Spatial component analysis
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(Nc, 2), stride=2, padding=0)
#         self.bn1 = nn.BatchNorm2d(Nu)
#         # self.square1 = SquareLayer()
#         # self.dropout1 = nn.Dropout(dropoutRate)

#         # Second convolutional block: Spatio-temporal filtering
#         self.conv2 = nn.Conv2d(in_channels=Nu, out_channels=20, kernel_size=(1, 12), stride=1, padding=(0, 115))
#         self.bn2 = nn.BatchNorm2d(20)
#         self.square2 = SquareLayer()
#         self.dropout2 = nn.Dropout(dropoutRate)

#         # Pooling layer: Temporal smoothing
#         self.pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))

#         # Fully connected layer
#         pooled_time = (timeSample - 62) // 12 + 1
#         self.fc = nn.Linear(in_features=20 * pooled_time, out_features=numClasses)

#     def forward(self, x):
#         # Add a channel dimension to the input (加上 batch dimension)
#         x = x.unsqueeze(1)  
#         # print(f'After unsqueeze: {x.shape}')  # 調試輸出形狀

#         # First convolutional block
#         x = self.conv1(x)
#         x = self.bn1(x)
#         # x = self.square1(x)
#         # x = self.dropout1(x)

#         # Second convolutional block
#         x = self.conv2(x)
#         # print(f'After conv2: {x.shape}')  # 調試輸出形狀
#         x = self.bn2(x)
#         x = self.square2(x)
#         x = self.dropout2(x)

#         # Pooling layer
#         x = self.pool(x)
#         # print(f'After pool: {x.shape}')  # 調試輸出形狀

#         # Flatten the tensor
#         x = x.view(x.size(0), -1)
#         # print(f'After flatten: {x.shape}')  # 調試輸出形狀

#         # Fully connected layer
#         x = self.fc(x)
#         # print(f'After fc: {x.shape}')  # 調試輸出形狀

#         return x
    

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=44, Nc=22, dropoutRate=0.5):
        super(SCCNet, self).__init__()

        # First convolutional block: Spatial component analysis
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(Nc, 4), stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(Nu)
        # self.square1 = SquareLayer()
        # self.dropout1 = nn.Dropout(dropoutRate)

        # Second convolutional block: Spatio-temporal filtering
        self.conv2 = nn.Conv2d(in_channels=Nu, out_channels=20, kernel_size=(1, 12), stride=1, padding=(0, 115))
        self.bn2 = nn.BatchNorm2d(20)
        self.square2 = SquareLayer()
        self.dropout2 = nn.Dropout(dropoutRate)

        # Pooling layer: Temporal smoothing
        self.pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))

        # Fully connected layer
        pooled_time = (timeSample - 62) // 12 + 1
        self.fc = nn.Linear(in_features=20 * pooled_time, out_features=numClasses)

    def forward(self, x):
        # Add a channel dimension to the input (加上 batch dimension)
        x = x.unsqueeze(1)  
        # print(f'After unsqueeze: {x.shape}')  # 調試輸出形狀

        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.square1(x)
        # x = self.dropout1(x)

        # Second convolutional block
        x = self.conv2(x)
        # print(f'After conv2: {x.shape}')  # 調試輸出形狀
        x = self.bn2(x)
        x = self.square2(x)
        x = self.dropout2(x)

        # Pooling layer
        x = self.pool(x)
        # print(f'After pool: {x.shape}')  # 調試輸出形狀

        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # print(f'After flatten: {x.shape}')  # 調試輸出形狀

        # Fully connected layer
        x = self.fc(x)
        # print(f'After fc: {x.shape}')  # 調試輸出形狀

        return x
    

# 測試用例
if __name__ == '__main__':
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, Nc=22, dropoutRate=0.5)
    print(model)
    sample_input = torch.randn(32, 22, 438)  # Batch size of 32, 22 EEG channels, 438 time samples
    output = model(sample_input)
    print(f"Output shape: {output.shape}")