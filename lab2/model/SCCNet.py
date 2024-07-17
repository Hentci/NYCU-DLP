import torch
import torch.nn as nn
import torch.nn.functional as F

class SquareLayer(nn.Module):
    def __init__(self):
        super(SquareLayer, self).__init__()

    def forward(self, x):
        return x ** 2

class SCCNet(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=22, Nc=20, dropoutRate=0.5):
        super(SCCNet, self).__init__()

        # First convolutional block: Spatial component analysis
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(22, 1), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(Nu)
        self.square1 = SquareLayer()
        self.dropout1 = nn.Dropout(dropoutRate)

        # Second convolutional block: Spatio-temporal filtering
        self.conv2 = nn.Conv2d(in_channels=Nu, out_channels=Nc, kernel_size=(1, 12), stride=1, padding=(0, 11))
        self.bn2 = nn.BatchNorm2d(Nc)
        self.square2 = SquareLayer()
        self.dropout2 = nn.Dropout(dropoutRate)

        # Pooling layer: Temporal smoothing
        self.pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 62))

        # Fully connected layer
        pooled_time = (timeSample - 62) // 62 + 1
        self.fc = nn.Linear(in_features=Nc * pooled_time, out_features=numClasses)

    def forward(self, x):
        # Add a channel dimension to the input
        x = x.unsqueeze(1)  # New shape: [batch_size, 1, 22, 438]
        # print(f'After unsqueeze: {x.shape}')  # 調試輸出形狀

        # First convolutional block
        x = self.conv1(x)
        # print(f'After conv1: {x.shape}')  # 調試輸出形狀
        x = self.bn1(x)
        # print(f'After bn1: {x.shape}')  # 調試輸出形狀
        x = self.square1(x)
        # print(f'After square1: {x.shape}')
        x = self.dropout1(x)

        # Permute dimensions to match the input shape of the second convolutional block
        # x = x.permute(0, 2, 1, 3)
        # print(f'After permute: {x.shape}')  # 調試輸出形狀

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
        
        return F.log_softmax(x, dim=1)
    

# SCCNet_v2 with ReLU activation functions
class SCCNet_v2(nn.Module):
    def __init__(self, numClasses=4, timeSample=438, Nu=22, Nc=20, dropoutRate=0.5):
        super(SCCNet_v2, self).__init__()

        # First convolutional block: Spatial component analysis
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(22, 1), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(Nu)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropoutRate)

        # Second convolutional block: Spatio-temporal filtering
        self.conv2 = nn.Conv2d(in_channels=Nu, out_channels=Nc, kernel_size=(1, 12), stride=1, padding=(0, 11))
        self.bn2 = nn.BatchNorm2d(Nc)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropoutRate)

        # Pooling layer: Temporal smoothing
        self.pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 62))

        # Fully connected layer
        pooled_time = (timeSample - 62) // 62 + 1
        self.fc = nn.Linear(in_features=Nc * pooled_time, out_features=numClasses)

    def forward(self, x):
        # Add a channel dimension to the input
        x = x.unsqueeze(1)  # New shape: [batch_size, 1, 22, 438]
        # print(f'After unsqueeze: {x.shape}')  # 調試輸出形狀

        # First convolutional block
        x = self.conv1(x)
        # print(f'After conv1: {x.shape}')  # 調試輸出形狀
        x = self.bn1(x)
        # print(f'After bn1: {x.shape}')  # 調試輸出形狀
        x = self.relu1(x)
        # print(f'After square1: {x.shape}')
        x = self.dropout1(x)

        # Permute dimensions to match the input shape of the second convolutional block
        # x = x.permute(0, 2, 1, 3)
        # print(f'After permute: {x.shape}')  # 調試輸出形狀

        # Second convolutional block
        x = self.conv2(x)
        # print(f'After conv2: {x.shape}')  # 調試輸出形狀
        x = self.bn2(x)
        x = self.relu2(x)
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
        
        return F.log_softmax(x, dim=1)
    

# 測試用例
if __name__ == '__main__':
    model = SCCNet(numClasses=4, timeSample=438, Nu=22, Nc=20, dropoutRate=0.5)
    print(model)
    sample_input = torch.randn(32, 22, 438)  # Batch size of 32, 22 EEG channels, 438 time samples
    output = model(sample_input)
    print(f"Output shape: {output.shape}")