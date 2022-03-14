import torch.nn as nn
import torch.nn.functional as F
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.main(x)


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dims=[64, 128, 256, 512, 1024]):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_dims = feature_dims

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        self.pool = nn.MaxPool2d(2,2)
        in_dims = self.in_channels
        for feature in feature_dims:
            self.down.append(DoubleConv(in_dims, feature))
            in_dims = feature

        for feature in reversed(feature_dims[:-1]):
            self.up.append(nn.ConvTranspose2d(in_dims, feature, 2,2))
            self.up.append(DoubleConv(2*feature, feature))
            in_dims = feature
        
        self.logits = nn.Conv2d(in_dims, out_channels, 1)

    def forward(self, x):
        intermediate = []
        for idx,layer in enumerate(self.down):
            x = layer(x)
            if idx != len(self.down) -1 :
                intermediate.append(x)
                x = self.pool(x)
        intermediate = list(reversed(intermediate))
        for idx in range(0,len(self.up), 2):
            x = self.up[idx](x)
            x = torch.cat([x, intermediate[idx//2]], dim=1)
            x = self.up[idx+1](x)
        logits = self.logits(x)
        return logits

if __name__ == "__main__":
    x = torch.randn((1,1,160,160))
    model = Unet(1,1)
    output = model(x)
    print(output.shape)

