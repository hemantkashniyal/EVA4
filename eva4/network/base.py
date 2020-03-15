import torch.nn as nn

class InternalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=False, dropout=0):
        super(InternalBlock,self).__init__()

        self.block = nn.Sequential(           
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                padding=padding, 
                dilation=dilation, 
                bias=bias,
            ), 
            nn.ReLU(),
            nn.BatchNorm2d(
                num_features=out_channels
            ),
            nn.Dropout(
                p=dropout
            )
        )

    def forward(self, x):
        x = self.block(x)
        return x    


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, bias=False):
        super(TransitionBlock,self).__init__()

        self.block = nn.Sequential(                 
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=padding, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.block(x)
        return x    

class DeepthSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, bias=False, dropout=0):
        super(DeepthSeparableBlock,self).__init__()

        self.block = nn.Sequential(           
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=in_channels, 
                kernel_size=kernel_size, 
                padding=padding, 
                dilation=dilation, 
                bias=bias,
                groups=in_channels,
            ), 
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=padding, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(
                num_features=out_channels
            ),
            nn.Dropout(
                p=dropout
            )
            
        )

    def forward(self, x):
        x = self.block(x)
        return x    

class GAPBlock(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(GAPBlock,self).__init__()

        self.block = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )    

    def forward(self, x):
        x = self.block(x)
        return x    