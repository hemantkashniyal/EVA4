import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config_dropout = config.get("dropout")
        self.config_bias_enabled = config.get("bias_enabled")
        self.config_input_channel = config.get("input_channel")

        # Input Block
        in_channel = self.config_input_channel
        out_channels = 8
        self.convblock0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=self.config_bias_enabled),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(self.config_dropout)
        ) # output_size = 28 RF = 3

        # CONVOLUTION BLOCK 1
        in_channel = out_channels
        out_channels = 8
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=self.config_bias_enabled),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(self.config_dropout)
        ) # output_size = 28 RF = 5

        # TRANSITION BLOCK 1
        in_channel = out_channels
        out_channels = 16
        self.transition1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=(1, 1), padding=1, bias=self.config_bias_enabled),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        ) # output_size = 28 RF = 5

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14 RF = 10

        # CONVOLUTION BLOCK 2
        in_channel = out_channels
        out_channels = 16
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=self.config_bias_enabled),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(self.config_dropout)
        ) # output_size = 14 RF = 12

        # CONVOLUTION BLOCK 3
        in_channel = out_channels
        out_channels = 16
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=self.config_bias_enabled),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(self.config_dropout)
        ) # output_size = 14 RF = 14

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7 RF = 28

        # CONVOLUTION BLOCK 4
        in_channel = out_channels
        out_channels = 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=(3, 3), padding=0, bias=self.config_bias_enabled),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(self.config_dropout)
        ) # output_size = 5 RF = 30

        # CONVOLUTION BLOCK 5
        in_channel = out_channels
        out_channels = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=(3, 3), padding=0, bias=self.config_bias_enabled),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(self.config_dropout)
        ) # output_size = 3 RF = 32

        # CONVOLUTION BLOCK 6
        in_channel = out_channels
        out_channels = 10
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=(3, 3), padding=0, bias=self.config_bias_enabled),
            # nn.ReLU(),
            # nn.BatchNorm2d(out_channels),
            # nn.Dropout(self.config_dropout)
        ) # output_size = 1 RF = 34

        self.gap = nn.AvgPool2d(1)

    def forward(self, x):
        x = self.convblock0(x)
        x = self.convblock1(x)
        x = self.transition1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool2(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)