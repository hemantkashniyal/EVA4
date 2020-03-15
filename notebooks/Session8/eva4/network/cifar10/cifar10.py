import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import InternalBlock, TransitionBlock, GAPBlock, DeepthSeparableBlock

class CIFAR10Net(nn.Module):
    def __init__(self, network_config):
        super(CIFAR10Net, self).__init__()
        self.config = network_config

        in_channel = self.config.input_channel
        dropout = self.config.dropout
        bias = self.config.bias_enabled

        self.input_block = nn.Sequential(
            InternalBlock(in_channels=in_channel, out_channels=16, kernel_size=(3,3), padding=1, dropout=dropout),   # in: 32x32x3 out: 32x32x16 rf: 3
            InternalBlock(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1, dropout=dropout),  # in: 32x32x16 out: 32x32x32 rf:5
            InternalBlock(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, dilation=2, dropout=dropout),   # in: 32x32x32 out: 32x32x64 rf: 9
            DeepthSeparableBlock(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1, dilation=2, dropout=dropout),   # in: 32x32x64 out: 32x32x128 rf: 11
        )

        self.input_transition_block = nn.Sequential(
            TransitionBlock(in_channels=128, out_channels=32, padding=1), # in: 32x32x128 out: 16x16x32 rf: 22
        )

        self.internal_block1 = nn.Sequential(
            InternalBlock(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, dropout=dropout),  # in: 16x16x32 out: 16x16x64 rf: 24
            InternalBlock(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1, dropout=dropout),   # in: 16x16x64 out: 16x16x128 rf: 26
            DeepthSeparableBlock(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1, dropout=dropout),  # in: 16x16x32 out: 16x16x64 rf: 24
        )

        self.internal_transition_block1 = nn.Sequential(
            TransitionBlock(in_channels=256, out_channels=32, padding=1), # in: 16x16x128 out: 8x8x32 rf: 52
        )

        self.internal_block2 = nn.Sequential(
            InternalBlock(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, dropout=dropout),  # in: 8x8x32 out: 8x8x64 rf: 54
            InternalBlock(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1, dropout=dropout),   # in:8x8x64 out: 8x8x128 rf: 56
        )

        self.internal_transition_block2 = nn.Sequential(
            TransitionBlock(in_channels=128, out_channels=32, padding=1), # in: 8x8x128 out: 4x4x32 rf: 112
        )

        self.output_block = nn.Sequential(
            GAPBlock(kernel_size=4),
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0,  bias=False),
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.input_transition_block(x)

        x = self.internal_block1(x)
        x = self.internal_transition_block1(x)

        x = self.internal_block2(x)
        x = self.internal_transition_block2(x)

        # output 
        x = self.output_block(x) # 

        # flatten the tensor so it can be passed to the dense layer afterward
        x = x.view(-1, 10)
        return F.log_softmax(x)