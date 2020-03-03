
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import InternalBlock, TransitionBlock, GAPBlock

class CIFAR10Net(nn.Module):
    def __init__(self, dropout=0):
        super(CIFAR10Net, self).__init__()

        dropout = 0.2

        self.input_block = nn.Sequential(
            InternalBlock(in_channels=3, out_channels=16, kernel_size=(3,3), padding=1, dropout=dropout),   # in: 32x32x3 out: 32x32x16 rf: 3
            InternalBlock(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1, dropout=dropout),  # in: 32x32x16 out: 32x32x32 rf:5
            InternalBlock(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, dropout=dropout)   # in: 32x32x32 out: 32x32x64 rf: 7
        )

        self.input_transition_block = nn.Sequential(
            TransitionBlock(in_channels=64, out_channels=16, padding=1) # in: 32x32x64 out: 16x16x16 rf: 14
        )

        self.internal_block1 = nn.Sequential(
            InternalBlock(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1, dropout=dropout),  # in: 16x16x16 out: 16x16x16 rf: 16
            InternalBlock(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1, dropout=dropout),  # in: 16x16x16 out: 16x16x32 rf: 18
            InternalBlock(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, dropout=dropout)   # in: 16x16x32 out: 16x16x64 rf: 20
        )

        self.internal_transition_block1 = nn.Sequential(
            TransitionBlock(in_channels=64, out_channels=16, padding=1) # in: 16x16x64 out: 8x8x16 rf: 40
        )

        self.internal_block2 = nn.Sequential(
            InternalBlock(in_channels=16, out_channels=32, kernel_size=(3,3), padding=0, dropout=dropout),  # in: 8x8x16 out: 6x6x32 rf: 42
            InternalBlock(in_channels=32, out_channels=64, kernel_size=(3,3), padding=0, dropout=dropout),  # in: 6x6x32 out: 4x4x64 rf: 44
            InternalBlock(in_channels=64, out_channels=16, kernel_size=(3,3), padding=0, dropout=dropout)   # in: 4x4x64 out: 2x2x16 rf: 46
        )

        self.output_block = nn.Sequential(
            GAPBlock(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1,1), padding=0,  bias=False),
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.input_transition_block(x)

        x = self.internal_block1(x)
        x = self.internal_transition_block1(x)

        x = self.internal_block2(x)

        # output 
        x = self.output_block(x) # 

        # flatten the tensor so it can be passed to the dense layer afterward
        x = x.view(-1, 10)
        return F.log_softmax(x)