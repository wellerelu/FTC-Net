import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3, 3), 1, (1, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch,out_ch,(3,3),1,(1,1),bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.transform = None

        if in_ch != out_ch:
            self.transform = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,1,1,bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, input):
        out = self.prelu(self.bn1(self.conv1(input)))
        out = self.conv2(out)
        if self.transform:
            input = self.transform(input)
        out = self.prelu(self.bn2(out+input))
        return out


class EncoderBlock(BasicBlock):
    def __init__(self, in_ch, out_ch):
        super(EncoderBlock, self).__init__(in_ch=in_ch, out_ch=out_ch)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, input):
        out = super(EncoderBlock, self).forward(input)
        out_encoder, indices = self.maxpool(out)
        return out_encoder, out, indices


class DecoderBlock(BasicBlock):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__(in_ch, out_ch)
        self.uppool = nn.MaxUnpool2d(2, 2)

    def forward(self, input, out_block, indices):
        uppool = self.uppool(input, indices)
        concat = torch.cat((out_block, uppool), dim=1)
        out_block = super(DecoderBlock, self).forward(concat)
        return out_block

class ConDecoderBlock(BasicBlock):
    def __init__(self, in_ch, out_ch):
        super(DecoderBlock, self).__init__(in_ch, out_ch)
        self.uppool = nn.MaxUnpool2d(2, 2)

    def forward(self, input, out_block, indices):
        uppool = self.uppool(input, indices)
        concat = torch.cat((out_block, uppool), dim=1)
        out_block = super(DecoderBlock, self).forward(concat)
        return out_block

class ClassifierBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=True)

    def forward(self, input):
        out_conv = self.conv(input)
        return out_conv
