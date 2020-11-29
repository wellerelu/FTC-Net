import torch
import torch.nn as nn
from network.Basic import BasicBlock,EncoderBlock,DecoderBlock,ClassifierBlock

class ConfidenceNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConfidenceNet, self).__init__()

        norm_ch = 64
        self.conv1 = nn.Conv2d(in_ch,norm_ch,(3,3),1,(1,1),bias=True)
        self.encoder1 = EncoderBlock(norm_ch, norm_ch)
        self.encoder2 = EncoderBlock(norm_ch, norm_ch)
        self.encoder3 = EncoderBlock(norm_ch, norm_ch)
        self.encoder4 = EncoderBlock(norm_ch, norm_ch)
        self.bottleneck = BasicBlock(norm_ch, norm_ch)
        self.decoder1 = DecoderBlock(norm_ch*2, norm_ch)
        self.decoder2 = DecoderBlock(norm_ch*2, norm_ch)
        self.decoder3 = DecoderBlock(norm_ch*2, norm_ch)
        self.decoder4 = DecoderBlock(norm_ch * 2, norm_ch)
        self.classifier = ClassifierBlock(norm_ch, out_ch)
        #self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, input):
        tmp = self.conv1(input)
        e1, out1, indices1 = self.encoder1(tmp)
        e2, out2, indices2 = self.encoder2(e1)
        e3, out3, indices3 = self.encoder3(e2)
        e4, out4, indices4 = self.encoder4(e3)
        bn = self.bottleneck(e4)

        d4 = self.decoder4(bn, out4, indices4)
        d3 = self.decoder3(d4, out3, indices3)
        d2 = self.decoder2(d3, out2, indices2)
        d1 = self.decoder1(d2, out1, indices1)
        out = self.classifier(d1)

        return out

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

    def get_grad_param(self):
        return self.parameters()







