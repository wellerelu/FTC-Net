import torch
import torch.nn as nn
from network.Basic import EncoderBlock, BasicBlock,DecoderBlock,ClassifierBlock
from network.FCModule import FCModule

class SegNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SegNet, self).__init__()

        norm_ch = 64
        self.conv = nn.Conv2d(in_ch,norm_ch,(3,3),1,(1,1),bias=False)

        self.encoder1 = EncoderBlock(norm_ch, norm_ch)
        self.encoder2 = EncoderBlock(norm_ch, norm_ch)
        self.encoder3 = EncoderBlock(norm_ch, norm_ch)

        self.bottleneck = BasicBlock(norm_ch, norm_ch)

        self.con_de1 = DecoderBlock(norm_ch*2,norm_ch)
        self.con_de2 = DecoderBlock(norm_ch*2, norm_ch)
        self.con_de3 = DecoderBlock(norm_ch*2, norm_ch)

        self.decoder1 = DecoderBlock(norm_ch*2, norm_ch)
        self.decoder2 = DecoderBlock(norm_ch*2, norm_ch)
        self.decoder3 = DecoderBlock(norm_ch*2, norm_ch)

        self.classifier1 = ClassifierBlock(norm_ch, out_ch)
        self.classifier2 = ClassifierBlock(norm_ch, out_ch-1)

        self.att1 = FCModule(norm_ch)
        self.att2 = FCModule(norm_ch)
        self.att3 = FCModule(norm_ch)


    def forward(self, input):
        tmp = self.conv(input)

        e1, out1, idx1 = self.encoder1(tmp)
        e2, out2, idx2 = self.encoder2(e1)
        e3, out3, idx3 = self.encoder3(e2)
        bn = self.bottleneck(e3)

        att3,out3 = self.att3(out3)
        conout3 = self.con_de3(bn, out3, idx3)
        d3 = self.decoder3(bn, out3, idx3)

        att2, out2 = self.att2(out2)
        conout2 = self.con_de2(conout3, out2, idx2)
        d2 = self.decoder2(d3, out2, idx2)

        att1, out1 = self.att1(out1)
        conout1 = self.con_de1(conout2, out1, idx1)
        d1 = self.decoder1(d2, out1, idx1)

        out = self.classifier1(d1)
        dst_map = self.classifier2(conout1)

        # return att1,dst_map,out

        return att1,dst_map,out

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def save(self, path):
        print('Saving model... %s' % path)
        torch.save(self, path)

    def get_grad_param(self):
        return self.parameters()






