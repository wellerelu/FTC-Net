import torch
import torch.nn as nn

class FCModule(nn.Module):
    def __init__(self,in_dim):
        super(FCModule,self).__init__()
        self.channel = in_dim

        self.global_features = 32

        self.C_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=self.global_features, kernel_size=1,bias=False),
            nn.BatchNorm2d(self.global_features),
            nn.PReLU()
        )
        self.B_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=self.global_features, kernel_size=1,bias=False),
            nn.BatchNorm2d(self.global_features),
            nn.PReLU()
        )
        self.A_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=self.global_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.global_features),
            nn.PReLU()
        )
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=self.global_features, out_channels=in_dim, kernel_size=1,bias=False),
            nn.BatchNorm2d(in_dim),
            # nn.PReLU()
        )

        self.softmax =nn.Softmax(dim=-1)
        self.prelu = nn.PReLU()

    def forward(self, shallow_in):
        m_batchsize, C, height, width = shallow_in.size()
        C = self.C_conv(shallow_in).permute(0, 3, 1, 2)
        C = C.contiguous().view(m_batchsize*width,self.global_features,height)
        B = self.B_conv(shallow_in).permute(0,3,2,1).contiguous().view(m_batchsize*width,height,self.global_features)
        energy = torch.bmm(B,C)
        # attmap = F.softmax(energy,dim=2) # (bw,h,h)
        attmap = torch.sigmoid(energy)

        A = self.A_conv(shallow_in).permute(0,3,2,1).contiguous().view(m_batchsize*width,height,self.global_features)
        selected_in = torch.bmm(attmap,A).view(m_batchsize,width,height,self.global_features).permute(0,3,2,1)
        out = self.prelu(self.transform(selected_in)+shallow_in)
        return attmap,out