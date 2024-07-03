import numpy as np
import torch
import torch.nn as nn

class u_shape(nn.Module):

    def __init__(self) -> None:
        super(u_shape, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.MaxPool2d(2)

        self.layer2 = nn.Sequential(
            SeparableConv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            tripleCov2d(128, 128, 3, 1)
        )

        self.down2 = nn.MaxPool2d(2)

        self.layer3 = nn.Sequential(
            SeparableConv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            tripleCov2d(256, 256, 3, 1)
        )

        self.down3 = nn.MaxPool2d(2)

        self.layer4 = nn.Sequential(
            SeparableConv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            tripleCov2d(512, 512, 3, 1)
        )

        self.down4 = nn.MaxPool2d(2)

        self.layer5 = nn.Sequential(
            SeparableConv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            tripleCov2d(1024, 1024, 3, 1)
        )

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.layer6 = nn.Sequential(
            SeparableConv2d(1024, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            tripleCov2d(512, 512, 3, 1)
        )

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.layer7 = nn.Sequential(
            SeparableConv2d(512, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            tripleCov2d(256, 256, 3, 1)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.layer8 = nn.Sequential(
            SeparableConv2d(256, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            tripleCov2d(128, 128, 3, 1)
        )

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.layer8 = nn.Sequential(
            SeparableConv2d(256, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            tripleCov2d(128, 128, 3, 1)
        )

        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):

        layer1_out = self.layer1(x)
        #print('layer1_out:', layer1_out.shape)
        d1= self.down1(layer1_out)
        #print('d1:', d1.shape)

        layer2_out = self.layer2(d1)
        #print('layer2_out:', layer2_out.shape)
        d2 = self.down2(layer2_out)
        #print('d2:', d2.shape)

        layer3_out = self.layer3(d2)
        #print('layer3_out:', layer3_out.shape)
        d3 = self.down3(layer3_out)
        #print('d3:', d3.shape)

        layer4_out = self.layer4(d3)
        #print('layer4_out:', layer4_out.shape)
        d4 = self.down4(layer4_out)
        #print('d4:', d4.shape)

        layer5_out = self.layer5(d4)
        #print('layer5_out:', layer5_out.shape)
        u1 = self.up1(layer5_out)
        #print('u1:', u1.shape)

        merge1 = torch.cat((layer4_out, u1), 1)
        #print('merge1:', merge1.shape)

        layer6_out = self.layer6(merge1)
        #print('layer6_out:', layer6_out.shape)
        u2 = self.up2(layer6_out)
        #print('u2:', u2.shape)

        merge2 = torch.cat((layer3_out, u2), 1)
        #print('merge2:', merge2.shape)

        layer7_out = self.layer7(merge2)
        #print('layer7_out:', layer7_out.shape)
        u3 = self.up3(layer7_out)
        #print('u3:', u3.shape)

        merge3 = torch.cat((layer2_out, u3), 1)
        #print('merge3:', merge3.shape)

        layer8_out = self.layer8(merge3)
        #print('layer8_out:', layer8_out.shape)
        u4 = self.up4(layer8_out)
        #print('u4:', u4.shape)

        merge4 = torch.cat((layer1_out, u4), 1)
        #print('merge4:', merge4.shape)

        final_out = self.final(merge4)
        #print('final_out:', final_out.shape)

        return final_out

class tripleCov2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, k, p):
        super(tripleCov2d, self).__init__()

        self.triple = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.triple(x)
        return out


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                               groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out