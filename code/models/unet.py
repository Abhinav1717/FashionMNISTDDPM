import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import os 

device = None

class DoubleConv(nn.Module):
    '''(convolution => [BN] => ReLU) * 2'''

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
            )
        
        self.conv2 = nn.Sequential (
            nn.Conv2d(mid_channels+1, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        

    def forward(self, x,time_embedding):

        projected_time_embedding = ProjectedTimeEmbedding(tuple(time_embedding.shape[1:]),tuple(x.shape[1:])).to(device)(time_embedding)
        x = torch.cat((x,projected_time_embedding),dim=1)
        x = self.conv1(x)
        x = torch.cat((x,projected_time_embedding),dim=1)
        x = self.conv2(x)
        
        return x
class Down(nn.Module):
    '''Downscaling with maxpool then double conv'''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
        

    def forward(self, x,time_embedding):
        x = self.maxpool(x)
        return self.conv(x,time_embedding)

class Up(nn.Module):
    '''Upscaling then double conv'''

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if using bilinear interpolation, use the faster method
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2,time_embedding):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x,time_embedding)

class ProjectedTimeEmbedding(nn.Module):
    def __init__(self,time_embeeding_shape,output_shape):
        super().__init__()
        
        self.output_shape = output_shape
        # print(output_shape)
        flattend_output_shape = output_shape[1]*output_shape[2]
        self.projection = nn.Linear(time_embeeding_shape[0],flattend_output_shape)
        
    def forward(self,time_embedding):
        projected_time_embedding = self.projection(time_embedding)
        reshaped_time_embedding = projected_time_embedding.view(-1,1,self.output_shape[1],self.output_shape[2])
        return reshaped_time_embedding
        
class NoisePredictor(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        
        global device
        device = torch.device(os.getenv('device'))
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.architecture = [in_channels,128,256,512,1024,512,256,128,out_channels]
        self.inc = DoubleConv(self.architecture[0]+1, self.architecture[1])
        self.down1 = Down(self.architecture[1]+1, self.architecture[2])
        self.down2 = Down(self.architecture[2]+1, self.architecture[3])
        self.down3 = Down(self.architecture[3]+1,self.architecture[4])
        self.up1 = Up(self.architecture[4]+self.architecture[3]+1, self.architecture[5], self.bilinear)
        self.up2 = Up(self.architecture[5]+self.architecture[2]+1,self.architecture[6], self.bilinear)
        self.up3 = Up(self.architecture[6]+self.architecture[1]+1,self.architecture[7],self.bilinear)
        self.outc = nn.Conv2d(self.architecture[7]+1, self.architecture[8], kernel_size=1)

    def forward(self, x,time_embed):
        
        x1 = self.inc(x,time_embed)
        # print(x1.shape,"output")
        x2 = self.down1(x1,time_embed)
        # print(x2.shape,"output")
        x3 = self.down2(x2,time_embed)
        # print(x3.shape,"output")
        x4 = self.down3(x3,time_embed)
        
        x = self.up1(x4, x3,time_embed)
        # print(x.shape,"output")
        x = self.up2(x, x2,time_embed)
        # print(x.shape,"output")
        # print(x.shape,"output")
        x = self.up3(x,x1,time_embed)
        projected_time_embedding = ProjectedTimeEmbedding(tuple(time_embed.shape[1:]),tuple(x.shape[1:])).to(device)(time_embed)
        x = torch.cat((x,projected_time_embedding),dim=1)
        x = self.outc(x)
        # x = torch.sigmoid(x)
        
        return x