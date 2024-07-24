# Change detection head

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from model.cd_modules.psp import _PSPModule
from model.cd_modules.se import ChannelSpatialSELayer
import numpy as np


class FFParser(nn.Module):
    def __init__(self, dim, h=128, w=65):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
        self.dim = dim

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        # assert H == W, "height and width are not equal"
        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size


        # x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        weight = weight.cuda()
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')

        x = x.reshape(B, C, H, W)

        return x




def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the change detection head.
    '''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3: #256 x 256
            in_channels += inner_channel*channel_multiplier[0]
        elif scale < 6: #128 x 128
            in_channels += inner_channel*channel_multiplier[1]
        elif scale < 9: #64 x 64
            in_channels += inner_channel*channel_multiplier[2]
        elif scale < 12: #32 x 32
            in_channels += inner_channel*channel_multiplier[3]
        elif scale < 15: #16 x 16
            in_channels += inner_channel*channel_multiplier[4]
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14') 
    return in_channels

class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(

            nn.Conv2d(dim*len(time_steps), dim, 1)
            if len(time_steps)>1
            else None,
            nn.ReLU()
            if len(time_steps)>1
            else None,
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(),

        )

    def forward(self, x):
        return self.block(x)




class laplacian(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(laplacian, self).__init__()
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.convl = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convl.weight.data.copy_(torch.from_numpy(kernel))

    def forward(self, x):
        x = self.convl(x)
        x = torch.abs(x)
        return x



class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x



class Block_grad(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.Sigmoid(),
        )

        self.block = nn.Sequential(
            nn.Conv2d(dim * len(time_steps), dim_out, 1),
            # if len(time_steps) > 1
            # else None,
            # nn.ReLU()
            # if len(time_steps) > 1
            # else None,
            nn.ReLU()

        )

        self.sobel = nn.Sequential(
            nn.Conv2d(dim * len(time_steps), dim, 1),
            nn.ReLU(),
            Sobelxy(dim),
            nn.Conv2d(dim, dim_out, 1),
            nn.ReLU(),
        )

        self.laplacian = nn.Sequential(
            nn.Conv2d(dim * len(time_steps), dim, 1),
            nn.ReLU(),
            laplacian(dim),
            nn.Conv2d(dim, dim_out, 1),
            nn.ReLU(),
        )

        self.convT = nn.Sequential(nn.Conv2d(dim_out*2, dim_out, 3, padding=1),
            nn.Sigmoid(),

        )


    def forward(self, x):
        x0 = self.block(x)
        x1 = self.block1(x0)
        x2 = self.sobel(x)
        x3 = self.laplacian(x)
        grad_x = torch.cat((x2, x3), dim=1)
        grad_x = self.convT(grad_x)

        return (x1+grad_x)*x0




class cd_head_v2(nn.Module):
    '''
    Change detection head (version 2).
    '''

    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, img_size=256, time_steps=None):
        super(cd_head_v2, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales    = feat_scales
        self.in_channels    = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size       = img_size
        self.time_steps     = time_steps

        self.ffparser = []

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_scales)):
            dim     = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

            self.decoder.append(
                #Block(dim=dim, dim_out=dim, time_steps=time_steps)
                Block_grad(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            d = len(self.feat_scales)-1-i

            # self.ffparser.append(FFParser(dim, 256 // (2 ** (d)), 256 // (2 ** (d + 1)) + 1))

            if i != len(self.feat_scales)-1:
                dim_out =  get_in_channels([self.feat_scales[i+1]], inner_channel, channel_multiplier)
                self.decoder.append(
                AttentionBlock(dim=dim, dim_out=dim_out)
            )

        # self.ffparser.append(FFParser(dim, 512 // (2 ** (d)), 640 // (2 ** (d + 1)) + 1))

        # Final classification head
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(dim_out, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # self.ffparser.reverse()

        # self.ffparser = nn.ModuleList(self.ffparser)


    def forward(self, feats_A):
        # Decoder
        lvl=0
        for layer in self.decoder:
            if isinstance(layer, Block_grad):
                f_A = feats_A[0][self.feat_scales[lvl]]
                # f_B = feats_B[0][self.feat_scales[lvl]]
                for i in range(1, len(self.time_steps)):
                    f_A = torch.cat((f_A, feats_A[i][self.feat_scales[lvl]]), dim=1)
                    # f_B = torch.cat((f_B, feats_B[i][self.feat_scales[lvl]]), dim=1)
                f_A = layer(f_A)
                # f_A = self.ffparser[lvl](f_A)
                # diff = torch.abs( layer(f_A)  - layer(f_B) )
                if lvl!=0:
                    f_A = f_A + x
                lvl+=1
            else:
                f_A = layer(f_A)
                x = F.interpolate(f_A, scale_factor=2, mode="bilinear")

        # x = self.ffparser[0](x)

        # Classifier
        cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))

        return cm

