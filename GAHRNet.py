import torch.nn as nn
import os

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import scipy.io as sio
import torch.nn.functional as F
import math
from functools import partial
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import time
from torch import einsum
from einops import rearrange

BN_MOMENTUM = 0.1

class PatchEmbed(nn.Module):        
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),
            
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),
                        
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),            
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),
            
        )

    def forward(self, x):
        x = self.proj(x)
        return x



class ResDWC1(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
                
        # self.conv_constant = nn.Parameter(torch.eye(kernel_size).reshape(dim, 1, kernel_size, kernel_size))
        # self.conv_constant.requires_grad = False
        
    def forward(self, x):
        # return F.conv2d(x, self.conv.weight+self.conv_constant, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim) # equal to x + conv(x)
        return x + self.conv(x)

class ResDWC3(nn.Module):
    def __init__(self, dim, kernel_size=1):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        # self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size) #(12,32,42,42)->45->2->
                    
    def forward(self, x):
        # return x + self.conv(x)
        # x = self.conv(x)
        return self.conv(x)

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=12, stride=6,padding=0):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        # self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride=stride,padding=padding,groups=dim) #(12,32,42,42)->45->2->
                    
    def forward(self, x):
        # return x + self.conv(x)
        return self.conv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
               
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = ResDWC1(hidden_features, 3)
        
    def forward(self, x):       
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention2(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
                
        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5
                
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
                
        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads *3, N).chunk(3, dim=2) # (B, num_heads, head_dim, N)
        
        attn = (k.transpose(-1, -2) @ q) * self.scale
        
        attn = attn.softmax(dim=-2) # (B, h, N, N)
        attn = self.attn_drop(attn)
        
        x = (v @ attn).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class StokenAttention2(nn.Module):
    def __init__(self, dim, refine=True, refine_attention=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.C1 = ResDWC(dim,4,2,1)
        self.C2 = ResDWC(dim,8,4,2)
        self.C3 = ResDWC(dim,12,6,3)
        # self.C4 = ResDWC(dim,12,6,9)
        self.dim = dim
        self.pws32 = nn.Conv2d(32, 8, kernel_size=1)
        self.pws64 = nn.Conv2d(64 , 16, kernel_size=1)
        self.pws128 = nn.Conv2d(128 , 64, kernel_size=1)
        self.pws256 = nn.Conv2d(256 , 128, kernel_size=1)
        # self.pws32 = nn.Conv2d(128, dim, kernel_size=1)

        self.con32 = ResDWC3(32)
        self.con64 = ResDWC3(64)
        self.con128 = ResDWC3(128)
        self.con256 = ResDWC3(256)

        self.refine_attention = refine_attention  
        
        self.scale1 = 8 ** - 0.5
        self.scale2 = 16 ** - 0.5
        self.scale3 = 64 ** - 0.5
        self.scale4 = 128 ** - 0.5
        
        self.unfold = Unfold(3)
        self.fold = Fold(3)
        
        self.stoken_refine32 = Attention(8, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.stoken_refine64 = Attention(16, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.stoken_refine128 = Attention(64, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.stoken_refine256 = Attention(128, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        
        
    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H, W = x.shape   
        
        # 976
         # (B, C, H, W)
        if C == 32:
            stoken_features1 = self.C1(x) # (B, C, h, w)
            stoken_features2 = self.C2(x) # (B, C, h, w)
            stoken_features3 = self.C3(x) # (B, C, h, w)
            stoken_features1 = self.pws32(stoken_features1)
            stoken_features2 = self.pws32(stoken_features2)
            stoken_features3 = self.pws32(stoken_features3)# (B, C/4, h, w)
            x = self.pws32(x)
        
            stoken_features4 = self.stoken_refine32(stoken_features1)
            stoken_features5 = self.stoken_refine32(stoken_features2)
            stoken_features6 = self.stoken_refine32(stoken_features3) # (B, C/4, h, w)
            B,c,h1,w1 = stoken_features1.shape
            B,c,h2,w2 = stoken_features2.shape
            B,c,h3,w3 = stoken_features3.shape
            x1 = x.reshape(B, c, h1, 2, w1, 2).permute(0, 2, 4, 3, 5, 1).reshape(B, h1*w1, 4, c)
            x2 = x.reshape(B, c, h2, 4, w2, 4).permute(0, 2, 4, 3, 5, 1).reshape(B, h2*w2, 16, c)
            x3 = x.reshape(B, c, h3, 6, w3, 6).permute(0, 2, 4, 3, 5, 1).reshape(B, h3*w3, 36, c)

            with torch.no_grad():
                y1 = self.unfold(stoken_features1) # (B, c*9, h1*w1)
                y2 = self.unfold(stoken_features2) # (B, c*9, h2*w2)
                y3 = self.unfold(stoken_features3) # (B, c*9, h3*w3)
                y1 = y1.transpose(1, 2).reshape(B, h1*w1, c, 9)
                y2 = y2.transpose(1, 2).reshape(B, h2*w2, c, 9)
                y3 = y3.transpose(1, 2).reshape(B, h3*w3, c, 9)
                affinity_matrix1 = x1 @ y1 * self.scale1 # (B, h1*w1, 4, 9)
                affinity_matrix2 = x2 @ y2 * self.scale1 # (B, h2*w2, 16, 9)
                affinity_matrix3 = x3 @ y3 * self.scale1 # (B, h3*w3, 36, 9)
                # 874
                affinity_matrix1 = affinity_matrix1.softmax(-1) # (B, h1*w1, 4, 9)
                affinity_matrix2 = affinity_matrix2.softmax(-1) # (B, h2*w2, 16, 9)
                affinity_matrix3 = affinity_matrix3.softmax(-1) # (B, h3*w3, 36, 9)  

                y4 = self.unfold(stoken_features4) # (B, c*9, h1*w1)
                y5 = self.unfold(stoken_features5) # (B, c*9, h2*w2)
                y6 = self.unfold(stoken_features6) # (B, c*9, h3*w3)
                y4 = y4.transpose(1, 2).reshape(B, h1*w1, c, 9)
                y5 = y5.transpose(1, 2).reshape(B, h2*w2, c, 9)
                y6 = y6.transpose(1, 2).reshape(B, h3*w3, c, 9)                                          

            pixel_features1 = y4 @ affinity_matrix1.transpose(-1, -2) # (B, h1*w1, C, h*w)
            pixel_features2 = y5 @ affinity_matrix2.transpose(-1, -2) # (B, hh*ww, C, h*w)
            pixel_features3 = y6 @ affinity_matrix3.transpose(-1, -2) # (B, hh*ww, C, h*w)
            # 687
            pixel_features1 = pixel_features1.reshape(B, h1, w1, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, c, H, W)
            pixel_features2 = pixel_features2.reshape(B, h2, w2, c, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(B, c, H, W)
            pixel_features3 = pixel_features3.reshape(B, h3, w3, c, 6, 6).permute(0, 3, 1, 4, 2, 5).reshape(B, c, H, W)
            
            pixel_features = torch.cat((pixel_features1, pixel_features2), dim=1)
            pixel_features = torch.cat((pixel_features, pixel_features3), dim=1)
            pixel_features = torch.cat((pixel_features, x), dim=1)
            # pixel_features = self.con32(pixel_features)
           
            return pixel_features
        elif C == 64:
            stoken_features1 = self.C1(x) # (B, C, h, w)
            stoken_features2 = self.C2(x) # (B, C, h, w)
            stoken_features3 = self.C3(x) # (B, C, h, w)
            stoken_features1 = self.pws64(stoken_features1)
            stoken_features2 = self.pws64(stoken_features2)
            stoken_features3 = self.pws64(stoken_features3)# (B, C/4, h, w)
            x = self.pws64(x)
        
            stoken_features4 = self.stoken_refine64(stoken_features1)
            stoken_features5 = self.stoken_refine64(stoken_features2)
            stoken_features6 = self.stoken_refine64(stoken_features3) # (B, C/4, h, w)
            B,c,h1,w1 = stoken_features1.shape
            B,c,h2,w2 = stoken_features2.shape
            B,c,h3,w3 = stoken_features3.shape
            x1 = x.reshape(B, c, h1, 2, w1, 2).permute(0, 2, 4, 3, 5, 1).reshape(B, h1*w1, 4, c)
            x2 = x.reshape(B, c, h2, 4, w2, 4).permute(0, 2, 4, 3, 5, 1).reshape(B, h2*w2, 16, c)
            x3 = x.reshape(B, c, h3, 6, w3, 6).permute(0, 2, 4, 3, 5, 1).reshape(B, h3*w3, 36, c)

            with torch.no_grad():
                y1 = self.unfold(stoken_features1) # (B, c*9, h1*w1)
                y2 = self.unfold(stoken_features2) # (B, c*9, h2*w2)
                y3 = self.unfold(stoken_features3) # (B, c*9, h3*w3)
                y1 = y1.transpose(1, 2).reshape(B, h1*w1, c, 9)
                y2 = y2.transpose(1, 2).reshape(B, h2*w2, c, 9)
                y3 = y3.transpose(1, 2).reshape(B, h3*w3, c, 9)
                affinity_matrix1 = x1 @ y1 * self.scale2 # (B, h1*w1, 4, 9)
                affinity_matrix2 = x2 @ y2 * self.scale2 # (B, h2*w2, 16, 9)
                affinity_matrix3 = x3 @ y3 * self.scale2 # (B, h3*w3, 36, 9)
                # 874
                affinity_matrix1 = affinity_matrix1.softmax(-1) # (B, h1*w1, 4, 9)
                affinity_matrix2 = affinity_matrix2.softmax(-1) # (B, h2*w2, 16, 9)
                affinity_matrix3 = affinity_matrix3.softmax(-1) # (B, h3*w3, 36, 9)                                            

                y4 = self.unfold(stoken_features4) # (B, c*9, h1*w1)
                y5 = self.unfold(stoken_features5) # (B, c*9, h2*w2)
                y6 = self.unfold(stoken_features6) # (B, c*9, h3*w3)
                y4 = y4.transpose(1, 2).reshape(B, h1*w1, c, 9)
                y5 = y5.transpose(1, 2).reshape(B, h2*w2, c, 9)
                y6 = y6.transpose(1, 2).reshape(B, h3*w3, c, 9)

            pixel_features1 = y4 @ affinity_matrix1.transpose(-1, -2) # (B, h1*w1, C, h*w)
            pixel_features2 = y5 @ affinity_matrix2.transpose(-1, -2) # (B, hh*ww, C, h*w)
            pixel_features3 = y6 @ affinity_matrix3.transpose(-1, -2) # (B, hh*ww, C, h*w)
            # 687
            pixel_features1 = pixel_features1.reshape(B, h1, w1, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, c, H, W)
            pixel_features2 = pixel_features2.reshape(B, h2, w2, c, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(B, c, H, W)
            pixel_features3 = pixel_features3.reshape(B, h3, w3, c, 6, 6).permute(0, 3, 1, 4, 2, 5).reshape(B, c, H, W)
            
            pixel_features = torch.cat((pixel_features1, pixel_features2), dim=1)
            pixel_features = torch.cat((pixel_features, pixel_features3), dim=1)
            pixel_features = torch.cat((pixel_features, x), dim=1)
            # pixel_features = self.con64(pixel_features)
            return pixel_features
        elif C == 128:
            stoken_features1 = self.C1(x) # (B, C, h, w)
            stoken_features1 = self.pws128(stoken_features1)
            x = self.pws128(x)
            stoken_features2 = self.stoken_refine128(stoken_features1)
            B,c,h1,w1 = stoken_features1.shape
            x1 = x.reshape(B, c, h1, 2, w1, 2).permute(0, 2, 4, 3, 5, 1).reshape(B, h1*w1, 4, c)
            with torch.no_grad():
                y1 = self.unfold(stoken_features1) # (B, c*9, h1*w1)
                y1 = y1.transpose(1, 2).reshape(B, h1*w1, c, 9)
                affinity_matrix1 = x1 @ y1 * self.scale3 # (B, h1*w1, 4, 9)
                # 874
                affinity_matrix1 = affinity_matrix1.softmax(-1) # (B, h1*w1, 4, 9)                                       

                y2 = self.unfold(stoken_features2) # (B, c*9, h1*w1)
                y2 = y2.transpose(1, 2).reshape(B, h1*w1, c, 9)
            
            pixel_features1 = y2 @ affinity_matrix1.transpose(-1, -2) # (B, h1*w1, C, h*w)
            # 687
            pixel_features1 = pixel_features1.reshape(B, h1, w1, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(B, c, H, W)
            pixel_features = torch.cat((pixel_features1, x), dim=1)
            # pixel_features = self.con128(pixel_features)
           
            return pixel_features
        elif C == 256:
            stoken_features1 = x
            stoken_features1 = self.pws256(stoken_features1)
            x = self.pws256(x)
            stoken_features1 = self.stoken_refine256(stoken_features1)
            pixel_features = torch.cat((stoken_features1, x), dim=1)
            # pixel_features = self.con256(pixel_features)
           
            return pixel_features



    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        if self.refine:
            if self.refine_attention:
                stoken_features = self.stoken_refine(stoken_features)
            else:
                stoken_features = self.stoken_refine(stoken_features)
        return stoken_features
        
    def forward(self, x):
        
        return self.stoken_forward(x)
        


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b*c, 1, h, w), self.weights, stride=1, padding=self.kernel_size//2)        
        return x.reshape(b, c*9, h*w)

class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)
           
        
    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size//2)        
        return x

class StokenAttentionLayer(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5):
        super().__init__()
                        
        self.layerscale = layerscale
        
        self.pos_embed = ResDWC1(dim, 3)
                                        
        self.norm1 = LayerNorm2d(dim)
        self.attn = StokenAttention2(dim, refine=True, refine_attention=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.)   
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)
                
        
        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1),requires_grad=True)
        
    def forward(self, x):
        x = self.pos_embed(x)
        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x))) 
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp2(self.norm2(x)))        
        return x




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print("x1=",x.size())
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print("x2=",out.size())
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print("x3=",out.size())
        out = self.conv3(out)
        out = self.bn3(out)
        # print("x4=",out.size())
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        # print("x5=",out.size())
        return out

class GlobalContext(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        # bottleneck information
        # "Compete to compute." NeurIPS 2013
        self.compete = False
        if self.compete:
            self.fc1 = nn.Linear(dim, 2*dim//8)
            self.fc2 = nn.Linear(dim//8, dim)
        else:
            self.fc = nn.Sequential(
                nn.Linear(dim, dim//8),
                act_layer(),
                nn.Linear(dim//8, dim)
            )
        self.weight_gc = True
        if self.weight_gc:
            self.head = 8
            self.scale = (dim//self.head) ** -0.5
            self.rescale_weight = nn.Parameter(torch.ones(self.head))
            self.rescale_bias = nn.Parameter(torch.zeros(self.head))
            self.epsilon = 1e-5

    def _get_gc(self, gap): # gap [b,c]
        if self.compete:
            b,c = gap.size()
            gc = self.fc1(gap).reshape([b,2,-1])
            gc, _ = gc.max(dim=1)
            gc = self.fc2(gc)
            return gc
        else:
            return self.fc(gap)


    def forward(self, x):
        if self.weight_gc:
            b,c,w,h = x.size()
            x = rearrange(x,"b c x y -> b c (x y)")
            gap = x.mean(dim=-1, keepdim=True)
            q, g = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h = self.head), [x,gap])  #[b,head, hdim, n]
            sim = einsum('bhdi,bhjd->bhij', q, g.transpose(-1, -2)).squeeze(dim=-1) * self.scale  #[b,head, w*h]
            std, mean = torch.std_mean(sim, dim=[1,2], keepdim=True)
            sim = (sim-mean)/(std+self.epsilon)
            sim = sim * self.rescale_weight.unsqueeze(dim=0).unsqueeze(dim=-1) + self.rescale_bias.unsqueeze(dim=0).unsqueeze(dim=-1)
            sim = sim.reshape(b,self.head,1, w, h) # [b, head, 1, w, h]
            gc = self._get_gc(gap.squeeze(dim=-1)).reshape(b,self.head,-1).unsqueeze(dim=-1).unsqueeze(dim=-1)  # [b, head, hdim, 1, 1]
            gc = rearrange(sim*gc, "b h d x y -> b (h d) x y")  # [b, head, hdim, w, h] - > [b,c,w,h]
        else:
            gc = self._get_gc(x.mean(dim=-1).mean(dim=-1)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return gc  # [b,c,w,h] for weighted or [b,c,1,1]

class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                StokenAttentionLayer(w, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        return x_fused

class StageModule3(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.GlobalContext1 = GlobalContext(c)
        self.GlobalContext2 = GlobalContext(c * (2 ** 1))
        self.GlobalContext3 = GlobalContext(c * (2 ** 2))

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                StokenAttentionLayer(w, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        x1 = self.fuse_layers[0][1](x[1]) + self.fuse_layers[0][2](x[2]) + self.fuse_layers[0][0](x[0])
        x2 = self.fuse_layers[1][1](x[1]) + self.fuse_layers[1][2](x[2]) + self.fuse_layers[1][0](x[0])
        x3 = self.fuse_layers[2][1](x[1]) + self.fuse_layers[2][2](x[2]) + self.fuse_layers[2][0](x[0])

        x_fused.append(
            self.relu(self.GlobalContext1(x1) + x1)
        )
        x_fused.append(
            self.relu(self.GlobalContext2(x2) + x2)
        )
        x_fused.append(
            self.relu(self.GlobalContext3(x3) + x3)
        )
        return x_fused
    
class StageModule4(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.GlobalContext1 = GlobalContext(c)
        self.GlobalContext2 = GlobalContext(c * (2 ** 1))
        self.GlobalContext3 = GlobalContext(c * (2 ** 2))
        self.GlobalContext4 = GlobalContext(c * (2 ** 3))

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                StokenAttentionLayer(w, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        x1 = self.fuse_layers[0][2](x[2]) + self.fuse_layers[0][3](x[3]) + self.fuse_layers[0][1](x[1])+ self.fuse_layers[0][0](x[0])
        x2 = self.fuse_layers[1][2](x[2]) + self.fuse_layers[1][3](x[3]) + self.fuse_layers[1][1](x[1])+ self.fuse_layers[1][0](x[0])
        x3 = self.fuse_layers[2][2](x[2]) + self.fuse_layers[2][3](x[3]) + self.fuse_layers[2][1](x[1])+ self.fuse_layers[2][0](x[0])
        x4 = self.fuse_layers[3][2](x[2]) + self.fuse_layers[3][3](x[3]) + self.fuse_layers[3][1](x[1])+ self.fuse_layers[3][0](x[0])

        x_fused.append(
            self.relu(self.GlobalContext1(x1) + x1)
        )
        x_fused.append(
            self.relu(self.GlobalContext2(x2) + x2)
        )
        x_fused.append(
            self.relu(self.GlobalContext3(x3) + x3)
        )
        x_fused.append(
            self.relu(self.GlobalContext4(x4) + x4)
        )
        return x_fused
    
class StageModule2(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 输入的分支数，每个分支对应一种尺度
        :param output_branches: 输出的分支数
        :param c: 输入的第一个分支通道数
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.GlobalContext1 = GlobalContext(c)
        self.GlobalContext2 = GlobalContext(c * (2 ** 1))

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            w = c * (2 ** i)  # 对应第i个分支的通道数
            branch = nn.Sequential(
                StokenAttentionLayer(w, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, layerscale=False, init_values = 1.0e-5)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()  # 用于融合每个分支上的输出
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        x1 = self.fuse_layers[0][0](x[0]) + self.fuse_layers[0][1](x[1])
        x2 = self.fuse_layers[1][0](x[0]) + self.fuse_layers[1][1](x[1])
        x_fused.append(
            self.relu(x1)
        )
        x_fused.append(
            self.relu(x2)
        )
        return x_fused
       

class GAHRNet(nn.Module):
    def __init__(self, base_channel: int = 32, num_joints: int = 29, drop_rate=0.):
        super().__init__()
        # Stem
        self.patch_embed = PatchEmbed(3, 64)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False), 
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )
        

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(  
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule2(input_branches=2, output_branches=2, c=base_channel)
        )

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule3(input_branches=3, output_branches=3, c=base_channel),
            StageModule3(input_branches=3, output_branches=3, c=base_channel),
            StageModule3(input_branches=3, output_branches=3, c=base_channel),
            StageModule3(input_branches=3, output_branches=3, c=base_channel)
        )

        # transition3
        self.transition3 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage4
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule4(input_branches=4, output_branches=4, c=base_channel),
            StageModule4(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=1, c=base_channel)
        )

        # Final layer
        self.final_layer = nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)

    def forward(self, x):
        
        x = self.patch_embed(x)
        # print('x2=',x.size())
        x = self.pos_drop(x)

        x = self.layer1(x)
        # print("x3=",x.size())
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list
        
        x = self.stage2(x)
        
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
    
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)
        
        x = self.upsample(x[0])
        x = self.final_layer(x)
        # print("x8=",x.size())

        return x


if __name__ == '__main__':
    model = GAHRNet(num_joints=19,base_channel=32).cuda('cuda:0')

    x = torch.rand(1,3,672,672).cuda('cuda:0')
    
    y = model(x)
    # model = Unfold(3)
    x1 = torch.rand(12,42*42,8,9)
    x2 = torch.rand(12,42*42,4,9)
    x3 = torch.rand(12,42*42,4,8)


    print(y.shape)
    
