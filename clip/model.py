from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch, math
import torch.nn.functional as F
from torch import nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.avgpool_adv = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.attnpool is not None:
            x = self.attnpool(x)
        else:
            ## add by AJ on 2023.8.9
            x = self.avgpool_adv(x)
            x = torch.flatten(x, 1)
        return x


class Convpass(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)
        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)    # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        x = x.permute(1, 0, 2)  # n，b，d - b，n，d
        B, N, C = x.shape
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv                   [36, 197, 8]
        x_down = self.act(x_down)
        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)  # [36, 8, 14, 14]
        x_patch = self.adapter_conv(x_patch)                                      # [36, 8, 14, 14]
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)       # [36, 196, 8]
        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)      # [36, 8, 1, 1]
        x_cls = self.adapter_conv(x_cls)                                          # [36, 8, 1, 1]
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)                 # [36, 1, 8]
        x_down = torch.cat([x_cls, x_patch], dim=1)                               # [36, 197, 8]
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv                  [36, 197, 768]
        x_up = x_up.permute(1, 0, 2)  # b，n，d - n，b，d
        return x_up

class AIM_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)  # [bs, dim, 1, 1]
        return y.expand_as(x)


### ViT backbone
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_cd, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        #     dilation=dilation, groups=groups, bias=bias)
    def forward(self, x, conv, theta):
        kernel_diff = conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=conv.bias, stride=conv.stride, padding=0, groups=conv.groups)
        return theta * out_diff
class CDCN_Adapter(nn.Module):
    def __init__(self, dim, xavier_init=False):
        super().__init__()
        self.adapter_conv_cd = Conv2d_cd(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.se = SELayer(dim, reduction=4)
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)

        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)
        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)    # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        #print("-------here-------------0517---")
        x = x.permute(1, 0, 2)  # n，b，d - b，n，d
        B, N, C = x.shape
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)
        x_patch = x_down[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)

        ### Adap-CDCN
        x_a = self.se(x_patch)
        x_patch_conv = self.adapter_conv(x_patch)
        x_patch_cd = self.adapter_conv_cd(x_patch, self.adapter_conv, x_a)
        x_patch = x_patch_conv - x_patch_cd
        ###############
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, 14 * 14, self.dim)
        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)
        x_down = torch.cat([x_cls, x_patch], dim=1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        x = x_up.permute(1, 0, 2)  # b，n，d - n，b，d
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        ### AJ
        self.attn = nn.MultiheadAttention(d_model, n_head)  ## vis
        # self.attn = MultiheadAttention(d_model, n_head)

        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        ### Add by AJ at 2023.8.16
        # if d_model == 768:  self.adapter = Convpass(8, xavier_init=False)
        self.grid = (14, 14)
        self.attn_probs = None
        self.attn_grad = None

    def set_attn_probs(self, attn_probs):
        self.attn_probs = attn_probs
    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    def get_mu_sig_4(self, beta, x):  #修改样本混合方式，即正样本混合，负样本混合；但正负样本不一起混合

        # 创建一个形状为[32, 197, 768]的随机张量
        x_ = x.permute(1, 0, 2)                # LND -> NLD   197,24,768--》24,197,168  其中，24是batch
        #print("get_mu_sig   x.shape----0611--", x_.shape)       
        #print("get_mu_sig   x.shape----0611--", x_.shape)
        shape = x_.shape
        # 检查变量的值,不是训练的数据，就直接返回
        if shape[0] != 24 or shape[0] != 32:
            #print(f"当前张量信息: 形状为 {shape}")
            return x_
               
        tensor = x_    

        # 按照第一维度分割为8个张量，每个张量形状为[4, 197, 768]
        chunks = torch.split(tensor, split_size_or_sections=4, dim=0)

        # 将第0、2、4、6个张量合并为一个形状为[16, 197, 768]的张量
        even_chunks = torch.cat([chunks[0], chunks[2], chunks[4], chunks[6]], dim=0)

        # 将第1、3、5、7个张量合并为一个形状为[16, 197, 768]的张量
        odd_chunks = torch.cat([chunks[1], chunks[3], chunks[5], chunks[7]], dim=0)
        even_chunks = even_chunks.permute(1, 0, 2)  
        odd_chunks= odd_chunks.permute(1, 0, 2)  

        # 打印合并后的张量形状以验证
        #print(f"Even chunks shape: {even_chunks.shape}")
        #print(f"Odd chunks shape: {odd_chunks.shape}")
        musig_even = self.get_mu_sig_half(beta, even_chunks)
        musig_odd = self.get_mu_sig_half(beta, odd_chunks)

        Tensor1 = musig_even
        Tensor2 = musig_odd

        # 将Tensor1按照第一维度分割为4个张量，每个张量形状为[4, 197, 768]
        T1, T2, T3, T4 = torch.split(Tensor1, split_size_or_sections=4, dim=0)

        # 将Tensor2按照第一维度分割为4个张量，每个张量形状为[4, 197, 768]
        T5, T6, T7, T8 = torch.split(Tensor2, split_size_or_sections=4, dim=0)

        # 按照指定的合并顺序合并这8个张量
        # 顺序是 T1, T5, T2, T6, T3, T7, T4, T8
        merged_tensor = torch.cat((T1, T5, T2, T6, T3, T7, T4, T8), dim=0)

        # 打印合并后的张量形状以验证
        #print(f"Merged tensor shape: {merged_tensor.shape}")
        return merged_tensor
        
    def get_mu_sig_3(self, beta, x):  #修改样本混合方式，即正样本混合，负样本混合；但正负样本不一起混合

        # 创建一个形状为[32, 197, 768]的随机张量
        x_ = x.permute(1, 0, 2)                # LND -> NLD   197,24,768--》24,197,168  其中，24是batch
        #print("get_mu_sig   x.shape----0611--", x_.shape)                      
        shape = x_.shape
         # 检查变量的值,不是训练的数据，就直接返回
        if shape[0] != 24 or shape[0] != 32:
            #print(f"当前张量信息: 形状为 {shape}")
            return x_
                   
        tensor = x_
        # 按照第一维度分割为8个张量，每个张量形状为[4, 197, 768]
        chunks = torch.split(tensor, split_size_or_sections=4, dim=0)

        # 将第0、2、4、6个张量合并为一个形状为[16, 197, 768]的张量
        even_chunks = torch.cat([chunks[0], chunks[2], chunks[4]], dim=0)

        # 将第1、3、5、7个张量合并为一个形状为[16, 197, 768]的张量
        odd_chunks = torch.cat([chunks[1], chunks[3], chunks[5]], dim=0)
        
        # 将第0、2、4、6个张量合并为一个形状为[16, 197, 768]的张量
        #even_chunks = torch.cat([chunks[0], chunks[2], chunks[4], chunks[6]], dim=0)

        # 将第1、3、5、7个张量合并为一个形状为[16, 197, 768]的张量
        #odd_chunks = torch.cat([chunks[1], chunks[3], chunks[5], chunks[7]], dim=0)
               
              
        even_chunks = even_chunks.permute(1, 0, 2)  
        odd_chunks= odd_chunks.permute(1, 0, 2)  

        # 打印合并后的张量形状以验证
        #print(f"Even chunks shape: {even_chunks.shape}")
        #print(f"Odd chunks shape: {odd_chunks.shape}")
        musig_even = self.get_mu_sig_half(beta, even_chunks)
        musig_odd = self.get_mu_sig_half(beta, odd_chunks)

        Tensor1 = musig_even
        Tensor2 = musig_odd

        # 将Tensor1按照第一维度分割为4个张量，每个张量形状为[4, 197, 768]
        #T1, T2, T3, T4 = torch.split(Tensor1, split_size_or_sections=4, dim=0)

        # 将Tensor2按照第一维度分割为4个张量，每个张量形状为[4, 197, 768]
        #T5, T6, T7, T8 = torch.split(Tensor2, split_size_or_sections=4, dim=0)

        # 按照指定的合并顺序合并这8个张量
        # 顺序是 T1, T5, T2, T6, T3, T7, T4, T8
        #merged_tensor = torch.cat((T1, T5, T2, T6, T3, T7, T4, T8), dim=0)
        
        
        # 将Tensor1按照第一维度分割为4个张量，每个张量形状为[4, 197, 768]
        T1, T2, T3 = torch.split(Tensor1, split_size_or_sections=4, dim=0)

        # 将Tensor2按照第一维度分割为4个张量，每个张量形状为[4, 197, 768]
        T4, T5, T6 = torch.split(Tensor2, split_size_or_sections=4, dim=0)

        # 按照指定的合并顺序合并这8个张量
        # 顺序是 T1, T5, T2, T6, T3, T7, T4, T8
        merged_tensor = torch.cat((T1, T4, T2, T5, T3, T6), dim=0)

        # 打印合并后的张量形状以验证
        #print(f"Merged tensor shape: {merged_tensor.shape}")
        return merged_tensor
        
        
    #def get_mu_sig(self, beta, x):
    def get_mu_sig_half(self, beta, x):
        #print("get_mu_sig   x.shape----0--", x.shape)
        # print("x_---shape---0---", x.shape)
        x_ = x.permute(1, 0, 2)                # LND -> NLD   197,24,768--》24,197,168  其中，24是batch
        #print("x_---shape---1---", x_.shape)
        B = x_.size(0)
        cls_token, pac_token = x_[:, 0:1], x_[:, 1:]
        # print("cls_token------", cls_token.shape)  #torch.Size([32, 1, 768])
        # print("pac_token------", pac_token.shape)  #torch.Size([32, 196, 768])
        #cls_token------ torch.Size([24, 1, 768])
        #pac_token------ torch.Size([24, 196, 768])
        #在PyTorch中，tensor.transpose(1, 2) 表示对张量（tensor）进行转置操作，交换张量的第1维和第2维的位置。
        mu_val =pac_token.transpose(1, 2)
        #print("get_mu_sig   mu_val.shape----0--", mu_val.shape)
        pac_tmp = torch.unsqueeze(pac_token.transpose(1, 2), dim=-1)   #pac_tmp------ torch.Size([24, 768, 196, 1])
        #print("pac_tmp------", pac_tmp.shape)
        x = pac_tmp.reshape((pac_tmp.shape[0], pac_tmp.shape[1],) + tuple(self.grid))
        #print("x------", x.shape)   #torch.Size([24, 768, 14, 14])

        mu = x.mean(dim=[2, 3], keepdim=True)  # compute instance mean  也就是在14*14这个小图的基础上处理
        var = x.var(dim=[2, 3], keepdim=True)  # compute instance variance
        #print("mu------", mu.shape)
        #print("var------", var.shape)

        sig = (var + 1e-6).sqrt()              # compute instance standard deviation
        mu, sig = mu.detach(), sig.detach()    # block gradients
        #在PyTorch中，tensor.detach() 方法用于从当前的计算图中分离出一个张量，使其不再参与梯度计算，也就是说，它不会在反向传播中接收梯度
        x_normed = (x - mu) / sig
        lmda = beta.sample((B, 1, 1, 1))  # sample instance-wise convex weights
        lmda = lmda.to(x.device)
        perm = torch.randperm(B)
        #torch.randperm 是 PyTorch 中的一个函数，用于生成一个随机排列的整数序列。
        # 这个序列中的每个数字都是从 0 到 n-1 的一个唯一整数，其中 n 是你指定的长度。
        mu2, sig2 = mu[perm], sig[perm]           # shuffling
        mu_mix = mu * lmda + mu2 * (1 - lmda)     # generate mixed mean
        sig_mix = sig * lmda + sig2 * (1 - lmda)  # generate mixed standard deviation

        #print("mu_mix.shape------", mu_mix.shape)  #torch.Size([24, 768, 1, 1])
        #print("sig_mix.shape------", sig_mix.shape)  #torch.Size([24, 768, 1, 1])
        # pac_tmp = x_normed * sig_mix + mu_mix
        # pac_token = pac_tmp.flatten(2).transpose(1, 2)
        # x = torch.cat((cls_token, pac_token), dim=1)
        # x_ = x.permute(1, 0, 2)  # LND -> NLD

        #return x_normed*sig_mix + mu_mix
        if self.training:
            #musig = torch.cat((mu_mix, sig_mix), dim=1) #torch.Size([24, 1536, 1, 1])
            musig = x_normed*sig_mix + mu_mix  #需要看看此时的X维度  章烈剽修改  [32, 768, 14, 14]
            #print("----training---enterherer 0521---")
            #print("musig.shape----0--", musig.shape)
        else:
            #musig = torch.cat((mu, sig), dim=1)
            #print("----eval---enterherer 0521---")
            musig = mu_val       #需要看看此时的X维度 章烈剽修改
            #print("musig.shape---1---", musig.shape)
        musig = musig.view(B, 768, -1)    #[32, 768, 196]  推理时 B = 15
        cls_token = cls_token .permute(0, 2, 1)   #[32, 768, 1]
        # print("musig.shape---20--", musig.shape)
        # print("cls_token.shape---20--", cls_token.shape)
        musig = torch.cat((cls_token, musig), dim=2)  #([32, 768, 197])
        musig = musig.permute(0, 2, 1)  # [32, 197,768]   和x保持一致的形状
        #musig = torch.flatten(musig, 1)  #压平从第二个维度开始的所有数据 [24, 1536]
        #print("musig.shape---0520---", musig.shape)
        return musig

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        ### AJ
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask,
        #                  attention_probs_forward_hook=self.set_attn_probs,
        #                  attention_probs_backwards_hook=self.set_attn_grad)[0]  ## vis
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x: torch.Tensor, beta, adapter):
        #print("x---forward---",x.shape)  #[197, 24, 768]   ([77, 24, 512])
        # #章烈剽备注：[197, 24, 768]对应的是图像信息   ([77, 24, 512])对应的是文本信息
        x = x + self.attention(self.ln_1(x))
        
        #print("x.shape[0]---forward---", x.shape[0])
        if x.shape[0] == 197:   ## adapter only in visual encoder ##  仅仅对文本信息进行处理
            #
            x_ = adapter(self.ln_1(x)).permute(1,0,2) # x_  [197, 32,,768]
            cdc_x =  x_.permute(1,0,2)  #[32, 197,,768]
           
            #print("x---forward--cdc-",x_.shape)   #[197, 32, 768]
            # cdc_x = cdc_x.permute(1, 0, 2)  # [32, 197,768]   和x保持一致的形状
            x_origin = x.permute(1,0,2) # [32, 197,768]  
            #x_origin = x_origin.permute(1, 0, 2)  # [32, 197,768]   和x保持一致的形状
            #print("cdc_x------",cdc_x.shape)   #[197, 32, 768]
            #import pdb; pdb.set_trace()
            musig = self.get_mu_sig_4(beta, x) + cdc_x  + x_origin  #+ self.mlp(self.ln_2(x_origin))              
            #musig = self.get_mu_sig_4(beta, x)
        else:  #这个是处理文本，所以musig为空
            x_ = 0
            musig = None

        x = x + self.mlp(self.ln_2(x))  + x_  #保留cdc 2024.06.23做flip_it实验，去掉cdc。
        #print("x.shape---end---", x.shape)  #([197, 24, 768])    ([77, 24, 512])
        return x, musig

from .vit_moe import Adapter_MoElayer2
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        #print("self.width,self.layers,heads-------",self.width,self.layers,heads)
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.beta = torch.distributions.Beta(0.1, 0.1)

        if width == 768:
            print("----come here--self.adapter = Adapter_MoElayers(dim=8)---")
            self.adapter = Adapter_MoElayer2()
        else:
            self.adapter = nn.Identity()

    def forward(self, x: torch.Tensor):
        mu_sig = []
        for i, blk in enumerate(self.resblocks):
            x, musig = blk(x, self.beta, self.adapter)
            #print("musig.shape---list content--", musig.shape)  #[32,1536]
            mu_sig.append(musig)
        #print("mu_sig---list--",len(mu_sig))  #12
        #tensors = [musig for _ in range(len(mu_sig))]
        # 将这些张量堆叠成一个更大的张量，形状为[12, 32, 197, 768]

        # if x.shape[0] == 197:  ## adapter only in visual encoder ##  仅仅对文本信息进行处理
        #     stacked_tensors = torch.stack(mu_sig)
        #     # 计算平均值，沿着第一个维度（即12个张量的维度）
        #     mean_tensor = torch.mean(stacked_tensors, dim=0)
        #     musig_out = mean_tensor

        # print("----mu_sig[0].shape", mu_sig[0].shape)
        # print("----len(mu_sig)", len(mu_sig))

        if mu_sig[0] is None:
            #print("mu_sig[0] == None")
            musig_out = None
        else:
            #print("mu_sig[0].shape", mu_sig[0].shape)
            stacked_tensors = torch.stack(mu_sig)
            # 计算平均值，沿着第一个维度（即12个张量的维度）
            mean_tensor = torch.mean(stacked_tensors, dim=0)
            musig_out = mean_tensor
        # 输出平均后的张量形状
        #print("musig_out.shape----00000---",musig_out.shape)  # 输出: torch.Size([32, 197, 768])
        #musig = torch.mean(torch.cat([ms.unsqueeze(0) for ms in mu_sig], dim=0), dim=0)
        return x, musig_out


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)  ## 768 12 12
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, qfpl=False):
        # print("--x.dtype--",x.dtype)
        # print("--self.conv1.weight.shape--", self.conv1.weight.shape)
        # print("--self.conv1.weight.dtype--", self.conv1.weight.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        if torch.isnan(x).any():
            print("output---x.shape---", x.shape)
            #print("output---x---", x)
            raise ValueError("输出张量包含NaN值")
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        #print("----0---x---", x.shape)
        x, mu_sig = self.transformer(x)
        #print("----1---x---", x.shape)
        #print("before --x.shape,mu_sig.shape---", x.shape, mu_sig.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        #mu_sig = mu_sig.permute(1, 0, 2)  # LND -> NLD
        #print("after---x.shape,mu_sig.shape---", x.shape, mu_sig.shape)

        cls = self.ln_post(x[:, 0, :])
        mu_sig = self.ln_post(mu_sig[:, 0, :])
        #print("0520 --x.shape,mu_sig.shape---", x.shape, mu_sig.shape)

        embed = x
        if qfpl: ## add by AJ 2023.9.23
            cls = cls @ self.proj
            mu_sig = mu_sig @ self.proj
            embed = embed @ self.proj
            return cls, embed, mu_sig

        if self.proj is not None:
            cls = cls @ self.proj
            mu_sig = mu_sig @ self.proj
            embed = embed @ self.proj
            #print("comee--here0518-----------")
        #print("0520 -after -x.shape,mu_sig.shape---", x.shape, mu_sig.shape)
        return mu_sig, cls


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64  ### 768//64=12
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        ### text
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x,_ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:  ### for k, v in state_dict.items():
        if key in state_dict:
            del state_dict[key]

    #convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


