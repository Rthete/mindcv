"""
MindSpore implementation of `MobileViT`.
Refer to: MobileViT：Light-weight, General-purpose, and Mobile-friendly Vision Transformer
"""

"""
module importing
"""
from threading import local
from typing import Callable, Optional, List
from einops import rearrange #是否可以用einops
import numpy

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Tensor
from mindspore import nn


from .layers.conv_norm_act import Conv2dNormActivation
from .layers.multi_head_attention import MultiHeadSelfAttention
from .registry import register_model



__all__ = [
    "mobilevit_xxs",
    "mobilevit_xs",
    "mobilevit_s",
]

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': '', 'classifier': '',
        **kwargs
    }
    
default_cfgs = {
    'mobilevit_xxs': _cfg(url=''),
    'mobilevit_xs': _cfg(url=''),
    'mobilevit_s': _cfg(url=''),  
}

    
class PreNorm(nn.Cell):
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm((dim,))
        self.fn = fn

    def construct(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

    
class FFN(nn.Cell):
    
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Dense(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def construct(self, x):
        return self.net(x)


class Transformer(nn.Cell):
    
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(nn.CellList([
                PreNorm(dim, MultiHeadSelfAttention(dim, heads, dim_head)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout))
            ]))

    def construct(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class InvertedResidual(nn.Cell):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expand_ratio: int,
                 norm: Optional[nn.Cell] = None,
                 ) -> None:
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        if not norm:
            norm = nn.BatchNorm2d

        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: List[nn.Cell] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(in_channels, hidden_dim, kernel_size=1, norm=norm, activation=None)
            )
        layers.extend([
            # dw
            Conv2dNormActivation(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm=norm,
                activation=nn.SiLU
            ),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1,
                      stride=1, has_bias=False),
            norm(out_channels)
        ])
        self.conv = nn.SequentialCell(layers)

    def construct(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            x = ops.add(identity, x)
        return x


class MobileVitBlock(nn.Cell):
    
    def __init__(self, in_channels, out_channels, d_model, layers, mlp_dim):
        super(MobileVitBlock, self).__init__()
        # Local representation
        self.local_representation = nn.SequentialCell(
            Conv2dNormActivation(in_channels, in_channels, 3, activation=nn.SiLU),
            Conv2dNormActivation(in_channels, d_model, 1, activation=nn.SiLU)
        )

        self.transformer = Transformer(d_model, layers, 1, 32, mlp_dim, 0.1)

        # Fusion block
        self.fusion_block1 = nn.Conv2d(d_model, in_channels, kernel_size = 1)
        self.fusion_block2 = nn.Conv2d(in_channels * 2, out_channels, 3, pad_mode = "pad", padding = 1)

    def construct(self, x):
        #TODO: rearrange处理
        
        local_repr = self.local_representation(x)
        # global_repr = self.global_representation(local_repr)
        _, _, h, w = local_repr.shape
        # local_repr = local_repr.asnumpy()
        # global_repr = rearrange(local_repr, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=2, pw=2)
        # global_repr = Tensor(global_repr)
        # global_repr = self.transformer(global_repr)
        # global_repr = global_repr.asnumpy()
        # global_repr = rearrange(global_repr, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//2, w=w//2, ph=2, pw=2)
        # global_repr = Tensor(global_repr)

        # Fuse the local and gloval features in the concatenation tensor
        # fuse_repr = self.fusion_block1(global_repr)
        fuse_repr = self.fusion_block1(local_repr)
        concat = ops.Concat(axis=1)
        result = self.fusion_block2(concat((x, fuse_repr)))
        return result

   
class MobileViT(nn.Cell):
    """
    主模型
    """
    
    """
    MobileViT model class, based on
    `"MobileViT：Light-weight, General-purpose, and Mobile-friendly Vision Transformer"
    <https://arxiv.org/abs/2110.02178>`_
    
    Args:

    """
    
    def __init__(self,
                 img_size, 
                 features_list, 
                 d_list, 
                 transformer_depth, 
                 expansion, 
                 num_classes = 1000):
        super(MobileViT, self).__init__()

        self.stem = nn.SequentialCell(
            nn.Conv2d(in_channels = 3, out_channels = features_list[0], kernel_size = 3, stride = 2, pad_mode = "pad", padding = 1),
            InvertedResidual(in_channels = features_list[0], out_channels = features_list[1], stride = 1, expand_ratio = expansion),
        )

        self.stage1 = nn.SequentialCell(
            InvertedResidual(in_channels = features_list[1], out_channels = features_list[2], stride = 2, expand_ratio = expansion),
            InvertedResidual(in_channels = features_list[2], out_channels = features_list[2], stride = 1, expand_ratio = expansion),
            InvertedResidual(in_channels = features_list[2], out_channels = features_list[3], stride = 1, expand_ratio = expansion)
        )

        self.stage2 = nn.SequentialCell(
            InvertedResidual(in_channels = features_list[3], out_channels = features_list[4], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[4], out_channels = features_list[5], d_model = d_list[0],
                           layers = transformer_depth[0], mlp_dim = d_list[0] * 2)
        )

        self.stage3 = nn.SequentialCell(
            InvertedResidual(in_channels = features_list[5], out_channels = features_list[6], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[6], out_channels = features_list[7], d_model = d_list[1],
                           layers = transformer_depth[1], mlp_dim = d_list[1] * 4)
        )

        self.stage4 = nn.SequentialCell(
            InvertedResidual(in_channels = features_list[7], out_channels = features_list[8], stride = 2, expand_ratio = expansion),
            MobileVitBlock(in_channels = features_list[8], out_channels = features_list[9], d_model = d_list[2],
                           layers = transformer_depth[2], mlp_dim = d_list[2] * 4),
            nn.Conv2d(in_channels = features_list[9], out_channels = features_list[10], kernel_size = 1, stride = 1, padding = 0)
        )

        self.avgpool = nn.AvgPool2d(kernel_size = img_size // 32)
        self.fc = nn.Dense(features_list[10], num_classes)


    def construct(self, x):
        # Stem
        x = self.stem(x)
        # Body
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # Head
        x = self.avgpool(x)
        
        x = x.view((x.shape[0], -1))
        x = self.fc(x)
        return x
 
model_cfg = {
    "xxs":{
        "features": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
        "d": [64, 80, 96],
        "expansion_ratio": 2,
        "layers": [2, 4, 3]
    },
    "xs":{
        "features": [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
        "d": [96, 120, 144],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
    "s":{
        "features": [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
        "d": [144, 192, 240],
        "expansion_ratio": 4,
        "layers": [2, 4, 3]
    },
} 
        
@register_model
def mobilevit_xxs(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileViT:
    default_cfg = default_cfgs['mobilevit_xxs']
    cfg_xxs = model_cfg["xxs"]
    model = MobileViT(img_size=256, 
                      features_list=cfg_xxs["features"], 
                      d_list=cfg_xxs["d"], 
                      transformer_depth=cfg_xxs["layers"], 
                      expansion=cfg_xxs["expansion_ratio"],
                      num_classes=num_classes)
    
    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

@register_model
def mobilevit_xs(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileViT:
    default_cfg = default_cfgs['mobilevit_xs']
    cfg_xs = model_cfg["xs"]
    model = MobileViT(img_size=256, 
                      features_liset=cfg_xs["features"], 
                      d_list=cfg_xs["d"], 
                      transformer_depth=cfg_xs["layers"], 
                      expansion=cfg_xs["expansion_ratio"],
                      num_classes=num_classes)
    
    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model

@register_model
def mobilevit_s(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> MobileViT:
    default_cfg = default_cfgs['mobilevit_s']
    cfg_s = model_cfg["s"]
    model = MobileViT(img_size=256, 
                      features_liset=cfg_s["features"], 
                      d_list=cfg_s["d"], 
                      transformer_depth=cfg_s["layers"], 
                      expansion=cfg_s["expansion_ratio"],
                      num_classes=num_classes)
    
    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model