from typing import Optional

import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops


class MultiHeadAttention(nn.Cell):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (float): Attention dropout. Default: 0.0
        bias (bool): Use bias or not. Default: ``True``

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        bias: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Dense(in_channels=embed_dim, out_channels=embed_dim, has_bias=bias)

        self.attn_dropout = nn.Dropout(keep_prob=1.0 - attn_dropout)
        self.out_proj = nn.Dense(in_channels=embed_dim, out_channels=embed_dim, has_bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(axis=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.batch_matmul = ops.BatchMatMul()

    def construct(self, x: Tensor) -> Tensor:
        # [N, P, C]
        b_sz, n_patches, in_channels = x.shape

        q = ops.reshape(self.qkv_proj(x), (b_sz, n_patches, self.num_heads, in_channels // self.num_heads))
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.reshape(self.qkv_proj(x), (b_sz, n_patches, self.num_heads, in_channels // self.num_heads))
        k = ops.transpose(k, (0, 2, 3, 1))
        v = ops.reshape(self.qkv_proj(x), (b_sz, n_patches, self.num_heads, in_channels // self.num_heads))
        v = ops.transpose(v, (0, 2, 1, 3))

        attn = self.batch_matmul(q, k)
        attn = ops.mul(attn, self.scaling)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        x = ops.transpose(self.batch_matmul(attn, v), (0, 2, 1, 3))
        x = ops.reshape(x, (b_sz, n_patches, in_channels))
        x = self.out_proj(x)
        # x = self.proj_drop(x)
        return x


class TransformerEncoder(nn.Cell):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        ffn_latent_dim (int): Inner dimension of the FFN
        num_heads (int) : Number of heads in multi-head attention. Default: 8
        attn_dropout (float): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers. Default: 0.0

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        *args,
        **kwargs
    ) -> None:

        super().__init__()

        attn_unit = MultiHeadAttention(
            embed_dim,
            num_heads,
            attn_dropout=attn_dropout,
            bias=True
        )

        self.pre_norm_mha = nn.SequentialCell(
            nn.LayerNorm((embed_dim,)),
            attn_unit,
            nn.Dropout(keep_prob=1.0 - dropout)
        )

        self.pre_norm_ffn = nn.SequentialCell(
            nn.LayerNorm((embed_dim,)),
            nn.Dense(in_channels=embed_dim, out_channels=ffn_latent_dim, has_bias=True),
            nn.SiLU(),
            nn.Dropout(keep_prob=1.0 - ffn_dropout),
            nn.Dense(in_channels=ffn_latent_dim, out_channels=embed_dim, has_bias=True),
            nn.Dropout(keep_prob=1.0 - dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout

    def construct(self, x: Tensor) -> Tensor:
        # multi-head attention
        # print("1",x.shape)
        res = x
        x = self.pre_norm_mha(x)
        x = x + res

        # feed forward network
        x = x + self.pre_norm_ffn(x)
        return x
