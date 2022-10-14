import imp
from mindspore import Tensor
from mindspore import nn
import mindspore.numpy as np
import mindspore.ops as ops
# from .layers.rearrange import rearrange
from einops import rearrange

class MultiHeadSelfAttention(nn.Cell):
    """
    Implement multi head self attention layer using the "Einstein summation convention".
    Paper: https://arxiv.org/abs/1706.03762
    Blog: https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    """

    def __init__(self, dim, num_heads=8, dim_head=None):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        _weight_dim = self.num_heads * self.dim_head
        self.to_qvk = nn.Dense(dim, _weight_dim * 3, has_bias=False)
        self.scale_factor = dim ** -0.5

        # Weight matrix for output, Size: num_heads*dim_head X dim
        # Final linear transformation layer
        self.w_out = nn.Dense(_weight_dim, dim, has_bias=False)

    def construct(self, x):
        split = ops.Split(-1, 3)
        qkv = split(self.to_qvk(x))

        q, k, v = map(lambda t: rearrange(t.asnumpy(), 'b p n (h d) -> b p h n d', h=self.num_heads), qkv)
        q = Tensor(q)
        k = Tensor(k)
        v = Tensor(v)

        k = np.swapaxes(k, -1, -2)
        dots = ops.matmul(q, k) * self.scale_factor
        softmax = nn.Softmax()
        attn = softmax(dots)
        out = ops.matmul(attn, v)
        # tensor类型转换为numpy
        out = rearrange(out.asnumpy(), 'b p h n d -> b p n (h d)')
        out = Tensor(out)
        return self.w_out(out)