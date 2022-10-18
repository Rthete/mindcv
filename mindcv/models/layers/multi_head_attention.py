from mindspore import Tensor
from mindspore import nn
import mindspore.numpy as np
import mindspore.ops as ops

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
        qkv = ops.split(self.to_qvk(x), -1, 3)
        b, p, n, c = qkv[0].shape

        # NOTE: remove einops
        # q, k, v = map(lambda t: rearrange(t.asnumpy(), 'b p n (h d) -> b p h n d', h=self.num_heads), qkv)
        # q = Tensor(q)
        # k = Tensor(k)
        # v = Tensor(v)
        
        # can't use map in GRAPH_MODE
        # q, k, v = map(lambda t: ops.reshape(t, (b, p, self.num_heads, n, c // self.num_heads)), qkv)
        q = ops.reshape(qkv[0], (b, p, self.num_heads, n, c // self.num_heads))
        k = ops.reshape(qkv[1], (b, p, self.num_heads, n, c // self.num_heads))
        v = ops.reshape(qkv[2], (b, p, self.num_heads, n, c // self.num_heads))

        k = np.swapaxes(k, -1, -2)
        dots = ops.matmul(q, k) * self.scale_factor
        softmax = nn.Softmax()
        attn = softmax(dots)
        out = ops.matmul(attn, v)

        # NOTE: remove einops
        # out = rearrange(out.asnumpy(), 'b p h n d -> b p n (h d)')
        # out = Tensor(out)
        
        out = ops.reshape(out, (b, p, n, c))
        return self.w_out(out)