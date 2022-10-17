from glob import glob
import numpy as np
import mindspore
from mindspore import Tensor
from mindcv.models.mobilevit import mobilevit_xxs
from einops import rearrange

mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
# mindspore.set_context(mode=mindspore.GRAPH_MODE)

# @mindspore.ms_function
def test():
    model = mobilevit_xxs()
    # model = densenet121()
    # print(model)
    dummy_input = Tensor(np.random.rand(1, 3, 256, 256), dtype=mindspore.float32)
    y = model(dummy_input)
    return y
    # print(y.shape)

def testrearrange1():
    local_repr = Tensor(np.random.rand(1, 64, 32, 32), dtype=mindspore.float32)
    b, d, h, w = local_repr.shape

    ph, pw = 2, 2
    h = h // ph
    w = w // pw
    global_repr = np.reshape(local_repr, (b, ph * pw, h * w, d))
    print(global_repr.shape)

    # local_repr = local_repr.asnumpy()
    # global_repr1 = rearrange(local_repr, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=2, pw=2)
    # global_repr1 = Tensor(global_repr1)
    # print(global_repr1.shape)

    global_repr = np.reshape(global_repr, (b, d, h * ph, w * pw))
    print(global_repr.shape)


if __name__ == '__main__':
    print(test().shape)
    # testrearrange1()