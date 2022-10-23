import numpy as np
import mindspore
from mindspore import Tensor
from mindcv.models.mobilevit import mobilevit_xxs

mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
mindspore.set_context(device_target="GPU", device_id=1)
# mindspore.set_context(mode=mindspore.GRAPH_MODE)
# @mindspore.ms_function
def test():
    model = mobilevit_xxs()
    # print(model) # can only print in PYNATIVE_MODE
    dummy_input = Tensor(np.random.rand(1, 3, 256, 256), dtype=mindspore.float32)
    y = model(dummy_input)
    return y

if __name__ == '__main__':
    print(test().shape)
