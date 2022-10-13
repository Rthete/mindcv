if __name__ == '__main__':
    import numpy as np
    import mindspore
    from mindspore import Tensor
    from mindcv.models.mobilevit import mobilevit_xxs
    from mindcv.models.densenet import densenet121

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    model = mobilevit_xxs()
    # model = densenet121()
    print(model)
    dummy_input = Tensor(np.random.rand(1, 3, 256, 256), dtype=mindspore.float32)
    y = model(dummy_input)
    print(y.shape)
    