import numpy as np
import torch


def numpy_dtype_to_pytorch_dtype(np_dtype):
    if np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.float16:
        return torch.float16
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == np.int16:
        return torch.int16
    elif np_dtype == np.int8:
        return torch.int8
    elif np_dtype == np.uint8:
        return torch.uint8
    elif np_dtype == np.bool_:
        return torch.bool
    else:
        raise ValueError("Unsupported NumPy dtype: {}".format(np_dtype))
