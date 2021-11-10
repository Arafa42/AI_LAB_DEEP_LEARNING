import torch

import utilities.utils as utils
import numpy as np

def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""
    MSE = np.square(np.subtract(input_tensor,target)).mean()
    return MSE
