import torch

import utilities.utils as utils
import numpy as np

def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Solution 1
    MSE = np.square(np.subtract(input_tensor,target)).mean()

    # Solution 2
    #loss = torch.nn.MSELoss()
    #MSE = torch.mean(loss(input_tensor, target))

    return MSE
