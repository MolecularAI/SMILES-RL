import time

import numpy as np
import torch

from typing import Union, Dict


def to_tensor(tensor: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Converts numpy ndarray to tensor and/or puts to GPU if available"""

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)

    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def set_default_device_cuda():
    """Sets the default device (cpu or cuda) used for all tensors."""
    cuda_is_available = torch.cuda.is_available()
    print(f"CUDA is available: {cuda_is_available}")
    if cuda_is_available == False:
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        return False
    else:  # device_name == "cuda":
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        torch.set_default_tensor_type(tensor)
        return True


def estimate_run_time(start_time: float, n_steps: int, step: int) -> Dict[str, float]:
    """Estimate the remaining run time

    Args:
        start_time (float): Start time
        n_steps (int): Total number of steps to perform
        step (int): Current step

    Returns:
        Dict[str,float]: Dictionary containing time elapsed and time left.
    """

    time_elapsed = int(time.time() - start_time)
    time_left = time_elapsed * ((n_steps - step) / (step + 1))
    summary = {"elapsed": time_elapsed, "left": time_left}
    return summary
