"""
Some functions for PyTorch
"""

import torch
import numpy as np


def get_device(device: str) -> torch.device:
    """
    Get the appropriate device for PyTorch.

    Args:
        device (str): Device string, e.g., "cpu", "cuda", or "cuda:{device_index}".

    Returns:
        torch.device: The specified device for PyTorch.
    """
    if "cpu" not in device:
        if torch.cuda.is_available():
            device = torch.device(device)
        else:
            exit("Cuda not available")
    else:
        device = torch.device(device)
    return device


def to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a NumPy array to a PyTorch tensor.

    Args:
        x (np.ndarray): Input array to convert.
        device (torch.device): The device to store the tensor on.
        dtype (torch.dtype): The desired data type of the tensor.

    Returns:
        torch.Tensor: Converted tensor.
    """
    return torch.from_numpy(x).to(device, dtype)

