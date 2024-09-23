"""
Utils for data preprocessing
https://en.wikipedia.org/wiki/Feature_scaling
"""

import torch
import numpy as np
import pandas as pd
from torch import Tensor


class StandardScale(object):
    """
    Standard normalization method for machine learning.
    Formula: x' = (x - mean(x)) / std(x)
    """
    mean: Tensor
    std: Tensor

    def fit(self, x: Tensor):
        """
        Calculate mean and standard deviation for scaling.
        
        Args:
            x (Tensor): Tensor of shape (num_samples, num_features).

        Returns:
            None
        """
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Scale the input tensor using the calculated mean and std.
        
        Args:
            x (Tensor): Tensor of shape (num_samples, num_features).

        Returns:
            Tensor: Scaled tensor of shape (num_samples, num_features).
        """
        return (x - self.mean) / self.std

    def to_dictionary(self) -> {str: Tensor}:
        """
        Save scaling parameters to a dictionary.
        
        Returns:
            dict: Dictionary with mean and std.
        """
        return {"mean": self.mean, "std": self.std}

    def from_dictionary(self, dictionary: {str: Tensor}, device: torch.device, dtype: torch.dtype):
        """
        Load scaling parameters from a dictionary.
        
        Args:
            dictionary (dict): Dictionary containing mean and std.
            device (torch.device): Device for tensor storage.
            dtype (torch.dtype): Data type for tensors.

        Returns:
            None
        """
        self.mean = dictionary["mean"].to(device, dtype)
        self.std = dictionary["std"].to(device, dtype)


class MinMaxScale(object):
    """
    Min-Max normalization method to scale features to the range [0, 1].
    Formula: x' = (x - min(x)) / (max(x) - min(x))
    """
    min: Tensor
    max: Tensor

    def fit(self, x: Tensor):
        """
        Calculate min and max for scaling.
        
        Args:
            x (Tensor): Tensor of shape (num_samples, num_features).

        Returns:
            None
        """
        self.min = x.min(dim=0).values
        self.max = x.max(dim=0).values

    def __call__(self, x: Tensor) -> Tensor:
        """
        Scale the input tensor using the calculated min and max.
        
        Args:
            x (Tensor): Tensor of shape (num_samples, num_features).

        Returns:
            Tensor: Scaled tensor of shape (num_samples, num_features).
        """
        return (x - self.min) / (self.max - self.min)

    def to_dictionary(self) -> {str: Tensor}:
        """
        Save scaling parameters to a dictionary.
        
        Returns:
            dict: Dictionary with min and max.
        """
        return {"min": self.min, "max": self.max}

    def from_dictionary(self, dictionary: {str: Tensor}, device: torch.device, dtype: torch.dtype):
        """
        Load scaling parameters from a dictionary.
        
        Args:
            dictionary (dict): Dictionary containing min and max.
            device (torch.device): Device for tensor storage.
            dtype (torch.dtype): Data type for tensors.

        Returns:
            None
        """
        self.min = dictionary["min"].to(device, dtype)
        self.max = dictionary["max"].to(device, dtype)


# Dictionary to access different scaling methods
scale = {
    "minmax": MinMaxScale(),
    "standard": StandardScale()
}


def fill_na(df: pd.DataFrame, courses: np.ndarray) -> pd.DataFrame:
    """
    Fill NaN values in DataFrame by replacing them with the mean value of their clusters.
    
    Clusters are formed based on "Birthday", "Best Hand", and course name.
    
    Args:
        df (pd.DataFrame): DataFrame to fill NaN values.
        courses (np.ndarray): Array of course names used for filling NaNs.

    Returns:
        pd.DataFrame: DataFrame with filled NaN values.
    """

    # Extract the year from the Birthday column
    years = np.empty(df["Birthday"].shape[0], dtype=np.int)
    for i, b in enumerate(df["Birthday"]):
        years[i] = int(b.split('-')[0])  # Convert to integer year
    df["Birthday"] = years

    # Fill NaN values based on clusters
    for year in df["Birthday"].unique():
        for hand in df["Best Hand"].unique():
            for course in courses:
                # Create a mask for current year and hand
                mask = (df["Birthday"] == year) & (df["Best Hand"] == hand)
                # Calculate mean value for the current course in the cluster
                val = df.loc[mask, course].mean()
                # Fill NaN values with the calculated mean
                df.loc[mask, course] = df.loc[mask, course].fillna(val)
                
    return df

