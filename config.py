"""
Class for loading and storing configurations from a YAML file.
"""

import yaml  # Import YAML to parse configuration files.
import numpy as np  # Import NumPy to handle arrays.


class Config(object):
    """
    A class to load and store configurations from a YAML file.
    """

    def __init__(self, filepath: str):
        """
        Initialize the Config object by loading configurations from the specified file.

        Args:
            filepath (str): The path to the YAML configuration file.
        """
        # Open the file and load the configurations using safe YAML loading.
        with open(filepath) as f:
            config = yaml.safe_load(f)  # Safely load the YAML file into a dictionary.

        # Dynamically assign each key-value pair from the loaded configuration
        # as attributes of the Config instance.
        for key in config.keys():
            setattr(self, key, config[key])  # Set each key as an attribute.

    def choosed_features(self) -> np.ndarray:
        """
        Return an array of features that have been selected (masked by True values).

        Returns:
            np.ndarray: Array of selected features.
        """
        # Convert the feature names (keys of self.features) to a NumPy array.
        features = np.array(list(self.features.keys()))
        # Convert the boolean values (values of self.features) to a NumPy array mask.
        mask = np.array(list(self.features.values()))
        # Return only the feature names where the mask is True.
        return features[mask]

