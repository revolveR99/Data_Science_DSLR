"""
One-vs-all logistic regression implementation using PyTorch tensors.
"""

import torch
import numpy as np

from dslr.classifier import LogisticRegression
from dslr.pytorch_utils import get_device, to_tensor


class OneVsAllLogisticRegression(object):
    def __init__(self,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 transform: callable = None,
                 lr: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = None,
                 seed: int = None,
                 save_hist: bool = False):
        """
        Initialize the One-vs-All Logistic Regression model.

        Args:
            device (str): Device to perform computations on (CPU or GPU).
            dtype (torch.dtype): Data type for the tensors.
            transform (callable): Function to transform input features.
            lr (float): Learning rate.
            epochs (int): Number of training epochs.
            batch_size (int): Number of samples per gradient update.
            seed (int): Random seed for reproducibility.
            save_hist (bool): If True, store loss history for visualization.
        """
        self.device = get_device(device)
        self.dtype = dtype
        self.transform = transform
        self.lr = lr
        self.epochs = epochs
        self.models = []  # List to hold models for each class
        self.labels = None  # Unique labels for classification
        self.batch_size = batch_size
        self.save_hist = save_hist
        if isinstance(seed, int):
            torch.manual_seed(seed)  # Set random seed

    def predict(self, x: torch.Tensor or np.ndarray) -> np.ndarray:
        """
        Predict labels for given input samples.

        Args:
            x (torch.Tensor or np.ndarray): Input samples of shape (num_samples, num_features).

        Returns:
            np.ndarray: Array of predicted labels of shape (num_samples).
        """
        if not isinstance(x, torch.Tensor):
            x = to_tensor(x, self.device, self.dtype)  # Convert to tensor

        # Scale input features if a transform is provided
        if self.transform is not None:
            x = self.transform(x)

        # Calculate the probability of assigning input samples to each class
        p = []
        for model in self.models:
            p.append(model.predict(x))

        # Assign labels according to the highest prediction probability
        p = torch.stack(p).t()  # Shape: (num_samples, num_classes)
        p = torch.argmax(p, dim=1).cpu()  # Get the index of the max probability
        labels = self.labels[p]  # Convert indices to actual labels
        return labels

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train multiple logistic regression models (one for each label) on the training set.

        Args:
            x (np.ndarray): Input samples of shape (num_samples, num_features).
            y (np.ndarray): Array of labels of shape (num_samples).

        Returns:
            None
        """
        if self.batch_size is None:
            self.batch_size = x.shape[0]  # Use all samples if batch size is not set

        x = to_tensor(x, self.device, self.dtype)  # Convert input to tensor

        # Split labels into one-vs-all binary sets
        bin_labels = self._split_labels(y)
        bin_labels = to_tensor(bin_labels, self.device, self.dtype)  # Convert binary labels to tensor

        # Scale features using the provided transform if available
        if self.transform is not None:
            self.transform.fit(x)
            x = self.transform(x)

        for labels in bin_labels:
            # Create a model for each class
            model = LogisticRegression(self.device,
                                       self.dtype,
                                       self.batch_size,
                                       self.epochs,
                                       self.lr,
                                       self.save_hist)
            # Train the model
            model.fit(x, labels)
            self.models.append(model)  # Store the model

    def _split_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Split labels into one-vs-all binary sets.

        Args:
            y (np.ndarray): Array of labels of shape (num_samples).

        Returns:
            np.ndarray: Binary array of shape (num_unique_labels, num_samples).
        """
        self.labels = np.unique(y)  # Get unique labels
        splitted_labels = np.zeros((self.labels.shape[0], y.shape[0]))

        for label, new_y in zip(self.labels, splitted_labels):
            new_y[np.where(y == label)] = 1  # Set 1 for current label
        return splitted_labels

    def save(self, path: str):
        """
        Save model parameters to a file.

        Args:
            path (str): Path where to save model weights.

        Returns:
            None
        """
        models_w = {"transform": self.transform.to_dictionary()}
        for model, label in zip(self.models, self.labels):
            models_w[label] = model.to_dictionary()  # Store model parameters by label
        torch.save(models_w, path)  # Save to file

    def load(self, path: str):
        """
        Load model parameters from a file.

        Args:
            path (str): Path to the saved model file.

        Returns:
            None
        """
        models_w = torch.load(path)  # Load model weights

        self.transform.from_dictionary(models_w.pop("transform"),
                                       self.device,
                                       self.dtype)  # Load transformation parameters
        self.labels = np.array(list(models_w.keys()))  # Retrieve labels

        for w in models_w.values():
            model = LogisticRegression(self.device,
                                       self.dtype,
                                       self.batch_size,
                                       self.epochs,
                                       self.lr)
            model.from_dictionary(w)  # Load model parameters
            self.models.append(model)  # Add the model to the list

