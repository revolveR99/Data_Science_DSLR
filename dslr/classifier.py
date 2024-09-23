"""
Logistic regression implementation using PyTorch tensors.

For more details, refer to: https://en.wikipedia.org/wiki/Logistic_regression
"""

import torch
from torch import Tensor


class LogisticRegression(object):
    a: Tensor  # Free member (intercept)
    b: Tensor  # Coefficients (weights)

    def __init__(self, device: torch.device,
                 dtype: torch.dtype,
                 batch_size: int,
                 epochs: int = 100,
                 lr: float = 0.001, save_hist: bool = False):
        """
        Initialize the logistic regression model.

        Args:
            device (torch.device): Device to perform computations on (CPU or GPU).
            dtype (torch.dtype): Data type for the tensors.
            batch_size (int): Number of samples per gradient update.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            save_hist (bool): If True, store loss history for visualization.
        """
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.iteration = 0
        self.hist = [] if save_hist else None  # History for loss values

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate the probability of input samples being assigned to the first class.

        Args:
            x (Tensor): Tensor of shape (num_samples, num_features).

        Returns:
            Tensor: Tensor of shape (num_samples) containing probabilities.
        """
        return 1.0 / (1.0 + torch.exp(x @ -self.b - self.a))  # Logistic function

    def fit(self, x: Tensor, y: Tensor):
        """
        Train the logistic regression model on a given training set using
        gradient descent.

        Args:
            x (Tensor): Tensor of shape (num_samples, num_features).
            y (Tensor): Tensor of shape (num_samples) - labels.

        Returns:
            None
        """
        # Initialize the free member and coefficients
        self.a = torch.randn(1).uniform_(-0.5, 0.5).to(self.device)
        self.b = torch.randn(x.shape[1]).uniform_(-0.5, 0.5).to(self.device)

        for self.iteration in range(self.epochs):
            # Random permutation for stochastic or mini-batch gradient descent
            perm = torch.randperm(x.shape[0])[:self.batch_size]

            # Calculate the gradient at the current point
            tmp_a, tmp_b = self._calculate_anti_gradient(x[perm], y[perm])

            # Update the free member and coefficients
            self.a += self.lr * tmp_a / perm.shape[0]
            self.b += self.lr * tmp_b / perm.shape[0]

            # Save loss history for visualization
            if self.hist is not None:
                self.hist.append(self._loss(x, y))

    def _calculate_anti_gradient(self,
                                 x: Tensor,
                                 y: Tensor) -> (Tensor, Tensor):
        """
        Calculate the anti-gradient of the log-likelihood function.

        Args:
            x (Tensor): Tensor of shape (num_samples, num_features).
            y (Tensor): Tensor of shape (num_samples) - labels.

        Returns:
            (Tensor, Tensor): Gradients for the free member and coefficients.
        """
        p = self.predict(x)  # Predicted probabilities
        dif = y - p  # Difference between true labels and predictions

        # Calculate the gradient for the free member
        da = torch.sum(dif)

        # Calculate the gradient for the coefficients
        db = x.t() @ dif
        return da, db

    def _loss(self, x: Tensor, y: Tensor) -> float:
        """
        Calculate the log-likelihood loss function.

        Args:
            x (Tensor): Tensor of shape (num_samples, num_features).
            y (Tensor): Tensor of shape (num_samples) - labels.

        Returns:
            float: Logarithm of the likelihood function.
        """
        # Prevent logarithm of zero by adding a small constant
        p = self.predict(x) + 0.000001

        # Calculate the loss
        loss = torch.sum(y * torch.log(p) +
                         (1.0 - y) * torch.log(1.0 - p)) / -x.shape[0]
        return float(loss.cpu().numpy())  # Return loss as a Python float

    def to_dictionary(self) -> {str: Tensor}:
        """
        Store model parameters in a dictionary.

        Returns:
            dict: Dictionary containing model parameters.
        """
        return {"a": self.a, "b": self.b}

    def from_dictionary(self, dictionary: {str: Tensor}):
        """
        Load model parameters from a dictionary.

        Args:
            dictionary (dict): Dictionary containing model parameters.

        Returns:
            None
        """
        self.a = dictionary["a"].to(self.device, self.dtype)
        self.b = dictionary["b"].to(self.device, self.dtype)

