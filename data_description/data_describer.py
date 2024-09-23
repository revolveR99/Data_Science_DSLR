"""
Class for data description.
This class extends Pandas DataFrame and provides methods for 
describing Hogwarts dataset with functions like mean, standard 
deviation, percentiles, etc.
"""

import numpy as np  # For numerical operations.
import pandas as pd  # For handling data in a DataFrame.
from abc import ABC  # For creating abstract base classes.


class HogwartsDataDescriber(pd.DataFrame, ABC):
    """
    HogwartsDataDescriber class is used for describing statistical properties
    of the dataset related to Hogwarts houses and features.
    It extends Pandas DataFrame and provides additional methods for 
    calculating descriptive statistics.
    """

    # Class-level attributes for Hogwarts houses and associated colors.
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    colors = ['red', 'green', 'blue', 'yellow']

    @staticmethod
    def read_csv(csv_path: str):
        """
        Static method to read a .csv file and return a HogwartsDataDescriber instance.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            HogwartsDataDescriber: A DataFrame-like object with descriptive capabilities.
        """
        return HogwartsDataDescriber(pd.read_csv(csv_path))

    def is_numeric(self, feature: str):
        """
        Check if a feature (column) contains only numeric values.

        Args:
            feature (str): Name of the column.

        Returns:
            bool: True if the column contains numeric values, False otherwise.
        """
        return np.issubdtype(self[feature].dtype, np.number)

    def count(self, feature: str) -> int:
        """
        Return the number of non-NaN elements in a column.

        Args:
            feature (str): Name of the column.

        Returns:
            int: Number of non-NaN elements.
        """
        return len(self[feature].dropna())

    def mean(self, feature: str) -> float:
        """
        Calculate the mean (average) value of a column, ignoring NaNs.

        Args:
            feature (str): Name of the column.

        Returns:
            float: Mean value of the column.
        """
        return sum(self[feature].dropna()) / self.count(feature)

    def std(self, feature: str) -> float:
        """
        Compute the standard deviation of the values in a column.
        The standard deviation measures the spread of the data.

        Formula:
            std = sqrt(mean(abs(x - x.mean())**2))

        Args:
            feature (str): Name of the column.

        Returns:
            float: Standard deviation of the column values.
        """
        dif = self[feature].dropna() - self.mean(feature)
        mean = sum(np.abs(dif) ** 2) / self.count(feature)
        return np.sqrt(mean)

    def min(self, feature: str) -> float:
        """
        Return the minimum value of a column, ignoring NaNs.

        Args:
            feature (str): Name of the column.

        Returns:
            float: Minimum value in the column.
        """
        tmp = np.nan
        for val in self[feature].dropna():
            tmp = tmp if val > tmp else val
        return tmp

    def max(self, feature: str) -> float:
        """
        Return the maximum value of a column, ignoring NaNs.

        Args:
            feature (str): Name of the column.

        Returns:
            float: Maximum value in the column.
        """
        tmp = -np.nan
        for val in self[feature].dropna():
            tmp = tmp if val < tmp else val
        return tmp

    def percentile(self, feature: str, percent: float) -> float:
        """
        Compute the percentile of the values in a column. Percentiles
        indicate the value below which a given percentage of the data falls.

        Args:
            feature (str): Name of the column.
            percent (float): The percentile to compute (between 0 and 100).

        Returns:
            float: The value at the given percentile.
        """
        arr = sorted(self[feature].dropna())  # Sort the values in the column.
        k = (len(arr) - 1) * percent / 100  # Calculate the position of the percentile.
        f = np.floor(k)  # Lower bound index.
        c = np.ceil(k)  # Upper bound index.
        
        # If the floor and ceiling are the same, return the exact value.
        if f == c:
            return arr[int(k)]
        
        # Otherwise, interpolate between the two closest values.
        d0 = arr[int(f)] * (c - k)
        d1 = arr[int(c)] * (k - f)
        return d0 + d1

