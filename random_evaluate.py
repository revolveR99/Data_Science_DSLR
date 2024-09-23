"""
Script to train and evaluate one-vs-all logistic regression
on dataset_train.csv.
"""

import numpy as np  # For numerical operations.
import pandas as pd  # For handling datasets.
from time import time  # For tracking time during execution.
from argparse import ArgumentParser  # For parsing command-line arguments.

# Importing custom modules for configuration, evaluation, and model operations.
from config import Config
from evaluate import accuracy_score
from logreg_train import plot_training
from dslr.preprocessing import scale, fill_na
from dslr.multi_classifier import OneVsAllLogisticRegression


def train_test_split(x: np.ndarray,
                     y: np.ndarray,
                     test_part=0.3,
                     random_state: int or None = None):
    """
    Split dataset into training and testing parts.
    
    Args:
        x (np.ndarray): Feature data.
        y (np.ndarray): Labels.
        test_part (float): Fraction of the data to use for testing.
        random_state (int or None): Seed for reproducibility.

    Returns:
        x_train, x_test, y_train, y_test: Split training and testing sets.
    """
    np.random.seed(random_state)

    # Shuffle the data.
    p = np.random.permutation(len(x))

    # Compute the offset for test data based on the test_part.
    x_offset = int(len(x) * test_part)
    y_offset = int(len(y) * test_part)

    # Split features and labels into training and testing sets.
    x_train = x[p][x_offset:]
    x_test = x[p][:x_offset]
    y_train = y[p][y_offset:]
    y_test = y[p][:y_offset]

    return x_train, x_test, y_train, y_test


def evaluate(data_path: str,
             config_path: str,
             test_part: float,
             v: bool = False):
    """
    Train and evaluate the one-vs-all logistic regression model.

    Args:
        data_path (str): Path to the dataset file.
        config_path (str): Path to the configuration YAML file.
        test_part (float): Fraction of the data to use for testing.
        v (bool): Flag to indicate if the training plot should be visualized.

    Returns:
        None
    """
    # Load configuration file to get selected features and other settings.
    config = Config(config_path)
    courses = config.choosed_features()

    # Start timer for preparation time.
    preparation_t = time()

    # Load the dataset and fill missing values for the selected features.
    df = pd.read_csv(data_path)
    df = fill_na(df, courses)

    # Extract feature values and labels from the dataset.
    x = df[courses].values
    y = df["Hogwarts House"].values

    # Split the data into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_part,
                                                        config.seed)

    # Create a one-vs-all logistic regression model using config settings.
    model = OneVsAllLogisticRegression(
        device=config.device,  # Device for training (e.g., CPU or GPU).
        transform=scale[config.scale],  # Scaling function for features.
        lr=config.lr,  # Learning rate.
        epochs=config.epochs,  # Number of training epochs.
        batch_size=config.batch_size,  # Batch size for training.
        seed=config.seed,  # Seed for reproducibility.
        save_hist=v  # Save loss history if visualization is enabled.
    )
    # Stop timer for preparation time.
    preparation_t = time() - preparation_t

    # Start timer for training time.
    train_t = time()
    # Train the model on the training set.
    model.fit(x_train, y_train)
    # Stop timer for training time.
    train_t = time() - train_t

    # Start timer for prediction time.
    predict_t = time()
    # Predict on the test set.
    p = model.predict(x_test)
    # Stop timer for prediction time.
    predict_t = time() - predict_t

    # Print the number of incorrect predictions and accuracy score.
    print("Wrong predictions:", sum(y_test != p))
    print("Accuracy:", np.round(accuracy_score(y_test, p), 4))

    # Print timing information.
    print('-' * 10 + "TIME" + '-' * 10)
    print("Preparation time:", np.round(preparation_t, 4))
    print("Training time:", np.round(train_t, 4))
    print("Prediction time:", np.round(predict_t, 4))
    print("All time:", np.round(preparation_t + train_t + predict_t, 4))

    # If visualization flag is set, plot the training loss.
    if v:
        plot_training(model)


if __name__ == "__main__":
    # Set up argument parser for command-line inputs.
    parser = ArgumentParser()

    # Optional argument for the dataset file path.
    parser.add_argument('--data_path',
                        type=str,
                        default='data/dataset_train.csv',
                        help='Path to dataset_train.csv file')

    # Optional argument for the configuration file path.
    parser.add_argument('--config_path',
                        type=str,
                        default='config.yaml',
                        help='Path to .yaml file')

    # Optional argument for specifying the test set fraction.
    parser.add_argument('--test_part',
                        type=float,
                        default=0.3,
                        help='Percent of test part. "0.3" means model will '
                             'train on 0.7 of data and evaluate at other 0.3')

    # Optional flag for visualizing the training plot.
    parser.add_argument('-v', action="store_true",
                        help='Visualize training')

    # Parse command-line arguments.
    args = parser.parse_args()

    # Call the evaluate function with the parsed arguments.
    evaluate(args.data_path, args.config_path, args.test_part, args.v)

