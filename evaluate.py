"""
Script to train and evaluate one-vs-all logistic regression
on ground truth dataset - dataset_truth.csv
"""

import os  # For operating system-related tasks like file paths.
import numpy as np  # For handling arrays and numerical operations.
import pandas as pd  # For data manipulation with dataframes.
from argparse import ArgumentParser  # For handling command-line arguments.

# Importing custom modules for training and predicting logistic regression models.
from logreg_train import train
from logreg_predict import predict


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy score of predictions.
    
    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
    
    Returns:
        float: Accuracy score (correct predictions / total predictions).
    """
    return sum(y_pred == y_true) / len(y_true)


def evaluate(train_path: str,
             test_path: str,
             truth_path: str,
             weights_path: str,
             output_folder: str,
             config_path: str,
             v: bool = False):
    """
    Train, predict, and evaluate the model.

    Args:
        train_path (str): Path to training data CSV.
        test_path (str): Path to testing data CSV.
        truth_path (str): Path to ground truth labels CSV.
        weights_path (str): Path to save the model weights.
        output_folder (str): Folder to save prediction output.
        config_path (str): Path to YAML configuration file.
        v (bool): Verbose flag for printing additional training information.
    """
    print("Training:")
    # Train the logistic regression model and save weights.
    train(train_path, weights_path, config_path, v)
    print('+' * 30)

    print("Predicting:")
    # Use the trained model to predict on test data and save predictions.
    predict(test_path, weights_path, output_folder, config_path)
    print('-' * 30)

    # Load the predicted house assignments and the ground truth.
    pred = pd.read_csv(os.path.join(output_folder, "houses.csv"))
    true = pd.read_csv(truth_path)

    # Extract the house predictions and actual house labels.
    y_pred = pred['Hogwarts House']
    y_true = true['Hogwarts House']

    # Print the number of wrong predictions.
    print("Wrong predictions:", np.sum(y_true != y_pred))
    # Calculate and print the accuracy of the model.
    print("Accuracy:", np.round(accuracy_score(y_true, y_pred), 4))


if __name__ == '__main__':
    # Set up command-line argument parsing.
    parser = ArgumentParser()

    # Add arguments for paths to datasets, weights, and output folder.
    parser.add_argument('--train_path', type=str,
                        default="data/dataset_train.csv",
                        help='Path to "dataset_train.csv" file')

    parser.add_argument('--test_path', type=str,
                        default="data/dataset_test.csv",
                        help='Path to "dataset_test.csv" file')

    parser.add_argument('--truth_path', type=str,
                        default="data/dataset_truth.csv",
                        help='Path to "dataset_truth.csv" file')

    parser.add_argument('--weights_path', type=str,
                        default="data/weights.pt",
                        help='Path to save weights file')

    parser.add_argument('--output_folder', type=str,
                        default="data",
                        help='Path to folder where to save houses.csv')

    parser.add_argument('--config_path', type=str,
                        default="config.yaml",
                        help='Path to .yaml file')

    # Add a flag for visualizing the training process.
    parser.add_argument('-v', action="store_true",
                        help='visualize training')

    # Parse command-line arguments.
    args = parser.parse_args()

    # Call the evaluate function with the parsed arguments.
    evaluate(args.train_path,
             args.test_path,
             args.truth_path,
             args.weights_path,
             args.output_folder,
             args.config_path,
             args.v)

