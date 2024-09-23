"""
Script to predict labels with one-vs-all logistic regression.
It saves predicted labels in houses.csv.
"""

import os  # For handling file paths and operations.
import numpy as np  # For numerical operations.
import pandas as pd  # For handling datasets.
from time import time  # For timing the execution of functions.
from argparse import ArgumentParser  # For handling command-line arguments.

# Importing custom modules for configuration, data preprocessing, and model.
from config import Config
from dslr.preprocessing import scale, fill_na
from dslr.multi_classifier import OneVsAllLogisticRegression


def predict(data_path: str,
            weights_path: str,
            output_folder: str,
            config_path: str):
    """
    Predict house labels using one-vs-all logistic regression.

    Args:
        data_path (str): Path to test dataset.
        weights_path (str): Path to saved model weights.
        output_folder (str): Folder to save predicted results (houses.csv).
        config_path (str): Path to configuration YAML file.
    """
    # Load configuration file.
    config = Config(config_path)
    # Select features from config for prediction.
    courses = config.choosed_features()

    # Start timer for preparation time.
    preparation_t = time()

    # Read test dataset and fill missing values.
    df = pd.read_csv(data_path)
    df = fill_na(df, courses)

    # Extract the selected feature values for prediction.
    x = df[courses].values

    # Initialize the one-vs-all logistic regression model.
    model = OneVsAllLogisticRegression(
        device=config.device,  # Use the specified device (e.g., CPU or GPU).
        transform=scale[config.scale],  # Apply scaling transformation.
    )

    # Load the model weights from the specified path.
    model.load(weights_path)

    # Stop timer for preparation time.
    preparation_t = time() - preparation_t

    # Start timer for prediction time.
    predict_t = time()
    # Perform prediction using the model.
    p = model.predict(x)
    # Stop timer for prediction time.
    predict_t = time() - predict_t

    # Save the predicted house labels in a CSV file.
    pred = pd.DataFrame(p, columns=["Hogwarts House"])
    pred.to_csv(os.path.join(output_folder, "houses.csv"),
                index_label="Index")

    # Print preparation and prediction times.
    print("Preparation time:", np.round(preparation_t, 4))
    print("Prediction time:", np.round(predict_t, 4))
    print("All time:", np.round(preparation_t + predict_t, 4))


if __name__ == "__main__":
    # Set up argument parser for command-line inputs.
    parser = ArgumentParser()

    # Add required argument for the path to test dataset.
    parser.add_argument('data_path', type=str,
                        help='Path to "dataset_test.csv" file')

    # Add required argument for the path to model weights.
    parser.add_argument('weights_path', type=str,
                        help='Path to "weights.pt" file')

    # Optional argument for specifying the output folder for the results.
    parser.add_argument('--output_folder', type=str, default="data",
                        help='Path to folder where to save houses.csv')

    # Optional argument for specifying the configuration YAML file.
    parser.add_argument('--config_path', type=str, default="config.yaml",
                        help='Path to .yaml file')

    # Parse command-line arguments.
    args = parser.parse_args()

    # Call the predict function with the parsed arguments.
    predict(args.data_path,
            args.weights_path,
            args.output_folder,
            args.config_path)

