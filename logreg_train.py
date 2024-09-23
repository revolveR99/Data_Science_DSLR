"""
Script to train one-vs-all logistic regression.
It saves the model's weights in weights.pt.
"""

import numpy as np  # For numerical operations.
import pandas as pd  # For handling datasets.
from time import time  # For timing the execution of functions.
from argparse import ArgumentParser  # For handling command-line arguments.
from matplotlib import pyplot as plt  # For plotting training loss.

# Importing custom modules for configuration, data preprocessing, and model.
from config import Config
from dslr.preprocessing import scale, fill_na
from dslr.multi_classifier import OneVsAllLogisticRegression


def plot_training(model: OneVsAllLogisticRegression):
    """
    Plot loss history after training the model.

    Args:
        model (OneVsAllLogisticRegression): Trained model with saved loss history.

    Returns:
        None
    """
    # Create a figure and axis for plotting.
    _, ax = plt.subplots()

    # Get the range of epochs (1 to number of epochs).
    epochs = range(1, model.epochs + 1)

    # Plot the loss history for each classifier in the one-vs-all model.
    for sub_model, label in zip(model.models, model.labels):
        ax.plot(epochs, sub_model.hist, label=label)

    # Set plot labels and title.
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(f'Logistic Regression, batch size: {model.batch_size}')
    ax.legend(loc="upper right")
    
    # Display the plot.
    plt.show()


def train(data_path: str,
          weights_path: str,
          config_path: str,
          v: bool = False):
    """
    Train a one-vs-all logistic regression model.

    Args:
        data_path (str): Path to training dataset.
        weights_path (str): Path to save the model's weights.
        config_path (str): Path to the configuration YAML file.
        v (bool): Flag to indicate whether to visualize training loss.

    Returns:
        None
    """
    # Load configuration file.
    config = Config(config_path)
    # Choose features from config to train on.
    courses = config.choosed_features()

    # Start timer for preparation time.
    preparation_t = time()

    # Read training dataset and fill missing values.
    df = pd.read_csv(data_path)
    df = fill_na(df, courses)

    # Extract feature values and labels for training.
    x = df[courses].values
    y = df["Hogwarts House"].values

    # Create a one-vs-all logistic regression model.
    model = OneVsAllLogisticRegression(
        device=config.device,  # Use the specified device (e.g., CPU or GPU).
        transform=scale[config.scale],  # Apply scaling transformation.
        lr=config.lr,  # Learning rate from the config.
        epochs=config.epochs,  # Number of training epochs.
        batch_size=config.batch_size,  # Batch size for training.
        seed=config.seed,  # Random seed for reproducibility.
        save_hist=v  # Save training history if visualization is enabled.
    )
    
    # Stop timer for preparation time.
    preparation_t = time() - preparation_t

    # Start timer for training time.
    train_t = time()
    # Train the model.
    model.fit(x, y)
    # Stop timer for training time.
    train_t = time() - train_t

    # Save the trained model's weights to the specified path.
    model.save(weights_path)

    # Print preparation, training, and total time taken.
    print("Preparation time:", np.round(preparation_t, 4))
    print("Training time:", np.round(train_t, 4))
    print("All time:", np.round(preparation_t + train_t, 4))

    # If visualization flag is set, plot the training loss.
    if v:
        plot_training(model)


if __name__ == "__main__":
    # Set up argument parser for command-line inputs.
    parser = ArgumentParser()

    # Add required argument for the path to the training dataset.
    parser.add_argument('data_path', type=str,
                        help='Path to "dataset_train.csv" file')

    # Optional argument for specifying the path to save model weights.
    parser.add_argument('--weights_path', type=str, default="data/weights.pt",
                        help='Path to save weights file')

    # Optional argument for specifying the configuration YAML file.
    parser.add_argument('--config_path', type=str, default="config.yaml",
                        help='Path to .yaml file')

    # Add optional flag for visualizing the training loss.
    parser.add_argument('-v', action="store_true",
                        help='Visualize training')

    # Parse command-line arguments.
    args = parser.parse_args()

    # Call the train function with the parsed arguments.
    train(args.data_path, args.weights_path, args.config_path, args.v)

