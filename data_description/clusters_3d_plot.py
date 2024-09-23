"""
3D plot of clusters mean values.
This script generates a 3D visualization of clusters based on
"Birthday", "Best Hand", and course names.
"""

import numpy as np  # For numerical operations.
import pandas as pd  # For handling datasets.
import matplotlib.pyplot as plt  # For plotting.
from argparse import ArgumentParser  # For parsing command-line arguments.
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting.

import sys
sys.path.append("..")  # Add the parent directory to system path for imports.
from config import Config  # For reading configuration settings.
from data_description.describe import abbreviation  # For course name abbreviation.


def clusters_3d(df: pd.DataFrame, courses: np.ndarray):
    """
    This function splits data into clusters and visualizes the clusters' mean values.
    The clustering is based on "Birthday", "Best Hand", and the provided courses.

    Clusters are structured as:
        2000 (Birthday) - "Right" (Best Hand) - course1
                        |                     |_ course2
                        |                     |_ ...
                        |_ "Left" - course1
                                  |_ ...
        1999 - ...

    Args:
        df (pd.DataFrame): Dataset containing "Birthday", "Best Hand", and course data.
        courses (np.ndarray): Array of course names to visualize.

    Returns:
        None
    """

    # Initialize 3D plot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.get_cmap('gist_rainbow')  # Color map for points.
    ax.set_xlabel('YEAR')
    ax.set_ylabel('HAND')
    ax.set_zlabel('MEAN')
    ax.set_title('CLUSTERS')

    # Replace full "Birthday" date with year only (e.g., "2000-03-30" -> 2000).
    years = np.empty(df["Birthday"].shape[0], dtype=np.int)
    for i, b in enumerate(df["Birthday"]):
        years[i] = b.split('-')[0]
    df["Birthday"] = years

    # Replace "Best Hand" with binary values (e.g., "Right" -> 0, "Left" -> 1).
    bin_hands = np.empty(df["Best Hand"].shape[0], dtype=np.int)
    for i, hand in enumerate(df["Best Hand"].unique()):
        bin_hands[df["Best Hand"] == hand] = i
    df["Best Hand"] = bin_hands

    # Iterate over unique birthdays, hands, and courses to compute cluster means.
    for year in df["Birthday"].unique():
        for hand in df["Best Hand"].unique():
            for i, course in enumerate(courses):
                # Filter the dataset by current birthday and hand.
                mask = (df["Birthday"] == year) & (df["Best Hand"] == hand)
                cluster = np.array(df.loc[mask, course].dropna())
                
                # Min-max scaling of the course values within the cluster.
                cluster = (cluster - cluster.min()) / (cluster.max() - cluster.min())

                # Calculate the mean of the course values within the cluster.
                mean = cluster.mean()

                # Abbreviate the course name if it's longer than 15 characters.
                if len(course) > 15:
                    course = abbreviation(course)

                # Plot the mean value of the cluster in 3D space.
                ax.scatter(year, hand, mean,
                           color=cmap(round(i / courses.shape[0], 2)),
                           label=course)
    
    # Adjust course names for the legend.
    courses = [c if len(c) < 15 else abbreviation(c) for c in courses]
    ax.legend(courses)
    
    # Display the plot.
    plt.show()


def vis_clusters_3d(data_path: str, config_path: str):
    """
    Visualize 3D clusters of mean course values based on "Birthday" and "Best Hand".

    Args:
        data_path (str): Path to the dataset file.
        config_path (str): Path to the configuration file.

    Returns:
        None
    """
    # Load configuration to select features (courses).
    config = Config(config_path)
    courses = config.choosed_features()

    # Load the dataset.
    df = pd.read_csv(data_path)

    # Plot the 3D clusters.
    clusters_3d(df, courses)


if __name__ == "__main__":
    # Set up argument parser for command-line inputs.
    parser = ArgumentParser()

    # Argument for dataset file path.
    parser.add_argument('--data_path', type=str,
                        default="../data/dataset_train.csv",
                        help='Path to "dataset_*.csv" file')

    # Argument for configuration file path.
    parser.add_argument('--config_path', type=str,
                        default="../config.yaml",
                        help='Path to .yaml file')

    # Parse the command-line arguments.
    args = parser.parse_args()

    # Call the function to visualize 3D clusters.
    vis_clusters_3d(args.data_path, args.config_path)

