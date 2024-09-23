"""
Plot the distribution of course marks for different Hogwarts houses.
The script reads the dataset, selects marks based on the specified course,
and generates a histogram for each house using different colors.
"""

import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
from argparse import ArgumentParser  # For handling command-line arguments

from data_describer import HogwartsDataDescriber  # Custom class for dataset description


def histogram(plot: plt.Axes, df: HogwartsDataDescriber, course: str):
    """
    Create a histogram for the specified course's marks, divided by Hogwarts house.

    Args:
        plot (plt.Axes): The plot object (axes) where the histogram will be drawn.
        df (HogwartsDataDescriber): The dataset object containing the student data.
        course (str): The name of the course whose marks distribution is being plotted.
    
    Returns:
        None
    """
    for house, color in zip(df.houses, df.colors):
        # Select course marks of students who belong to the current house
        marks = df[course][df['Hogwarts House'] == house].dropna()

        # Plot the histogram for the house's marks, setting transparency with alpha
        plot.hist(marks, color=color, alpha=0.5)


def show_course_marks_distribution(csv_path: str, course: str):
    """
    Read the dataset and plot the marks distribution for the specified course.

    Args:
        csv_path (str): The path to the dataset .csv file.
        course (str): The name of the course whose marks distribution is to be plotted.
    
    Returns:
        None
    """
    # Read the dataset using the custom data describer
    df = HogwartsDataDescriber.read_csv(csv_path)

    # Create a subplot for the histogram
    _, ax = plt.subplots()

    # Plot the histogram for the specified course
    histogram(ax, df, course)

    # Set plot title and labels
    ax.set_title(course)
    ax.legend(df.houses, frameon=False)  # Add legend for the houses
    ax.set_xlabel('Marks')
    ax.set_ylabel('Number of Students')

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Create a parser to handle command-line arguments
    parser = ArgumentParser()

    # Argument for specifying the path to the dataset .csv file
    parser.add_argument('--data_path',
                        type=str,
                        default='../data/dataset_train.csv',
                        help='Path to dataset_train.csv file')

    # Argument for specifying the course name to plot
    parser.add_argument('--course',
                        type=str,
                        default='Care of Magical Creatures',
                        help='Name of the course to plot')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function to display the marks distribution for the given course
    show_course_marks_distribution(args.data_path, args.course)

