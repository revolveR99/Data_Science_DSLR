"""
Generate a scatter plot to visualize the relationship between two courses' marks.
"""

import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
from argparse import ArgumentParser  # For handling command-line arguments

from data_describer import HogwartsDataDescriber  # Custom class for dataset description


def scatter_plot(plot: plt,
                 df: HogwartsDataDescriber,
                 course1: str,
                 course2: str):
    """
    Create a scatter plot for two courses.

    Args:
        plot (plt): The Matplotlib axes to plot on.
        df (HogwartsDataDescriber): The dataset containing course marks.
        course1 (str): The name of the first course (x-axis).
        course2 (str): The name of the second course (y-axis).
    
    Returns:
        None
    """
    # Loop through each house and plot their marks for the two courses
    for house, color in zip(df.houses, df.colors):
        # Select marks for course1 and course2 for students in the current house
        x = df[course1][df['Hogwarts House'] == house]
        y = df[course2][df['Hogwarts House'] == house]

        # Create a scatter plot for the selected marks
        plot.scatter(x, y, color=color, alpha=0.5)  # Using transparency for better visibility


def show_scatter_plot(csv_path: str, course1: str, course2: str):
    """
    Display a scatter plot for two specified courses.

    Args:
        csv_path (str): The path to the dataset .csv file.
        course1 (str): The name of the first course (x-axis).
        course2 (str): The name of the second course (y-axis).
    
    Returns:
        None
    """
    # Read the dataset using the custom data describer
    df = HogwartsDataDescriber.read_csv(csv_path)
    
    # Create a new plot
    _, ax = plt.subplots()

    # Generate the scatter plot
    scatter_plot(ax, df, course1, course2)
    
    # Set axis labels
    ax.set_xlabel(course1)
    ax.set_ylabel(course2)
    
    # Add a legend for the Hogwarts houses
    ax.legend(df.houses)
    
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

    # Argument for specifying the first course (x-axis)
    parser.add_argument('--course1',
                        type=str,
                        default='Astronomy',
                        help='Name of the course for x axis')

    # Argument for specifying the second course (y-axis)
    parser.add_argument('--course2',
                        type=str,
                        default='Defense Against the Dark Arts',
                        help='Name of the course for y axis')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function to display the scatter plot for the two courses
    show_scatter_plot(args.data_path, args.course1, args.course2)

