"""
Display pair plots for all courses in the dataset.
The script creates a matrix of plots, where each course's distribution is plotted
on the diagonal, and scatter plots between pairs of courses are plotted in the off-diagonal cells.
"""

import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
from argparse import ArgumentParser  # For handling command-line arguments

from histogram import histogram  # Function to create histograms
from scatter_plot import scatter_plot  # Function to create scatter plots
from data_describer import HogwartsDataDescriber  # Custom class for dataset description


def show_pair_plot(csv_path: str):
    """
    Create pair plots for all courses in the dataset.

    Args:
        csv_path (str): The path to the dataset .csv file.
    
    Returns:
        None
    """
    # Read the dataset using the custom data describer
    df = HogwartsDataDescriber.read_csv(csv_path)
    
    # Get the list of courses (assumed to start at column index 6)
    courses = list(df.columns[6:])

    # Create a 13x13 grid of subplots for pair plotting
    _, axs = plt.subplots(13, 13, figsize=(25.6, 14.4), tight_layout=True)
    
    # Iterate over each pair of courses to generate the appropriate plot
    for row_course, row_plt in zip(courses, axs):
        for col_course, col_plt in zip(courses, row_plt):
            # If the current course is on the diagonal, plot a histogram
            if row_course == col_course:
                histogram(col_plt, df, row_course)
            else:
                # Otherwise, plot a scatter plot for course pairs
                scatter_plot(col_plt, df, row_course, col_course)

            # Remove axis tick labels for cleanliness
            col_plt.tick_params(labelbottom=False)
            col_plt.tick_params(labelleft=False)

            # Set x-axis labels only for the last row
            if col_plt.is_last_row():
                col_plt.set_xlabel(col_course.replace(' ', '\n'))

            # Set y-axis labels only for the first column
            if col_plt.is_first_col():
                label = row_course.replace(' ', '\n')
                length = len(label)
                
                # Adjust label to fit within the grid by adding a line break
                if length > 14 and '\n' not in label:
                    label = label[:int(length / 2)] + "\n" + label[int(length / 2):]
                col_plt.set_ylabel(label)

    # Add legend for the Hogwarts houses on the side of the plot
    plt.legend(df.houses,
               loc='center left',
               frameon=False,
               bbox_to_anchor=(1, 0.5))

    # Display the pair plot matrix
    plt.show()


if __name__ == "__main__":
    # Create a parser to handle command-line arguments
    parser = ArgumentParser()

    # Argument for specifying the path to the dataset .csv file
    parser.add_argument('--data_path',
                        type=str,
                        default='../data/dataset_train.csv',
                        help='Path to dataset_train.csv file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function to display the pair plot for all courses
    show_pair_plot(args.data_path)

