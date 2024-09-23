"""
Display descriptive information for all numerical features in the dataset.
This script reads the dataset from a .csv file, and for each numerical feature,
it prints out its count, mean, standard deviation, min, max, and percentiles (25%, 50%, 75%).
"""

from argparse import ArgumentParser  # For handling command-line arguments.
from data_describer import HogwartsDataDescriber  # Custom class for dataset description.


def abbreviation(string: str) -> str:
    """
    Create an abbreviation for a long string by using the first letter of each word.
    
    Args:
        string (str): The input string to abbreviate.

    Returns:
        str: The abbreviation of the string.
    """
    string_list = string.split(" ")  # Split the string by spaces.
    abb = ""
    for word in string_list:
        abb += word[0]  # Append the first letter of each word.
    return abb


def describe(csv_path: str):
    """
    Display information for all numerical features in the dataset, including count, mean,
    standard deviation, min, max, and percentiles.

    Args:
        csv_path (str): Path to the .csv file containing the dataset.
    """
    data = HogwartsDataDescriber.read_csv(csv_path)  # Read the dataset using the custom data describer.
    
    # Print header for the output table.
    print(f'{"":15} |{"Count":>12} |{"Mean":>12} |{"Std":>12} |{"Min":>13}'
          f'|{"Max":>12} |{"25%":>12} |{"50%":>12} |{"75%":>12} |')
    
    # Loop through all columns in the dataset.
    for feature in data.columns:
        # Print the feature name, abbreviated if it's longer than 15 characters.
        if len(feature) > 15:
            print(f'{abbreviation(feature):15.15}', end=' |')
        else:
            print(f'{feature:15.15}', end=' |')
        
        # Print the count of non-NaN values in the feature.
        print(f'{data.count(feature):>12.4f}', end=' |')
        
        # Check if the feature is numeric and has non-zero elements.
        if data.is_numeric(feature) and data.count(feature) != 0:
            # Print mean, standard deviation, min, max, and percentiles (25%, 50%, 75%).
            print(f'{data.mean(feature):>12.4f}', end=' |')
            print(f'{data.std(feature):>12.4f}', end=' |')
            print(f'{data.min(feature):>12.4f}', end=' |')
            print(f'{data.max(feature):>12.4f}', end=' |')
            print(f'{data.percentile(feature, 25):>12.4f}', end=' |')
            print(f'{data.percentile(feature, 50):>12.4f}', end=' |')
            print(f'{data.percentile(feature, 75):>12.4f}', end=' |\n')
        else:
            # Print a message if the feature is not numeric.
            print(f'{"No numerical value to display":>64}')


if __name__ == "__main__":
    # Create a parser to handle command-line arguments.
    parser = ArgumentParser()

    # Argument for specifying the path to the dataset.
    parser.add_argument('data_path', type=str, help='Path to .csv file')

    # Parse the arguments.
    args = parser.parse_args()

    # Call the describe function with the provided dataset path.
    describe(args.data_path)

