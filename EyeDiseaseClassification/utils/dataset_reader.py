import os


def count_files_in_directory(directory: str) -> int:
    """
    Counts all the files in a directory and its subdirectories.

    :param directory: The path to the directory.
    :type directory: str

    :return: The total number of files in the directory and its subdirectories.
    """

    total_files = 0

    # Walk through all directories and files in the given directory
    for root, dirs, files in os.walk(directory):
        total_files += len(files)

    return total_files