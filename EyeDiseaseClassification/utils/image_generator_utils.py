from typing import List

import polars as pl
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_image_generator(dataset: pl.DataFrame
                           , batch_size: int
                           , image_size: tuple = (256, 256)
                           , columns: tuple = ('image-path', 'image-label')) -> ImageDataGenerator:
    """
    Creates an image generator from a polars dataframe.
    Model required data to be in the form of an image data generator.
    Image Data Generators are used to create Tensors.

    Tensor:
    At its core, a tensor is a multidimensional array.
    It's a generalization of vectors and matrices to potentially higher dimensions.
    In TensorFlow, tensors are used to represent data like numbers, strings, or booleans.

    :param dataset: Dataframe to convert into an image generator
    :type dataset: pd.DataFrame

    :param batch_size: Size of the batch.
    :type batch_size: int

    :param image_size: Size of the
    :type image_size: tuple

    :param columns: x and y columns names
    :type columns: List[str]

    :return: Image generator (ImageDataGenerator)
    """

    # Ensure the dataframe contains the necessary columns
    if not all(col in dataset.columns for col in columns):
        raise ValueError("Polars DataFrame must contain the specified columns.")

    # Define the ImageDataGenerator (add any specific transformations or preprocessing here)
    image_generator = ImageDataGenerator()

    # Create the generator (Image Generator expects Pandas dataframe convert with `to_pandas()`)
    generator = image_generator.flow_from_dataframe(dataset.to_pandas(),
                                                    x_col=columns[0],
                                                    y_col=columns[1],
                                                    target_size=image_size,
                                                    class_mode='categorical',
                                                    color_mode='rgb',
                                                    shuffle=True,
                                                    batch_size=batch_size)

    return generator


def display_image_generator(image_generator: ImageDataGenerator) -> None:
    """
    Display the images within a given Image Data Generator
    :param image_generator: Image Data Generator with images to display.
    :type image_generator: ImageDataGenerator

    :return: None
    """

    classes = list(image_generator.class_indices.keys())