import polars as pl
import math


def train_test_validation_split(dataframe: pl.DataFrame
                                , test_percentage: float = .2
                                , validation_percentage: float = .0) -> (pl.DataFrame, pl.DataFrame, pl.DataFrame):
    """
    Split data into training, testing and/or validation datasets.
    :param dataframe: Polars dataframe that will be split into Training, Test and/or Validation Datasets
    :type dataframe: pd.DataFrame

    :param test_percentage: percentage of data to be used for testing
    :type test_percentage: float

    :param validation_percentage: percentage of data to be used for validation
    :type validation_percentage: float

    :return: train, validation, and test datasets that are pl.DataFrame's
    """

    # Check that testing + validation percentage aren't greater than 100%
    training_percentage = 100 - (test_percentage + validation_percentage)
    assert training_percentage > 0.0, f'test_percentage[{test_percentage} + validation_percentage[{validation_percentage}] is greater than 100'

    test_size = math.ceil(len(dataframe) * test_percentage)
    validation_size = math.ceil(len(dataframe) * validation_percentage)
    training_size = math.ceil(len(dataframe) - (test_size + validation_size))

    shuffled = dataframe.with_columns([pl.all().shuffle(seed=42)])

    train = shuffled.slice(0, training_size)
    test = shuffled.slice(training_size, test_size)
    validation = shuffled.slice(training_size + test_size, validation_size)

    return train, test, validation