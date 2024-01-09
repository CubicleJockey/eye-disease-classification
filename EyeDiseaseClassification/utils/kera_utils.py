import tensorflow as tf
import matplotlib.pyplot as plt


def create_image_datasets(directory: str,
                          batch_size: int,
                          image_size: tuple = (256, 256),
                          validation_split: float = 0.2,
                          test_split: float = 0.1) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    """
    Creates image datasets for training, validation, and testing from a directory structure.

    :param directory: Path to the main dataset directory.
    :type directory: str

    :param batch_size: Size of the batch.
    :type batch_size: int

    :param image_size: Size of the images (width, height).
    :type image_size: tuple

    :param validation_split: Float between 0 and 1, fraction of data to reserve for validation.
    :type validation_split: float

    :param test_split: Float between 0 and 1, fraction of validation data to reserve for testing.
    :type test_split: float

    :return: A tuple of three `tf.data.Dataset` objects for training, validation, and testing.
    """

    # Creating the training and initial validation dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=123,
        validation_split=validation_split,
        subset='training',
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=123,
        validation_split=validation_split,
        subset='validation',
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    # Calculating the number of batches to take for test dataset from the validation dataset
    total_val_batches = validation_dataset.cardinality().numpy()
    test_batches = int(total_val_batches * test_split)

    # Splitting the validation dataset into validation and test datasets
    test_dataset = validation_dataset.take(test_batches)
    validation_dataset = validation_dataset.skip(test_batches)

    return train_dataset, validation_dataset, test_dataset


def display_batch_of_images(dataset: tf.data.Dataset
                            , class_names: list[str]
                            , rows: int = 5
                            , cols: int = 5
                            , fig_size: tuple[float, float] = (10, 10)) -> None:
    """
    Displays a batch of images in a grid format.

    :param class_names: Classification labels based on folders.
    :type class_names: list[str]

    :param dataset: A TensorFlow Dataset object containing image batches.
    :type dataset: tf.data.Dataset

    :param rows: Number of rows in the grid.
    :type rows: int

    :param cols: Number of columns in the grid.
    :type cols: int

    :param fig_size: Size of the figure to plotted
    :type fig_size: tuple[float, float]

    :return: None
    """

    # Take one batch from the dataset
    for images, labels in dataset.take(1):
        # Initialize the plot
        plt.figure(figsize=fig_size)

        # Total number of images to display (rows * cols)
        total_images = rows * cols

        for i in range(total_images):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            label_index = labels[i].numpy().argmax()
            plt.title(class_names[label_index])
            plt.axis("off")


def plot_accuracy_metrics(model_history: History, fig_size: tuple = (14, 6)) -> None:
    """
    :param model_history: Sequence Model History (Accuracy Metrics)
    :type model_history: History

    :param fig_size: Figure Dimensions for Display
    :type fig_size: tuple

    :return: None
    """
    legend_labels = ['Training', 'Validation']
    title_template = 'Training Dataset vs. Validation Dataset ({})'

    _, (accuracy_axes, loss_axes) = plt.subplots(1, 2, figsize=fig_size)

    accuracy_axes.set_title(title_template.format('Accuracy'))
    accuracy_axes.set_xlabel('Epoch')
    accuracy_axes.set_ylabel('Accuracy')
    accuracy_axes.plot(model_history['loss'], label=legend_labels[0])
    accuracy_axes.plot(model_history['val_loss'], label=legend_labels[1])
    accuracy_axes.legend(legend_labels)

    loss_axes.set_title(title_template.format('Loss'))
    loss_axes.set_xlabel('Epoch')
    loss_axes.set_ylabel('Loss')
    loss_axes.plot(model_history['loss'], label=legend_labels[0])
    loss_axes.plot(model_history['val_loss'], label=legend_labels[1])
    loss_axes.legend(legend_labels)

    plt.tight_layout()
    plt.show()