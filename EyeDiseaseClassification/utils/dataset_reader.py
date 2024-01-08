import os
from typing import List


class EyeImage:
    def __init__(self, image_path: str, image_label: str):
        self.image_path = image_path
        self.image_label = image_label

    @property
    def path(self):
        return self.image_path

    @property
    def label(self):
        return self.image_label


def load_eye_dataset(images_folder: str) -> List[EyeImage]:
    """
    :param images_folder: string of the parent folder containing the images
    :type images_folder: str

    :return: A dictionary containing the image path and its corresponding label.
    """
    eye_dataset: List[EyeImage] = list()

    folders = os.listdir(images_folder)
    for folder in folders:
        current_path = os.path.join(images_folder, folder)
        image_files = os.listdir(current_path)
        for file in image_files:
            image_path = os.path.join(current_path, file)
            if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                eye_image = EyeImage(image_path, folder)
                eye_dataset.append(eye_image)
            else:
                continue

    return eye_dataset


def eye_dataset_to_dictionary(dataset: List[EyeImage]) -> dict:
    """
    :param dataset: A list of EyeImage instances
    :type dataset: List[EyeImage]

    :return: Dictionary version of the list of EyeImages
    """
    eye_dataset_as_dict = {
        'image_path': list(),
        'image-label': list()
    }
    for eye_image in dataset:
        eye_dataset_as_dict['image_path'].append(eye_image.path)
        eye_dataset_as_dict['image-label'].append(eye_image.label)

    return eye_dataset_as_dict