# Standard library imports
import glob

# Third party imports
import pandas as pd
import numpy as np
from skimage.io import imread
import cv2


def load_train_csv(path_to_csv, path_to_frames):
    """
    Loads the training csv file and returns a dataframe with the following columns:
    - frame_id: path to the frame image
    - bounding_boxes: list of bounding boxes coordinates

    Arguments:
    ----------
        path_to_csv (str): Path to the csv file.
        path_to_frames (str): Path to the frames folder.

    Returns:
    --------
        df_ground_truth (pandas.DataFrame): Dataframe containing the bounding boxes coordinates for each frame image.
        no_bbox_counter (int): Number of frames with NO bounding boxes.
    """

    no_bbox_counter = 0
    df_ground_truth = pd.read_csv(path_to_csv)
    # Append './data/initial/raw/frames/' to the beginning of each frame_id
    df_ground_truth["frame_id"] = df_ground_truth["frame_id"].apply(
        lambda x: path_to_frames + x
    )

    # Convert bounding boxes to numpy array
    for i, val in enumerate(df_ground_truth["bounding_boxes"]):
        if val is not np.nan:
            df_ground_truth["bounding_boxes"][i] = np.array(
                [float(i) for i in str(val).split(" ")]
            )
        else:
            no_bbox_counter += 1

    # List of chunks of 4 coordinates
    for i, val in enumerate(df_ground_truth["bounding_boxes"]):
        if val is not np.nan:
            df_ground_truth["bounding_boxes"][i] = [
                np.array(val[i : i + 4]) for i in range(0, len(val), 4)
            ]
        else:
            no_bbox_counter += 1

    # Replace NaN with empty list
    df_ground_truth["bounding_boxes"].fillna("").apply(list)

    return df_ground_truth, no_bbox_counter


def get_images(directory, data_aug=False):
    """
    Returns a list of images from a directory.

    Arguments:
    ----------
        dir (str): Path to the directory.
        data_aug (bool): Whether to perform data augmentation or not.

    Returns:
    --------
        images (list): List of images.
    """

    images = []

    for filename in glob.glob(directory, recursive=True):
        if data_aug:
            # Perform Horizontal Flip
            images.append(
                cv2.flip(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB), 1)
            )

            # Perform Brightness Change randomly
            value = np.random.randint(0, 20)
            images.append(increase_brightness(cv2.imread(filename), value=value))

            # Perform Center Cropping at 30% of margin and resize to 64x64
            width = cv2.imread(filename).shape[1]
            height = cv2.imread(filename).shape[0]
            margin = 0.3
            images.append(
                cv2.resize(
                    cv2.imread(filename)[
                        int(height * margin) : int(height * (1 - margin)),
                        int(width * margin) : int(width * (1 - margin)),
                    ],
                    (64, 64),
                )
            )

        else:
            images.append(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))

    return images


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
