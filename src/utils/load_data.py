import pandas as pd
import numpy as np
from skimage.io import imread
import glob


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


def get_images(directory):
    """
    Returns a list of images from a directory.

    Arguments:
    ----------
        dir (str): Path to the directory.

    Returns:
    --------
        images (list): List of images.
    """

    images = []

    for filename in glob.glob(directory, recursive=True):
        # images.append(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
        images.append(imread(filename))

    return images
