import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from skimage.io import imread
from tqdm import tqdm
import math
import time


def read_frame(df_annotation, frame):
    """Read frames and create integer frame_id"""
    file_path = df_annotation[df_annotation.index == frame]["frame_id"].values[0]
    return imread(file_path)


def annotations_for_frame(df_annotation, frame):
    assert frame in df_annotation.index
    bbs = df_annotation[df_annotation.index == frame].bounding_boxes.values[0]

    if pd.isna(bbs):  # some frames contain no vehicles
        return []

    bbs = list(map(lambda x: int(x), bbs.split(" ")))
    return np.array_split(bbs, len(bbs) / 4)


def show_annotation(df_annotation, frame):
    assert frame in df_annotation.index
    img = read_frame(df_annotation, frame)
    # bbs = annotations_for_frame(df_annotation, frame)
    bbs = df_annotation[df_annotation.index == frame].bounding_boxes.values[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    for x, y, dx, dy in bbs:

        rect = patches.Rectangle((x, y), dx, dy, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

    ax.imshow(img)
    ax.set_title(f"Annotations for frame {frame}.")


def draw_annotation(df_annotation, frame):
    """
    Draws annotations for a given frame.
    """
    img = read_frame(df_annotation, frame)
    bboxs = df_annotation[df_annotation.index == frame].bounding_boxes.values[0]

    # Check if value of bboxs is empty list
    if not isinstance(bboxs, list) and math.isnan(bboxs):
        return img

    img_with_bounding_boxes = draw_bounding_boxes(img, bboxs)

    return img_with_bounding_boxes


def draw_bounding_boxes(img, bboxs, color=(0, 0, 255)):
    """
    Draws bounding boxes on the given image.

    Arguments:
    ----------
        img (numpy.ndarray): Image to draw bounding boxes on.
        bboxs (list): List of bounding boxes coordinates.
        color (tuple): Color of the bounding boxes.

    Returns:
    --------
        copy_img (numpy.ndarray): Image with bounding boxes drawn on.
    """
    # Copy image
    copy_img = np.copy(img)

    # Iterate on coordinates
    for x, y, dx, dy in bboxs:
        cv2.rectangle(copy_img, (int(x), int(y)), (int(x + dx), int(y + dy)), color, 2)

    return copy_img


def extract_vehicle_images_from_bboxs(
    csv_df: pd.DataFrame, resolution: tuple = (64, 64)
):
    """
    Extracts vehicle images from each frame image using the bounding boxes coordinates indicated the csv file.

    Arguments:
    ----------
        csv_df (pandas.DataFrame): Dataframe containing the bounding boxes coordinates for each frame image.

    Returns:
    --------
        None
    """
    counter = 0
    # Loop through the image frames in the dataset
    start = time.time()
    for i in tqdm(csv_df.index):
        # Read the image frame
        frame = read_frame(csv_df, i)

        # Get the bounding boxes for the current frame
        bboxs = csv_df[csv_df.index == i].bounding_boxes.values[0]

        # Check if value of bboxs is empty list
        if not isinstance(bboxs, list) and math.isnan(bboxs):
            continue

        try:
            # Loop through the bounding boxes
            for x, y, dx, dy in bboxs:

                # Check if the bounding box is bigger than threshold
                if not (dx > 20 and dy > 20):
                    continue

                # Crop the image
                crop_img = frame[int(y) : int(y + dy), int(x) : int(x + dx)]

                # Resize the image
                crop_img = cv2.resize(crop_img, resolution)

                # Save the image
                cv2.imwrite(f"./data/initial/vehicles/{i}.png", crop_img)
                counter += 1

        except Exception as exc:
            print(bboxs, type(bboxs))
            print(exc)

    end = time.time()
    print(f"Number of images extracted: {counter}")
    print(f"Time taken: {end - start} seconds")

    return None
