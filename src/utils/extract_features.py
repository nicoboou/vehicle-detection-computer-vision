import numpy as np
from tqdm import tqdm
import pickle

# Local imports
from src.utils.preprocessing import convert_color

from src.classes.color_extractor import ColorFeaturesExtractor
from src.classes.hog_extractor import HoGFeaturesExtractor
from src.classes.spatial_extractor import SpatialFeaturesExtractor


def extract_features(
    path,
    color_space: str = "RGB",
    spatial_size: tuple = (32, 32),
    color_histogram_bins: int = 32,
    hog_orientations: int = 9,
    pixels_per_cell: int = 8,
    cells_per_block: int = 2,
    hog_channel_to_extract: int = 0,
    extract_spatial_features: bool = True,
    extract_color_features: bool = True,
    extract_hog_features: bool = True,
    color_histogram_range: tuple = (0, 256),
):
    """
    Description:
    ------------
    Extract features from a list of images using three methods:
    - Spatial binning
    - Color histogram
    - Histogram of oriented gradients (HOG)

    Parameters:
    -----------
    imgs (list): list of images
    color_space (str): color space to use
    spatial_size (tuple): spatial binning dimensions
    hist_bins (int): number of histogram bins
    orient (int): HOG orientations
    pix_per_cell (int): HOG pixels per cell
    cell_per_block (int): HOG cells per block
    hog_channel (int): HOG channel
    spatial_feat (bool): spatial features on or off
    hist_feat (bool): histogram features on or off
    hog_feat (bool): HOG features on or off
    hist_range (tuple): histogram range

    Returns:
    --------
    features (list): list of feature vectors
    """

    # List of features vectors per image
    features = []

    # Iterate through the list of images
    for image in tqdm(path):
        image_features = []

        # Print max and min value of image array
        # print(f"Max value: {np.max(image)}")
        # print(f"Min value: {np.min(image)}")

        # Modify image if other than 'RGB'
        # img *= 255.0
        # img = image.astype(np.uint8)

        # Convert color if other than 'RGB'
        feature_img = (
            convert_color(image, conversion=color_space)
            if color_space != "RGB"
            else np.copy(image)
        )

        # Spatial features extraction
        if extract_spatial_features:
            spatial_features_extractor = SpatialFeaturesExtractor(size=spatial_size)
            spatial_features, _ = spatial_features_extractor.extract(
                img=feature_img, visualize=False
            )
            image_features.append(spatial_features)

        # Color histogram features extraction
        if extract_color_features:
            color_hist_features_extractor = ColorFeaturesExtractor(
                n_bins=color_histogram_bins, histogram_range=color_histogram_range
            )
            color_hist_features, _ = color_hist_features_extractor.extract(
                img=feature_img, visualize=False
            )
            image_features.append(color_hist_features)

        # HOG features extraction
        if extract_hog_features:
            hog_features_extractor = HoGFeaturesExtractor(
                orientations=hog_orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                channel_to_extract=hog_channel_to_extract,
            )
            hog_features, _ = hog_features_extractor.extract(
                img=feature_img, visualize=False
            )
            image_features.append(hog_features)

        features.append(np.concatenate(image_features))

    # Return list of feature vectors
    return features


def save_features(path, file_name: str, features):
    """
    Save extracted features for easy access.

    Parameters:
    -----------
    path (str): path to save the data
    file_name (str): name of the file
    features (list): list of features

    Returns:
    --------
    None
    """

    features_pickle_file = path + file_name + ".pkl"
    print(f"Saving features to pickle file {features_pickle_file}...")

    try:
        with open(features_pickle_file, "wb") as pfile:
            pickle.dump(
                {
                    "features": features,
                },
                pfile,
                pickle.HIGHEST_PROTOCOL,
            )

    except Exception as exc:
        print(f"Unable to save features to {features_pickle_file} : {exc}")
        raise

    print("Features cached in pickle file.")

    return None
