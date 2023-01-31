import numpy as np
import cv2
from skimage.feature import hog

from typing import Union, List

from src.classes.base_features_extractor import BaseFeaturesExtractor


class HoGFeaturesExtractor(BaseFeaturesExtractor):
    """
    A class for that extracts Histogram of Oriented Gradients (HoG) features from an image.
    Can return a vizualization of the HoG features if desired.

    Attributes:
        None
    """

    def __init__(
        self,
        orientations: int = 12,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (3, 3),
        channel_to_extract: any = "all",
    ):
        """
        Initializes a HoGFeaturesExtractor object.
        """
        super().__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.channel_to_extract = channel_to_extract

    def extract(self, img: str, visualize: bool = False) -> Union[List, np.ndarray]:

        if self.channel_to_extract == "all":
            features_list = []
            for ch in range(img.shape[2]):

                if visualize:
                    features, hog_image = hog(
                        img[:, :, ch],
                        orientations=self.orientations,
                        pixels_per_cell=self.pixels_per_cell,
                        cells_per_block=self.cells_per_block,
                        visualize=visualize,
                    )
                    features_list.append(features)

                else:
                    features = hog(
                        img[:, :, ch],
                        orientations=self.orientations,
                        pixels_per_cell=self.pixels_per_cell,
                        cells_per_block=self.cells_per_block,
                        visualize=visualize,
                    )
                    hog_image = None
                    features_list.append(features)

            features_list = np.ravel(features_list)

            return features_list, hog_image

        else:
            if visualize:
                features_list, hog_image = hog(
                    img[:, :, self.channel_to_extract],
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    visualize=visualize,
                )
            else:
                features_list = hog(
                    img[:, :, self.channel_to_extract],
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    visualize=visualize,
                )
                hog_image = None

            return features_list, hog_image
