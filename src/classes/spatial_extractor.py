# Standard library imports
from typing import Union, List

# Third party imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Local application imports
from src.classes.base_features_extractor import BaseFeaturesExtractor


class SpatialFeaturesExtractor(BaseFeaturesExtractor):
    """
    A class for that extracts

    Attributes:
        None
    """

    def __init__(
        self,
        size: tuple = (32, 32),
    ):
        """
        Initializes a SpatialFeaturesExtractor object.
        """
        super().__init__()
        self.size = size

    def extract(
        self, img: np.ndarray, visualize: bool = False
    ) -> Union[List, np.ndarray]:

        # Use cv2.resize().ravel() to resize each channel and output the feature vector
        color1 = cv2.resize(img[:, :, 0], self.size).ravel()
        color2 = cv2.resize(img[:, :, 1], self.size).ravel()
        color3 = cv2.resize(img[:, :, 2], self.size).ravel()

        features = np.hstack((color1, color2, color3))

        if visualize:
            spatial_bin_plot = self._plot_histogram(
                histogram=features,
                title="Spatial Binned Color Histogram",
            )

        else:
            spatial_bin_plot = None

        return features, spatial_bin_plot

    def _plot_histogram(
        self, histogram: List[np.ndarray], title: str = None
    ) -> plt.figure:
        """
        Plots a Spatial bin histogram

        Parameters
        ----------
        histogram (np.ndarray): histogram to plot
        ylabels (List[str]): labels for the y-axis
        title (str): title of the plot

        Returns
        -------
        fig (plt.figure): the figure
        """
        # Create a figure
        fig = plt.figure(figsize=(12, 3))

        # Plot histogram
        plt.plot(histogram, color="r")

        # Set title
        if title:
            plt.title(title)

        # Return figure
        return fig
