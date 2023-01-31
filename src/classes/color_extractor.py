import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Union, List

from src.classes.base_features_extractor import BaseFeaturesExtractor


class ColorFeaturesExtractor(BaseFeaturesExtractor):
    """
    A class for that extracts

    Attributes:
        None
    """

    def __init__(
        self,
        n_bins: int = 32,
        histogram_range: tuple = (0, 256),
    ):
        """
        Initializes a ColorFeaturesExtractor object.
        """
        super().__init__()
        self.n_bins = n_bins
        self.histogram_range = histogram_range

    def extract(
        self, img: np.array, visualize: bool = False
    ) -> Union[List, np.ndarray]:

        # Compute the histogram of the color channels separately
        histogram_channel_1 = np.histogram(
            img[:, :, 0], bins=self.n_bins, range=self.histogram_range
        )

        histogram_channel_2 = np.histogram(
            img[:, :, 1], bins=self.n_bins, range=self.histogram_range
        )

        histogram_channel_3 = np.histogram(
            img[:, :, 2], bins=self.n_bins, range=self.histogram_range
        )

        features = np.concatenate(
            (histogram_channel_1[0], histogram_channel_2[0], histogram_channel_3[0])
        )

        if visualize:
            color_bin_plot = self._plot_histogram(
                histograms=[
                    histogram_channel_1,
                    histogram_channel_2,
                    histogram_channel_3,
                ],
                ylabels=["R", "G", "B"],
                title="Color Histogram",
            )

        else:
            color_bin_plot = None

        return features, color_bin_plot

    def _plot_histogram(
        self, histograms: List[np.ndarray], ylabels: List[str], title: str = None
    ) -> plt.figure:
        """
        Plots a histogram

        Parameters
        ----------
        xvalues (np.ndarray): values to plot
        ylabels (List[str]): labels for the y-axis
        title (str): title of the plot

        Returns
        -------
        fig (plt.figure): the figure
        """
        # Create a figure
        fig, axes = plt.subplots(
            1, 3, figsize=(12, 3), dpi=100, facecolor="w", edgecolor="k"
        )

        # Plot histogram 1 ("R")
        axes[0].bar(histograms[0][1][:-1], histograms[0][0], width=1)
        axes[0].set_title("R Histogram")
        axes[0].set_xlabel("Bins")
        axes[0].set_ylabel("Frequency")

        # Plot histogram 2 ("G")
        axes[1].bar(histograms[1][1][:-1], histograms[1][0], width=1)
        axes[1].set_title("G Histogram")
        axes[1].set_xlabel("Bins")
        axes[1].set_ylabel("Frequency")

        # Plot histogram 3 ("B")
        axes[2].bar(histograms[2][1][:-1], histograms[2][0], width=1)
        axes[2].set_title("B Histogram")
        axes[2].set_xlabel("Bins")
        axes[2].set_ylabel("Frequency")

        # Return figure
        return fig
