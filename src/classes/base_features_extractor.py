# Standard library imports
from abc import ABC, abstractmethod
from typing import List, Union

# Third party imports
import numpy as np


class BaseFeaturesExtractor(ABC):
    """
    Base class to build a feature extractor on.
    To build a child of this class and inherit the methods, need to implement the extract method.
    """

    @abstractmethod
    def extract(
        self, img: np.array, visualize: bool = False
    ) -> Union[List, np.ndarray]:
        """
        Method to extract the features from an image by giving a path.

        Parameters
        ----------
        img :  np.array
            Input image
        visualize : bool, optional
            Option to visualize the predictions, by default False

        Returns
        -------
        Union[List[Tuple[int, int]], np.ndarray]
            Or List of coordinates or base image with some circles or square on it

        Raises
        ------
        NotImplementedError
            If the method is not implement
        """
        raise NotImplementedError
