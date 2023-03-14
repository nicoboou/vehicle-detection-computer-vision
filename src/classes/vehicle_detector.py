# Standard library imports
import time

import numpy as np
import cv2

from scipy.ndimage import label


# Local application imports
from src.utils.preprocessing import convert_color

from src.classes.spatial_extractor import SpatialFeaturesExtractor
from src.classes.color_extractor import ColorFeaturesExtractor
from src.classes.hog_extractor import HoGFeaturesExtractor

from src.utils.annotate import draw_bounding_boxes


class VehicleDetector:
    """
    Class to build a vehicle detector.

    Algorithm:
    ----------
    For each window search area:
        1. Extract windows of defined size in the search area
        2. Extract features from each window
        3. Predict if the window contains a vehicle
        4. If the window contains a vehicle, add it to the list of detected windows
    """

    def __init__(
        self,
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
        Initialize the vehicle detector.

        Parameters:
        -----------
        color_space (str): color space to use
        spatial_size (tuple): spatial binning dimensions
        hist_bins (int): number of color histogram bins
        orient (int): number of HOG orientations
        pix_per_cell (int): number of pixels per HOG cell
        cell_per_block (int): number of HOG cells per block
        hog_channel (int): channel to extract HOG features from
        spatial_feat (bool): extract spatial features
        hist_feat (bool): extract color histogram features
        hog_feat (bool): extract HOG features
        hist_range (tuple): color histogram range
        """
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.color_histogram_bins = color_histogram_bins
        self.hog_orientations = hog_orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.hog_channel_to_extract = hog_channel_to_extract
        self.extract_spatial_features = extract_spatial_features
        self.extract_color_features = extract_color_features
        self.extract_hog_features = extract_hog_features
        self.color_histogram_range = color_histogram_range

    def img_features_extractor(self, img: np.ndarray):
        """
        Description:
        ------------
        Extract features from a SINGLE image using three methods:
        - Spatial binning
        - Color histogram
        - Histogram of oriented gradients (HOG)

        Parameters:
        -----------
        path (str): path to image
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
        image_features (np.array): vector of features
        """

        image_features = []

        # Convert color
        feature_img = (
            convert_color(img, conversion=self.color_space)
            if self.color_space != "RGB"
            else np.copy(img)
        )

        # Spatial features extraction
        if self.extract_spatial_features:
            spatial_features_extractor = SpatialFeaturesExtractor(
                size=self.spatial_size
            )
            spatial_features, _ = spatial_features_extractor.extract(
                img=feature_img, visualize=False
            )
            image_features.append(spatial_features)

        # Color histogram features extraction
        if self.extract_color_features:
            color_hist_features_extractor = ColorFeaturesExtractor(
                n_bins=self.color_histogram_bins,
                histogram_range=self.color_histogram_range,
            )
            color_hist_features, _ = color_hist_features_extractor.extract(
                img=feature_img, visualize=False
            )
            image_features.append(color_hist_features)

        # HOG features extraction
        if self.extract_hog_features:
            hog_features_extractor = HoGFeaturesExtractor(
                orientations=self.hog_orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                channel_to_extract=self.hog_channel_to_extract,
            )
            hog_features, _ = hog_features_extractor.extract(
                img=feature_img, visualize=False
            )
            image_features.append(hog_features)

        #    Return img feature vectors
        return np.concatenate(image_features)

    def slide_window(
        self,
        img,
        window_search_area=(None, None),
        window_size=(64, 64),
        overlap=0.75,
    ):
        """
        Description:
        ------------
        Slide a window of defined size on a defined search area across the image and return a list of windows.

        Parameters:
        -----------
        img (np.array): image to slide window across
        window_search_area (tuple): y start and stop positions
        window_size (tuple): window size (x, y)
        overlap (float): overlap fraction

        Returns:
        --------
        windows (list): list of windows
        """

        # If y start/stop positions not defined, set to image height
        x_start_stop = [0, img.shape[1]]
        y_start_stop = [
            0 if window_search_area[0] is None else window_search_area[0],
            img.shape[0] if window_search_area[1] is None else window_search_area[1],
        ]

        # Calculate area to browse
        x_length = x_start_stop[1] - x_start_stop[0]
        y_length = y_start_stop[1] - y_start_stop[0]

        # Compute the number of pixels per step
        number_pixels_per_step_x = np.int(window_size[0] * (1 - overlap))
        number_pixels_per_step_y = np.int(window_size[1] * (1 - overlap))

        # Number of windows for x and y
        number_windows_x = np.int(x_length / number_pixels_per_step_x) - 1
        number_windows_y = np.int(y_length / number_pixels_per_step_y) - 1

        # List of windows to extract features from
        window_list = []

        for window_y in range(number_windows_y):
            for window_x in range(number_windows_x):

                # Compute window coords
                x = window_x * number_pixels_per_step_x + x_start_stop[0]
                y = window_y * number_pixels_per_step_y + y_start_stop[0]
                dx = window_size[0]
                dy = window_size[1]

                # Append window position to list
                window_list.append([x, y, dx, dy])

        return window_list

    def search_windows(self, img, windows, clf, scaler, proba=False):
        """
        Description:
        ------------
        Search for cars in a list of windows.

        Parameters:
        -----------
        img (np.array): image to search for cars
        windows (list): list of windows to search for cars
        clf (sklearn classifier): classifier to use for prediction
        scaler (sklearn scaler): scaler to use for prediction

        Returns:
        --------
        on_windows (list): list of windows where cars were detected
        """
        # List of windows in which a vehicle was detected
        windows_with_vehicle = []

        for window in windows:

            # Extract window from input image
            img_window = cv2.resize(
                img[
                    window[1] : window[1] + window[3], window[0] : window[0] + window[2]
                ],
                (64, 64),
                interpolation=cv2.INTER_AREA,
            )
            # Extract features from img window
            features = self.img_features_extractor(img_window)

            # Scale features to pass in the classifier
            scaled_features = scaler.transform(np.array(features).reshape(1, -1))

            # Predict
            if proba and (clf == "SGDClassifier" or clf == "RandomForestClassifier"):
                prediction = clf.predict_proba(scaled_features)

            else:
                prediction = clf.predict(scaled_features)

            # If positive (prediction == 1), save window
            if prediction == 1:
                windows_with_vehicle.append(window)

        return windows_with_vehicle

    def browse_img_multiple_scales(
        self, img, window_search_areas, window_sizes, overlap, clf, scaler, proba=False
    ):
        """
        Description:
        ------------
        Browse an image at multiple scales and return a list of windows where cars were detected.

        Parameters:
        -----------
        img (np.array): image to browse
        window_search_areas (list): list of y start and stop positions
        window_sizes (list): list of window sizes (x, y)
        overlap (float): overlap fraction
        clf (sklearn classifier): classifier to use for prediction
        scaler (sklearn scaler): scaler to use for prediction

        Returns:
        --------
        windows_with_vehicle (list): list of windows where cars were detected
        """
        hot_windows = []
        all_windows = []

        for i, area in enumerate(window_search_areas):
            windows = self.slide_window(
                img,
                window_search_area=area,
                window_size=window_sizes[i],
                overlap=overlap,
            )

            all_windows += [windows]

            hot_windows += self.search_windows(
                img=img, windows=windows, clf=clf, scaler=scaler, proba=proba
            )

        return hot_windows, all_windows

    @staticmethod
    def add_heat(heatmap, bbox_list, value=1):
        """
        Description:
        ------------
        Add heat to a heatmap.

        Parameters:
        -----------
            heatmap (np.array): heatmap
            bbox_list (list): list of bounding boxes
            value (int): value to add to the heatmap

        Returns:
        --------
            heatmap (np.array): updated heatmap
        """
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += value for all pixels inside each bbox
            heatmap[box[1] : box[1] + box[3], box[0] : box[0] + box[2]] += value

        # Return updated heatmap
        return heatmap

    @staticmethod
    def apply_threshold(heatmap, threshold):
        """
        Description:
        ------------
        Apply a threshold to the heatmap.

        Parameters:
        -----------
        heatmap (np.array): heatmap
        threshold (int): threshold to apply

        Returns:
        --------
        heatmap (np.array): thresholded heatmap
        """

        print(f"np.max(heatmap) = {np.max(heatmap)}")

        # if np.max(heatmap) >= 4:
        #     thresh = 4
        # else:
        #     thresh = np.max(heatmap) * threshold

        thresh = np.max(heatmap) * threshold
        print(f"Thresholding at: {thresh}")

        heatmap[heatmap <= thresh] = 0

        return heatmap

    @staticmethod
    def draw_labeled_bboxes(img: np.array, labels) -> np.array:
        """
        Description:
        ------------
        Draw bounding boxes around detected cars.

        Parameters:
        -----------
        img (np.array): image
        labels (np.array): labels

        Returns:
        --------
        img (np.array): image with bounding boxes
        """

        final_bboxs = []
        for i in range(1, labels[1] + 1):

            nonzero = (labels[0] == i).nonzero()

            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # Define a bounding box based on min/max x and y
            bbox = (
                (np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)),
            )

            # Check if the bounding box is big enough
            if bbox[1][0] - bbox[0][0] > 20 and bbox[1][1] - bbox[0][1] > 20:

                cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

                returned_bbox = [
                    bbox[0][0],
                    bbox[0][1],
                    bbox[1][0] - bbox[0][0],
                    bbox[1][1] - bbox[0][1],
                ]

                final_bboxs.append(returned_bbox)

        return img, final_bboxs

    def analyse_frame(
        self,
        frame,
        window_search_areas,
        window_sizes,
        overlap,
        clf,
        scaler,
        proba,
        heat_queue,
        threshold=5,
    ):
        """
        Description:
        ------------
        Analyse a frame.

        Parameters:
        -----------
        frame (np.array): frame
        heat_queue (deque): queue of heatmaps
        threshold (int): threshold to apply to the heatmap

        Returns:
        --------
        allwindows_img (np.array): image with all windows
        window_img (np.array): image with windows where cars were detected
        heatmap (np.array): heatmap
        detected_img (np.array): image with bounding boxes
        bounding_boxes (list): list of bounding boxes
        time_to_detect (float): time to detect cars
        """

        frame = np.copy(cv2.imread(frame))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Instanciate timer
        start = time.time()

        # Detect vehicles
        hot_windows, all_windows = self.browse_img_multiple_scales(
            img=frame,
            window_search_areas=window_search_areas,
            window_sizes=window_sizes,
            overlap=overlap,
            clf=clf,
            scaler=scaler,
            proba=proba,
        )
        time_to_detect = round(time.time() - start, 2)

        detected_windows_img = draw_bounding_boxes(
            img=frame, bboxs=hot_windows, color=(0, 0, 1)
        )

        allwindows_img = np.copy(frame)

        for i, _ in enumerate(all_windows):

            # Pick random color
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )

            allwindows_img = draw_bounding_boxes(
                allwindows_img, all_windows[i], color=color
            )

        # Heatmap
        heat = np.zeros_like(frame[:, :, 0]).astype(float)

        # Add heat to each box in box list
        heat = self.add_heat(heat, hot_windows)

        # Add heat to circular buffer and find total
        heat_queue.append(heat)

        total_heat = np.sum(heat_queue, axis=0).astype(np.uint8)

        # Apply threshold to help remove false positives
        total_heat = self.apply_threshold(heatmap=total_heat, threshold=threshold)
        heatmap = np.clip(total_heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        detected_img, bounding_boxes = self.draw_labeled_bboxes(np.copy(frame), labels)

        return (
            allwindows_img,
            detected_windows_img,
            heatmap,
            detected_img,
            bounding_boxes,
            time_to_detect,
        )
