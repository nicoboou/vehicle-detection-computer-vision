# Standard library imports
import numpy as np

# Third party imports
import cv2
from matplotlib import pyplot as plt

# Local application imports
from src.classes.detected_object import DetectedObject
from src.utils.annotate import draw_bounding_boxes
from src.utils.detailed_frame import detailed_frame


class FrameMemory:
    """
    Description
    -----------
    Class which keeps track of the detected objects in the previous frames.

    Attributes
    ----------
    current_frame_nb (int): the current frame number
    frame_sampling_rate (int): the number of frames to skip before processing a frame
    previous_detected_objects (list): the list of detected objects in the previous frame
    current_detected_objects (list): the list of detected objects in the current frame
    save_frame (bool): whether to save the current frame
    min_detection_count (int): the minimum number of times an object must be detected before being considered as a real object
    debug_imgs (bool): whether to display debug images

    Methods
    -------
    process_frame(original_img, allwindows_img, detected_windows_img, heatmap, detected_img, bounding_boxes): returns a new image where the vehicles detected on the original image have been highlighted via a bounding box

    """

    def __init__(
        self,
        frame_sampling_rate=5,
        save_frame=False,
        min_detection_count=8,
        min_centroids_dist=40,
    ):

        self.current_frame_nb = 0
        self.frame_sampling_rate = frame_sampling_rate
        self.previous_detected_objects = []
        self.current_detected_objects = []
        self.save_frame = save_frame
        self.min_detection_count = min_detection_count
        self.min_centroids_dist = min_centroids_dist

    def process_frame(
        self,
        original_img,
        allwindows_img,
        detected_windows_img,
        heatmap,
        detected_img,
        bounding_boxes,
    ):
        """
        Returns a new image where the vehicles detected on the original image have been highlighted via a bounding box
        """
        # ====================================================================================================================== #
        # ================================== Get path of image and get np.array in RGB format ================================== #
        # ====================================================================================================================== #

        frame = np.copy(cv2.imread(original_img))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ====================================================================================================================== #
        # ========================================= CASE 1: frame_sampling_rate = 0 ============================================ #
        # ====================================================================================================================== #

        if self.frame_sampling_rate == 0:
            drawn_smoothed_bboxes = draw_bounding_boxes(frame, bounding_boxes)
            frame_with_overlays = detailed_frame(
                img=drawn_smoothed_bboxes,
                overlays=[heatmap, detected_windows_img],
                overlay_size=(320, 200),
                x_offset=20,
                y_offset=10,
            )
            return frame_with_overlays, bounding_boxes

        # ====================================================================================================================== #
        # ========================================= CASE 2: frame_sampling_rate =! 0 ============================================ #
        # ====================================================================================================================== #

        else:

            # ============================================================================================================== #
            # ========================================= Check given bounding_boxes ========================================= #
            # ============================================================================================================== #

            for bounding_box in bounding_boxes:

                # Retriece patch from bounding_box
                patch = frame[
                    bounding_box[1] : bounding_box[1] + bounding_box[3],
                    bounding_box[0] : bounding_box[0] + bounding_box[2],
                ]

                # Create a DetectedObject
                do = DetectedObject(
                    bounding_box=np.asarray(bounding_box),
                    img_patch=patch,
                    frame_nb=self.current_frame_nb,
                    min_centroids_dist=self.min_centroids_dist,
                )

                # Filter small bounding_boxes
                if do.bounding_box_area() < 16 * 16:
                    print("Skipping bounding box")
                    continue

                to_append = True

                # Check matches
                for pdo in self.current_detected_objects:
                    if pdo.same_object(do):
                        print("**Found a match**")
                        pdo.smoothe_bounding_box(do)
                        pdo.add_similar_object(do)
                        to_append = False

                        break

                # If new object, append to current_detected_objects
                if to_append:
                    print("Appending new object with centroid:", do.centroid)
                    self.current_detected_objects.append(do)

            # ==================================================================================================== #
            # ================================== Frame Sampling (at defined rate) ================================ #
            # ===> takes all detected objects in previous frames and only retain those that have more than X matches
            # ==================================================================================================== #

            print("previously detected objects=", len(self.previous_detected_objects))

            # If time to sample & not first frame, refresh and prune
            if (
                self.current_frame_nb > 0
                and self.current_frame_nb % self.frame_sampling_rate == 0
            ):
                retained_detected_objects = []

                for pdo in self.current_detected_objects:
                    print(
                        "number of times detected object appeared: ",
                        pdo.similar_objects_count(),
                    )
                    if pdo.similar_objects_count() >= self.min_detection_count:
                        print(
                            f"** Adding similar object to one with centroid {pdo.centroid} **"
                        )
                        retained_detected_objects.append(pdo)

                bounding_boxes = []
                for ro in retained_detected_objects:
                    print(
                        np.asarray(
                            [
                                ro.bounding_box[0],
                                ro.bounding_box[1],
                                ro.bounding_box[2],
                                ro.bounding_box[3],
                            ]
                        )
                    )

                    bounding_boxes.append(
                        np.asarray(
                            [
                                ro.bounding_box[0],
                                ro.bounding_box[1],
                                ro.bounding_box[2],
                                ro.bounding_box[3],
                            ]
                        )
                    )

                self.previous_detected_objects = retained_detected_objects
                self.current_detected_objects = []
                print(
                    f"Refresh at frame n. {self.current_frame_nb}: newly detected objects {len(self.previous_detected_objects)}"
                )

            # If not time to sample, continue
            else:
                print(
                    "so far newly detected objects=", len(self.current_detected_objects)
                )

                bounding_boxes = []
                for ro in self.current_detected_objects:
                    bounding_boxes.append(
                        np.asarray(
                            [
                                ro.bounding_box[0],
                                ro.bounding_box[1],
                                ro.bounding_box[2],
                                ro.bounding_box[3],
                            ]
                        )
                    )

            # Increment nb of frame
            self.current_frame_nb += 1

            drawn_smoothed_bboxes = draw_bounding_boxes(frame, bounding_boxes)

            frame_with_overlays = detailed_frame(
                img=drawn_smoothed_bboxes,
                overlays=[detected_windows_img],
                overlay_size=(320, 200),
                x_offset=20,
                y_offset=10,
            )

            return frame_with_overlays, bounding_boxes
