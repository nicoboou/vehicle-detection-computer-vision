# Standard library imports
import math

# Third party imports
import cv2
import numpy as np


class DetectedObject:
    """
    Description
    -----------
    Class which represents a detected object

    Attributes
    ----------
    bounding_box (tuple): the bounding box of the detected object
    img_patch (numpy.ndarray): the image patch of the detected object
    frame_nb (int): the frame number in which the object was detected
    centroid (tuple): the centroid of the bounding box
    similar_objects (list): the list of similar objects detected in the previous frames
    min_centroids_dist (int): minimum distance between two centroids to consider an object to be the same

    Methods
    -------
    bounding_box_area(): returns the area of the bounding box
    same_object(detected_object): returns whether the current detected object is the same as the newly detected one passed as argument
    smoothe_bounding_box(same_object): smoothes the bounding boxes between the current object and the same one from another frame
    """

    def __init__(self, bounding_box, img_patch, frame_nb, min_centroids_dist):
        self.bounding_box = bounding_box
        self.img_patch = img_patch
        self.frame_nb = frame_nb
        self.min_centroids_dist = min_centroids_dist

        self.centroid = (
            int((bounding_box[0] + (bounding_box[0] + bounding_box[2])) / 2),
            int((bounding_box[1] + (bounding_box[1] + bounding_box[3])) / 2),
        )

        self.similar_objects = []

    def bounding_box_area(self):
        """
        Returns the area of the bounding box
        """
        return (self.bounding_box[2]) * (self.bounding_box[3])

    def same_object(self, detected_object):
        """
        Returns whether the current detected object is the same as the newly detected one passed as argument
        """
        # The distance between the centroids
        dist = math.hypot(
            detected_object.centroid[0] - self.centroid[0],
            detected_object.centroid[1] - self.centroid[1],
        )

        if dist <= self.min_centroids_dist:
            return True

        return False

    def smoothe_bounding_box(self, same_object):
        """
        Smoothes the bounding boxes between the current object and the same one from another frame
        """
        print("old centroid=", self.centroid)
        self.centroid = (
            int((same_object.centroid[0] + self.centroid[0]) / 2),
            int((same_object.centroid[1] + self.centroid[1]) / 2),
        )
        print("new centroid=", self.centroid)

        print("old bbox=", self.bounding_box)

        current_width = self.bounding_box[2]
        current_height = self.bounding_box[3]

        obj_width = same_object.bounding_box[2]
        obj_height = same_object.bounding_box[3]

        max_width = max(current_width, obj_width)
        max_height = max(current_height, obj_height)

        # mean_width = (current_width + obj_width) / 2
        # mean_height = (current_height + obj_height) / 2

        half_width = int(max_width / 2)
        half_height = int(max_height / 2)

        new_bounding_box = np.asarray(
            (
                self.centroid[0] - half_width,
                self.centroid[1] - half_height,
                max_width,
                max_height,
            )
        )

        self.bounding_box = new_bounding_box
        print("new bbox=", self.bounding_box)

    def add_similar_object(self, similar_object):
        """
        Accumulates a similar object
        """
        self.similar_objects.append(similar_object)

    def similar_objects_count(self):
        """
        Returns the number of similar objects
        """
        return len(self.similar_objects)
