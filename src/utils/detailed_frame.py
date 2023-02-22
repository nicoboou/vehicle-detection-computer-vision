# Standard library imports
import copy

# Third party imports
import cv2

# Local application imports
from src.classes.vehicle_detector import VehicleDetector


def detailed_frame(img, overlays, overlay_size, x_offset, y_offset):
    """
    Returns an image that combines the original image with a set of overlaid images
    The overlaid images are resized to the supplied dimensions and overlaid onto the original image.
    Note that the returned image is a modification of the original image.
    """

    for i, overlay_img in enumerate(overlays):
        sc = i + 1
        small_overlay_img = cv2.resize(overlay_img, overlay_size)
        overlay = small_overlay_img

        # If it is the heatmap (nb of channels = 1), we apply threshold
        if len(overlay.shape) < 3:
            # Create the mini-heatmap
            overlay = VehicleDetector.apply_threshold(overlay, threshold=5)

        new_y_offset = y_offset
        new_x_offset = sc * x_offset + i * overlay_size[0]

        print(f"img.shape: {img.shape}")
        print(f"overlay.shape: {overlay.shape}")

        img[
            new_y_offset : new_y_offset + overlay_size[1],
            new_x_offset : new_x_offset + overlay_size[0],
        ] = overlay

    return img
