import cv2


def convert_color(img, conversion="RGB2YCrCb"):
    """
    Convert image from one color space to another
    """
    if conversion == "BGR2RGB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if conversion == "RGB2YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conversion == "BGR2YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conversion == "RGB2LUV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conversion == "RGB2Lab":
        return cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    if conversion == "RGB2HSV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conversion == "RGB2HLS":
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conversion == "RGB2YUV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    if conversion == "BGR2YUV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
