import numpy as np


def bounding_boxes_to_mask(bounding_boxes, H, W):
    """
    Converts set of bounding boxes to a binary mask.

    Parameters
    ----------
    bounding_boxes (np.ndarray): Array of bounding boxes
    H (int): Height of the mask
    W (int): Width of the mask

    Returns
    -------
    np.ndarray: Binary mask
    """

    mask = np.zeros((H, W))
    for x, y, dx, dy in bounding_boxes:
        mask[int(y) : int(y + dy), int(x) : int(x + dx)] = 1

    return mask


def run_length_encoding(mask):
    """
    Produces run length encoding for a given binary mask.

    Parameters
    ----------
    mask (np.ndarray): Binary mask

    Returns
    -------
    str: Run length encoding
    """

    # find mask non-zeros in flattened representation
    non_zeros = np.nonzero(mask.flatten())[0]
    padded = np.pad(non_zeros, pad_width=1, mode="edge")

    # find start and end points of non-zeros runs
    limits = (padded[1:] - padded[:-1]) != 1
    starts = non_zeros[limits[:-1]]
    ends = non_zeros[limits[1:]]
    lengths = ends - starts + 1

    return " ".join(["%d %d" % (s, l) for s, l in zip(starts, lengths)])
