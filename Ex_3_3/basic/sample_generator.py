from typing import Tuple

import cv2
import numpy as np


def create_synthetic_image(size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Create a synthetic test image with various features.
    """
    img = np.zeros(size, dtype=np.uint8)

    # Add some geometric shapes
    cv2.rectangle(img, (50, 50), (100, 100), 255, -1)
    cv2.circle(img, (150, 150), 30, 200, -1)
    cv2.line(img, (0, 200), (256, 200), 150, 5)

    # Add text
    cv2.putText(img, 'TEST', (20, 230), cv2.FONT_HERSHEY_SIMPLEX,
                1, 255, 2, cv2.LINE_AA)

    return img
