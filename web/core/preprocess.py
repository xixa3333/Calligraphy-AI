import cv2
import numpy as np


def calligraphy_preprocess(image, target_size=128):
    """Crop, center, thicken and skeletonize a grayscale calligraphy image."""
    if image is None:
        raise ValueError("Image is empty.")

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coordinates = cv2.findNonZero(cv2.bitwise_not(binary))
    if coordinates is None:
        resized = cv2.resize(image, (target_size, target_size))
    else:
        x, y, width, height = cv2.boundingRect(coordinates)
        cropped = image[y : y + height, x : x + width]
        ratio = width / height
        if ratio > 1:
            new_width, new_height = target_size, max(1, int(target_size / ratio))
        else:
            new_width, new_height = max(1, int(target_size * ratio)), target_size
        glyph = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized = np.full((target_size, target_size), 255, dtype=np.uint8)
        top = (target_size - new_height) // 2
        left = (target_size - new_width) // 2
        resized[top : top + new_height, left : left + new_width] = glyph

    inverted = cv2.bitwise_not(resized)
    thickened = cv2.dilate(inverted, np.ones((2, 2), np.uint8), iterations=1)
    skeleton = cv2.ximgproc.thinning(
        thickened, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
    )
    return cv2.bitwise_not(skeleton).astype(np.float32) / 255.0
