"""
BlueDream v3plus — View Transformer
Perspective transformation between screen and pitch coordinates.
Self-contained — no roboflow/sports dependency.
"""

from typing import Tuple

import cv2
import numpy as np
from numpy.typing import NDArray


class ViewTransformer:
    """Perspective transform via homography between source and target points."""

    def __init__(
        self,
        source: NDArray[np.float32],
        target: NDArray[np.float32],
    ):
        if source.shape != target.shape:
            raise ValueError(
                f"Source and target must have same shape. "
                f"Got {source.shape} vs {target.shape}"
            )
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError(
                f"Expected (N, 2) arrays. Got {source.shape}"
            )
        if len(source) < 4:
            raise ValueError(
                f"Need at least 4 point pairs. Got {len(source)}"
            )
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography computation failed")

    def transform_points(self, points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Transform screen points to pitch coordinates."""
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)

    def transform_image(
        self, image: NDArray[np.uint8], resolution_wh: Tuple[int, int]
    ) -> NDArray[np.uint8]:
        """Warp image to target resolution using homography."""
        return cv2.warpPerspective(image, self.m, resolution_wh)
