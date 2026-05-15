from __future__ import annotations

from typing import overload

import cv2
import numpy as np


@overload
def preprocess_image(image: None) -> None: ...
@overload
def preprocess_image(image: np.ndarray) -> cv2.typing.MatLike: ...


def preprocess_image(image: np.ndarray | None) -> cv2.typing.MatLike | None:
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)
