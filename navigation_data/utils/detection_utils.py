from typing import Union

import numpy as np


def calculate_boundingbox_size(boundingbox: list) -> float:
    x_min, y_min, x_max, y_max = boundingbox
    return (x_max - x_min) * (y_max - y_min)


def calculate_boundingbox_center(bounding_box: Union[list, np.ndarray]) -> tuple:
    instance_center = (
        (bounding_box[0] + bounding_box[2]) / 2,
        (bounding_box[1] + bounding_box[3]) / 2,
    )

    return instance_center
