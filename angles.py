import numpy as np


def calculate_angle(a, b, c):
    """
    Return the angle in degrees at point b, formed by the vectors b→a and b→c.

    Uses arctan2 rather than dot-product / arccos to handle all four quadrants
    correctly and avoid numerical instability near 0° and 180°.
    Result is folded into [0, 180].
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    angle = np.degrees(
        np.arctan2(c[1] - b[1], c[0] - b[0]) -
        np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle
