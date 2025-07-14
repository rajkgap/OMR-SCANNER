import numpy as np

from src import logger


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between_vectors(v1, v2):
    cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def calculate_error_rates_of_aspect_and_angles(p1, p2, p3, p4, ideal_aspect_ratio, ideal_angle):
    logger.info("Calculating error rates of aspect and angles....")
    p1, p2, p3, p4 = map(np.array, [p1, p2, p3, p4])

    aspect_ratio_1 = distance(p2, p4) / distance(p1, p2)
    aspect_ratio_2 = distance(p1, p3) / distance(p3, p4)

    angles = [
        angle_between_vectors(p2 - p1, p3 - p1),
        angle_between_vectors(p1 - p2, p4 - p2),
        angle_between_vectors(p1 - p3, p4 - p3),
        angle_between_vectors(p2 - p4, p3 - p4)
    ]

    error_aspect_ratio1 = float(100*abs(ideal_aspect_ratio-aspect_ratio_1)/ideal_aspect_ratio)
    error_aspect_ratio2 = float(100*abs(ideal_aspect_ratio-aspect_ratio_2)/ideal_aspect_ratio)
    error_angles = [float(100*abs(ideal_angle-ang)/ideal_angle) for ang in angles]
    logger.info("Calculated error rates of aspect and angles.")
    return {
        "error_aspect_ratio_1": error_aspect_ratio1,
        "error_aspect_ratio_2": error_aspect_ratio2,
        "error_angles": error_angles
    }
