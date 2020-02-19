import numpy as np

# Calculates a mean and standard deviation of array elements.


def mean_std_dev(image):
    mean = [
        np.mean(image[:, :, 0]),
        np.mean(image[:, :, 1]),
        np.mean(image[:, :, 2])
    ]
    std_dev = [
        np.std(image[:, :, 0]),
        np.std(image[:, :, 1]),
        np.std(image[:, :, 2])
    ]

    return mean, std_dev
