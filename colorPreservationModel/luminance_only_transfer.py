"""
Based on the fact that visual perceptors are more sensitive on changes in luminosity than in colour.
From content and style image we extract luminosity characteristics.

In most cases, computationally less expensive than color histogram matching.
If the process of histogram matching is more complicated it can be less effective than color histogram matching.
"""

import numpy as np
import cv2
from utils import mean_std_dev

CONTENT_IMAGE_NAME = "blagaj.jpg"
STYLE_IMAGE_NAME = "StarryNightOverTheRoneVanGogh.jpg"

LOGARITHMIC_COLOR_SPACE = cv2.COLOR_BGR2LAB
RGB_COLOR_SPACE = cv2.COLOR_LAB2BGR


class LuminanceOnlyTransfer(object):
    def __init__(self):
        print("LuminanceOnlyTransfer initialized.")

    @staticmethod
    def open_image_in_log_space(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, LOGARITHMIC_COLOR_SPACE)
        return image

    @staticmethod
    def match(style_image_path, content_image_path):
        style_image = LuminanceOnlyTransfer.open_image_in_log_space(style_image_path)
        s_height, s_width, s_depth = style_image.shape
        content_image = LuminanceOnlyTransfer.open_image_in_log_space(content_image_path)

        mu_s, sigma_s = mean_std_dev(style_image)
        mu_c, sigma_c = mean_std_dev(content_image)

        for h in range(0, s_height):
            for w in range(0, s_width):
                for d in range(0, s_depth):
                    """
                    Each luminance pixel is updated with formula:
                    Ls' = (sigma_c / sigma_s) * (Ls - mu_s) + mu_c
                    """
                    luminance_pixel = style_image[h, w, d]
                    luminance_pixel = (sigma_c[d] / sigma_s[d]) * \
                                      (luminance_pixel - mu_s[d]) + mu_c[d]

                    if luminance_pixel < 0:
                        luminance_pixel = 0
                    if luminance_pixel > 255:
                        luminance_pixel = 255

                    style_image[h, w, d] = luminance_pixel

        style_image = cv2.cvtColor(style_image, RGB_COLOR_SPACE)
        cv2.imwrite("../images/results/LOT_" + STYLE_IMAGE_NAME, style_image)

        return style_image


LuminanceOnlyTransfer.match("../images/style/" + STYLE_IMAGE_NAME, "../images/content/" + CONTENT_IMAGE_NAME)
