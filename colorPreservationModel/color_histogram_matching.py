"""
For inputs: content and style image, we generate new style image based on input style image, such that
color histogram of new style image is 'aligned' with content's image histogram.

If xi = (R, G, B)^T is transpose vector that represents one pixel of an input image , we transform every pixel
in a way: Xs' <- AXs + b, where A is matrix of type 3x3, b is a column vector made up of 3 elements.

This approach is used so that mean and variance of RGB values in new style image is 'aligned' with values from
content image.
"""

import numpy as np
import imageio

CONTENT_IMAGE_NAME = "blagaj.jpg"
STYLE_IMAGE_NAME = "StarryNightOverTheRoneVanGogh.jpg"


class ColorHistogramMatching(object):
    def __init__(self):
        print("ColorHistogramMatching initialized.")

    @staticmethod
    def open_image_as_rgb_file(image_path):
        # https://stackoverflow.com/questions/49701626/how-to-open-image-as-rgb-file-when-using-imageio
        image = imageio.imread(image_path, pilmode="RGB")
        return image

    @staticmethod
    def normalize_image(image):
        # get in 0.0 - 1.0 range
        return image.astype(float) / 256

    @staticmethod
    def pixel_covariance(image):
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance
        mu = image.mean(0).mean(0)
        relative_difference = (image - mu).transpose(2, 0, 1).reshape(3, -1)
        return relative_difference.dot(
            relative_difference.T) / relative_difference.shape[1] + (10**(-5)) * np.eye(relative_difference.shape[0])

    @staticmethod
    def cholesky_decomposition(pixel_covariance_style, pixel_covariance_content, style_image):
        Ls = np.linalg.cholesky(pixel_covariance_style)
        Lc = np.linalg.cholesky(pixel_covariance_content)
        mu = style_image.mean(0).mean(0)
        A_chol = Lc.dot(np.linalg.inv(Ls)).dot((style_image - mu).transpose(2, 0, 1).reshape(3, -1))
        return A_chol

    @staticmethod
    def generate_style_prime_image(style_image, A_chol):
        style_prime_image = A_chol.reshape(*style_image.transpose(2, 0, 1).shape).transpose(1, 2, 0)
        style_prime_image += style_image.mean(0).mean(0)
        style_prime_image[style_prime_image > 1] = 1
        style_prime_image[style_prime_image < 0] = 0
        return style_prime_image

    @staticmethod
    def match(style_image_path, content_image_path):
        """
        We choose this transformation so that the mean and covariance of the RGB values in the new style image
        S' match those of C.
        :param style_image_path: file path to the input style image that will be transformed
        :param content_image_path: file path to the content image that will be used for transformation of style image
        :return: new_style_image (style image is transformed in order to match colors of the content image)
        """
        unnormalized_style_image = ColorHistogramMatching.open_image_as_rgb_file(style_image_path)
        style_image = ColorHistogramMatching.normalize_image(unnormalized_style_image)

        unnormalized_content_image = ColorHistogramMatching.open_image_as_rgb_file(content_image_path)
        content_image = ColorHistogramMatching.normalize_image(unnormalized_content_image)

        pixel_covariance_style = ColorHistogramMatching.pixel_covariance(style_image)
        pixel_covariance_content = ColorHistogramMatching.pixel_covariance(content_image)

        A_chol = ColorHistogramMatching.cholesky_decomposition(pixel_covariance_style, pixel_covariance_content,
                                                               style_image)

        style_prime_image = ColorHistogramMatching.generate_style_prime_image(style_image, A_chol)
        imageio.imwrite("../images/results/CHM_" + STYLE_IMAGE_NAME, style_prime_image)

        return style_prime_image


ColorHistogramMatching.match("../images/style/" + STYLE_IMAGE_NAME, "../images/content/" + CONTENT_IMAGE_NAME)
