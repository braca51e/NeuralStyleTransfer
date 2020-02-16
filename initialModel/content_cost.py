from matplotlib.pyplot import imread
import torch

content_image_name = "thelouvre.jpg"

content_image = imread("../images/content/" + content_image_name)
# imshow(content_image)
# plt.show()


class ContentCost(object):
    def __init__(self):
        print("ContentCost initialized.")

    @staticmethod
    def compute(o, g, convolutional_layer):
        # o, g - feature representations of original and generated image on given convolutional layer
        return torch.mean((g[convolutional_layer] - o[convolutional_layer])**2)
