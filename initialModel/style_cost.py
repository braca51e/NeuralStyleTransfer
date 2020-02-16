import torch
from matplotlib.pyplot import imread

style_image_name = "dvabasibozuka_paja.jpg"

style_image = imread("../images/style/" + style_image_name)
# imshow(style_image)
# plt.show()

"""
We don't want to capture the style from only one layer, but to "merge" style costs from multiple
different layers. We are going to use weights for style layers as described in paper and combine
style costs for different layers in a Jstyle(style_image, generated_image).
"""
STYLE_LAYERS = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2}

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

"""
We compute style representation by computing the correlations between different filter outputs.
These correlations can be computed in terms of features by using the Gram matrix G^l (multiply of
matrix and transposed matrix), where G^l(i, j) is a scalar product between vectorized feature maps 
i and j in layer l. 
G^l(i,j) = sum_k(F^l(ik) * F^l(jk))
"""


def gram_matrix(input_l):
    """
    :param input_l: input tensor representing target feature
    :return: gram matrix at layer l
    """
    # depth represents a number of feature maps
    # height, width are dimensions of a particular map
    batch_size, depth, height, width = input_l.size()

    input_k = input_l.view(depth, height * width)

    gram_m = torch.mm(input_k, input_k.t())

    return gram_m


class LayerStyleCost(object):
    def __init__(self):
        print("LayerStyleCost initialized.")

    @staticmethod
    def calculate_gram_matrices(style_features):
        gram_matrices = {}
        for layer in style_features:
            gram_matrices[layer] = gram_matrix(style_features[layer])
        return gram_matrices

    @staticmethod
    def compute_layer_style_cost(layer, generated_image_gram_matrix, style_image_gram_matrix):
        return STYLE_LAYERS[layer] * torch.mean((generated_image_gram_matrix - style_image_gram_matrix) ** 2)

    @staticmethod
    def compute(J_style_layer, depth, height, width):
        J_style = J_style_layer / (depth * height * width)

        return J_style
