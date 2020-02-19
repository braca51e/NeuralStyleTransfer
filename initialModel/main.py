import torch
from content_cost import ContentCost
from style_cost import LayerStyleCost
from style_cost import STYLE_LAYERS
from model import model
from model import load_image
from model import tensor_to_image
from model import ITERATIONS
from model import features_l
from model import MOST_INFLUENTIAL_CONV_LAYER
from style_cost import gram_matrix
from model import compute_total_loss
from model import generated_image_update
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import imageio

COLOR_HISTOGRAM_MATCHING = True

ALPHA = 1
BETA = 10 ** 6

ratio = ALPHA / BETA

PATH = "../savedModels/initial/initialmodel.pth"

CONTENT_IMAGE_NAME = "blagaj.jpg"
STYLE_IMAGE_NAME = "StarryNightOverTheRoneVanGogh.jpg"

content_cost = ContentCost()
style_cost = LayerStyleCost()

"""
We need to freeze all the network except the final layer. We need to set requires_grad == False to freeze 
the parameters so that the gradients are not computed in backward().
In this case, we only want to keep the convolution's part of VGG-19.
We are optimizing pixels of generated (G) image.
"""
for param in model.parameters():
    param.requires_grad = False

print("VGG-19 architecture:\n\n" + str(model))

device = torch.device("cuda")
model.to(device)

# CONV and pooling layers
torch_layers = model._modules.items()

# inputs
content_image = load_image("../images/content/" + CONTENT_IMAGE_NAME).to(device)
style_image = load_image("../images/style/" + STYLE_IMAGE_NAME, shape=content_image.shape[-2:]).to(device)

if COLOR_HISTOGRAM_MATCHING:
    style_image = load_image("../images/results/CHM_" + STYLE_IMAGE_NAME, shape=content_image.shape[-2:]).to(device)

assert content_image.size() == style_image.size(), \
    "content and style images need to be of the same size"

plot_image = np.concatenate((tensor_to_image(content_image), tensor_to_image(style_image)), axis=1)

plt.imshow(plot_image)
plt.show()

content_features = features_l(content_image, torch_layers)
style_features = features_l(style_image, torch_layers)

# gram matrices for each layer
gram_matrices = LayerStyleCost.calculate_gram_matrices(style_features)

generated_image = content_image.clone().requires_grad_(True).to(device)


def run():
    """
    cost function that minimizes both the style and the content cost
    J(G) = ALPHA * Jcontent(C, G) + BETA * Jstyle(S, G)
    :return:
    """
    save_image_steps = 1000

    optimizer = optim.Adam([generated_image], lr=0.003)
    losses = []

    for step in range(1, ITERATIONS + 1):
        generated_image_features = features_l(generated_image, torch_layers)

        J_style = 0
        J_content = ContentCost.compute(generated_image_features, content_features,
                                       MOST_INFLUENTIAL_CONV_LAYER)

        for layer in STYLE_LAYERS:
            generated_image_feature_l = generated_image_features[layer]
            generated_image_gram_matrix_l = gram_matrix(generated_image_feature_l)

            batch_size, depth, height, width = generated_image_feature_l.shape

            style_image_gram_matrix = gram_matrices[layer]

            J_style_layer = LayerStyleCost.compute_layer_style_cost(layer, generated_image_gram_matrix_l,
                                                                    style_image_gram_matrix)

            J_style += LayerStyleCost.compute(J_style_layer, depth, height, width)

        # alpha, content_loss, beta, style_loss
        Jg = compute_total_loss(ALPHA, J_content, BETA, J_style)

        generated_image_update(optimizer, Jg)

        if step % save_image_steps == 0:
            print("Jg = " + str(Jg.item()))
            imageio.imwrite("../images/results/blagajvangogh/" + str(step) + ".jpg",
                            tensor_to_image(generated_image))
            losses.append("Jg = " + str(Jg.item()))

    with open("../images/results/blagajvangogh/losses.txt", "w") as f:
        for loss in losses:
            f.write(str(loss) + "\n")

    torch.save(model, PATH)


run()
