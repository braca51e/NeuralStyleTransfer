from __future__ import print_function
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

cnn_normalization_mean = (0.485, 0.456, 0.406)
cnn_normalization_std = (0.229, 0.224, 0.225)
ITERATIONS = 8000
MOST_INFLUENTIAL_CONV_LAYER = 'conv4_2'


def compute_total_loss(alpha, content_loss, beta, style_loss):
    """

    :param alpha: content's weight
    :param content_loss: computed content loss function
    :param beta: style's weight
    :param style_loss: computed style loss function
    :return: loss for minimization of content and style loss functions
    """
    return alpha * content_loss + beta * style_loss


def generated_image_update(optimizer, total_loss):
    # zero the gradient buffers of all parameters
    optimizer.zero_grad()
    # back-propagate the error
    total_loss.backward()
    # update
    optimizer.step()


"""
Original Neural Style Transfer paper uses a pre-trained CNN, and builds on top of that (transfer learning).
By following the original paper, we are going to use the VGG-19 network.
"""

assert(torch.cuda.is_available() is True)

model = models.vgg19(pretrained=True).features


def discard_transparency_channel(transform, image):
    # https://stackoverflow.com/questions/35902302/discarding-alpha-channel-from-images-stored-as-numpy-arrays
    return transform(image)[:3, :, :]


def load_image(file_path, shape=None):
    image = Image.open(file_path).convert("RGB")

    if max(image.size) > 800:
        size = 800
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    """
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB 
    images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded 
    in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and 
    std = [0.229, 0.224, 0.225].
    https://pytorch.org/docs/stable/torchvision/models.html
    """
    image_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image = discard_transparency_channel(image_transforms, image).unsqueeze(0)

    return image


def tensor_to_image(tensor, title=None):
    image = tensor.cpu().clone().detach()
    image = image.numpy().squeeze(0)  # remove the fake batch dimension

    image = image.transpose(1, 2, 0)
    image = image * np.array(cnn_normalization_std) + np.array(cnn_normalization_mean)

    image = image.clip(0, 1)

    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(3)
    return image


def features_l(input_image, torch_layers):
    layer_names = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    mapped_features = {}

    img = input_image

    for torch_name, layer_object in torch_layers:
        img = layer_object(img)
        if torch_name in layer_names:
            mapped_features[layer_names[torch_name]] = img

    return mapped_features
