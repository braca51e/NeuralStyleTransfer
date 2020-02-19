# NeuralStyleTransfer
Implementation of 2D Neural Style Transfer in a manner proposed in an original paper, along with optimization for preserving content image colors.

## Programs & libraries needed in order to run this project 
* [Pillow](https://python-pillow.org/) : Python Imaging Library
* [matplotlib](https://matplotlib.org/) : Plotting library for the Python programming language and its numerical mathematics extension NumPy
* [NumPy](https://www.numpy.org/) : Fundamental package for scientific computing with Python
* [OpenCV](https://opencv.org/) : Library of programming functions mainly aimed at real-time computer vision
* [PyTorch](https://pytorch.org/) : Open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing

## How to run?
Just run `main.py` in `initialModel` package.<br>
If `COLOR_HISTOGRAM_MATCHING` variable is set to True, then it will do the color preservation optimization with technique: Color Histogram Matching.<br>
If `LUMINANCE_ONLY_TRANSFER` variable is set to True, then it will do the color preservation optimization with technique: Luminance Only Transfer.

## Example with color preservation
For given inputs:
![alt text](https://raw.githubusercontent.com/reinai/NeuralStyleTransfer/master/images/results/resultExamples/third/contentstyle3.PNG)
Outputs with different ratios:
![alt text](https://raw.githubusercontent.com/reinai/NeuralStyleTransfer/master/images/results/resultExamples/third/collage.jpg)
