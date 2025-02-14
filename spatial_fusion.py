import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

from alignment import *

"""
Fuses image1 with image2 using Laplacian and Gaussian filtering

:param image1: The ndarray of the first image
:type image1: ndarray
:param image2: The ndarray of the second image
:type image2: ndarray
:param sigma: The magnitude of blur used with Gaussian filtering
:type sigma: float
:param alpha: The magnitude of first image within the overlapped image
:type alpha: float

:return: A fused image between image1 and image2
:rtype: ndarray
"""
def fuse_photos_spatial(image1=None, image2=None, sigma=3.0, alpha=0.5):
    if image1 is None or image2 is None:
        raise Exception("Missing an image!")
    
    gaussian = ski.filters.gaussian(image1, sigma=sigma)
    laplacian = ski.filters.laplace(image2)

    overlapped_image = overlap_images(gaussian, laplacian, alpha=alpha)
    plt.imsave("spatial_hybrid.jpg", overlapped_image)