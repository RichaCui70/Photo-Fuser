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
:param gaussian_tier: The tier of the Gaussian pyramid we want our image to be
:type gaussian_tier: number
:param laplacian_tier: The tier of the Laplacian pyramid we want our image to be
:type laplacian_tier: number
:param alpha: The magnitude of first image within the overlapped image
:type alpha: float

:return: A fused image between image1 and image2
:rtype: ndarray
"""
def fuse_photos_spatial(image1=None, image2=None, gaussian_tier=0, laplacian_tier=0, downscale=2, alpha=0.5):
    if image1 is None or image2 is None:
        raise Exception("Missing an image!")
    if alpha < 0 or alpha > 1:
        raise Exception("Alpha should be within 0..1")


    image1_guassian_pyramid = tuple(ski.transform.pyramid_gaussian(image1, downscale=downscale, channel_axis=-1))
    image1_gaussian_small = image1_guassian_pyramid[1:][gaussian_tier]
    image1_gaussian_small_normalized = np.ubyte((image1_gaussian_small - image1_gaussian_small.min()) / (image1_gaussian_small.max() - image1_gaussian_small.min()) * 255)
    print(image1_gaussian_small_normalized)
    image1_gaussian = ski.transform.resize(image1_gaussian_small_normalized, image1.shape)
    print(image1_gaussian)

    image2_laplacian_pyramid = tuple(ski.transform.pyramid_laplacian(image2, downscale=downscale, channel_axis=-1))
    image2_laplacian_small = image2_laplacian_pyramid[1:][laplacian_tier]
    image2_laplacian_small_normalized = np.ubyte((image2_laplacian_small - image2_laplacian_small.min()) / (image2_laplacian_small.max() - image2_laplacian_small.min()) * 255)
    image2_laplacian = ski.transform.resize(image2_laplacian_small_normalized, image2.shape)


    overlapped_image = overlap_images(image1_gaussian, image2_laplacian, alpha=alpha)
    plt.imsave("spatial_hybrid.jpg", overlapped_image)