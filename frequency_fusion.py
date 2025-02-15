import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

from alignment import *

"""
Fuses image1 with image2 using Frquency Fournier Transform (FFT) based filtering

:param image1: The ndarray of the first image
:type image1: ndarray
:param image2: The ndarray of the second image
:type image2: ndarray
:param low_pass_cutoff: The position of the cut-off relative to the shape of the FFT for the low pass images. Receives a value between [0, 0.5].
:type low_pass_cutoff: float
:param high_pass_cutoff: The position of the cut-off relative to the shape of the FFT for the high pass images. Receives a value between [0, 0.5].
:type high_pass_cutoff: float
:param alpha: The magnitude of first image within the overlapped image
:type alpha: float

:return: A fused image between image1 and image2
:rtype: ndarray
"""
def fuse_photos_freq(image1=None, image2=None, low_pass_cutoff=0.08, high_pass_cutoff=0.08, alpha=0.5):
    if image1 is None or image2 is None:
        raise Exception("Missing an image!")

    low_pass_filtered = ski.filters.butterworth(
        image=image1,
        cutoff_frequency_ratio=low_pass_cutoff,
        high_pass=False
    )
    low_pass_normalized = np.ubyte((low_pass_filtered - low_pass_filtered.min()) / (low_pass_filtered.max() - low_pass_filtered.min()) * 255)
    low_pass = ski.transform.resize(low_pass_normalized, image1.shape)

    high_pass_filtered = ski.filters.butterworth(
        image=image2,
        cutoff_frequency_ratio=high_pass_cutoff,
        high_pass=True
    )
    high_pass_normalized = np.ubyte((high_pass_filtered - high_pass_filtered.min()) / (high_pass_filtered.max() - high_pass_filtered.min()) * 255)
    high_pass = ski.transform.resize(high_pass_normalized, image1.shape)

    overlapped_image = overlap_images(low_pass, high_pass, alpha=alpha)
    plt.imsave("frequency_hybrid.jpg", overlapped_image)
