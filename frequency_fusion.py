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
def fuse_photos_freq(image1=None, image2=None, low_pass_cutoff=0.08, high_pass_cutoff=0.02, alpha=0.5):
    if image1 is None or image2 is None:
        raise Exception("Missing an image!")

    low_pass = ski.filters.butterworth(
        image=image1,
        cutoff_frequency_ratio=low_pass_cutoff,
        high_pass=False
    )
    low_pass_normalized = np.ubyte((low_pass - low_pass.min()) / (low_pass.max() - low_pass.min()) * 255)
    print(low_pass.dtype)
    # image1_gaussian = ski.transform.resize(low_pass_normalized, image1.shape)

    high_pass = ski.filters.butterworth(
        image=image2,
        cutoff_frequency_ratio=high_pass_cutoff,
        high_pass=True
    )
    
    print(high_pass.dtype)

    # overlapped_image = overlap_images(low_pass, high_pass, alpha=alpha)
    # plt.imsave("frequency_hybrid.jpg", overlapped_image)
