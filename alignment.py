import numpy as np
import skimage as ski
import matplotlib.pyplot as plt


"""
Aligns keypoints from image 2 to image 1, warping image 2 based off the resulting alignment

:param image1_keypoints: The keypoints of the first image
:type image1_keypoints: array
:param image2_keypoints: The keypoints of the second image
:type image2_keypoints: array
:param image2: The ndarray of the second image
:type image2: ndarray

:return: The warped version of image 2
:rtype: ndarray
"""
def image_warping(image1_keypoints=[], image2_keypoints=[], image2=None):
    if image2 is None:
        raise Exception("Missing image2!")
    elif len(image1_keypoints) == 0 or len(image1_keypoints) == 0:
        raise Exception("Missing keypoints!")

    eye1_left = np.array(image1_keypoints[0])  # Left eye in image 1
    eye1_right = np.array(image1_keypoints[1])  # Right eye in image 1

    eye2_left = np.array(image2_keypoints[0])  # Left eye in image 2
    eye2_right = np.array(image2_keypoints[1])  # Right eye in image 2

    # Compute the scale factor (eye distance ratio)
    eye_dist1 = np.linalg.norm(eye1_right - eye1_left)
    eye_dist2 = np.linalg.norm(eye2_right - eye2_left)
    scale = eye_dist1 / eye_dist2

    # Compute the translation vector (align left eye)
    translation = eye1_left - scale * eye2_left

    # Construct the Affine transformation matrix
    tform = ski.transform.AffineTransform(scale=(scale, scale), translation=translation)

    # Apply transformation
    aligned_image2 = ski.transform.warp(image2, tform.inverse, output_shape=image2.shape)

    return aligned_image2

"""
Overlaps image1 with image2

:param image1: The ndarray of the first image
:type image1: ndarray
:param image2: The ndarray of the second image
:type image2: ndarray
:param alpha: The magnitude of first image within the overlapped image
:type alpha: float
:param save_image: Toggle to save a .jpg file of the returned image
:type save_image: boolean

:return: The warped version of image 2
:rtype: ndarray
"""
def overlap_images(image1=None, image2=None, alpha=0.5, save_image=False):
    if image1 is None or image2 is None:
        raise Exception("Missing an image!")
    
    byte_image1 = ski.util.img_as_ubyte(image1)
    byte_image2 = ski.util.img_as_ubyte(image2)

    overlapped_image = np.ubyte(alpha*byte_image1 + (1-alpha)*byte_image2)

    if save_image: plt.imsave("aligned.jpg", overlapped_image)

    return overlapped_image