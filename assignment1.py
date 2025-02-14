import os
import sys
import skimage as ski

from alignment import *
from spatial_fusion import *
from frequency_fusion import *

IMAGE_ONE_ARG_INDEX = 1
IMAGE_TWO_ARG_INDEX = 2
KEYPOINTS_ONE_ARG_INDEX = 3
KEYPOINTS_TWO_ARG_INDEX = 4

PHOTO_DIR = './photos'

"""
Read the keypoints of a text file via the path to the file

:param path: The path to the text file
:type path: string

:return: An array of tuples shown through keypoints
:rtype: array
"""
def read_keypoints(path=""):

    keypoints = []
    with open(path) as f:
        all_lines = f.readlines()
        for i in range(0, len(all_lines)):
            keypoints.append(tuple(map(int, all_lines[i][:-1].split())))

    return keypoints

if __name__ == "__main__":
    image1 = os.path.join(PHOTO_DIR, sys.argv[IMAGE_ONE_ARG_INDEX])
    image2 = os.path.join(PHOTO_DIR, sys.argv[IMAGE_TWO_ARG_INDEX])

    image1 = ski.io.imread(image1)
    image2 = ski.io.imread(image2)

    keypoints1 = read_keypoints(f'{PHOTO_DIR}/{sys.argv[KEYPOINTS_ONE_ARG_INDEX]}')
    keypoints2 = read_keypoints(f'{PHOTO_DIR}/{sys.argv[KEYPOINTS_TWO_ARG_INDEX]}')

    # Resize and warp the second image to the first image based off the keypoints and first image dimensions
    image2 = ski.transform.resize(image2, image1.shape)
    image2 = image_warping(image1_keypoints=keypoints1, image2_keypoints=keypoints2, image2=image2)

    overlap_images(image1, image2, save_image=True)
    fuse_photos_spatial(image1=image1, image2=image2, alpha=0.5)
    fuse_photos_freq(image1=image1, image2=image2, alpha=0.5)
