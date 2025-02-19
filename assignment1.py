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

if __name__ == "__main__":
    image1 = os.path.join(PHOTO_DIR, sys.argv[IMAGE_ONE_ARG_INDEX])
    image2 = os.path.join(PHOTO_DIR, sys.argv[IMAGE_TWO_ARG_INDEX])

    image1 = ski.io.imread(image1)
    image2 = ski.io.imread(image2)

    keypoints1 = read_keypoints(f'{PHOTO_DIR}/{sys.argv[KEYPOINTS_ONE_ARG_INDEX]}')
    keypoints2 = read_keypoints(f'{PHOTO_DIR}/{sys.argv[KEYPOINTS_TWO_ARG_INDEX]}')
    
    if keypoints1 is None or keypoints2 is None:
        image2 = ski.transform.resize(image2, image1.shape)
    else:
        image2 = image_warping(image1_keypoints=keypoints1, image2_keypoints=keypoints2, image2=image2)
        
    overlap_images(ski.util.img_as_float64(image1), ski.util.img_as_float64(image2), save_image=True)
    fuse_photos_spatial(image1=image1, image2=image2)
    fuse_photos_freq(image1=image1, image2=image2)
