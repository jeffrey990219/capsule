from numpy import *
from PIL import Image
import time
import os
import cv2
from skimage import io


def preprocessing(google_photos_urls):
    for url in google_photos_urls:
        img = io.imread(url)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite("uploads/google_photos/" + str(i) + ".jpg", img)

        # Extract surf features
        surf = cv2.xfeatures2d.SURF_create()
        (kps, descs) = surf.detectAndCompute(img, None)
        print (len(kps), descs.shape)

        # kps are surf keypoints (https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html)
        # descs is a 2d array with dimensions (# features , 64) (which is the same as CAPSULE)
        # probably will need to get only 512 features(rows) out of each image
        # which ones would we get then??? first 512? what about images that have less features


def group(filenames):
    # Return list of lists where each list is an "album"
    return [["1", "2", "3"], ["4", "5", "6"], ["7", "8"]]

