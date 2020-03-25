import numpy as np
from PIL import Image
import time
import os
import cv2
from skimage import io
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def preprocessing(google_photos_urls):
    surf = cv2.xfeatures2d.SURF_create(400)
    surf.setHessianThreshold(5000)
    print("======")
    for url1 in google_photos_urls:
        img1 = io.imread(url1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        #cv2.imwrite("uploads/google_photos/" + str(i) + ".jpg", img)
        (kp1, des1) = surf.detectAndCompute(img1, None)
        print(len(kp1))
        img2 = cv2.drawKeypoints(img1, kp1, None, (255,0,0),4)
        # plt.imshow(img2), plt.show()

def group(filenames):
    # Return list of lists where each list is an "album"
    return [["1", "2", "3"], ["4", "5", "6"], ["7", "8"]]

