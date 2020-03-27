import numpy as np
from PIL import Image
import time
import os
import cv2
from skimage import io
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.utils import murmurhash3_32
import array
import glob
import mmh3
from array import array
from collections import defaultdict
import heapq

# 2D matrix is used for each of the L hashtables.
# Height is 2^K keys
# Width is number of train images (as bit array)
class Array2D(object):
    def __init__(self, w, h):
        self.data = array("B", [0] * w * h) # 2d matrix of unsigned bits
        self.width = w
        self.height = h
    def __getitem__(self, index):
        return self.data[index[1] * self.width + index[0]]
    def __setitem__(self, index, value):
          self.data[index[1] * self.width + index[0]] = value
    def printArray(self):
        for i in range(self.height):
            s = ""
            for j in range(self.width):
                s += str(self.data[i * self.width + j])
            print(s)
    def probeBucket(self, index):
        # Probes bit array for one of the buckets/keys.
        # Returns indices of the images that are hashed into same bucket.
        s = []
        for i in range(self.width):
            if self.data[index * self.width + i] == True:
                s.append(i)
        return s


# Hash function factory for each hashtable.
# TODO: not sure if this is right hash function to use. Any ideas?
def hashfuncFactoryH(m, i):
    return lambda key: mmh3.hash_from_buffer(key, seed = i) % m;

# LSH class
class LSH():
    def __init__(self, L, K, num_features, num_images):
        self.surf = cv2.xfeatures2d.SURF_create(500) # SURF
        self.L = L # number of hashtables
        self.K = K # number of bits for each hashtable's keys
        self.num_features = num_features # number of features to extract
        self.num_images = num_images # total number of images
        self.bit_array_lst = [Array2D(self.num_images, 2 ** self.K) for _ in range(self.L)] # hashtables
        self.hash_func_lst = [hashfuncFactoryH(2 ** self.K, i) for i in range(self.L)] # hash functions

    def query(self, key): 
        counts_dict = defaultdict(int)
        (kp, des) = self.surf.detectAndCompute(key, None) # extract features
        # For each feature, hash it and probe buckets to find similar images' indices.
        for i in range(len(kp)): 
            surf_feature = des[i:i+1][0]
            for j in range(self.L):
                hash_index = self.hash_func_lst[j](key)
                sim_lst = self.bit_array_lst[j].probeBucket(hash_index)
                for img_index in sim_lst:
                    counts_dict[img_index] += 1
        # Rank image indices according to count.
        sorted_dict = sorted(counts_dict.items(), key = lambda x: x[1])
        return sorted_dict

    def insert(self, key, bit_index):
        (kp, des) = self.surf.detectAndCompute(key, None)
        # if (len(kp) < self.num_features): ## TODO: need reasonable fix for this
        #     print("Not enough features: ", len(kp))
        #     return

        # For each feature, hash it and store in each hashtable.
        for i in range(len(kp)):
            surf_feature = des[i:i+1][0]
            for j in range(self.L):
                hash_index = self.hash_func_lst[j](key)
                self.bit_array_lst[j][(bit_index, hash_index)] = True

    def storeHash(self, key):
        # TODO
        self.num_images += 1
        return 0

    def printArray(self):
        for arr in self.bit_array_lst:
            arr.printArray()
            print("-----")

def preprocessing(google_photos_urls):
    surf = cv2.xfeatures2d.SURF_create(400)
    #surf.setHessianThreshold(400)

    print("======")
    for i, url in enumerate(google_photos_urls):
        img = io.imread(url)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("uploads/google_photos/" + str(i) + ".jpg", img)
        (kp, des) = surf.detectAndCompute(img, None)
        
        #img2 = cv2.drawKeypoints(img1, kp1, None, (255,0,0),4)
        # plt.imshow(img2), plt.show()

def group(filenames):
    # Return list of lists where each list is an "album"
    return [["1", "2", "3"], ["4", "5", "6"], ["7", "8"]]

def preprocessing2():
    filenames = []
    num_images = len(glob.glob('uploads/google_photos/*.jpg'))
    lsh = LSH(2, 4, 400, num_images) # TODO: find a good K and L
    for i, filename in enumerate(glob.glob('uploads/google_photos/*.jpg')):
        print(i, filename)
        filenames.append(filename)
        img = cv2.imread(filename)
        lsh.insert(img, i)
    #lsh.printArray()

    # Query a single image for any similar images.
    query_image = cv2.imread("uploads/google_photos/17.jpg")
    g = lsh.query(query_image)
    for item in g:
        print(filenames[item[0]])


preprocessing2()
