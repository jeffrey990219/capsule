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
from datasketch import MinHash
from bitstring import BitArray
from sklearn.metrics.pairwise import euclidean_distances
import imagehash

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


def hashfuncFactoryH(m, i, j):
    def f(key):
        return mmh3.hash(key, seed = i * 4 + j) % m;
    return f

def hashfuncFactoryG(m, i, j):
    def f(key):
        return -1 if mmh3.hash(key, seed = (i * 4 + j) + 9999) % 2 == 0 else 1
    return f


# Hash function factory for each hashtable.
# TODO: not sure if this is right hash function to use. Any ideas?
# def hashfuncFactoryH(m, i):
#     def f(arr):
#         minn = float('inf')
#         for item in arr:
#             h = mmh3.hash(item, seed = i) % m
#             if h < minn:
#                 minn = h
#         return minn
#     return f


def MinHash(s, table_index, m):
    hashcodes = []
    for i in range(m):
        minn = float('inf')
        for attr in s:
            hash_value = mmh3.hash(attr, seed = table_index * m + i)
            if hash_value < minn:
                minn = hash_value
        hashcodes.append(minn)
    return hashcodes

# Original (possibly flawed minhash)
# class LSH():
#     def __init__(self, L, K, num_features, num_images):
#         self.surf = cv2.xfeatures2d.SURF_create(300) # SURF
#         self.L = L # number of hashtables
#         self.K = K # number of bits for each hashtable's keys
#         self.num_features = num_features # number of features to extract
#         self.num_images = num_images # total number of images
#         self.bit_array_lst = [Array2D(self.num_images, 2 ** self.K) for _ in range(self.L)] # hashtables
#         self.hash_func_lst = [hashfuncFactoryH(2 ** self.K, i) for i in range(self.L)] # hash functions
#         self.num_features_dict = {}
#         self.count = 0

#     def query(self, key): 
#         counts_dict = defaultdict(int)
#         (kp, des) = self.surf.detectAndCompute(key, None) # extract features
#         if (len(kp) < self.num_features): ## TODO: need reasonable fix for this
#             print("Not enough features: ", len(kp))
#             return {}

#         # For each feature, hash it and probe buckets to find similar images' indices.
#         for i in range(self.num_features): 
#             surf_feature = des[i:i+1][0]
#             for j in range(self.L):
#                 hash_index = self.hash_func_lst[j](surf_feature)
#                 sim_lst = self.bit_array_lst[j].probeBucket(hash_index)
#                 for img_index in sim_lst:
#                     counts_dict[img_index] += 1
#                     # counts_dict[img_index] += 1.0 / self.num_features_dict[img_index]
#         # Rank image indices according to count.
#         sorted_dict = sorted(counts_dict.items(), key = lambda x: x[1])
#         print(sorted_dict)
#         return sorted_dict

#     def insert(self, key, bit_index):
#         (kp, des) = self.surf.detectAndCompute(key, None)
#         self.num_features_dict[self.count] = len(kp)
#         self.count += 1
#         # img2 = cv2.drawKeypoints(key,kp,None,(255,0,0),4)
#         # cv2.imshow('ok', img2)
#         # cv2.waitKey(0)
#         if (len(kp) < self.num_features): ## TODO: need reasonable fix for this
#             print("Not enough features: ", len(kp))
#             return

#         # For each feature, hash it and store in each hashtable.
#         for i in range(self.num_features):
#             surf_feature = des[i:i+1][0]
#             for j in range(self.L):
#                 hash_index = self.hash_func_lst[j](surf_feature)
#                 self.bit_array_lst[j][(bit_index, hash_index)] = True
#         return des


# signed random projection
class LSH():
    def __init__(self, L, K, num_features, num_images, num_hash_funcs):
        self.surf = cv2.xfeatures2d.SURF_create(300) # SURF
        self.L = L # number of hashtables
        self.K = K # number of bits for each hashtable's keys
        self.num_features = num_features # number of features to extract
        self.num_images = num_images # total number of images
        self.num_hash_funcs = num_hash_funcs
        self.bit_array_lst = [Array2D(self.num_images, 2 ** self.K) for _ in range(self.L)] # hashtables
        # list of L lists that each contain 16 hash functions
        self.hash_func_lst = [[hashfuncFactoryH(self.K // self.num_hash_funcs, i, j) for j in range(self.num_hash_funcs)] for i in range(self.L)] # hash functions
        
        self.g_func_lst = [[hashfuncFactoryG(2, i, j) for j in range(self.num_hash_funcs)] for i in range(self.L)] # hash functions
        self.count = 0

    def query(self, key): 
        counts_dict = defaultdict(int)
        (kp, des) = self.surf.detectAndCompute(key, None) # extract features
        if (len(kp) < self.num_features): ## TODO: need reasonable fix for this
            print("Not enough features: ", len(kp))
            return None

        # For each feature, hash it and probe buckets to find similar images' indices.
        for i in range(self.num_features):
            surf_feature = des[i:i+1][0]
            for j in range(self.L):
                B = np.zeros(self.K)
                for attr in surf_feature:
                    for h in range(self.num_hash_funcs):
                        B[self.K // self.num_hash_funcs * h + self.hash_func_lst[j][h](attr)] += attr * self.g_func_lst[j][h](attr)
                b = np.where(B > 0, 1, 0)
                b = BitArray(b)
                hash_index = b.uint
                sim_lst = self.bit_array_lst[j].probeBucket(hash_index)
                for img_index in sim_lst:
                    counts_dict[img_index] += 1
                    # counts_dict[img_index] += 1.0 / self.num_features_dict[img_index]
        # Rank image indices according to count.
        sorted_dict = sorted(counts_dict.items(), key = lambda x: x[1])
        print(sorted_dict)
        return sorted_dict

    def insert(self, key, bit_index):
        (kp, des) = self.surf.detectAndCompute(key, None)
        # img2 = cv2.drawKeypoints(key,kp,None,(255,0,0),4)
        # cv2.imshow('ok', img2)
        # cv2.waitKey(0)
        if (len(kp) < self.num_features): ## TODO: need reasonable fix for this
            print("Not enough features: ", len(kp))
            return None

        # For each feature, hash it and store in each hashtable.
        for i in range(self.num_features):
            surf_feature = des[i:i+1][0]
            for j in range(self.L):
                B = np.zeros(self.K)
                for attr in surf_feature:
                    for h in range(self.num_hash_funcs):
                      B[self.K // self.num_hash_funcs * h + self.hash_func_lst[j][h](attr)] += attr * self.g_func_lst[j][h](attr)
                        # 80 / 16 * [0 ... 15] 
                b = np.where(B > 0, 1, 0)
                b = BitArray(b)
                hash_index = b.uint
                self.bit_array_lst[j][(bit_index, hash_index)] = True
        return des

# Minwise Hashing
# class LSH():
#     def __init__(self, L, K, num_features, num_images, num_hash_funcs):
#         self.surf = cv2.xfeatures2d.SURF_create(300) # SURF
#         self.L = L # number of hashtables
#         self.K = K # number of bits for each hashtable's keys
#         self.num_features = num_features # number of features to extract
#         self.num_images = num_images # total number of images
#         self.num_hash_funcs = num_hash_funcs
#         self.bit_array_lst = [Array2D(self.num_images, 2 ** self.K) for _ in range(self.L)] # hashtables
#         # list of L lists that each contain 16 hash functions

#     def query(self, key): 
#         counts_dict = defaultdict(int)
#         (kp, des) = self.surf.detectAndCompute(key, None) # extract features
#         if (len(kp) < self.num_features): ## TODO: need reasonable fix for this
#             print("Not enough features: ", len(kp))
#             return {}
#         # For each feature, hash it and probe buckets to find similar images' indices.
#         for i in range(self.num_features):
#             surf_feature = des[i:i+1][0]
#             for j in range(self.L):
#                 hashcodes = MinHash(surf_feature, j, self.num_hash_funcs)
#                 # hash_index = self.hash_func_lst[table_index](hashcodes)
#                 hash_index = hash(tuple(hashcodes)) % (2 ** self.K)
#                 sim_lst = self.bit_array_lst[j].probeBucket(hash_index)
#                 for img_index in sim_lst:
#                     counts_dict[img_index] += 1
#                     # counts_dict[img_index] += 1.0 / self.num_features_dict[img_index]
#         # Rank image indices according to count.
#         sorted_dict = sorted(counts_dict.items(), key = lambda x: x[1])
#         print(sorted_dict)
#         return sorted_dict

#     def insert(self, key, bit_index):
#         (kp, des) = self.surf.detectAndCompute(key, None)
#         # img2 = cv2.drawKeypoints(key,kp,None,(255,0,0),4)
#         # cv2.imshow('ok', img2)
#         # cv2.waitKey(0)
#         if (len(kp) < self.num_features): ## TODO: need reasonable fix for this
#             print("Not enough features: ", len(kp))
#             return {}

#         # For each feature, hash it and store in each hashtable.
#         for i in range(self.num_features):
#             surf_feature = des[i:i+1][0]
#             for j in range(self.L):
#                 hashcodes = MinHash(surf_feature, j, self.num_hash_funcs)
#                 # hash_index = self.hash_func_lst[table_index](hashcodes)
#                 hash_index = hash(tuple(hashcodes)) % (2 ** self.K)
#                 self.bit_array_lst[j][(bit_index, hash_index)] = True
#         return des

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

# def group(filenames):
#     # Return list of lists where each list is an "album"
#     return [["1", "2", "3"], ["4", "5", "6"], ["7", "8"]]

def preprocessing2():
    filenames = []
    for i in range(1, 10):
        filenames.extend(glob.glob('uploads/training/' + str(i) + '/*.jpg'))
    print(len(filenames))
    lsh = LSH(14, 12, 487, len(filenames), 4)
    for i, filename in enumerate(filenames):
        print(i, filename)
        img = cv2.imread(filename)
        lsh.insert(img, i)

    # num_images = len(glob.glob('uploads/google_photos/*.jpg'))
    # lsh = LSH(10, 18, 300, num_images, 6) # TODO: find a good K and L
    # for i, filename in enumerate(glob.glob('uploads/google_photos/*.jpg')):
    #     print(i, filename)
    #     filenames.append(filename)
    #     img = cv2.imread(filename)
    #     lsh.insert(img, i)
    #lsh.printArray()

    print("-----Querying-----")
    # Query a single image for any similar images.
    query_image = cv2.imread("uploads/training/1/2_1.jpg")
    g = lsh.query(query_image)
    for item in g:
        print(filenames[item[0]])

def bruteforce():
    lsh = LSH(10,12,285,3)


    query_image = cv2.imread("uploads/google_photos/17.jpg")
    des1 = lsh.insert(query_image, 0)
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    # with open("17.txt", 'w') as f:
    #     f.write(np.array2string(des1, separator=', '))
    # query_image2 = cv2.imread("uploads/google_photos/18.jpg")

    # h1 = imagehash.average_hash(Image.open('uploads/google_photos/17.jpg'))
    # h2 = imagehash.average_hash(Image.open('uploads/google_photos/18.jpg'))
    # h3 = imagehash.average_hash(Image.open('uploads/google_photos/14.jpg'))

    # print(h1 - h2)
    # print(h1 - h3)
    # des2 = lsh.insert(query_image2, 1)
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    # with open("18.txt", 'w') as f:
    #     f.write(np.array2string(des2, separator=', '))    
    # query_image3 = cv2.imread("uploads/google_photos/14.jpg")    
    # des3 = lsh.insert(query_image3, 3)
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
    # with open("14.txt", 'w') as f:
    #     f.write(np.array2string(des3, separator=', '))
    # print(des1.shape)
    # print(des2.shape)
    # print(des3.shape)



    # m = 1 - scipy.spatial.distance.cdist(des1[0:200], des1[0:200], 'cosine').reshape(-1)
    # print(np.mean(m))
    filenames = []
    num_images = len(glob.glob('uploads/google_photos/*.jpg'))
    s = cv2.xfeatures2d.SURF_create(400) # SURF
    for i, filename in enumerate(glob.glob('uploads/google_photos/*.jpg')):
        print(i, filename)
        filenames.append(filename)
        img = cv2.imread(filename)
        (kp, des2) = s.detectAndCompute(img, None)
        dist_matrix = euclidean_distances(des2, des1)
        threshold = 0.01
        count = 0
        for row in dist_matrix:
            for elem in row:
                if elem > threshold:
                    count += 1
        print (count)


        # des2 = des2.reshape(1,-1)


if __name__ == "__main__":
    preprocessing2()