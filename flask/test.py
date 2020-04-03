
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

B = [0] * 20

v = [-3.27791134e-03, -3.41947866e-03,  4.70909802e-03,  3.81646026e-03,  2.94525363e-02, -1.43781630e-02,  5.30112386e-02,  3.85769978e-02, -5.72701590e-03, -3.66003951e-03,  5.38849831e-02,  3.32627445e-02,  1.85788341e-03, -5.21973358e-04,  2.13911268e-03,  2.13938695e-03,  2.54409593e-02,  1.98501181e-02,  4.03837264e-02,  2.90356502e-02,  2.00652972e-01,  1.96015149e-01,  3.62618119e-01,  2.48579696e-01, -1.58818975e-01,  9.58998650e-02,  3.71981829e-01,  2.57111520e-01,  1.38525823e-02,  1.29556758e-02,  2.12249849e-02,  1.55392317e-02,  1.78294405e-02, -7.89302960e-03,  3.47821340e-02,  3.47523727e-02,  1.93461567e-01, -8.50505829e-02,  3.54335755e-01,  2.27079734e-01, -1.17654391e-01, -1.74504101e-01,  3.73727858e-01,  2.39808947e-01,  7.44890468e-03, -4.05941019e-03,  1.52499368e-02,  7.17612961e-03, -2.22730776e-03, -8.89271905e-04,  7.59484107e-03,  4.51709842e-03,  8.14298727e-03,  4.77145333e-03,  5.28440177e-02,  2.95667779e-02, -2.87444075e-03,  1.90887880e-02,  5.50547875e-02,  3.75872292e-02,  1.16998970e-03, -7.38819144e-05,  1.24412973e-03,  4.15586081e-04]

def hashfunc(key, m, i):
	return mmh3.hash(np.float32(key), seed = i) % m;

def g(key, m, i):
	return -1 if mmh3.hash(np.float32(key), seed = i + 10) % 2 == 0 else 1

for i in range(64):
	for j in range(4):
		B[5 * j + hashfunc(v[i], 5, j)] += v[i] * g(i, 2, j)

print (B)