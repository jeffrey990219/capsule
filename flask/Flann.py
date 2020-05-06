import cv2
from glob import glob
import numpy as np
import heapq
from collections import defaultdict
import time


class Capsule():
    def __init__(self):
        FLANN_INDEX_LSH = 6
        index_params_LSH = dict(algorithm = FLANN_INDEX_LSH, 
                            table_number = 6,      # 12
                            key_size = 12,          # 20
                            multi_probe_level = 1)  #2
        search_params = {}  # dict(checks=5)

        self.EXTRACTOR = cv2.ORB_create(512)
        # FLANN Matcher: Used to compute KNN
        self.FLANN = cv2.FlannBasedMatcher(index_params_LSH,search_params)
        # Brute-force Matcher: Used to calculate exact similarity
        self.BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.IMG_DATABASE = [] # Stores the img urls

    def insert(self, new_img_url):
        img = cv2.imread(new_img_url)
        (kp, des) = self.EXTRACTOR.detectAndCompute(img, None)
        if des is not None:
            des = np.uint8(des)
            # Add into FLANN's descriptor set
            self.FLANN.add([des])
            # Store the img_url
            self.IMG_DATABASE.append(new_img_url)

    def query_similar_imgs(self, query_img_url):
        query_img = cv2.imread(query_img_url)
        (kp, q_des) = self.EXTRACTOR.detectAndCompute(query_img, None)
        # Look for k most similar features in FLANN's descriptor set per feature in 'q_des'
        knn_matches = self.FLANN.knnMatch(q_des, k=10) 
        # Count the number of occurrences of each image whose feature is one of the k most similar features to the query features
        frequency_count = defaultdict(int)
        for match_vector in knn_matches:
            # Uncomment this to print the detailed info in 'knn_matches'
            # print([(self.IMG_DATABASE[x.imgIdx], x.distance) for x in match_vector])
            for m in match_vector:
                frequency_count[m.imgIdx] += 1
        min_dist_heap = []
        # Get top-10 images in all candidate images
        for (k, v) in frequency_count.items():
            heapq.heappush(min_dist_heap, (v, k))
        
        top_10 = heapq.nlargest(30, min_dist_heap)
        # Compute the exact similartiy of each of the top-10 images and keep only those with score < THRESHOLD_SIM
        THRESHOLD_SIM = 10
        results = []
        for top_img_url in [self.IMG_DATABASE[ele[1]] for ele in top_10]:
            top_img = cv2.imread(top_img_url)
            (kp, des) = self.EXTRACTOR.detectAndCompute(top_img, None)
            matches = self.BF.match(q_des, des)
            top_10_matches = sorted(matches, key = lambda x: x.distance)[:10]
            if sum([m.distance for m in top_10_matches]) / len(top_10_matches) < THRESHOLD_SIM:
                results.append(top_img_url)
        return results

CAPSULE = Capsule()
def preprocessing(google_photo_urls):
    for url in google_photo_urls:
        CAPSULE.insert(url)

def group(query_img_urls):
    # Return list of lists where each list is an "album"
    print(len(query_img_urls))
    results = []
    for q_url in query_img_urls:
        print("QUERY: %s" %q_url)
        start_time = time.time()
        results.append(CAPSULE.query_similar_imgs(q_url))
        end_time = time.time()
        print("Finished in %.2f seconds" %((end_time - start_time)))
    return results


# if __name__ == "__main__":
#     CAPSULE = Capsule()
#     print(cv2.__version__)
#     training_img_urls = sorted(glob('./uploads/google_photos/*.jpg'))
#     num_all_imgs = len(training_img_urls)
#     inserted = 0
#     for url in training_img_urls:
#         CAPSULE.insert(url)
#         inserted += 1
#         print("%d / %d INSERTED" %(inserted, num_all_imgs))

#     query_img_urls = sorted(glob('./uploads/google_photos/*test*'))
#     for q_url in query_img_urls:
#         print("QUERY: %s" %q_url)
#         start_time = time.time()
#         CAPSULE.query_similar_imgs(q_url)
#         end_time = time.time()
#         print("Finished in %.2f seconds" %((end_time - start_time) / 10**9))
