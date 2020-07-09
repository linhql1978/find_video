import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist

np.seterr(divide='ignore', invalid='ignore')
from numpy import linalg as LA
import glob
from sklearn.cluster import KMeans
from scipy.spatial import distance

#HOG
def hog(img_gray, cell_size=8, block_size=2, bins=9):
    img = img_gray
    h, w = img.shape  # 128, 64

    # gradient
    xkernel = np.array([[-1, 0, 1]])
    ykernel = np.array([[-1], [0], [1]]) #
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel)

    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan(np.divide(dy, dx + 0.00001))  # radian
    orientation = np.degrees(orientation)  # -90 -> 90
    orientation += 90  # 0 -> 180

    num_cell_x = w // cell_size  # 8 cell
    num_cell_y = h // cell_size  # 16 cell
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins])  # 16 x 8 x 9
    for cx in range(num_cell_x): # 0 - 7
        for cy in range(num_cell_y): # 0 - 15
            ori = orientation[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            mag = magnitude[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag)  # 1-D vector
            hist_tensor[cy, cx, :] = hist
        pass
    pass

    # normalization
    redundant_cell = block_size - 1
    feature_tensor = np.zeros(
        [num_cell_y - redundant_cell, num_cell_x - redundant_cell, block_size * block_size * bins])
    for bx in range(num_cell_x - redundant_cell):  # 7
        for by in range(num_cell_y - redundant_cell):  # 15
            by_from = by
            by_to = by + block_size
            bx_from = bx
            bx_to = bx + block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten()  # to 1-D array (vector)
            feature_tensor[by, bx, :] = v / LA.norm(v, 2)
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any():  # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v

    return feature_tensor.flatten()

# Kmeans
def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) ==
        set([tuple(a) for a in new_centers]))

def predict(centers,input):
    distance_min=distance.euclidean(centers[0],input)
    index=0
    for i in range(1,len(centers)):
        distance_i=distance.euclidean(centers[i],input)
        if(distance_i<distance_min):
            distance_min=distance_i
            index=i
            pass
        pass
    return index

def kmeans(X, K):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
        pass
    return (centers[-1], labels[-1], it)

dim=64
K=7
list_videos=[]
list_distance_videos=[]

img_test=cv2.imread('images_test/cycle/bycycle.jpeg',cv2.IMREAD_GRAYSCALE)
img_test=cv2.resize(src=img_test,dsize=(dim,dim*2))
vector_img_test=hog(img_test)

videos_dir=glob.glob('../demo2/frames_dir/*.mp4')
for video_dir in videos_dir:
    video_basename=os.path.basename(video_dir)
    frames=glob.glob('../demo2/frames_dir/'+video_basename+'/*.jpg')
    X=[]
    list_distance=[]
    for frame in frames:
        img=cv2.imread(frame,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(src=img,dsize=(dim,dim*2))
        vector_img=hog(img)
        X.append(vector_img)
        pass
    centers,labels,_=kmeans(np.array(X),K)
    center_index=predict(centers,vector_img_test)
    for i,feature_img in enumerate(X):
        if(labels[i]==center_index):
            list_distance.append(distance.euclidean(feature_img,vector_img_test))
            pass
        pass
    list_distance_videos.append(min(list_distance))
    list_videos.append(video_basename)
    pass
print('list_videos: ',list_videos)
print('list_distance_videos: ',list_distance_videos)
dis_min=min(list_distance_videos)
for i in range(len(list_distance_videos)):
    if(dis_min==list_distance_videos[i]):
        print('result video: ',list_videos[i])
        pass
    pass
