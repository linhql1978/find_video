import os
import cv2
import numpy as np
from numpy import linalg as LA
import glob
from sklearn.cluster import KMeans
from scipy.spatial import distance

def hog(img_gray, cell_size=8, block_size=2, bins=9):
    img = img_gray
    h, w = img.shape  # 128, 64

    # gradient
    xkernel = np.array([[-1, 0, 1]])
    ykernel = np.array([[-1], [0], [1]])
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel)

    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan(np.divide(dy, dx + 0.00001))  # radian
    orientation = np.degrees(orientation)  # -90 -> 90
    orientation += 90  # 0 -> 180

    num_cell_x = w // cell_size  # 8
    num_cell_y = h // cell_size  # 16
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins])  # 16 x 8 x 9
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            mag = magnitude[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag)  # 1-D vector, 9 elements
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

    return feature_tensor.flatten()  # 3780 features

dim=64
frames=glob.glob('../demo2/frames_dir/cycle.mp4/*.jpg')
X=[]
for frame in frames:
    img=cv2.imread(frame,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(src=img,dsize=(dim,dim*2))
    vector_img=hog(img)
    X.append(vector_img)

kmeans=KMeans(n_clusters=7,random_state=0).fit(X)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(X)
# print('pred_label: ',pred_label)

X_test=[]
img_test=cv2.imread('../demo2/images_test/cycle/bycycle.jpeg',cv2.IMREAD_GRAYSCALE)
img_test=cv2.resize(src=img_test,dsize=(dim,dim*2))
vector_img_test=hog(img_test)
X_test.append(vector_img_test)
pred_label_test=kmeans.predict(X_test)
print('pred_label_test: ',pred_label_test[0])

# print(X[pred_label==pred_label_test[0]])
labels=(pred_label==pred_label_test[0])
print('labels: ',labels)

list_dist=[]
# Xk=X[pred_label == pred_label_test[0],:]
for i,vector_img in enumerate(X):
    if(labels[i]==True):
        print('i: ',i)
        list_dist.append(distance.euclidean(vector_img_test,vector_img))
#
print('min: ',min(list_dist))