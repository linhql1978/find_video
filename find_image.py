import os
import cv2
import numpy as np
from numpy import linalg as LA
import glob
from scipy.spatial import distance


def hog(img_gray, cell_size=8, block_size=2, bins=9):
    img = img_gray
    h, w = img.shape  # 128, 64

    # print('h: ',h,',w: ',w)

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
                # print('x: ',bx,', by: ',by)
                print('v\n',v)

    return feature_tensor.flatten()  # 3780 features

dim=64
img_test=cv2.imread('../demo2/images_test/keyboard/keyboard1.jpeg',cv2.IMREAD_GRAYSCALE)
img_test=cv2.resize(src=img_test,dsize=(dim,dim*2)) #dsize=(w,h)
vector_img_test=hog(img_test)

list_videos=[]
list_distance_videos=[]

dir_videos=glob.glob('../demo2/frames_dir/*.mp4')
for dir_video in dir_videos:
    dir_video_basename=os.path.basename(dir_video)
    # frames=glob.glob('../demo1/test_frames/'+dir_video_basename+'/*.jpg')
    frames=glob.glob('../demo2/frames_dir/'+dir_video_basename+'/*.jpg')
    list_distance=[]
    for frame in frames:
        img=cv2.imread(frame,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(src=img,dsize=(dim,dim*2))
        vector_img=hog(img)
        list_distance.append(distance.euclidean(vector_img,vector_img_test))

    list_videos.append(dir_video_basename)
    list_distance_videos.append(min(list_distance))

print('list_videos: ',list_videos)
print('list_distance_videos: ',list_distance_videos)

index=0
for i,dis in enumerate(list_distance_videos):
    if(dis==min(list_distance_videos)):
        index=i
        break;

print('video: ',list_videos[index])