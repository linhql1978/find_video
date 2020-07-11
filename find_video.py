import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist

np.seterr(divide='ignore', invalid='ignore')
from numpy import linalg as LA
import glob
from scipy.spatial import distance

# Hàm trích xuất đặc trưng sử dụng Histograms of Oriented Gradients
def hog(img_gray, cell_size=8, block_size=2, bins=9):
    img = img_gray
    h, w = img.shape  # 128, 64

    # Tính đạo hàm image theo ox và oy
    xkernel = np.array([[-1, 0, 1]])
    ykernel = np.array([[-1], [0], [1]]) #
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel)

    # Tính biên độ và hướng của image dựa trên đạo hàm
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

    # Chuẩn hóa và trả về vector đặc trưng của image
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

# Khởi tạo ngẫu nhiên các centers
def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

# Gán nhãn
def kmeans_assign_labels(X, centers):
    D = cdist(X, centers)
    return np.argmin(D, axis = 1)

# Cập nhật lại các centers
def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster
        Xk = X[labels == k, :]
        # take average
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers

# Kiểm tra điều kiện dừng thuật toán Kmeans
def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) ==
        set([tuple(a) for a in new_centers]))

# Dự đoán input sẽ thuộc cụm nào
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

# Hàm Kmeans
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
K=7 # Giá trị k để phân cụm
list_videos=[] # Danh sách lưu tên các videos
list_distance_videos=[] # Danh sách lưu khoảng cách ngắn nhất của input cho mỗi video

img_test=cv2.imread('images_test/cycle/bycycle.jpeg',cv2.IMREAD_GRAYSCALE) # Đọc input ở chế độ ảnh xám
img_test=cv2.resize(src=img_test,dsize=(dim,dim*2)) # Resize ảnh về (64,128) <=> (w,h)
vector_img_test=hog(img_test) # Sử dụng HOG trích xuất đắc trưng của input

videos_dir=glob.glob('frames_dir/*.mp4') # Lấy danh sách các folder chứa các frame của các video
for video_dir in videos_dir:
    video_basename=os.path.basename(video_dir)
    X=[] # Danh sách chứa các đặc trưng cho images trong video
    list_distance=[]
    frames=glob.glob('frames_dir/'+video_basename+'/*.jpg') # Lấy danh sách các frame của video
    for frame in frames:
        img=cv2.imread(frame,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(src=img,dsize=(dim,dim*2))
        vector_img=hog(img) # Trích xuất đặc trưng hình dạnh sử dụng HOG
        X.append(vector_img) # Lưu vector đặc trưng của image để phân cụm và tính toán
        pass
    centers,labels,_=kmeans(np.array(X),K) # Sử dụng Kmeans để phân cụm các frame của video với số cụm bằng K
    center_index=predict(centers,vector_img_test) # Dự đoán input thuộc cụm nào
    for i,feature_img in enumerate(X): # Duyệt tất cả các vector đặc trưng có trong X
        if(labels[i]==center_index): # Kiểm tra xem vector đặc trưng có thuộc cụm đã tiên đoán cho input hay không?
            list_distance.append(distance.euclidean(feature_img,vector_img_test)) # Tính toán bằng L2-norm và lưu lại
            pass
        pass
    list_distance_videos.append(min(list_distance)) # Tìm ra khoảng cách ngắn nhất của input tới frame trong video
    list_videos.append(video_basename) # Lưu lại tên video
    pass
print('list_videos: ',list_videos) # In ra tên các video
print('list_distance_videos: ',list_distance_videos) # In ra khoảng cách tương ứng
dis_min=min(list_distance_videos) # Lấy khoảng cách nhỏ nhất của input tới frame trong các video
for i in range(len(list_distance_videos)):
    if(dis_min==list_distance_videos[i]):
        print('result video: ',list_videos[i]) # In ra video
        break
        pass
    pass
