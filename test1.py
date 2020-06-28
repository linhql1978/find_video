import os
import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from numpy import linalg as LA
import glob
from sklearn.cluster import KMeans # thư viện kmeans
from scipy.spatial import distance # thư viện dùng để tính L-norm

def hog(img_gray, cell_size=8, block_size=2, bins=9):
    img = img_gray
    h, w = img.shape  # 128, 64

    # gradient
    xkernel = np.array([[-1, 0, 1]]) # kernel dùng để tính đạo hàm theo ox
    ykernel = np.array([[1], [0], [-1]]) # kernel dùng để tính đạo hàm theo oy
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel) # đạo hàm của ảnh theo ox
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel) # đạo hàm của ảnh theo oy

    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy)) # tính biên độ của pixel
    orientation = np.arctan(np.divide(dy, dx + 0.00001))  # radian, dùng 0.00001 tránh 0/0
    orientation = np.degrees(orientation)  # -90 -> 90
    orientation += 90  # 0 -> 180

    num_cell_x = w // cell_size  # 8 cell
    num_cell_y = h // cell_size  # 16 cell
    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins])  # 16 x 8 x 9
    for cx in range(num_cell_x): # 0 - 7
        for cy in range(num_cell_y): # 0 - 15
            ori = orientation[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            mag = magnitude[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag)  # 1-D vector, 9 elements, dùng để thống kê hist
            hist_tensor[cy, cx, :] = hist # lưu hist của mỗi cell
        pass
    pass

    # normalization
    redundant_cell = block_size - 1 # do kích thước block = 2
    feature_tensor = np.zeros(
        [num_cell_y - redundant_cell, num_cell_x - redundant_cell, block_size * block_size * bins])
    for bx in range(num_cell_x - redundant_cell):  # 7
        for by in range(num_cell_y - redundant_cell):  # 15
            by_from = by
            by_to = by + block_size
            bx_from = bx
            bx_to = bx + block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten()  # to 1-D array (vector), vector này chứa 4*9=36 phần tử
            feature_tensor[by, bx, :] = v / LA.norm(v, 2) # chuẩn hóa lại v
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any():  # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v

    return feature_tensor.flatten()  # 3780 features trích rút theo hog

dim=64 # số chiều = 64
list_kmeans=[] # dùng để chứa các đối tượng đã phân cụm cho từng video
list_preds=[] # chứa các nhãn của cụm của video
list_X=[] # chứa các đặc đựng của cụm của video
list_video_name=[] # chứa tên video
videos_dir=glob.glob('../demo2/frames_dir/*.mp4') # đọc các thư mục video chứa các frame
for video_dir in videos_dir: # lặp từng thư mục
    video_basename=os.path.basename(video_dir)
    frames=glob.glob('../demo2/frames_dir/'+video_basename+'/*.jpg') # lặp từng frame trong video
    X=[] # dùng để chứa các đặc trưng của frame của video
    for frame in frames: # lặp từng frame
        img=cv2.imread(frame,cv2.IMREAD_GRAYSCALE) # đọc ảnh xám của frame
        img=cv2.resize(src=img,dsize=(dim,dim*2)) # convert frame về 64*128 (w*h)
        vector_img=hog(img) # nhận vector đặc trưng của frame
        X.append(vector_img) # thêm vào X
    kmeans=KMeans(n_clusters=5,random_state=0).fit(X) # tiến hành phân cụm với k=5
    list_X.append(X) # lưu trữ X
    list_kmeans.append(kmeans) # lưu đối tượng đã phân cụm của video
    list_preds.append(kmeans.predict(X)) # lưu các nhãn của cụm của video
    list_video_name.append(video_basename) # lưu basename của video
    print('kmeans success: ',video_basename)

list_videos=[] # basename của các video
list_distance_videos=[] # khoảng cách nhỏ nhất của input tới frame gần nhất của video tính theo L-norm 2

img_test=cv2.imread('images_test/chair/chair.jpeg', cv2.IMREAD_GRAYSCALE) # đọc input theo ảnh xám
img_test=cv2.resize(src=img_test,dsize=(dim,dim*2)) # convert input
vector_img_test=hog(img_test) # lấy vector đặc trưng của input
X_test=[] # dùng để tiên đoán
X_test.append(vector_img_test) # thêm input cần tiên đoán

for i,kmeans in enumerate(list_kmeans): # lặp theo index và kmeans
    video_basename=os.path.basename(list_video_name[i])
    pred_label_test=kmeans.predict(X_test) # tiên đoán X_test
    list_distance=[] # lưu trữ các khoảng cách của input tới các frame trong cụm được tiên đoán
    for j,value in enumerate(list_preds[i]==pred_label_test[0]): # lặp các frame
        if(value==True): # kiểm tra frame có thuộc cụm đã tiên đoán cho input hay không?
            list_distance.append(distance.euclidean(list_X[i][j],X_test[0])) # nếu đúng thì tính L-norm rồi lưu vào list_distance
    list_videos.append(video_basename) # lưu basename của video
    list_distance_videos.append(min(list_distance)) # lưu khoảng cách nhỏ nhất của input tới frame nằm trong cụm đã tiên đoán của video

print('list_videos: ',list_videos) # danh sách video
print('list_distance_videos: ',list_distance_videos) # danh sách khoảng cách nhỏ nhất của input tới từng video

index=None
for i,dis in enumerate(list_distance_videos): # lặp khoảng cách nhỏ nhất của input tới từng video
    if(dis==min(list_distance_videos)): # nếu khoảng cách nhỏ nhất của input tới video là khoảng cách nhỏ nhất của input tới tất cả các video
        index=i # lấy ra index
        break

print('video: ',list_videos[index]) # in video chứa vật thể