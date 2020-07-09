import cv2
import glob
import os
import numpy as np

listVideos=glob.glob('videos/*.mp4') # Đọc các video trong folder videos
frames_dir= 'frames_dir' # Khai báo tên folder để chứa các frames
for video in listVideos:
    vidcap=cv2.VideoCapture(video) # Xử lý video
    count=0
    success=True
    video_baseName=os.path.basename(video)
    os.makedirs(os.path.join(frames_dir, video_baseName), exist_ok=True) # Tạo folder chứa các frames cho mỗi video
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) # Thiết lập đọc frame cho mỗi giây
        success,image=vidcap.read()

        save_path=os.path.join(frames_dir, video_baseName, 'frame%d.jpg' % count) # Tạo được dẫn lưu frame

        image_last=cv2.imread(save_path+'frame{}.jpg'.format(count-1)) # Lấy ra frame được tách gần nhất trong video
        if np.array_equal(image,image_last): # So sánh nếu 2 frame hiện tại và frame gần nhất giống nhau thì thoát vòng lặp
            break

        cv2.imwrite(save_path,image) # Lưu frame vào đường dẫn đã được tạo

        print('{}.sec reading a new frame: {}'.format(count,True))
        count+=1

    vidcap.release() # Giải phòng video
    print('Extract Success video: ', video_baseName)