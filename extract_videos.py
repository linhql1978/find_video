import cv2
import glob
import os
import numpy as np

listVideos=glob.glob('../demo2/videos/*.mp4') # đọc các video trong thư mục videos
frames_dir= 'frames_dir' # tạo thư mục lưu trữ các video chứa các frames
for video in listVideos: # lặp từng video
    print('video: ',video)
    vidcap=cv2.VideoCapture(video)
    count=0
    success=True
    video_baseName=os.path.basename(video) # lấy basename của video
    os.makedirs(os.path.join(frames_dir, video_baseName), exist_ok=True) # tạo thư mục video chứa các frames
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000)) # bắt frame từng phút
        success,image=vidcap.read() # hàm này trả về bool và frame bắt được

        save_path=os.path.join(frames_dir, video_baseName, 'frame%d.jpg' % count) # tạo đường dẫn để lưu frame

        image_last=cv2.imread(save_path+'frame{}.jpg'.format(count-1)) # lấy ra frame cuối cùng được lưu
        if np.array_equal(image,image_last): # kiểm tra nếu 2 frame là 1 thì thoát vòng lặp
            break

        cv2.imwrite(save_path,image) # lưu frame và đường dẫn đã tạo

        print('{}.sec reading a new frame: {}'.format(count,True))
        count+=1

    vidcap.release() # giải phóng video
    print('Extract Success video: ', video_baseName)