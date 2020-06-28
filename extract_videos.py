import cv2
import glob
import os
import numpy as np

listVideos=glob.glob('../demo2/videos/*.mp4')
frames_dir= 'frames_dir'
for video in listVideos:
    print('video: ',video)
    vidcap=cv2.VideoCapture(video)
    count=0
    success=True
    video_baseName=os.path.basename(video)
    os.makedirs(os.path.join(frames_dir, video_baseName), exist_ok=True)
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        success,image=vidcap.read()

        save_path=os.path.join(frames_dir, video_baseName, 'frame%d.jpg' % count)

        image_last=cv2.imread(save_path+'frame{}.jpg'.format(count-1))
        if np.array_equal(image,image_last):
            break

        cv2.imwrite(save_path,image)

        print('{}.sec reading a new frame: {}'.format(count,True))
        count+=1

    vidcap.release()
    print('Extract Success video: ', video_baseName)