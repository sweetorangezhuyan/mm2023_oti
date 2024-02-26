'''
获取k400的帧图片
'''
import os
import sys
from multiprocessing import Pool
import cv2
path='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/zhuyan29/hmdb51_org'
# now_frame_action=[i for i in sorted(os.listdir('/opt/kinetics400_frames'))]
savepath='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/zhuyan29/hmdb51_frame_org'
def uniform_extract_frames(video_path,action_name,save_name):
    c = 1
    cap = cv2.VideoCapture(video_path)
#     print(cap.isOpened())
    frames_num = cap.get(7)
#     print(frames_num)
    save_path=os.path.join(savepath,action_name)
    save_path=os.path.join(save_path,save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
#     print(save_path)
    timeF = 1
    if frames_num < 64:
        timeF = 1
    else:
        timeF = int(frames_num / 64.)

    if cap.isOpened():
        rval, frame = cap.read()
    else:
        rval = False

    while rval:
        rval, frame = cap.read()
        if frame is not None:
            if (c % timeF == 0):
                cv2.imwrite(save_path + '/' + str(c).zfill(5) + '.jpg', frame)
        c = c + 1
        #cv2.waitKey(1)
    cap.release()

if __name__ == "__main__":

    p_names = sorted(os.listdir(path))
    print(len(p_names))
#     print(now_frame_action)
    for p_name in p_names:
#         if p_name not in now_frame_action:
        print(p_name)
        if p_name!='.DS_Store':
            camera_root_path = os.path.join(path, p_name)
            camera_names = sorted(os.listdir(camera_root_path))
            count=0
            for camera_name in camera_names[:len(camera_names)]:
                count+=1
    #             print(camera_name)
    #             print(count)
                video_root_path = os.path.join(camera_root_path, camera_name)
                save_name=camera_name.split('.')[0]
                uniform_extract_frames(video_root_path,p_name,save_name)
