'''
对kinetic 400的训练数据装载方式
'''
import torch
import torch.utils.data as data
# import decord
import os
import numpy as np
from numpy.random import randint
import io
# import pandas as pd
import random
from PIL import Image
import math
import copy


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[-1])

def getallvideos(path,num_segments,seg_length):
    tmp = [x.strip().split(' ') for x in open(path)]
#     print(tmp[0])
    video_list=[]
    for item in tmp:
        if int(item[1]) > num_segments*seg_length:
            video_list.append(VideoRecord(item))
    print('video number:%d' % (len(video_list)))
    return video_list
class Video_dataset(data.Dataset):
    def __init__(self, root_path, list_file, labels_file,
                 num_segments=1, modality='RGB', new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 index_bias=1, dense_sample=False, test_clips=3):

        self.root_path = root_path  ## /opt/
        self.list_file = list_file  ## k400_train.txt
        self.num_segments = num_segments  ## 8
        self.modality = modality  ##RGB
        self.seg_length = new_length  ##  片段长度
        self.image_tmpl = image_tmpl  ##'{:05d}.jpg'
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.sample_range = 128
        self.video_list = getallvideos(self.list_file,self.num_segments,self.seg_length)
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.test_clips = test_clips
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')

        if self.index_bias is None:
            if self.image_tmpl == "{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
#         self._parse_list()
        self.initialized = False

    @property
    #@property装饰器会将方法转换为相同名称的只读属性,可以与所定义的属性配合使用，这样可以防止属性被修改
    def total_length(self):
        return self.num_segments * self.seg_length

    @property
    def classes(self):
        '''
        output: class name
        '''
        classes_all = [i.strip().split(',')[1] for i in open(self.labels_file).readlines()]
        return classes_all

    def sample_indices1(self, video_list):
        sample_set = [int(i.split('.')[0]) for i in video_list]
        sample_set = np.array(sorted(sample_set))
        # random sample 
        indexs = sorted(np.random.choice(len(sample_set), self.num_segments*self.seg_length, replace=False))

        # uniform sample
        # inter=len(sample_set)//self.num_segments
        # # indexs=np.arange(len(sample_set))[::inter][:self.num_segments]
        # indexs = sorted(np.random.randint(0, len(sample_set), (self.num_segments * self.seg_length,)))

        sample_set = sample_set[indexs]
        return sample_set


    def __getitem__(self, index):
        record = self.video_list[index]  # 一条视频的记录
#         path,frames_num,label=record[0],record[1],record[2]
        video_list = os.listdir(os.path.join(self.root_path, record.path))
        
        segment_indices = self.sample_indices1(video_list)
        return self.get(record, segment_indices)

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            seg_imgs = self._load_image(record.path, seg_ind)
            images.extend(seg_imgs)
        process_data, record_label = self.transform((images, record.label))## process_data (frames*3)*224*224
        return process_data, record_label

    def __len__(self):
        return len(self.video_list)
