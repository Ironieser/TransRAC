import numpy as np
from PIL import Image
from torchvision import transforms
import os
import cv2
import torch
import pandas as pd
import math
from label2rep import rep_label
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# cv2.setNumThreads(0)
import time


class MyDataset(Dataset):

    def __init__(self, root_dir, label_dir, frames, method):
        """
        :param root_dir: 数据集根目录
        :param label_dir: 数据集子目录
         # data_root = r'/data/train'
         # data_lable = data_root + 'label.xlsx'
         # data_video = data_root + 'video/'
        """
        super().__init__()
        self.root_dir = root_dir
        if method == 'train':
            self.video_dir = os.path.join(self.root_dir, r'train')
        elif method == 'valid':
            self.video_dir = os.path.join(self.root_dir, r'valid')
        elif method == 'test':
            self.video_dir = os.path.join(self.root_dir, r'test')
        else:
            raise ValueError('module is wrong.')
        # self.path = os.path.join(self.root_dir, self.child_dir)
        self.video_filename = os.listdir(self.video_dir)
        self.label_filename = os.path.join(self.root_dir, label_dir)
        self.file_list = []
        self.label_list = []
        self.num_idx = 4
        self.num_frames = frames  # model frames
        self.num_period = 64
        self.error = 0
        self.image_size = 224
        df = pd.read_csv(self.label_filename)
        for i in range(0, len(df)):
            filename = df.loc[i, 'name']
            label_tmp = df.values[i][self.num_idx:].astype(np.float64)
            label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)
            # if self.isdata(filename, label_tmp):  # data is right format ,save in list
            self.file_list.append(filename)
            self.label_list.append(label_tmp)

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            video_imgs: loaded video to imgs    :FloatTensor    [1,64,3,224,224]
            y1: period length  0 - max period   :LongTensor     [1,64,64]
            y2: within period  0 - 1            :LongTensor     [1,64]
            num_period : count by processing labels:(y2/y1)[~np.isnan(y2/y1).bool()].sum() : ndarray [1]
            ps: if count !=  num_period : return num_period
        """

        filename = self.file_list[index]
        video_path = os.path.join(self.video_dir, filename)
        video_imgs, original_frames_length = self.read_video(video_path)
        y1, y2, num_period = self.adjust_label(self.label_list[index], original_frames_length, self.num_period)
        return video_imgs, y1, y2, num_period

    def __len__(self):
        """返回数据集的大小"""
        return len(self.file_list)

    def read_video(self, video_filename, width=224, height=224):
        """Read video from file."""
        # print('-------------------------------------')
        # print(video_filename, 'to open')

        try:
            cap = cv2.VideoCapture(video_filename)
            frames = []
            if cap.isOpened():
                while True:
                    success, frame_bgr = cap.read()
                    if success is False:
                        break
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_rgb = cv2.resize(frame_rgb, (width, height))
                    frames.append(frame_rgb)
            cap.release()
            original_frames_length = len(frames)

            frames = self.adjust_frames(frames)  # [f,w,h,c]
            frames = np.asarray(frames)  # [f,hw,,c]
            frames = frames.transpose(0, 3, 2, 1)  # [f,c,h,w]
            frames = torch.FloatTensor(frames)  # tensor:[f,c,h,w]
            frames -= 127.5
            frames /= 127.5
        except:
            raise ValueError('cant open  video file')
        return frames, original_frames_length

    def adjust_frames(self, frames):
        frames_adjust = []
        frame_length = len(frames)
        if self.num_frames <= len(frames):
            for i in range(1, self.num_frames + 1):
                frame = frames[i * frame_length // self.num_frames - 1]
                frames_adjust.append(frame)

        else:  # 当帧数不足时，补足帧数
            for i in range(frame_length):
                frame = frames[i]
                frames_adjust.append(frame)
            for i in range(self.num_frames - frame_length):
                if len(frames) > 0:
                    frame = frames[-1]
                    frames_adjust.append(frame)
        return frames_adjust  # [f,w,h,3]

    def adjust_label(self, label, frame_length, num_frames=64):
        """
        original cycle list to label
        Args:
            frame_length:  original frames length
            label: label point example [6,31,31,44,44,54] or [0]
            num_frames: 64
        Returns: [6,31,31,44,44,54]
        """
        new_crop = []
        for i in range(len(label)):  # frame_length -> 64
            item = min(math.ceil((float(label[i]) / float(frame_length)) * num_frames), num_frames - 1)
            new_crop.append(item)
        new_crop = np.asarray(new_crop)
        # new_label = normalize_label(new_crop, num_frames)
        y1, y2, num_period = rep_label(new_crop, num_frames)
        y1_tensor = torch.LongTensor(y1)
        y2_tensor = torch.LongTensor(y2)
        return y1_tensor, y2_tensor, num_period

#
# data_root = r'./data/LSP_npz(64)'
# label_file = 'train.csv'
# data = MyDataset(data_root, label_file, 64, 'train')
# for i in range(len(data)):
#     imgs, y1, y2, count = data[0]
# # # a, b, c, d = test[1]
# # # tag = ['train', 'valid', 'test']
# # #
# # data_root = r'/p300/LSP'
