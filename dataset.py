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
            video_imgs: loaded video to imgs
            y1: period length  0 - max period
            y2: within period  0 - 1
            num_period : count by processing labels:(y2/y1)[~np.isnan(y2/y1).bool()].sum()
            ps: if count !=  num_period : return num_period
        """
        filename = self.file_list[index]
        npz_name = os.path.splitext(filename)[0] + '.npz'
        npz_path = os.path.join(self.video_dir, npz_name)
        video_imgs, original_frames_length = self.read_npz(npz_path)
        y1, y2, num_period = self.adjust_label(self.label_list[index], original_frames_length, self.num_period)
        return video_imgs, y1, y2, num_period

    def __len__(self):
        """返回数据集的大小"""
        return len(self.file_list)

    def read_npz(self, npz_path):
        """

        Args:
            npz_path: the absolute path to video.npz

        Returns:
            frames: tensor [f,c,h,w] which has been normalized.
                    h and w are 224;
            fps: the original of video.fps


        """
        with np.load(npz_path) as data:
            frames = data['imgs']
            fps = data['fps'].item()
        frames = torch.FloatTensor(frames)  # tensor:[f,c,h,w]; h,w is 224
        frames -= 127.5
        frames /= 127.5
        return frames, fps

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

# data_root = r'./data/LSP_npz(64)'
# label_file = 'train.csv'
# data = MyDataset(data_root, label_file, 64, 'train')
# for i in range(len(data)):
#     imgs,y1,y2,count = data[i]
# # a, b, c, d = test[1]
# # tag = ['train', 'valid', 'test']
# #
# data_root = r'/p300/LSP'
