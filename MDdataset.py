import numpy as np
from PIL import Image
from torchvision import transforms
import os
import cv2
import torch
import pandas as pd
import math
from label2rep import rep_label
from label2normal import normalize_label, normalize_label2, normalize_label3
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
        # to save data in ram
        self.file_imgs_np = np.zeros([len(self.file_list), self.num_frames, 3, self.image_size, self.image_size])
        self.file_fps_np = np.zeros([len(self.file_list)])

        for i in range(len(self.file_list)):
            filename = self.file_list[i]
            npz_name = os.path.splitext(filename)[0] + '.npz'
            npz_path = os.path.join(self.video_dir, npz_name)
            with np.load(npz_path) as data:
                frames = data['imgs']
                fps = data['fps'].item()
            self.file_imgs_np[i] = frames
            self.file_fps_np[i] = fps

    def __getitem__(self, index):
        """

        Args:
            index:

        Returns:
            video_imgs: loaded video to imgs    :FloatTensor    [1,64,3,224,224]
            y1: density  :Tensor     [64,]
            num_period : count by y1.sum() : float
            HINT: if count !=  num_period : return num_period
        """

        filename = self.file_list[index]
        npz_name = os.path.splitext(filename)[0] + '.npz'
        npz_path = os.path.join(self.video_dir, npz_name)
        # video_imgs, original_frames_length = self.read_npz(npz_path, index)
        video_imgs, original_frames_length = self.load_file(index)
        y1, num_period, y2, num_period2, y3, num_period3 = self.adjust_label(index,
                                                                             original_frames_length, self.num_period)

        return video_imgs, y1, num_period, y2, num_period2, y3, num_period3
        # return video_imgs, y1,  y2, y3

    def __len__(self):
        """返回数据集的大小"""
        return len(self.file_list)

    def load_file(self, index):
        frames = self.file_imgs_np[index]
        fps = self.file_fps_np[index]
        frames = torch.FloatTensor(frames)  # tensor:[f,c,h,w]; h,w is 224
        frames -= 127.5
        frames /= 127.5
        return frames, fps

    def read_npz(self, npz_path, index):
        """

        Args:
            npz_path: the absolute path to video.npz

        Returns:
            frames: tensor [f,c,h,w] which has been normalized.
                    h and w are 224;
            fps: the original of video.fps


        """

        ts = time.time()
        with np.load(npz_path) as data:
            frames = data['imgs']
            fps = data['fps'].item()
        # self.file_imgs_np[index] = frames
        # self.file_fps_np[index] = fps

        frames = torch.FloatTensor(frames)  # tensor:[f,c,h,w]; h,w is 224
        frames -= 127.5
        frames /= 127.5
        return frames, fps

    def adjust_label(self, index, frame_length, num_frames=64):
        """
        original cycle list to label
        Args:
            frame_length:  original frames length
            label: label point example [6,31,31,44,44,54] or [0]
            num_frames: 64
        Returns:
             y1_tensor shape: [num_frames]
             num_period:
        """
        label = self.label_list[index]
        filename = self.file_list[index]
        new_crop = []
        for i in range(len(label)):  # frame_length -> 64
            item = min(math.ceil((float(label[i]) / float(frame_length)) * num_frames), num_frames - 1)
            new_crop.append(item)
        new_crop = np.asarray(new_crop)
        # new_label = normalize_label(new_crop, num_frames)

        y1, num_period = normalize_label(new_crop, num_frames)
        y2, num_period2 = normalize_label2(new_crop, num_frames)
        y3, num_period3 = normalize_label3(new_crop, num_frames)
        y1_tensor = torch.tensor(y1)
        y2_tensor = torch.tensor(y2)
        y3_tensor = torch.tensor(y3)
        if len(new_crop) / 2 - num_period > 0.01:
            print("video name:{}, original label:{},num period:{}".format(filename, len(new_crop),
                                                                          num_period))
        elif len(new_crop) / 2 - num_period2 > 0.01:
            print("video name:{}, original label:{},num period:{}".format(filename, len(new_crop),
                                                                          num_period2))
        elif len(new_crop) / 2 - num_period2 > 0.01:
            print("video name:{}, original label:{},num period:{}".format(filename, len(new_crop),
                                                                          num_period3))

        return y1_tensor, num_period, y2_tensor, num_period2, y3_tensor, num_period3


#
def test():
    data_root = r'/p300/LLSP_npz(64)'
    # data_root = r'./data/LSP_npz(64)'
    label_file = 'train.csv'
    data = MyDataset(data_root, label_file, 64, 'train')
    for i in range(0, len(data)):
        datas, target1, count1, target2, count2, target3, count3 = data[i]

# if __name__ == '__main__':
#     test()
# # # a, b, c, d = test[1]
# # # tag = ['train', 'valid', 'test']
# # #
# # data_root = r'/p300/LSP'