import numpy as np
from PIL import Image
from torchvision import transforms
import os
import cv2
import torch
import pandas as pd
import math
from label2rep import rep_label
from label2normal import normalize_label
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
        self.gap = np.zeros([512])
        for i in range(0, len(df)):
            filename = df.loc[i, 'name']
            label_tmp = df.values[i][self.num_idx:].astype(np.float64)
            label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)
            # if self.isdata(filename, label_tmp):  # data is right format ,save in list
            label = label_tmp
            for i in range(0, len(label), 2):
                s = label[i]
                e = label[i + 1]
                if e - s < 512:
                    self.gap[e - s] += 1
                else:
                    print("e-s：", e - s)
            self.file_list.append(filename)
            self.label_list.append(label_tmp)
        # to save data in ram
        # self.file_imgs_np = np.zeros([len(self.file_list), self.num_frames, 3, self.image_size, self.image_size])
        # self.file_imgs_np = np.zeros([4, self.num_frames, 3, self.image_size, self.image_size])
        # self.file_fps_np = np.zeros([len(self.file_list)])
        # self.memory_tag = np.full(len(self.file_list), False)

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
        # npz_name = os.path.splitext(filename)[0] + '.npz'
        # npz_path = os.path.join(self.video_dir, npz_name)
        # video_imgs, original_frames_length = self.read_npz(npz_path, index)
        label = self.label_list[index]
        count64 = 0
        count32 = 0
        count128 = 0
        gap = []
        for i in range(0, len(label), 2):
            s = label[i]
            e = label[i + 1]
            if e - s > 64:
                if e - s > 128:
                    count128 += 1
                else:
                    count64 += 1
            elif e - s > 32:
                count32 += 1
            gap.append(e - s)
        # y1, num_period = self.adjust_label(self.label_list[index], original_frames_length, self.num_period)
        return count32, count64, count128, gap

    def __len__(self):
        """返回数据集的大小"""
        return len(self.file_list)

    def read_npz(self, npz_path, index):
        """

        Args:
            npz_path: the absolute path to video.npz

        Returns:
            frames: tensor [f,c,h,w] which has been normalized.
                    h and w are 224;
            fps: the original of video.fps


        """
        # if self.memory_tag[index]:
        if False:
            # tag = 'loading form ram'
            # ts = time.time()
            # frames = self.file_imgs_np[index]
            # fps = self.file_fps_np[index]
            pass
        else:
            tag = 'loading form disk'
            ts = time.time()
            with np.load(npz_path) as data:
                frames = data['imgs']
                fps = data['fps'].item()
            # self.file_imgs_np[index] = frames
            # self.file_fps_np[index] = fps
            # self.memory_tag[index] = True
        te = time.time()
        if te - ts > 5:
            print('file:', self.file_list[index], ' loaded take ', te - ts, 's and it is ', tag)
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
        Returns:
             y1_tensor shape: [num_frames]
             num_period:
        """
        new_crop = []
        for i in range(len(label)):  # frame_length -> 64
            item = min(math.ceil((float(label[i]) / float(frame_length)) * num_frames), num_frames - 1)
            new_crop.append(item)
        new_crop = np.asarray(new_crop)
        # new_label = normalize_label(new_crop, num_frames)
        y1, num_period = normalize_label(new_crop, num_frames)
        y1_tensor = torch.tensor(y1)

        return y1_tensor, num_period


#
data_root = r'./data/LSP_npz(64)'
label_file = 'train.csv'
dataset = MyDataset(data_root, label_file, 64, 'train')
# dataloader = DataLoader(dataset=dataset, pin_memory=True, batch_size=1,
#                         drop_last=False, shuffle=False, num_workers=0)
# for i,c32, c64, c128,gap in enumerate(dataloader):
gap = dataset.gap
plt.plot(range(512), gap, '-*', markersize=0.3, linewidth=0.2)
plt.show()
# avg_gap = []
# avg_c64 = []
# avg_c128 = []
# for i in range(len(dataset)):
#     c32, c64, c128, gap = dataset[i]
#     avg_gap.append(max(gap))
#     avg_c64.append(c64)
#     avg_c128.append(c128)
#     print(i, ' max gap:', max(gap), ' c32:', c32, ' c64:', c64, ' c128:', c128)
# print(np.mean(avg_gap))
# print(np.mean(avg_c64))
# print(np.mean(avg_c128))
# # # a, b, c, d = test[1]
# # # tag = ['train', 'valid', 'test']
# # #
# # data_root = r'/p300/LSP'
