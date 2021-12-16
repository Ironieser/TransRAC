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


def isdata(self, file, label):
    video_path = os.path.join(self.video_dir, file)
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7)
    if label.size == 0:
        # print('empty data:', file)
        self.error += 1
        return False
    elif frames_num >= max(label):
        return True
    else:
        # print("error data:", file, frames_num)
        # print('error data:', label)
        # print("error data:", len(label))
        self.error += 1
        return False


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
            num_period : count by processing labels:(y2/y1)[~np.isnan(y2/y1)].sum()
            ps: if count !=  num_period : return num_period
        """
        filename = self.file_list[index]
        video_path = os.path.join(self.video_dir, filename)
        npy_pth = npy_dir + os.path.splitext(filename)[0] + '.npz'
        if not os.path.exists(npy_pth):
            print('video :', filename, ' does not been saved to npz.')
            a = self.read_video(video_path, filename)
            if a == 1:
                print(index, filename, 'have been saved to npz.')
            else:
                print('error:', index, filename, 'can not be saved to npz.')
        else:
            try:
                data = np.load(npy_pth)
                print('good:', index, filename, 'loading success')
            except:
                print('error:', index, filename, 'is wrong npz which cant opened.')
        # y1, y2, num_period = self.adjust_label(self.label_list[index], original_frames_length, self.num_frames)
        # count = len(self.label_list[index]) / 2
        # if count == 0:
        #     print('file:', filename)
        # if count - num_period > 0.1:
        #     print('file:', filename)
        #     print('y1:', y1)
        #     print('y2:', y2)
        #     print('count:', count)
        #     print('sum_y:', num_period)
        return 1

    def __len__(self):
        """返回数据集的大小"""
        return len(self.file_list)

    def read_video(self, video_filename, filename, width=224, height=224):
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

            # cap = cv2.VideoCapture(video_filename)
            # original_frames_length = cap.get(7)
            # cap.release()
            # frames = 1
            frames = self.adjust_frames(frames)  # [f,w,h,c]
            frames = np.asarray(frames)  # [f,hw,,c]
            if (frames.size != 0):
                frames = frames.transpose(0, 3, 2, 1)  # [f,c,h,w]
            else:
                print(filename, ' is wrong video. size = 0')
                return -1
            # npy_pth = r'/p300/LSP_npz/npy_data/' + os.path.splitext(filename)[0]
            npy_pth = npy_dir + os.path.splitext(filename)[0]
            np.savez(npy_pth, imgs=frames, fps=original_frames_length)  # [f,c,h,w]

        except:
            print('error: ', video_filename, ' cannot open')
            # raise video_filename
        # print(video_filename, 'frames finished')
        return 1

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

    def adjust_label(self, label, frame_length, num_frames):
        """
        original cycle list to label
        Args:
            frame_length: frame_length
            label: label point example [6,31,31,44,44,54] or [0]
            num_frames: 64
        Returns: [6,31,31,44,44,54]
        """
        new_crop = []
        for i in range(len(label)):  # frame_length -> 64
            item = min(math.ceil((float(label[i]) / float(frame_length)) * num_frames), num_frames - 1)
            new_crop.append(item)
        # new_crop = np.sort(new_crop)
        new_crop = np.asarray(new_crop)
        # new_label = normalize_label(new_crop, num_frames)
        y1, y2, num_period = rep_label(new_crop, num_frames)
        y1_tensor = torch.LongTensor(y1)
        y2_tensor = torch.LongTensor(y2)
        return y1_tensor, y2_tensor, num_period


data_root = r'/p300/LSP_extra'

tag = ['train', 'valid', 'test']

# data_root = r'/dfs/data/LSP/LSP'
npy_dir_128 = r'/p300/LSP_extra_npz(128)/'
npy_dir_64 = r'/p300/LSP_extra_npz(64)/'
error_dir = r'/p300/LLSP_npz(64)/'
for _ in range(3):
    error_pth = error_dir + tag[_]
    # files = os.listdir(error_dir)
    print('============================================')
    print(tag[_])
    print('============================================')
    for file in os.listdir(error_pth):
        file_pth = os.path.join(error_pth,file)
        try:
            data = np.load(file_pth)
            print('good', file_pth, ':finished loading')
        except:
            print('error',file_pth, ':failed loading')
# if not os.path.exists(npy_dir_128):
#     os.mkdir(npy_dir_128)
# if not os.path.exists(npy_dir_64):
#     os.mkdir(npy_dir_64)
# for _ in range(3):
#     mod = tag[_]
#     label_file = mod + '.csv'
#     test = MyDataset(data_root, label_file, 128, mod)
#     npy_dir = npy_dir_128 + mod + '/'
#     if not os.path.exists(npy_dir):
#         os.mkdir(npy_dir)
#     print('=========================================')
#     print(mod)
#     for i in range(len(test)):
#         a = test[i]
#
# for _ in range(3):
#     mod = tag[_]
#     label_file = mod + '.csv'
#     test = MyDataset(data_root, label_file, 64, mod)
#     npy_dir = npy_dir_64 + mod + '/'
#     print('=========================================')
#     print(mod, ' : ', npy_dir)
#     if not os.path.exists(npy_dir):
#         os.mkdir(npy_dir)
#     for i in range(len(test)):
#         a = test[i]
