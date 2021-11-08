import os
import torch
from Model_inn import RepNet
from Repnet_dataset import MyData
from repnet_looping import train_loop

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_ids = [1]

# root_dir = r'/p300/data/LSPdataset'
root_dir = r'D:\人体重复运动计数\LSPdataset'
train_video_dir = 'train'
train_label_dir = 'train.csv'
valid_video_dir = 'valid'
valid_label_dir = 'valid.csv'
lastckpt = '/p300/checkpoint/1105_1_64_9.pt'

NUM_FRAME = 64

train_dataset = MyData(root_dir, train_video_dir, train_label_dir, num_frame=NUM_FRAME)
valid_dataset = MyData(root_dir, valid_video_dir, valid_label_dir, num_frame=NUM_FRAME)

model = RepNet(NUM_FRAME)
NUM_EPOCHS = 100
LR = 6e-6
BATCH_SIZE=1

train_loop(NUM_EPOCHS, model, train_dataset, valid_dataset, train=True, valid=True,
           batch_size=BATCH_SIZE, lr=LR, saveCkpt=True, ckpt_name='1108_3',log_dir='scalar1108_3',device_ids=device_ids)

