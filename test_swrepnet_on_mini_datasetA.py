import cv2
import torch
import torch.nn as nn
from build_swrepnet import swrepnet
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from test_swrepnet_dataset import MyDataset
import os
import numpy as np
# import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from tqdm import tqdm
import platform
import random
# from torchsummary import summary
# import shutil
# import ipdb
import ipdb

cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if platform.system() == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device_ids = [0, 3]
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

epoch_size = 5000
FRAME = 128
BATCH_SIZE = 4
random.seed(1)
LR = 6e-6
W1 = 5
W2 = 0
NUM_WORKERS = 32


def eval_model(count, y1, y2, w1, w2):
    label_count = count.detach().cpu().numpy().flatten()
    pre_count = (w1 * y1.sum(dim=1).detach().cpu().numpy() + w2 * y2.detach().cpu().numpy().flatten()) / (w1 + w2)
    # pre_count = (w1 * y1.sum(dim=1).detach().cpu().numpy()) / (w1)
    # print(label_count)
    gap_abs = abs(pre_count - label_count) / label_count
    MAE = np.sum(gap_abs) / len(label_count)
    OBO = 0
    for i in range(len(label_count)):
        if abs(pre_count - label_count)[i] < 1:
            OBO += 1
    OBO /= len(label_count)
    return MAE, OBO, pre_count, label_count
    pass


if __name__ == '__main__':

    model = swrepnet(frame=FRAME)
    model = torch.nn.DataParallel(model.to(device), device_ids=device_ids)
    # model = MMDataParallel(model.to(device), device_ids=device_ids)

    data_dir1 = r'/p300/LSP_extra'
    data_dir2 = r'./data/LSP'
    if os.path.exists(data_dir1):
        data_root = data_dir1
    elif os.path.exists(data_dir2):
        data_root = r'./data/LSP'
    else:
        raise ValueError('NO data root')

    test_label = r'test.csv'
    test_dataset = MyDataset(root_dir=data_root, label_dir=test_label, frames=FRAME, method='test')
    test_loader = DataLoader(dataset=test_dataset, pin_memory=True, batch_size=BATCH_SIZE,
                             drop_last=False, shuffle=False, num_workers=NUM_WORKERS)
    criterion1 = nn.MSELoss()
    criterion2 = nn.SmoothL1Loss()
    w1 = W1
    w2 = W2
    scaler = GradScaler()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.8)  # three step decay
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.to(device)

    # tensorboard
    new_train = False
    log_dir = './swrnet_log2'
    if new_train is True:
        del_list = os.listdir(log_dir)
        for f in del_list:
            file_path = os.path.join(log_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    log_dir = './swrnet_log2'
    writer = SummaryWriter(log_dir=log_dir)

    # lastCkptPath = '/p300/SWRNET/checkpoint/ckpt470_trainMAE_0.6788799460571364.pt'
    lastCkptPath = '/p300/SWRNET/checkpoint2/ckpt3_trainMAE_1.4209200608455077.pt'
    # lastCkptPath = None
    if lastCkptPath is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastCkptPath)
        model.load_state_dict(checkpoint['state_dict'], strict=True)

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint
    else:
        print('without ckpt')

    # print("********* Training begin *********")

    # save evaluation metrics
    total_MAE, total_OBO = [], []
    pbar = tqdm(test_loader, total=len(test_loader))
    batch_idx = 0
    model.eval()
    # model.train()
    print('begin test')
    for datas, labels, count in pbar:
        # ipdb.set_trace()
        # print('///////////////////////{} begin to train/////////////////////////////'.format(batch_idx))
        # torch.cuda.empty_cache()
        datas = datas.to(device)
        labels = labels.to(device)  # output: [b,f]
        count = count.to(device).reshape([-1, 1])
        with torch.no_grad():
            try:

                y1, y2 = model(datas)
                # print('test success')
            except:
                print('wrong')
            else:
                loss1 = criterion1(y1, labels)
                loss2 = criterion2(y2, count.float())
                loss = w1 * loss1 + w2 * loss2
        MAE, OBO, pre_count, label_count = eval_model(count, y1, y2, w1, w2)
        total_MAE.append(MAE)
        total_OBO.append(OBO)
    ipdb.set_trace()
    print('total_MAE:', np.mean(total_MAE))
    print('total_OBO:', np.mean(total_OBO))
