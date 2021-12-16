import cv2
import torch
import torch.nn as nn
# from build_swrepnet import swrepnet
from build_repnet_MDM import RepNetPeriodEstimator
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from MDdataset import MyDataset
import os
import numpy as np
# import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from tqdm import tqdm
import platform
import random
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.backends.cudnn
from vision import sm_heat_map
import seaborn as sns
import time
from torchvision.transforms import ToTensor
import matplotlib.pyplot  as plt
# from tool import data_utils

# from torchsummary import summary
# import shutil
import ipdb

# cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if platform.system() == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
    device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device_ids = [1]
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

epoch_size = 5000
FRAME = 64
BATCH_SIZE = 8
random.seed(1)
LR = 6e-6
W1 = 1
W2 = 0
W3 = 0
NUM_WORKERS = 2
LOG_DIR = './repnetMDM_log1124_3'
NPZ = True
CKPT_DIR = './ckp_repnetMDM_3'
P = 0.2
lastCkptPath = '/p300/SWRNET/ckp_repnetMDM_3/ckpt30_trainMAE_1.558.pt'
# lastCkptPath = None
if platform.system() == 'Windows':
    NUM_WORKERS = 0
    lastCkptPath = None
    BATCH_SIZE = 2


def get_loss_weight(l1, l2):
    k = l1 / l2
    if k > 1:
        w1 = 1
        w2 = int(k)
    else:
        w1 = int(1 / k)
        w2 = 1
    return w1, w2


def eval_model(pre_count, label_count):
    """

    Args:
        y1: per frame density map, shape = tensor:[b,f] . e.g. [4,64]
        label_count: the count of grand truth ,just use y.sum()

    Returns:
        MAE: the MAE of one batch.
        OBO: the OBO of one batch.
        pre_count: the count of per video, shape = np:[b]

    """
    label_count = label_count.cpu().view(-1)

    gap_abs = abs((pre_count - label_count) / label_count)
    MAE = gap_abs.sum() / len(label_count)
    OBO = float((abs(pre_count - label_count) < 1).sum() / len(label_count))

    return float(MAE), OBO


def show_sim_matrix(matrix):
    show = ToPILImage()
    if len(matrix.shape) == 3:
        show(matrix).show()
    else:
        print("error: the length of shape of sim_matrix  is not 3")


if __name__ == '__main__':

    model = RepNetPeriodEstimator()
    # model = torch.nn.DataParallel(model.to(device), device_ids=device_ids)
    if torch.cuda.device_count() > 1:
        print("Let's use", len(device_ids), "GPUs!")
        model = MMDataParallel(model, device_ids=device_ids)
    model.to(device)
    data_dir1 = r'/p300/LLSP'
    data_dir2 = r'./data/LSP_npz(64)'
    data_dir3 = r'/dfs/data/LSP/LSP'
    if os.path.exists(data_dir1 + '_npz(64)'):
        data_root = data_dir1
        if NPZ:
            if FRAME == 64:
                data_root += '_npz(64)'
            elif FRAME == 128:
                data_root += '_npz(128)'
            else:
                raise ValueError('without npz with FRAME:', FRAME)
    elif os.path.exists(data_dir2):
        data_root = data_dir2
    elif os.path.exists(data_dir3):
        data_root = data_dir3
    else:
        raise ValueError('NO data root')

    valid_label = r'test.csv'

    valid_dataset = MyDataset(root_dir=data_root, label_dir=valid_label, frames=FRAME, method='test')
    # test_dataset = MyDataset(root_dir=data_root, label_dir=valid_label, frames=FRAME, method='test')

    valid_loader = DataLoader(dataset=valid_dataset, pin_memory=True, batch_size=BATCH_SIZE,
                              drop_last=False, shuffle=True, num_workers=NUM_WORKERS)

    if lastCkptPath is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastCkptPath)
        currEpoch = checkpoint['epoch']
        trainLosses = checkpoint['trainLoss']
        validLosses = checkpoint['valLoss']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        currTrainstep = checkpoint['train_step']
        currValidationstep = checkpoint['valid_step']
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
    else:
        currEpoch = 0
        currTrainstep = 0
        currValidationstep = 0

    # save evaluation metrics
    valid_MAE, valid_OBO = [], []

    w1, w2, w3 = W1, W2, W3
    test_MAE = [[] for i in range(40)]
    test_OBO = [[] for i in range(40)]
    # valid
    pbar = tqdm(valid_loader, total=len(valid_loader))
    # print("********* Validation *********")
    model.eval()
    for datas, target1, count1, target2, count2, target3, count3, original_frames_length in pbar:
        datas = datas.to(device)

        count1 = count1.cpu().reshape([-1, 1])

        count2 = count2.cpu().reshape([-1, 1])

        count3 = count3.cpu().reshape([-1, 1])

        # 向前传播
        with torch.no_grad():
            y1, y2, y3, embedding_code, sim_matrix = model(
                datas)  # output : y1 :[b,f]  sim_matrix:[b,1,f,f]

        pre_count1 = torch.sum(y1, dim=1).detach().cpu()
        pre_count2 = torch.sum(y2, dim=1).detach().cpu()
        pre_count3 = torch.sum(y3, dim=1).detach().cpu()
        pre_count = (w1 * pre_count1 + w2 * pre_count2 + w3 * pre_count3) / (w1 + w2 + w3)
        count = (w1 * count1 + w2 * count2 + w3 * count3) / (w1 + w2 + w3)
        MAE, OBO = eval_model(pre_count, count)
        valid_MAE.append(MAE)
        valid_OBO.append(OBO)
        for i in range(len(original_frames_length)):
            test_MAE[int(original_frames_length[i] // 64)].append(MAE)
            test_OBO[int(original_frames_length[i] // 64)].append(OBO)

        # pbar.set_postfix({
        #         #     'Valid MAE': float(np.mean(valid_MAE)),
        #         #     'Valid OBO': float(np.mean(valid_OBO))}
        #         # )
    D_MAE, D_OBO = [], []
    print(np.mean(valid_MAE))
    print(np.mean(valid_OBO))
    for i in range(40):
        D_MAE.append(np.mean(test_MAE[i]) if not np.isnan(np.mean(test_MAE[i])) else 0)
        D_OBO.append(np.mean(test_OBO[i]) if not np.isnan(np.mean(test_OBO[i])) else 0)

    x = []
    for i in range(40):
        s = i * 64
        e = s + 64
        if i < 39:
            st = str(s) + '-' + str(e)
        else:
            st = str(64 * 40) + "+"
        x.append(st)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    # plt.barh(x, D_MAE)
    for i in range(len(D_MAE)):
        if D_MAE[i] != 0:
            plt.plot(i, D_MAE[i], 'r*')
    # plt.plot(range(len(D_MAE)),D_MAE,'r-')
    # import ticker
    #
    # tick_spacing = 10
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    # ax.set_aspect('auto')
    plt.xlabel('MAE')
    plt.title('MAE in different duration')

    ax = fig.add_subplot(122)
    # plt.barh(x, D_OBO)
    # plt.plot(range(len(D_OBO)), D_OBO, 'r-')

    for i in range(len(D_OBO)):
        if D_OBO[i] != 0:
            plt.plot(i, D_OBO[i], 'r*')

    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    # ax.set_aspect('auto')
    plt.xlabel('OBO')
    plt.title('OBO in different duration')
    # plt.show()
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.5
                        , hspace=0)
    plt.savefig('MAE_and_OBO.png')
    print('test MAE:', np.mean(valid_MAE))
    print('test OBO:', np.mean(valid_OBO))
