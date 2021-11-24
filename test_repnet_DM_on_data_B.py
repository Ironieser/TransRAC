import cv2
import torch
import torch.nn as nn
# from build_swrepnet import swrepnet
from build_repnet_DM import RepNetPeriodEstimator
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from test_dataB_for_repnetDM_dataset import MyDataset
import os
import numpy as np
# import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from tqdm import tqdm
import platform
import random
from torchvision.transforms import ToPILImage

import torch.backends.cudnn
import matplotlib.pyplot  as plt

# from torchsummary import summary
# import shutil
# import ipdb


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if platform.system() == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
    device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device_ids = [1, 0]
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

epoch_size = 5000
FRAME = 64
BATCH_SIZE = 1
random.seed(1)
LR = 6e-6
W1 = 1
W2 = 10
NUM_WORKERS = 32
# LOG_DIR = './repnet_log1116_2'
NPZ = True
# CKPT_DIR = './ckp_repDnet_test_A'
P = 0.2
# lastCkptPath = '/p300/SWRNET/ckp_repDnet_4/ckpt36_trainMAE_1.442.pt'  # not train feature layers
lastCkptPath = '/p300/SWRNET/ckp_repDnet_4/ckpt33_trainMAE_1.559.pt'  # not train feature layers
# lastCkptPath = '/p300/SWRNET/ckp_repDnet_6/ckpt99_trainMAE_0.288.pt'  # train feature layers
# lastCkptPath = None
if platform.system() == 'Windows':
    NUM_WORKERS = 0
    lastCkptPath = None
    BATCH_SIZE = 1


def get_loss_weight(l1, l2):
    k = l1 / l2
    if k > 1:
        w1 = 1
        w2 = int(k)
    else:
        w1 = int(1 / k)
        w2 = 1
    return w1, w2


def eval_model(y1, label_count):
    """

    Args:
        y1: per frame density map, shape = tensor:[b,f] . e.g. [4,64]
        label_count: the count of grand truth ,just use y.sum()

    Returns:
        MAE: the MAE of one batch.
        OBO: the OBO of one batch.
        pre_count: the count of per video, shape = np:[b]

    """
    pre_count = torch.sum(y1, dim=1).detach().cpu()  # [b]
    label_count = label_count.cpu().view(-1)

    gap_abs = abs((pre_count - label_count) / label_count)
    MAE = gap_abs.sum() / len(label_count)
    OBO = float((abs(pre_count - label_count) < 1).sum() / len(label_count))

    return float(MAE), OBO, pre_count.numpy()


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
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = MMDataParallel(model, device_ids=device_ids)
    model.to(device)
    data_dir1 = r'/p300/dataset_B'

    if os.path.exists(data_dir1):
        data_root = data_dir1
    else:
        raise ValueError("input dir wrong")

    test_label = r'dataset-B.csv'

    test_dataset = MyDataset(root_dir=data_root, label_dir=test_label, frames=FRAME, method='test')

    test_loader = DataLoader(dataset=test_dataset, pin_memory=True, batch_size=BATCH_SIZE,
                             drop_last=False, shuffle=False, num_workers=NUM_WORKERS)

    if lastCkptPath is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastCkptPath)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
    else:
        print('without ckpt')

    test_MAE = [[] for i in range(13)]
    test_OBO = [[] for i in range(13)]

    total_MAE, total_OBO = [], []
    pbar = tqdm(test_loader, total=len(test_loader))
    model.eval()
    idx = 0
    for datas, target1, count in pbar:
        datas = datas.to(device)
        target1 = target1.to(device)  # output: [b,f]
        count = count.cpu().reshape([-1, 1])
        # 向前传播
        with torch.no_grad():
            y1, embedding_code, sim_matrix = model(datas)  # output : [b,f,period]
        if count < 1:
            continue
        duration = FRAME / count.numpy()
        MAE, OBO, pre_count = eval_model(y1, count)
        test_MAE[int(duration // 5)].append(MAE)
        test_OBO[int(duration // 5)].append(OBO)
        total_MAE.append(MAE)
        total_OBO.append(OBO)
        # if platform.system() == 'Windows':
        #     if idx > 3:
        #         break
        #     else:
        #         idx += 1
    D_MAE, D_OBO = [], []
    print(np.mean(total_MAE))
    print(np.mean(total_OBO))
    for i in range(13):
        D_MAE.append(np.mean(test_MAE[i]) if not np.isnan(np.mean(test_MAE[i])) else 0)
        D_OBO.append(np.mean(test_OBO[i]) if not np.isnan(np.mean(test_OBO[i])) else 0)
    np.save('D_MAE_2.npy', D_MAE)
    np.save('D_OBO_2.npy', D_OBO)

    # fig = plt.figure()
    x = []
    for i in range(13):
        s = i * 5
        e = s + 5
        if i != 12:
            st = str(s) + '-' + str(e)
        else:
            st = "60+"
        x.append(st)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    plt.barh(x, D_MAE)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    plt.xlabel('MAE')
    plt.title('MAE in different duration')

    ax = fig.add_subplot(122)
    plt.barh(x, D_OBO)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    plt.xlabel('OBO')
    plt.title('OBO in different duration')
    # plt.show()
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.savefig('test_repnet_on_dataA without training of features.png')
    # print(np.mean(total_MAE))
    # print(np.mean(total_OBO))
    pbar.set_postfix({
        'test MAE': float(np.mean(total_MAE)),
        'test OBO': float(np.mean(total_OBO))}
    )
