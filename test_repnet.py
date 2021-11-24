import cv2
import torch
import torch.nn as nn
# from build_swrepnet import swrepnet
from build_repnet import RepNetPeriodEstimator
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import MyDataset
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

# from tool import data_utils

# from torchsummary import summary
# import shutil
# import ipdb
import matplotlib.pyplot  as plt

# cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if platform.system() == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
    device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device_ids = [3, 0]
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
W2 = 10
NUM_WORKERS = 32

NPZ = True
P = 0.2
lastCkptPath = '/p300/SWRNET/ckp_repnet_1119_1/ckpt15_trainMAE_1.822.pt'
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


def eval_model(y1, y2, label_count):
    """

    Args:
        y1: per frame period length, shape = tensor:[b,f,max_period] . e.g. [4,64,32]
        y2: per frame within period, shape = tensor:[b,f]. e.g.[4,64]
        label_count: the count of grand truth which was built by label2rep.py

    Returns:
        MAE: the MAE of one batch.
        OBO: the OBO of one batch.
        pre_count: the count of per video, shape = np:[b]

        e.g.
        y1.shape = [2,64,32]
           y1: tensor([[[-0.1527,  0.0970,  0.0122,  ..., -0.1046,  0.4646,  0.0536],
                     [-0.1073, -0.1229, -0.0724,  ...,  0.1291,  0.4023,  0.4369],
                     [ 0.2510,  0.0822, -0.1133,  ...,  0.0591,  0.1691, -0.0701],
                     ...,
                     [ 0.4523,  0.1650, -0.3716,  ...,  0.3348,  0.2594, -0.2106],
                     [-0.0569, -0.2922,  0.0599,  ..., -0.0198,  0.5320,  0.3258],
                     [-0.3526,  0.3049,  0.4505,  ..., -0.0690,  0.1838,  0.4717]],

                    [[ 0.3433, -0.4929,  0.3322,  ...,  0.0730,  0.5719, -0.4434],
                     [-0.0430,  0.2925, -0.0581,  ...,  0.1498,  0.0712,  0.1374],
                     [ 0.1892,  0.1509,  0.2136,  ..., -0.4939,  0.2189, -0.2659],
                     ...,
                     [ 0.1503, -0.2162, -0.2845,  ..., -0.0260,  0.2399, -0.4369],
                     [-0.1763, -0.0746,  0.0118,  ..., -0.1149, -0.0094,  0.2220],
                     [-0.0989, -0.0205,  0.1667,  ...,  0.2107,  0.2170, -0.0162]]])
        y2.shape = [2,64]
        then  by y1 = torch.argmax(y1, dim=2)
           y1: tensor([[31, 37, 38, 40, 62, 14, 62, 53, 45, 13, 35, 32, 16,  6, 38, 49, 35, 42,
                     54, 30, 52, 57, 38, 16, 35, 14, 46, 63, 37, 53, 62, 62, 55, 32, 16,  9,
                     32, 13, 52, 12,  0, 59, 51, 41, 57, 39, 34, 15, 54, 62, 34, 43, 60, 62,
                     14, 32, 14, 54, 45, 58, 52,  3, 62, 25],
                    [62, 54,  4,  7, 22,  7, 18, 62, 28, 26, 52, 62, 52, 32, 60, 59, 28, 28,
                      5, 52, 54, 25, 45,  3, 54,  7, 53, 13, 16, 33, 14, 11, 57, 39, 15,  9,
                     62, 53, 37, 23, 45, 57, 14, 62, 34, 53, 28, 44, 52, 36, 62, 55,  2,  5,
                     38, 62, 16, 19, 32, 53, 38, 37, 32, 37]])
           y2: tensor([[-1.9085e-02, -1.1825e-01,  5.3301e-01, -2.2395e-01,  9.3441e-02,
                      3.3679e-01,  9.3310e-02,  1.8909e-01,  2.8656e-01,  3.7536e-01,
                      2.5514e-02,  5.0600e-03,  2.0166e-01, -9.5531e-02, -1.0591e-01,
                     -5.8728e-01,  9.5945e-02, -3.7965e-02, -6.1789e-02, -1.4481e-01,
                      2.6248e-01,  3.1214e-02,  1.1477e-01, -4.2951e-02, -1.9043e-01,
                      4.0703e-01, -4.9590e-02, -5.6835e-02, -4.4173e-01,  3.7733e-01,
                      4.4231e-01,  2.4140e-01,  1.5698e-01,  1.6453e-01, -2.1800e-01,
                     -1.2975e-02, -1.3237e-01,  1.4669e-01, -5.6803e-02,  1.8498e-01,
                      1.4089e-01, -1.2717e-03,  4.7374e-01,  3.1298e-02,  7.5854e-02,
                      1.2213e-02, -1.2622e-01, -2.2001e-01, -1.6361e-01, -1.1558e-01,
                     -1.6277e-02,  1.0667e-01,  3.0014e-01, -5.8727e-01,  3.1545e-02,
                     -2.9533e-01,  4.8003e-01,  1.4045e-01,  3.9375e-01, -8.2327e-02,
                     -2.3117e-01,  1.5900e-01,  3.4784e-03,  2.4025e-01],
                    [-1.2943e-01,  3.1282e-01, -5.8563e-02,  2.0546e-01,  1.9258e-01,
                     -1.6292e-03,  3.6191e-02,  1.8655e-01,  1.8982e-01,  1.7922e-01,
                      3.4486e-01, -1.1937e-01,  2.3351e-01,  5.1337e-02, -9.6292e-02,
                      1.5240e-01,  4.6470e-01,  1.5024e-03, -1.0442e-01,  3.4582e-02,
                      9.0641e-02,  1.7103e-01, -1.2973e-01,  4.2786e-01,  1.9274e-01,
                      2.1767e-01,  3.7242e-01,  6.5135e-02, -3.6086e-01,  1.7195e-01,
                      5.7926e-01,  4.0312e-01, -2.1481e-03, -1.4346e-01, -1.5655e-02,
                      4.6510e-01,  3.0693e-04,  2.0094e-02, -9.0856e-02, -7.1425e-02,
                      4.4809e-02, -1.7435e-01,  4.1627e-01,  2.1307e-01, -1.9213e-01,
                      2.7987e-02,  2.8832e-01,  2.1462e-01,  5.2479e-02,  8.8549e-02,
                      2.0141e-01, -9.0030e-02,  6.5863e-02,  2.0524e-02, -1.0676e-01,
                      6.5326e-02,  5.2676e-01,  4.0676e-01,  1.5199e-01,  2.2177e-01,
                     -5.9902e-02,  1.0546e-01, -1.3028e-01,  4.5258e-02]])
            so :
                pre_count = tensor([0., 0.])
                label_count = tensor([ 6.0000, 23.0000], dtype=torch.float64)
            then adjust the data type and return.

    """
    y1 = torch.argmax(y1, dim=2).detach().cpu()  # [b,f]
    y2 = y2.detach().cpu()
    pre_count = torch.zeros([y1.shape[0]]).cpu()
    label_count = label_count.cpu().view(-1)
    for _ in range(y1.shape[0]):
        for i in range(y1.shape[1]):
            if y1[_][i] >= 1 and y2[_][i] > P:
                pre_count[_] += 1 / y1[_][i]

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

    test_label = r'test.csv'

    test_dataset = MyDataset(root_dir=data_root, label_dir=test_label, frames=FRAME, method='test')

    test_loader = DataLoader(dataset=test_dataset, pin_memory=True, batch_size=BATCH_SIZE,
                             drop_last=False, shuffle=False, num_workers=NUM_WORKERS)

    if lastCkptPath is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastCkptPath)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

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
    for datas, target1, target2, count in pbar:
        datas = datas.to(device)
        target1 = target1.to(device)  # output: [b,f]
        target2 = target2.to(device)  # output: [b,f]
        count = count.cpu().reshape([-1, 1])
        # 向前传播
        with torch.no_grad():
            y1, y2, embedding_code, sim_matrix = model(datas)  # output : [b,f,period]
        # if count < 1:
        #     continue
        MAE, OBO, pre_count = eval_model(y1, y2, count)
        # duration = FRAME / count.numpy()
        # test_MAE[int(duration // 5)].append(MAE)
        # test_OBO[int(duration // 5)].append(OBO)
        total_MAE.append(MAE)
        total_OBO.append(OBO)
    D_MAE, D_OBO = [], []
    print(np.mean(total_MAE))
    print(np.mean(total_OBO))

    #
    # print('D_MAE:', D_MAE)
    # print('D_OBO:', D_OBO)
    # np.save('C_MAE_2.npy', D_MAE)
    # np.save('C_OBO_2.npy', D_OBO)
    # x = []
    # for i in range(13):
    #     s = i * 5
    #     e = s + 5
    #     if i != 12:
    #         st = str(s) + '-' + str(e)
    #     else:
    #         st = "60+"
    #     x.append(st)
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # plt.barh(x, D_MAE)
    # ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    # plt.xlabel('MAE')
    # plt.title('MAE in different duration')
    #
    # ax = fig.add_subplot(122)
    # plt.barh(x, D_OBO)
    # ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
    # plt.xlabel('OBO')
    # plt.title('OBO in different duration')
    # # plt.show()
    # # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.5, hspace=0)
    # plt.savefig('test_repCnet_dataA without training of features.png')
    # # print(np.mean(total_MAE))
    # # print(np.mean(total_OBO))
    # pbar.set_postfix({
    #     'test MAE': float(np.mean(total_MAE)),
    #     'test OBO': float(np.mean(total_OBO))}
    # )
