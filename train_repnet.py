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

# from tool import data_utils

# from torchsummary import summary
# import shutil
# import ipdb

cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if platform.system() == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
    device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device_ids = [1, 2, 3, 4, 5, 6, 7]
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

epoch_size = 5000
FRAME = 64
BATCH_SIZE = 14
random.seed(1)
LR = 1e-5
W1 = 1
W2 = 20
NUM_WORKERS = 24
LOG_DIR = './repnet_log1112_2'
CKPT_DIR = './ckp_repnet_2'
NPZ = True
P = 0.2
lastCkptPath = None

if platform.system() == 'Windows':
    NUM_WORKERS = 0


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
    data_dir1 = r'/p300/LSP'
    data_dir2 = r'./data/LSP_npz(64)'
    data_dir3 = r'/dfs/data/LSP/LSP'
    if os.path.exists(data_dir1):
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

    train_label = r'train.csv'
    valid_label = r'valid.csv'

    train_dataset = MyDataset(root_dir=data_root, label_dir=train_label, frames=FRAME, method='train')
    valid_dataset = MyDataset(root_dir=data_root, label_dir=valid_label, frames=FRAME, method='valid')

    train_loader = DataLoader(dataset=train_dataset, pin_memory=True, batch_size=BATCH_SIZE,
                              drop_last=False, shuffle=True, num_workers=NUM_WORKERS)
    # if torch.cuda.is_available():
    #     train_loader = data_utils.CudaDataLoader(train_loader, device=0)

    valid_loader = DataLoader(dataset=valid_dataset, pin_memory=True, batch_size=BATCH_SIZE,
                              drop_last=False, shuffle=True, num_workers=NUM_WORKERS)

    # tensorboard
    new_train = False
    log_dir = LOG_DIR
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)
    if new_train is True:
        del_list = os.listdir(log_dir)
        for f in del_list:
            file_path = os.path.join(log_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    writer = SummaryWriter(log_dir=log_dir)

    # lastCkptPath = '/p300/SWRNET/checkpoint/ckpt350_trainMAE_0.8896425843327483.pt'
    # lastCkptPath = '/p300/SWRNET/checkpoints5/ckpt9_trainMAE_0.663.pt'

    if lastCkptPath is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastCkptPath)
        currEpoch = checkpoint['epoch']
        trainLosses = checkpoint['trainLoss']
        validLosses = checkpoint['valLoss']
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
    else:
        currEpoch = 0
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': 1e-5}], lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=currEpoch)
    # print("********* Training begin *********")
    ep_pbar = tqdm(range(currEpoch, epoch_size))
    train_step = 0
    valid_step = 0
    for epoch in ep_pbar:
        # save evaluation metrics
        train_MAE, train_OBO, valid_MAE, valid_OBO, train_loss, train_loss1, train_loss2 = [], [], [], [], [], [], []
        valid_loss1, valid_loss2, valid_loss = [], [], []
        train_pre, train_label, train_gap = [], [], []
        pbar = tqdm(train_loader, total=len(train_loader))
        batch_idx = 0
        model.train()
        train_step = 0
        for datas, target1, target2, count in pbar:
            # ipdb.set_trace()

            datas = datas.to(device)
            target1 = target1.to(device)  # output: [b,f]
            target2 = target2.to(device)  # output: [b,f]
            count = count.cpu().reshape([-1, 1])
            optimizer.zero_grad()

            # 向前传播
            with autocast():
                y1, y2, embedding_code, sim_matrix = model(
                    datas, epoch)  # output : y1 :[b,f,period] y2: [b,f] sim_matrix:[b,1,f,f]
                loss1 = criterion1(y1.transpose(1, 2), target1)  # y = [b,c,d1,d2,d2,d3,...], target: [b,d1,d2,d3]
                loss2 = criterion2(y2, target2.float())
                w1, w2 = get_loss_weight(loss1, loss2)
                loss = w1 * loss1 + w2 * loss2

            # 反向传播 scaler 放大梯度
            if platform.system() != 'Windows':
                scaler.scale(loss.float()).backward()
                scaler.step(optimizer)
                scaler.update()

            # 输出结果
            MAE, OBO, pre_count = eval_model(y1, y2, count)
            train_label.append(count.numpy()[0])
            train_pre.append(pre_count[0])
            train_gap.append(pre_count[0] - count[0])

            train_MAE.append(MAE)
            train_OBO.append(OBO)
            train_loss.append(float(loss))
            train_loss1.append(float(w1 * loss1))
            train_loss2.append(float(w2 * loss2))

            # if batch_idx % 10 == 0:
            pbar.set_postfix({'Epoch': epoch,
                              'loss': float(np.mean(train_loss)),
                              'Train MAE': float(np.mean(train_MAE)),
                              'Train OBO': float(np.mean(train_OBO)),
                              'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            k_batch = int(len(train_loader) / 10)
            if batch_idx % k_batch == 0 and batch_idx != 0:
                b_loss = np.mean(train_loss[batch_idx - k_batch:batch_idx])
                b_loss1 = np.mean(train_loss1[batch_idx - k_batch:batch_idx])
                b_loss2 = np.mean(train_loss2[batch_idx - k_batch:batch_idx])
                b_MAE = np.mean(train_MAE[batch_idx - k_batch:batch_idx])
                b_OBO = np.mean(train_OBO[batch_idx - k_batch:batch_idx])
                writer.add_image('sim_matrix',
                                 make_grid(sim_matrix.detach().cpu(), nrow=3, padding=20,
                                           normalize=True, pad_value=1), train_step)
                writer.add_scalars('train/batch_pre',
                                   {"pre": float(np.mean(train_pre)), "label": float(np.mean(train_label))},
                                   train_step)
                train_label, train_pre = [], []
                writer.add_scalars('train/batch_MAE', {"MAE": float(b_MAE)}, train_step)
                writer.add_scalars('train/batch_OBO', {"OBO": float(b_OBO)}, train_step)
                writer.add_scalars('train/batch_Loss', {"Loss": float(b_loss)}, train_step)
                writer.add_scalars('train/batch_Loss1', {"Loss1": float(w1 * b_loss1)}, train_step)
                writer.add_scalars('train/batch_Loss2', {"Loss2": float(w2 * b_loss2)}, train_step)
                train_step += 1
            batch_idx += 1
        # valid
        pbar = tqdm(valid_loader, total=len(valid_loader))
        # print("********* Validation *********")
        model.eval()
        batch_idx = 0
        for datas, target1, target2, count in pbar:
            datas = datas.to(device)
            target1 = target1.to(device)  # output: [b,f]
            target2 = target2.to(device)  # output: [b,f]
            count = count.cpu().reshape([-1, 1])
            # 向前传播
            with torch.no_grad():
                y1, y2, embedding_code, sim_matrix = model(datas)  # output : [b,f,period]
                loss1 = criterion1(y1.transpose(1, 2), target1)
                loss2 = criterion2(y2, target2.float())
                loss = w1 * loss1 + w2 * loss2
            MAE, OBO, pre_count = eval_model(y1, y2, count)

            valid_MAE.append(MAE)
            valid_OBO.append(OBO)
            valid_loss.append(float(loss))
            valid_loss1.append(float(w1 * loss1))
            valid_loss2.append(float(w1 * loss2))

            pbar.set_postfix({'Epoch': epoch,
                              'loss': float(np.mean(valid_loss)),
                              'Valid MAE': float(np.mean(valid_MAE)),
                              'Valid OBO': float(np.mean(valid_OBO))}
                             )

            k_batch = int(len(valid_loader) / 5)
            if k_batch <= 0:
                k_batch = 1
            if batch_idx % k_batch == 0 and batch_idx != 0:
                b_loss = np.mean(valid_loss[batch_idx - k_batch:batch_idx])
                b_loss1 = np.mean(valid_loss1[batch_idx - k_batch:batch_idx])
                b_loss2 = np.mean(valid_loss2[batch_idx - k_batch:batch_idx])
                b_MAE = np.mean(valid_MAE[batch_idx - k_batch:batch_idx])
                b_OBO = np.mean(valid_OBO[batch_idx - k_batch:batch_idx])

                writer.add_scalars('valid/batch_MAE', {"MAE": float(b_MAE)}, valid_step)
                writer.add_scalars('valid/batch_OBO', {"OBO": float(b_OBO)}, valid_step)
                writer.add_scalars('valid/batch_Loss', {"Loss": float(b_loss)}, valid_step)
                writer.add_scalars('valid/batch_Loss1', {"Loss1": float(w1 * b_loss1)}, valid_step)
                writer.add_scalars('valid/batch_Loss2', {"Loss2": float(w2 * b_loss2)}, valid_step)
                train_step += 1
            batch_idx += 1

        # per epoch of train and valid over
        scheduler.step()
        ep_pbar.set_postfix({'Epoch': epoch,
                             'loss': float(np.mean(train_loss)),
                             'Tr_MAE': float(np.mean(train_MAE)),
                             'Tr_OBO': float(np.mean(train_OBO)),
                             'ValMAE': float(np.mean(valid_MAE)),
                             'ValOBO': float(np.mean(valid_OBO))})

        # tensorboardX  per epoch
        writer.add_scalars('train/MAE',
                           {"train_MAE": float(np.mean(train_MAE)), "valid_MAE": float(np.mean(valid_MAE))},
                           epoch)
        writer.add_scalars('train/OBO',
                           {"train_OBO": float(np.mean(train_OBO)), "valid_OBO": float(np.mean(valid_OBO))}, epoch)
        writer.add_scalars('train/Loss',
                           {"train_Loss": float(np.mean(train_loss)), "valid_Loss": float(np.mean(valid_loss))}, epoch)
        writer.add_scalars('train/Loss1',
                           {"train_Loss1": float(np.mean(train_loss1)), "valid_Loss1": float(np.mean(valid_loss1))},
                           epoch)
        writer.add_scalars('train/Loss2',
                           {"train_Loss2": float(np.mean(train_loss2)), "valid_Loss2": float(np.mean(valid_loss2))},
                           epoch)
        writer.add_scalars('train/learning rate', {"learning rate": optimizer.state_dict()['param_groups'][0]['lr']},
                           epoch)
        # writer.add_scalars('valid/MAE', {"MAE": float(np.mean(valid_MAE))}, epoch)
        # writer.add_scalars('valid/OBO', {"OBO": float(np.mean(valid_OBO))}, epoch)
        # writer.add_scalars('valid/Loss', {"MSELoss": float(np.mean(val_loss))}, epoch),
        # writer.add_scalars('valid/Loss1', {"MSELoss": float(np.mean(val_loss1))}, epoch),
        # writer.add_scalars('valid/Loss2', {"MSELoss": float(np.mean(val_loss2))}, epoch),

        # save model weights

        saveCkpt = True
        ckpt_name = '/ckpt'
        if saveCkpt and epoch % 3 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLoss': float(np.mean(train_loss)),
                'trainMAE': float(np.mean(train_MAE)),
                'trainOBO': float(np.mean(train_OBO)),
                'valLoss': float(np.mean(valid_loss)),
                'valMAE': float(np.mean(valid_MAE)),
                'valOBO': float(np.mean(valid_OBO)),
            }
            torch.save(checkpoint,
                       CKPT_DIR + ckpt_name + str(epoch) + '_trainMAE_' +
                       str(float(np.around(np.mean(train_MAE), 3))) + '.pt')
    writer.close()
