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

# from tool import data_utils

# from torchsummary import summary
# import shutil
import ipdb

if platform.system() == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]
    device = torch.device('cpu')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device_ids = [0, 1]
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
W2 = 1
W3 = 1
NUM_WORKERS = 2
LOG_DIR = './repnetMDM_log1124_4'
NPZ = True
CKPT_DIR = './ckp_repnetMDM_4'
P = 0.2
# lastCkptPath = '/p300/SWRNET/ckp_repDnet_6/ckpt99_trainMAE_0.288.pt'
lastCkptPath = None
if platform.system() == 'Windows':
    NUM_WORKERS = 0
    lastCkptPath = None
    BATCH_SIZE = 2


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
    # pre_count = torch.sum(y1, dim=1).detach().cpu()  # [b]
    label_count = label_count.cpu().view(-1)

    gap_abs = abs((pre_count - label_count) / label_count)
    MAE = gap_abs.sum() / len(label_count)
    OBO = float((abs(pre_count - label_count) < 1).sum() / len(label_count))

    return float(MAE), OBO


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

    train_label = r'test.csv'
    # ipdb.set_trace()
    t1 = time.time()
    print('init dataset')
    train_dataset = MyDataset(root_dir=data_root, label_dir=train_label, frames=FRAME, method='test')
    t2 = time.time()
    print('init dataloader')
    train_loader = DataLoader(dataset=train_dataset, pin_memory=True, batch_size=BATCH_SIZE,
                              drop_last=False, shuffle=True, num_workers=NUM_WORKERS)
    t3 = time.time()

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

    print('t2-t1={},t3-t2={}'.format(t2 - t1, t3 - t2))
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
    criterion1 = nn.MSELoss()
    scaler = GradScaler()

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': 6e-6}], weight_decay=0.01, lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=currEpoch - 1)
    print("********* Training begin *********")
    ep_pbar = tqdm(range(currEpoch, epoch_size))
    train_step = currTrainstep
    valid_step = currValidationstep

    for epoch in ep_pbar:
        # save evaluation metrics
        train_MAE, train_OBO, valid_MAE, valid_OBO, train_loss, train_loss1, train_loss2, train_loss3 = \
            [], [], [], [], [], [], [], []
        valid_loss1, valid_loss2, valid_loss3, valid_loss = [], [], [], []
        train_pre, train_label, train_gap = [], [], []
        # pbar = tqdm(train_loader, total=len(train_loader))
        batch_idx = 0
        model.train()
        w1, w2, w3 = W1, W2, W3
        tag = 0
        for datas, target1, count1, target2, count2, target3, count3 in train_loader:
            # ipdb.set_trace()
            if tag % 2 == 0:
                ts = time.time()
            else:
                td = time.time()
                print('td-ts:', td - ts)
            tag += 1
            datas = datas.to(device)
            target1 = target1.to(device)  # output: [b,f]
            count1 = count1.cpu().reshape([-1, 1])
            target2 = target2.to(device)  # output: [b,f]
            count2 = count2.cpu().reshape([-1, 1])
            target3 = target3.to(device)  # output: [b,f]
            count3 = count3.cpu().reshape([-1, 1])
            optimizer.zero_grad()
            with autocast():

                y1, y2, y3, embedding_code, sim_matrix = model(
                    datas, epoch)  # output : y1 :[b,f]  sim_matrix:[b,1,f,f]
                loss1 = criterion1(y1, target1.float())  # y = [b,f], target: [b,f]
                loss2 = criterion1(y2, target1.float())  # y = [b,f], target: [b,f]
                loss3 = criterion1(y3, target1.float())  # y = [b,f], target: [b,f]
            loss = w1 * loss1 + w2 * loss2 + w3 * loss3

            if platform.system() != 'Windows':
                scaler.scale(loss.float()).backward()
                scaler.step(optimizer)
                scaler.update()
            # 输出结果
            pre_count1 = torch.sum(y1, dim=1).detach().cpu()
            pre_count2 = torch.sum(y2, dim=1).detach().cpu()
            pre_count3 = torch.sum(y3, dim=1).detach().cpu()
            pre_count = (w1 * pre_count1 + w2 * pre_count2 + w3 * pre_count3) / (w1 + w2 + w3)
            # ipdb.set_trace()
            count = (w1 * count1 + w2 * count2 + w3 * count3) / (w1 + w2 + w3)

            MAE, OBO = eval_model(pre_count, count)
            pre_count = pre_count.numpy()

            train_label.append(count.numpy()[0])
            train_pre.append(pre_count[0])
            train_gap.append(pre_count[0] - count.numpy()[0])

            train_MAE.append(MAE)
            train_OBO.append(OBO)
            train_loss.append(float(loss))
            train_loss1.append(float(w1 * loss1))
            train_loss2.append(float(w2 * loss2))
            train_loss3.append(float(w2 * loss3))
            # t3 = time.time()
            # if batch_idx % 10 == 0:
            # pbar.set_postfix({'Epoch': epoch,
            #                   'loss': float(np.mean(train_loss)),
            #                   'Train MAE': float(np.mean(train_MAE)),
            #                   'Train OBO': float(np.mean(train_OBO)),
            #                   'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            t1 = time.time()
            length = int(len(train_loader) / 10)
            # t2 = time.time()

            if length > 0:
                k_batch = length
            else:
                k_batch = 1
            # t3 = time.time()
            # if batch_idx % k_batch == 0 and batch_idx != 0:
            b_loss = np.mean(train_loss[batch_idx - k_batch:batch_idx])
            b_loss1 = np.mean(train_loss1[batch_idx - k_batch:batch_idx])
            b_loss2 = np.mean(train_loss2[batch_idx - k_batch:batch_idx])
            b_loss3 = np.mean(train_loss3[batch_idx - k_batch:batch_idx])
            b_MAE = np.mean(train_MAE[batch_idx - k_batch:batch_idx])

            b_OBO = np.mean(train_OBO[batch_idx - k_batch:batch_idx])
            # t4 = time.time()
            sim_img = sm_heat_map.get_sm_hm(sim_matrix.detach().cpu(), pre_count, count.view(-1).numpy())
            # t5 = time.time()
            # writer.add_image('sim_matrix/sim_img', sim_img, train_step)
            writer.add_scalars('train/batch_pre',
                               {"pre": float(np.mean(train_pre)), "label": float(np.mean(train_label))},
                               train_step)
            writer.add_scalars('train/batch_gap',
                               {"gap": float(np.mean(train_gap))}, train_step)
            train_label, train_pre, train_gap = [], [], []
            writer.add_scalars('train/batch_MAE', {"MAE": float(b_MAE)}, train_step)
            writer.add_scalars('train/batch_OBO', {"OBO": float(b_OBO)}, train_step)
            writer.add_scalars('train/batch_Loss', {"Loss": float(b_loss)}, train_step)
            writer.add_scalars('train/batch_Loss1', {"Loss1": float(b_loss1)}, train_step)
            writer.add_scalars('train/batch_Loss2', {"Loss2": float(b_loss2)}, train_step)
            writer.add_scalars('train/batch_Loss3', {"Loss3": float(b_loss3)}, train_step)
            t6 = time.time()
            train_step += 1
            batch_idx += 1
            print('t61:', t6 - t1)
            # print('tag:{},t21={},t32={},t43={},t54={},t65={}'.format(tag, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5))
