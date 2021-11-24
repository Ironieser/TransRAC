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

from torchvision.transforms import ToTensor

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
NUM_WORKERS = 20
LOG_DIR = './repnetMDM_log1124_3'
NPZ = True
CKPT_DIR = './ckp_repnetMDM_3'
P = 0.2
# lastCkptPath = '/p300/SWRNET/ckp_repDnet_6/ckpt99_trainMAE_0.288.pt'
lastCkptPath = None
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
    # pre_count = torch.sum(y1, dim=1).detach().cpu()  # [b]
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

    train_label = r'train.csv'
    valid_label = r'valid.csv'

    train_dataset = MyDataset(root_dir=data_root, label_dir=train_label, frames=FRAME, method='train')
    valid_dataset = MyDataset(root_dir=data_root, label_dir=valid_label, frames=FRAME, method='valid')
    # test_dataset = MyDataset(root_dir=data_root, label_dir=valid_label, frames=FRAME, method='test')

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
    # optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': 1e-5}], lr=LR)
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': 6e-6}], weight_decay=0.01, lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=currEpoch - 1)
    # print("********* Training begin *********")
    ep_pbar = tqdm(range(currEpoch, epoch_size))
    train_step = currTrainstep
    valid_step = currValidationstep
    for epoch in ep_pbar:
        # save evaluation metrics
        train_MAE, train_OBO, valid_MAE, valid_OBO, train_loss, train_loss1, train_loss2, train_loss3 = \
            [], [], [], [], [], [], [], []
        valid_loss1, valid_loss2, valid_loss3, valid_loss = [], [], [], []
        train_pre, train_label, train_gap = [], [], []
        pbar = tqdm(train_loader, total=len(train_loader))
        batch_idx = 0
        model.train()
        w1, w2, w3 = W1, W2, W3
        for datas, target1, count1, target2, count2, target3, count3 in pbar:
            # ipdb.set_trace()

            datas = datas.to(device)
            target1 = target1.to(device)  # output: [b,f]
            count1 = count1.cpu().reshape([-1, 1])
            target2 = target2.to(device)  # output: [b,f]
            count2 = count2.cpu().reshape([-1, 1])
            target3 = target3.to(device)  # output: [b,f]
            count3 = count3.cpu().reshape([-1, 1])
            optimizer.zero_grad()

            # 向前传播
            with autocast():
                y1, y2, y3, embedding_code, sim_matrix = model(
                    datas, epoch)  # output : y1 :[b,f]  sim_matrix:[b,1,f,f]
                loss1 = criterion1(y1, target1.float())  # y = [b,f], target: [b,f]
                loss2 = criterion1(y2, target1.float())  # y = [b,f], target: [b,f]
                loss3 = criterion1(y3, target1.float())  # y = [b,f], target: [b,f]
            loss = w1 * loss1 + w2 * loss2 + w3 * loss3
            # 反向传播 scaler 放大梯度
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

            # if batch_idx % 10 == 0:
            pbar.set_postfix({'Epoch': epoch,
                              'loss': float(np.mean(train_loss)),
                              'Train MAE': float(np.mean(train_MAE)),
                              'Train OBO': float(np.mean(train_OBO)),
                              'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            k_batch = int(len(train_loader) / 10) if int(len(train_loader) / 10) > 0 else 1
            if batch_idx % k_batch == 0 and batch_idx != 0:
                b_loss = np.mean(train_loss[batch_idx - k_batch:batch_idx])
                b_loss1 = np.mean(train_loss1[batch_idx - k_batch:batch_idx])
                b_loss2 = np.mean(train_loss2[batch_idx - k_batch:batch_idx])
                b_loss3 = np.mean(train_loss3[batch_idx - k_batch:batch_idx])
                b_MAE = np.mean(train_MAE[batch_idx - k_batch:batch_idx])
                b_OBO = np.mean(train_OBO[batch_idx - k_batch:batch_idx])
                sim_img = sm_heat_map.get_sm_hm(sim_matrix.detach().cpu(), pre_count, count.view(-1).numpy())
                writer.add_image('sim_matrix/sim_img', sim_img, train_step)
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

                train_step += 1
            batch_idx += 1
        # valid
        pbar = tqdm(valid_loader, total=len(valid_loader))
        # print("********* Validation *********")
        model.eval()
        batch_idx = 0
        for datas, target1, count1, target2, count2, target3, count3 in pbar:
            datas = datas.to(device)
            target1 = target1.to(device)  # output: [b,f]
            count1 = count1.cpu().reshape([-1, 1])
            target2 = target2.to(device)  # output: [b,f]
            count2 = count2.cpu().reshape([-1, 1])
            target3 = target3.to(device)  # output: [b,f]
            count3 = count3.cpu().reshape([-1, 1])
            optimizer.zero_grad()
            # 向前传播
            with torch.no_grad():
                y1, y2, y3, embedding_code, sim_matrix = model(
                    datas, epoch)  # output : y1 :[b,f]  sim_matrix:[b,1,f,f]
                loss1 = criterion1(y1, target1.float())  # y = [b,f], target: [b,f]
                loss2 = criterion1(y2, target1.float())  # y = [b,f], target: [b,f]
                loss3 = criterion1(y3, target1.float())  # y = [b,f], target: [b,f]
            loss = w1 * loss1 + w2 * loss2 + w3 * loss3
            pre_count1 = torch.sum(y1, dim=1).detach().cpu()
            pre_count2 = torch.sum(y2, dim=1).detach().cpu()
            pre_count3 = torch.sum(y3, dim=1).detach().cpu()
            pre_count = (w1 * pre_count1 + w2 * pre_count2 + w3 * pre_count3) / (w1 + w2 + w3)
            count = (w1 * count1 + w2 * count2 + w3 * count3) / (w1 + w2 + w3)
            MAE, OBO = eval_model(pre_count, count)
            pre_count = pre_count.numpy()
            valid_MAE.append(MAE)
            valid_OBO.append(OBO)
            valid_loss.append(float(loss))
            valid_loss1.append(float(w1 * loss1))
            valid_loss2.append(float(w1 * loss2))
            valid_loss3.append(float(w1 * loss3))

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
                b_loss3 = np.mean(valid_loss3[batch_idx - k_batch:batch_idx])
                b_MAE = np.mean(valid_MAE[batch_idx - k_batch:batch_idx])
                b_OBO = np.mean(valid_OBO[batch_idx - k_batch:batch_idx])

                writer.add_scalars('valid/batch_MAE', {"MAE": float(b_MAE)}, valid_step)
                writer.add_scalars('valid/batch_OBO', {"OBO": float(b_OBO)}, valid_step)
                writer.add_scalars('valid/batch_Loss', {"Loss": float(b_loss)}, valid_step)
                writer.add_scalars('valid/batch_Loss1', {"Loss1": float(b_loss1)}, valid_step)
                writer.add_scalars('valid/batch_Loss2', {"Loss2": float(b_loss2)}, valid_step)
                writer.add_scalars('valid/batch_Loss3', {"Loss3": float(b_loss3)}, valid_step)
                valid_step += 1
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
        writer.add_scalars('train/Loss3',
                           {"train_Loss3": float(np.mean(train_loss3)), "valid_Loss3": float(np.mean(valid_loss3))},
                           epoch)
        writer.add_scalars('train/learning rate', {"learning rate": optimizer.state_dict()['param_groups'][0]['lr']},
                           epoch)

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
                'train_step': train_step,
                'valid_step': valid_step
            }
            torch.save(checkpoint,
                       CKPT_DIR + ckpt_name + str(epoch) + '_trainMAE_' +
                       str(float(np.around(np.mean(train_MAE), 3))) + '.pt')
    writer.close()
