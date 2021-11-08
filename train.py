import cv2
import torch
import torch.nn as nn
# from build_swrepnet import swrepnet
from build_swcp import swcp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset import MyDataset
import os
import numpy as np
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from tqdm import tqdm
import platform
import random
from torchsummary import summary
import shutil
import ipdb

cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if platform.system() == 'Windows':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device_ids = [0]

else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device_ids = [0, 1, 2, 3]
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)

epoch_size = 5000
FRAME = 128
BATCH_SIZE = 2
random.seed(1)
LR = 6e-6
W1 = 5
W2 = 1
NUM_WORKERS = 32


def eval_model(y1, y2, label_count):
    y1 = torch.argmax(y1, dim=2)  # [b,f]
    pre_count = torch.zeros([y1.shape[0]]).cuda()
    for _ in range(y1.shape[0]):
        for i in range(y1.shape[1]):
            if y1[_][i] != 0:
                pre_count[_] += y2[_][i] / y1[_][i]

    gap_abs = abs(pre_count - label_count) / label_count
    MAE = gap_abs.sum() / len(label_count)
    OBO = float((abs(pre_count - label_count) < 1).sum())
    # for i in range(len(label_count)):
    #     if abs(pre_count - label_count)[i] < 1:
    #         OBO += 1
    OBO /= float(len(label_count))
    return float(MAE), OBO, pre_count.cpu().detach().numpy()


if __name__ == '__main__':

    model = swcp(frame=FRAME)
    model = torch.nn.DataParallel(model.to(device), device_ids=device_ids)
    # model = MMDataParallel(model.to(device), device_ids=device_ids)

    data_dir1 = r'/p300/LSP'
    data_dir2 = r'./data/LSP'
    if os.path.exists(data_dir1):
        data_root = r'/p300/LSP'
    elif os.path.exists(data_dir2):
        data_root = r'./data/LSP'
    else:
        raise ValueError('NO data root')

    train_label = r'train.csv'
    valid_label = r'valid.csv'

    train_dataset = MyDataset(root_dir=data_root, label_dir=train_label, frames=FRAME, method='train')
    valid_dataset = MyDataset(root_dir=data_root, label_dir=valid_label, frames=FRAME, method='valid')

    train_loader = DataLoader(dataset=train_dataset, pin_memory=True, batch_size=BATCH_SIZE, drop_last=False,
                              shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, pin_memory=True, batch_size=BATCH_SIZE, drop_last=False,
                              shuffle=True, num_workers=NUM_WORKERS)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCEWithLogitsLoss()
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

    # lastCkptPath = '/p300/SWRNET/checkpoint/ckpt350_trainMAE_0.8896425843327483.pt'
    # lastCkptPath = '/p300/SWRNET/checkpoint2/ckpt15_trainMAE_1.498641728136049.pt'
    lastCkptPath = None
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
        # sw_path = r'swt_log/swin_base_patch4_window12_384_22k.pth'
        sw_path = r'swt_log/swin_tiny_patch4_window7_224.pth'
        pre_dict = torch.load(sw_path)['model']
        new_dict = {}
        for k, v in pre_dict.items():
            if k != 'head.weight' and k != 'head.bias':  # 舍弃head 层
                key = 'module.sw_1.' + k
                new_dict[key] = v
            # model.state_dict()[key] = v
        model.load_state_dict(new_dict, strict=False)

    # print("********* Training begin *********")
    ep_pbar = tqdm(range(currEpoch, epoch_size))
    for epoch in ep_pbar:
        # save evaluation metrics
        train_MAE, train_OBO, valid_MAE, valid_OBO, avg_loss, avg_loss1, avg_loss2 = [], [], [], [], [], [], []
        val_loss1, val_loss2, val_loss = [], [], []
        train_pre, train_label, train_gap = [], [], []
        pbar = tqdm(train_loader, total=len(train_loader))
        batch_idx = 0
        model.train()
        for datas, target1, target2, count in pbar:
            # ipdb.set_trace()
            # print('///////////////////////{} begin to train/////////////////////////////'.format(batch_idx))
            # torch.cuda.empty_cache()
            datas = datas.to(device)
            target1 = target1.to(device)  # output: [b,f]
            target2 = target2.to(device)  # output: [b,f]
            count = count.to(device).reshape([-1, 1])
            optimizer.zero_grad()

            # 向前传播
            with autocast():

                y1, y2 = model(datas)  # output : [b,f,period]
                loss1 = criterion1(y1.transpose(1, 2), target1)
                loss2 = criterion2(y2, target2.float())
                loss = w1 * loss1 + w2 * loss2
            # 反向传播 scaler 放大梯度
            scaler.scale(loss.float()).backward()
            scaler.step(optimizer)
            scaler.update()

            # 输出结果
            MAE, OBO, pre_count = eval_model(y1, y2, count)
            count = count.cpu()
            train_label.append(count[0])
            train_pre.append(pre_count[0])
            train_gap.append(pre_count[0] - count[0])

            train_MAE.append(MAE)
            train_OBO.append(OBO)
            avg_loss.append(float(loss))
            avg_loss1.append(float(loss1))
            avg_loss2.append(float(loss2))

            if batch_idx % 10 == 0:
                pbar.set_postfix({'Epoch': epoch,
                                  'loss': float(np.mean(avg_loss)),
                                  'Train MAE': float(np.mean(train_MAE)),
                                  'Train OBO': float(np.mean(train_OBO)),
                                  'lr': optimizer.state_dict()['param_groups'][0]['lr']})
            if batch_idx % 20 == 0:
                writer.add_scalars('train/batch_pre',
                                   {"pre": float(np.mean(train_pre)), "label": float(np.mean(train_label))},
                                   epoch * len(train_loader) + batch_idx / 20)
                train_label, train_pre = [], []
                # writer.add_scalars('train/batch_label', {"MAE": float(MAE)}, epoch * len(train_loader)+ batch_idx)
            writer.add_scalars('train/batch_MAE', {"MAE": float(MAE)}, epoch * len(train_loader) + batch_idx)
            writer.add_scalars('train/batch_OBO', {"OBO": float(OBO)}, epoch * len(train_loader) + batch_idx)
            writer.add_scalars('train/batch_Loss', {"Loss": float(loss)}, epoch * len(train_loader) + batch_idx)
            writer.add_scalars('train/batch_Loss1', {"Loss1": float(loss1)}, epoch * len(train_loader) + batch_idx)
            writer.add_scalars('train/batch_Loss2', {"Loss2": float(loss2)}, epoch * len(train_loader) + batch_idx)
            batch_idx += 1
            break
        # valid
        pbar = tqdm(valid_loader, total=len(valid_loader))
        # print("********* Validation *********")
        model.eval()
        batch_idx = 0
        for datas, target1, target2, count in pbar:
            datas = datas.to(device)
            target1 = target1.to(device)  # output: [b,f]
            target2 = target2.to(device)  # output: [b,f]
            count = count.to(device).reshape([-1, 1])
            # 向前传播
            with torch.no_grad():
                y1, y2 = model(datas)  # output : [b,f,period]
                loss1 = criterion1(y1.transpose(1, 2), target1)
                loss2 = criterion2(y2, target2.float())
                loss = w1 * loss1 + w2 * loss2
            MAE, OBO, pre_count = eval_model( y1, y2, count)
            count = count.cpu()
            valid_MAE.append(MAE)
            valid_OBO.append(OBO)

            val_loss.append(float(loss))
            val_loss1.append(float(loss1))
            val_loss2.append(float(loss2))
            pbar.set_postfix({'Epoch': epoch,
                              'loss': float(loss),
                              'Valid MAE': MAE,
                              'Valid OBO': OBO})

            writer.add_scalars('valid/batch_MAE', {"MAE": float(MAE)}, epoch * len(valid_loader) + batch_idx)
            writer.add_scalars('valid/batch_OBO', {"OBO": float(OBO)}, epoch * len(valid_loader) + batch_idx)
            writer.add_scalars('valid/batch_Loss', {"Loss": float(loss)}, epoch * len(valid_loader) + batch_idx)
            writer.add_scalars('valid/batch_Loss1', {"Loss1": float(loss1)}, epoch * len(valid_loader) + batch_idx)
            writer.add_scalars('valid/batch_Loss2', {"Loss2": float(loss2)}, epoch * len(valid_loader) + batch_idx)

            batch_idx += 1

        # one epoch train and valid over
        ep_pbar.set_postfix({'Epoch': epoch,
                             'loss': float(np.mean(avg_loss)),
                             'Tr_MAE': np.mean(train_MAE),
                             'Tr_OBO': np.mean(train_OBO),
                             'ValMAE': np.mean(valid_MAE),
                             'ValOBO': np.mean(valid_OBO)})

        # tensorboardX
        writer.add_scalars('train/MAE', {"MAE": float(np.mean(train_MAE))}, epoch)
        writer.add_scalars('train/OBO', {"OBO": float(np.mean(train_OBO))}, epoch)
        writer.add_scalars('train/Loss', {"Loss": float(np.mean(avg_loss))}, epoch)
        writer.add_scalars('train/Loss1', {"Loss1": float(np.mean(avg_loss1))}, epoch)
        writer.add_scalars('train/Loss2', {"Loss2": float(np.mean(avg_loss2))}, epoch)

        writer.add_scalars('valid/MAE', {"MAE": float(np.mean(valid_MAE))}, epoch)
        writer.add_scalars('valid/OBO', {"OBO": float(np.mean(valid_OBO))}, epoch)
        writer.add_scalars('valid/Loss', {"MSELoss": float(np.mean(val_loss))}, epoch),
        writer.add_scalars('valid/Loss1', {"MSELoss": float(np.mean(val_loss1))}, epoch),
        writer.add_scalars('valid/Loss2', {"MSELoss": float(np.mean(val_loss2))}, epoch),

        writer.add_scalars('train/learning rate', {"learning rate": optimizer.state_dict()['param_groups'][0]['lr']},
                           epoch)
        # save model weights
        saveCkpt = True
        ckpt_name = 'ckpt'
        if saveCkpt and epoch % 3 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLoss': float(np.mean(avg_loss)),
                'trainMAE': float(np.mean(train_MAE)),
                'trainOBO': float(np.mean(train_OBO)),
                'valLoss': float(np.mean(val_loss)),
                'valMAE': float(np.mean(valid_MAE)),
                'valOBO': float(np.mean(valid_OBO)),
            }
            torch.save(checkpoint,
                       'checkpoint2/' + ckpt_name + str(epoch) + '_trainMAE_' +
                       str(float(np.mean(train_MAE))) + '.pt')
    writer.close()
