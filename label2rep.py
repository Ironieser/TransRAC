import math
import numpy as np


def rep_label(y_frame, y_length):
    """

    Args:
        y_frame: [2,5,5,8,9,12]  -> [2,4] [5,8],[9,11]
        e.g.
            if [2,3,3,8] => [2,2],[3,8]
            if [2,2,2,3] => [2,2],[3,3]
            else [2,2,2,2] => [2,2]
        y_length: 总帧数
    Returns:
        y1: 0 - y_length
        y2: 0 - 1
    """
    y1 = np.zeros(y_length, dtype=float)  # 坐标轴长度，即帧数
    y2 = np.zeros(y_length, dtype=float)
    # for i in range(0,y_length.size,2):
    for i in range(0, y_frame.size, 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]
        if i + 2 < y_frame.size:
            if x_b == y_frame[i + 2]:
                if x_a != x_b:
                    x_b -= 1
                elif y_frame[i + 2] != y_frame[i + 3]:
                    y_frame[i + 2] += 1
                else:
                    continue
        p = x_b - x_a + 1
        for j in range(x_a, x_b + 1):
            y1[j] = p
            y2[j] = 1
    #     avg = (x_b + x_a) / 2
    #     sig = (x_b - x_a) / 6
    #     num = x_b - x_a + 1  # 帧数量
    #     if num != 1:
    #         for j in range(num):
    #             x_1 = x_a - 0.5 + j
    #             x_2 = x_a + 0.5 + j
    #
    #     else:
    #         y_label[x_a] = 1
    period = 0
    for i in range(len(y1)):
        if y1[i] != 0:
            period += y2[i] / y1[i]
    return y1, y2, period


# # # # test
# y_frame = np.array([4, 4, 4, 4, 4, 5, 13, 15, 15, 16])  # 关键帧
# # # # # y_frame = np.array([1067, 3303, 3303, 15, 15, 22, 25, 40])  # 关键帧
# y1, y2 , c = rep_label(y_frame, 18)
# import torch
#
# y1_tensor = torch.FloatTensor(y1)
# y2_tensor = torch.FloatTensor(y2)

# # print(y)
# #
