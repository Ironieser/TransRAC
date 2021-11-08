# %%


import math
import numpy as np


def rep_label(y_frame, y_length):
    """

    Args:
        y_frame: [2,5,5,8,9,12]
        y_length: 总帧数
    Returns:
        y1: 0 - 1
        y2: 0 - y_length
    """
    y1 = np.zeros(y_length, dtype=float)  # 坐标轴长度，即帧数
    y2 = np.zeros(y_length, dtype=float)
    # for i in range(0,y_length.size,2):

    for i in range(0, y_frame.size, 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]
        p = x_b - x_a + 1
        for j in range(x_a, x_b + 1):
            y1[j] = 1
            y2[j] += 1 / p
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
    return y1, y2

# # # test
y_frame = np.array([4, 4, 6, 8, 9, 12, 13, 15, 15, 16])  # 关键帧
# # # y_frame = np.array([1067, 3303, 3303, 15, 15, 22, 25, 40])  # 关键帧
y = rep_label(y_frame, 18)
print(y)
# # print(y)
# #
