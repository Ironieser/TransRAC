# %%

from scipy import integrate
import math
import numpy as np


# 正态分布概率密度函数
def PDF(x, u, sig):
    return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)


# 概率密度函数积分
def get_integrate(x_1, x_2, avg, sig):
    y, err = integrate.quad(PDF, x_1, x_2, args=(avg, sig))
    return y


def normalize_label(y_frame, y_length):
    """

    Args:
        y_frame: 帧
        y_length: 帧总数

    Returns:
        y_label:  size:narray(y_length,)
        num_period: the period calculated by normal
    """

    y_label = np.zeros(y_length, dtype=float)  # 坐标轴长度，即帧数
    for i in range(0, y_frame.size, 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]

        avg = (x_b + x_a) / 2
        sig = (x_b - x_a) / 6
        num = x_b - x_a + 1  # 帧数量
        if num != 1:
            for j in range(num):
                x_1 = x_a - 0.5 + j
                x_2 = x_a + 0.5 + j
                # TODO： 优化函数，时间复杂度降低为O(1)
                y_ing = get_integrate(x_1, x_2, avg, sig)
                y_label[x_a + j] += y_ing
        else:
            y_label[x_a] += 1
    num_period = 0
    for i in range(len(y_label)):
        num_period += y_label[i]
    return y_label, num_period


def normalize_label2(y_frame, y_length):
    """

    Args:
        y_frame: 帧
        y_length: 帧总数

    Returns:
        y_label:  size:narray(y_length,)
        num_period: the period calculated by normal
    """

    y_label = np.zeros(y_length, dtype=float)  # 坐标轴长度，即帧数
    for i in range(0, y_frame.size, 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]

        # avg = (x_b + x_a) / 2
        avg = x_a
        sf = math.ceil(x_a - (x_b - x_a) / 2)
        ed = math.floor(x_a + (x_b - x_a) / 2)
        sig = (ed - sf) / 6
        num = ed - sf + 1
        if num != 1:
            if sf < 0:
                s_ing = get_integrate(sf - 1, -0.5, avg, sig)
                add_s_ing = s_ing / (ed - 0 + 1)
                sf = 0
            else:
                add_s_ing = 0
            for j in range(ed - sf + 1):
                x_1 = sf - 0.5 + j
                x_2 = sf + 0.5 + j
                # TODO： 优化函数，时间复杂度降低为O(1)
                y_ing = get_integrate(x_1, x_2, avg, sig)
                y_label[sf + j] += y_ing + add_s_ing
        else:
            y_label[x_a] += 1
    num_period = 0
    for i in range(len(y_label)):
        num_period += y_label[i]
    return y_label, num_period


def normalize_label3(y_frame, y_length):
    """

    Args:
        y_frame: 帧
        y_length: 帧总数

    Returns:
        y_label:  size:narray(y_length,)
        num_period: the period calculated by normal
    """
    y_label = np.zeros(y_length, dtype=float)  # 坐标轴长度，即帧数
    for i in range(0, y_frame.size, 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]

        # avg = (x_b + x_a) / 2
        avg = x_b
        sf = math.ceil(x_b - (x_b - x_a) / 2)
        ed = math.floor(x_b + (x_b - x_a) / 2)
        sig = (ed - sf) / 6
        num = ed - sf + 1
        if num != 1:
            if ed > y_length - 1:
                s_ing = get_integrate(y_length - 1 + 0.5, ed + 1, avg, sig)
                add_s_ing = s_ing / (y_length - 1 - sf + 1)
                ed = y_length - 1
            else:
                add_s_ing = 0
            for j in range(ed - sf + 1):
                x_1 = sf - 0.5 + j
                x_2 = sf + 0.5 + j
                # TODO： 优化函数，时间复杂度降低为O(1)
                y_ing = get_integrate(x_1, x_2, avg, sig)
                y_label[sf + j] += y_ing + add_s_ing
        else:
            y_label[x_a] += 1
    num_period = 0
    for i in range(len(y_label)):
        num_period += y_label[i]
    return y_label, num_period

# if __name__ == '__main__':
#     y_frame = np.array([0, 8, 8, 9, 9, 10])
#     y = normalize_label2(y_frame, 10)

# # # test
# y_frame = np.array([4, 4, 6, 8, 9, 12, 13, 15, 15, 16])  # 关键帧
# # # y_frame = np.array([1067, 3303, 3303, 15, 15, 22, 25, 40])  # 关键帧
# y = normalize_label(y_frame, 18)
# # print(y)
# #
