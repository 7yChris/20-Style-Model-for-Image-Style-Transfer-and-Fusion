# -*- coding: UTF-8 -*-
from PIL import Image
import numpy as np
import scipy.misc as misc
import os


# backward需要用到的3个函数
filenames = []
# 内容图片的提取
def random_batch(path, batch_size, shape):
    # 列出文件路径
    global filenames
    if len(filenames) == 0:
        filenames = os.listdir(path)
    # 在0和len间随机取两个数，即随机给出两个图片编号
    rand_samples = np.random.randint(0, len(filenames), [batch_size])
    # 初始化存储两张图片的矩阵
    batch = np.zeros([batch_size, shape[0], shape[1], shape[2]])
    c = 0
    # 按照随机给出的图片编号，读取图片
    # 并进行裁剪、放缩
    for idx in rand_samples:
        try:
            batch[c, :, :, :] = misc.imresize(crop(np.array(Image.open(path + filenames[idx]))), [shape[0], shape[1]])
        except:
            img = crop(np.array(Image.open(path + filenames[0])))
            batch[c, :, :, :] = misc.imresize(img, [shape[0], shape[1]])
        c += 1
    return batch


# 风格图片的提取
def random_select_style(path, batch_size, shape, c_nums):
    # 列出风格图片文件名
    filenames = os.listdir(path)
    # 随机选出一张风格图片
    rand_sample = np.random.randint(0, len(filenames))
    # 读取风格图片，并进行裁剪、resize
    img = misc.imresize(crop(np.array(Image.open(path + str(rand_sample + 1) + ".png"))), [shape[0], shape[1]])
    # 初始化风格图片存储矩阵
    batch = np.zeros([batch_size, shape[0], shape[1], shape[2]])
    # 标记选中了哪张风格图片
    y = np.zeros([1, c_nums])
    y[0, rand_sample] = 1
    # 风格图片存储矩阵，存储batch_size个相同的风格图片
    for i in range(batch_size):
        batch[i, :, :, :] = img[:, :, :3]
    return batch, y


# 图片剪裁
def crop(img):
    # 得到图像的高
    h = img.shape[0]
    # 得到图像的宽
    w = img.shape[1]

    # 如果是长方形，则进行随机裁剪，使之介于正方形与原始图像尺寸之间
    if h < w:
        x = 0
        y = np.random.randint(0, w - h + 1)
        length = h
    elif h > w:
        x = np.random.randint(0, h - w + 1)
        y = 0
        length = w

    # 如果是正方形，则不进行裁剪
    else:
        x = 0
        y = 0
        length = h
    return img[x:x + length, y:y + length, :]
