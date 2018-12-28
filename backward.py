# -*- coding: UTF-8 -*-

from forward import vggnet, forward, content_loss, style_loss
import tensorflow as tf

from generateds import get_content_tfrecord
from utils import random_batch, random_select_style
from PIL import Image
import numpy as np
import scipy.misc as misc
import argparse
import time
import os

# 初始化各种参数
parser = argparse.ArgumentParser()
# 输入图像尺寸
parser.add_argument("--IMG_H", type=int, default=256)
parser.add_argument("--IMG_W", type=int, default=256)
parser.add_argument("--IMG_C", type=int, default=3)
# 风格图像尺寸
parser.add_argument("--STYLE_H", type=int, default=512)
parser.add_argument("--STYLE_W", type=int, default=512)
# 风格图像张数
parser.add_argument("--C_NUMS", type=int, default=10)
# Batch大小，默认为2
parser.add_argument("--BATCH_SIZE", type=int, default=2)
# 学习率
parser.add_argument("--LEARNING_RATE", type=float, default=0.001)
# 内容权重和风格权重
parser.add_argument("--CONTENT_WEIGHT", type=float, default=1.0)
parser.add_argument("--STYLE_WEIGHT", type=float, default=5.0)
# 训练内容图像路径，train2014
parser.add_argument("--PATH_CONTENT", type=str, default=".MSCOCO")
# 风格图像路径
parser.add_argument("--PATH_STYLE", type=str, default="./style_imgs/")
# 生成模型路径
parser.add_argument("--PATH_MODEL", type=str, default="./save_para/")
# VGG16路径
parser.add_argument("--PATH_VGG16", type=str, default="./vgg_para/")
# 训练轮数
parser.add_argument("--steps", type=int, default=50000)
args = parser.parse_args()


def backward(IMG_H=256, IMG_W=256, IMG_C=3, STYLE_H=512, STYLE_W=512, C_NUMS=10, batch_size=2, learning_rate=0.001,
             content_weight=1.0, style_weight=5.0, path_content="./MSCOCO/", path_style="./style_imgs/",
             model_path="./save_para/", vgg_path="./vgg_para/", path_data='./data',
             dataset_name='coco_train.tfrecords'):
    # 内容图像：batch为2，图像大小为256*256*3
    content = tf.placeholder(tf.float32, [batch_size, IMG_H, IMG_W, IMG_C])
    # 风格图像：batch为2，图像大小为512*512*3
    style = tf.placeholder(tf.float32, [batch_size, STYLE_H, STYLE_W, IMG_C])
    # 风格1：喂入一种风格的标签
    y = tf.placeholder(tf.float32, [1, C_NUMS])
    # 风格2：0
    y_1 = tf.zeros([1, C_NUMS])
    # 风格3：0
    y_2 = tf.zeros([1, C_NUMS])
    # 风格4：0
    y_3 = tf.zeros([1, C_NUMS])
    # 初始化三个alpha
    alpha1 = tf.constant([1.])
    alpha2 = tf.constant([1.])
    alpha3 = tf.constant([1.])

    # 图像生成网络：前向传播
    target = forward(content, y, y_1, y_2, y_3, alpha1, alpha2, alpha3, True)
    # 生成图像、内容图像、风格图像特征提取
    Phi_T = vggnet(target, vgg_path)
    Phi_C = vggnet(content, vgg_path)
    Phi_S = vggnet(style, vgg_path)
    # Loss计算
    # 总Loss
    Loss = content_loss(Phi_C, Phi_T) * content_weight + style_loss(Phi_S, Phi_T) * style_weight
    # 风格Loss
    Style_loss = style_loss(Phi_S, Phi_T)
    # 内容Loss
    Content_loss = content_loss(Phi_C, Phi_T)

    # 定义当前训练轮数变量
    global_step = tf.Variable(0, trainable=False)

    # 优化器：Adam优化器，损失最小化
    Opt = tf.train.AdamOptimizer(learning_rate).minimize(Loss, global_step=global_step)

    content_batch = get_content_tfrecord(batch_size, os.path.join(path_data, dataset_name), IMG_H)

    # 实例化saver对象，便于之后保存模型
    saver = tf.train.Saver()
    # 开始计时
    time_start = time.time()

    with tf.Session() as sess:
        # 初始化全局变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 在路径中查询有无checkpoint
        ckpt = tf.train.get_checkpoint_state(model_path)
        # 从checkpoint恢复模型
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Restore Model Successfully')

        # 开启多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 开始训练，共进行5万轮
        for itr in range(args.steps):
            # 计时
            time_step_start = time.time()

            # 随机读取batch_size张内容图片，存储在四维矩阵中（batch_size*h*w*c）
            # batch_content = random_batch(path_content, batch_size, [IMG_H, IMG_W, IMG_C])
            batch_content = sess.run(content_batch)
            batch_content = np.reshape(batch_content, [batch_size, IMG_W, IMG_H, IMG_C])
            # 随机选择1个风格图片，并返回风格图片存储矩阵（batch_size*h*w*c，每个h*w*c都相同），y_labels为风格标签
            batch_style, y_labels = random_select_style(path_style, batch_size, [STYLE_H, STYLE_W, IMG_C], C_NUMS)

            # 喂数据，开始训练
            sess.run(Opt, feed_dict={content: batch_content, style: batch_style, y: y_labels})
            step = sess.run(global_step)

            # 打印相关信息
            if itr % 100 == 0:
                # 为之后打印信息进行相关计算
                [loss, Target, CONTENT_LOSS, STYLE_LOSS] = sess.run([Loss, target, Content_loss, Style_loss],
                                                                    feed_dict={content: batch_content,
                                                                               style: batch_style, y: y_labels})
                # 连接3张图片（内容图片、风格图片、生成图片）
                save_img = np.concatenate((batch_content[0, :, :, :],
                                           misc.imresize(batch_style[0, :, :, :], [IMG_H, IMG_W]),
                                           Target[0, :, :, :]), axis=1)
                # 打印轮数、总loss、内容loss、风格loss
                print("Iteration: %d, Loss: %e, Content_loss: %e, Style_loss: %e" %
                      (step, loss, CONTENT_LOSS, STYLE_LOSS))
                # 展示训练效果：打印3张图片，内容图+风格图+风格迁移图
                Image.fromarray(np.uint8(save_img)).save(
                    "save_imgs/" + str(itr) + "_" + str(np.argmax(y_labels[0, :])) + ".jpg")

            time_step_stop = time.time()
            # 存储模型
            a = 5
            if itr % a == 0:
                saver.save(sess, model_path + "model", global_step=global_step)
                print('Iteration: %d, Save Model Successfully, single step time = %.2fs, total time = %.2fs' % (
                    step, time_step_stop - time_step_start, time_step_stop - time_start))

        # 关闭多线程
        coord.request_stop()
        coord.join(threads)


def main():
    backward()


if __name__ == '__main__':
    main()
