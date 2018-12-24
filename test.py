# -*- coding: UTF-8 -*-
import tensorflow as tf               #导入tensorflow模块
import numpy as np                    #导入numpy模块
from PIL import Image                 #导入PIL模块
from forward import forward           #导入前向网络模块
import argparse                       #导入参数选择模块

# 设置参数
parser = argparse.ArgumentParser()    #定义一个参数设置器
parser.add_argument("--C_NUMS", type=int, default=10)     #参数：图片数量，默认值为10
parser.add_argument("--PATH_MODEL", type=str, default="./save_para/")    #参数：模型存储路径
parser.add_argument("--PATH_RESULTS", type=str, default="./results/")    #参数：结果存储路径
parser.add_argument("--PATH_IMG", type=str, default="./imgs/5.jpg")      #参数：选择测试图像
parser.add_argument("--LABEL_1", type=int, default=9)                    #参数：风格1
parser.add_argument("--LABEL_2", type=int, default=4)                    #参数：风格2
parser.add_argument("--ALPHA", type=float, default=0.5)                  #参数：Alpah，风格权重，默认为0.5
args = parser.parse_args()                                               #定义参数集合args


def Init(c_nums=10, model_path=args.PATH_MODEL):                         #初始化图片生成模型参数
    content = tf.placeholder(tf.float32, [1, None, None, 3])             #图片输入定义
    y1 = tf.placeholder(tf.float32, [1, c_nums])                         #初始化风格1选择范围
    y2 = tf.placeholder(tf.float32, [1, c_nums])                         #初始化风格2选择范围
    alpha = tf.placeholder(tf.float32)                                   #风格与内容的权重比
    target = forward(content, y1, y2, alpha)                             #定义将要生成的图片
    sess = tf.Session()                                                  #定义一个sess
    sess.run(tf.global_variables_initializer())                          #模型初始化
    saver = tf.train.Saver()                                             #模型存储器定义
    ckpt = tf.train.get_checkpoint_state(model_path)                     #从模型存储路径中获取模型
    if ckpt and ckpt.model_checkpoint_path:                              #从检查点中恢复模型
        saver.restore(sess, ckpt.model_checkpoint_path)                  #从检查点的路径名中分离出训练轮数
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]   #获取训练步数
    return target, sess, content, y1, y2, alpha                          #返回目标图片，模型session，输入图片，风格选择，风格权重


def stylize(img_path, result_path, label1, label2, alpha, target, sess, content_ph, y1_ph, y2_ph, alpha_ph): # 风格迁移
    img = np.array(Image.open(img_path))               #将输入图片序列化
    Y1 = np.zeros([1, 10])                             #数组置0
    Y2 = np.zeros([1, 10])                             #数组置0
    Y1[0, label1] = 1                                  #第label1个风格置1
    Y2[0, label2] = 1                                  #第label2个风格置1
    img = sess.run(target, feed_dict={content_ph: img[np.newaxis, :, :, :], y1_ph: Y1, y2_ph: Y2, alpha_ph: alpha})  #生成图片
    Image.fromarray(np.uint8(img[0, :, :, :])).save(result_path + "result" + str(alpha) + ".jpg")                   #保存风格迁移后的图片

#测试程序
def test():
    target, sess, content, y1, y2, alpha = Init(args.C_NUMS, args.PATH_MODEL)          #初始化生成模型
    for a in range(11):                                                                #将风格权重分成10份，计算10种风格不同的图片
        args.ALPHA = a / 10
        stylize(args.PATH_IMG, args.PATH_RESULTS, args.LABEL_1, args.LABEL_2, args.ALPHA, target, sess, content, y1, y2, alpha)

#主程序
def main():
    test()

#主程序入口
if __name__ == '__main__':
    main()
