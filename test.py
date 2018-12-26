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
parser.add_argument("--LABEL_1", type=int, default=2)                    #参数：风格1
parser.add_argument("--LABEL_2", type=int, default=4)                    #参数：风格2
parser.add_argument("--LABEL_3", type=int, default=6)                    #参数：风格3
parser.add_argument("--LABEL_4", type=int, default=8)                    #参数：风格4
parser.add_argument("--ALPHA1", type=float, default=0.1)                  #参数：Alpah1，风格权重，默认为0.1
parser.add_argument("--ALPHA2", type=float, default=0.1)                  #参数：Alpah2，风格权重，默认为0.1
parser.add_argument("--ALPHA3", type=float, default=0.1)                  #参数：Alpah2，风格权重，默认为0.1
args = parser.parse_args()                                               #定义参数集合args


def Init(c_nums=10, model_path=args.PATH_MODEL):                         #初始化图片生成模型参数
    content = tf.placeholder(tf.float32, [1, None, None, 3])             #图片输入定义
    y1 = tf.placeholder(tf.float32, [1, c_nums])                         #初始化风格1选择范围
    y2 = tf.placeholder(tf.float32, [1, c_nums])                         #初始化风格2选择范围
    y3 = tf.placeholder(tf.float32, [1, c_nums])
    y4 = tf.placeholder(tf.float32, [1, c_nums])
    alpha1 = tf.placeholder(tf.float32)
    alpha2 = tf.placeholder(tf.float32)                                  #风格与内容的权重比
    alpha3 = tf.placeholder(tf.float32)
    target = forward(content, y1, y2, y3, y4, alpha1, alpha2, alpha3, False)     #定义将要生成的图片
    sess = tf.Session()                                                  #定义一个sess
    sess.run(tf.global_variables_initializer())                          #模型初始化
    saver = tf.train.Saver()                                             #模型存储器定义
    ckpt = tf.train.get_checkpoint_state(model_path)                     #从模型存储路径中获取模型
    if ckpt and ckpt.model_checkpoint_path:                              #从检查点中恢复模型
        saver.restore(sess, ckpt.model_checkpoint_path)                  #从检查点的路径名中分离出训练轮数
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]   #获取训练步数
    return target, sess, content, y1, y2, y3, y4, alpha1, alpha2, alpha3               #返回目标图片，模型session，输入图片，风格选择，风格权重


def stylize(img_path, result_path, label1, label2, label3, label4, alpha1, alpha2, alpha3, target, sess, content_ph, y1_ph, y2_ph, y3_ph, y4_ph, alpha1_ph, alpha2_ph, alpha3_ph): # 风格迁移
    print('%.2f'%alpha1,'%.2f'%alpha2,'%.2f'%alpha3,'%.2f'%(1.0-alpha1-alpha2-alpha3))
    img = np.array(Image.open(img_path))               #将输入图片序列化
    Y1 = np.zeros([1, 10])                             #数组置0
    Y2 = np.zeros([1, 10])                             #数组置0
    Y3 = np.zeros([1, 10])
    Y4 = np.zeros([1, 10])
    Y1[0, label1] = 1                                  #第label1个风格置1
    Y2[0, label2] = 1                                  #第label2个风格置1
    Y3[0, label3] = 1
    Y4[0, label4] = 1
    img = sess.run(target, feed_dict={content_ph: img[np.newaxis, :, :, :], y1_ph: Y1, y2_ph: Y2, y3_ph: Y3, y4_ph: Y4, alpha1_ph: alpha1, alpha2_ph: alpha2, alpha3_ph: alpha3})  #生成图片
    Image.fromarray(np.uint8(img[0, :, :, :])).save(result_path + 'result' + '_' + '%.2f'%(alpha1) + '_' + '%.2f'%(alpha2) + '_' + '%.2f'%(alpha3) + '_' + '%.2f'%(1-alpha1-alpha2-alpha3)+'.jpg')                   #保存风格迁移后的图片
    return img

#测试程序
def test():
    target, sess, content, y1, y2, y3, y4, alpha1, alpha2, alpha3 = Init(args.C_NUMS, args.PATH_MODEL)  #初始化生成模型
    size = 5
    i=0
    while i < size:
        x_sum=i*25.0
        y_sum = 100.0 - x_sum
        x_step = x_sum / 4.0
        y_step = y_sum / 4.0

        j=0
        while j < size:
            ap1=j*x_step
            ap2=x_sum-ap1
            ap3 = j*y_step
            args.ALPHA1 = float('%.2f'%(ap1/100.0))
            args.ALPHA2 = float('%.2f'%(ap2/100.0))
            args.ALPHA3 = float('%.2f'%(ap3/100.0))

            img_return = stylize(args.PATH_IMG, args.PATH_RESULTS, args.LABEL_1, args.LABEL_2, args.LABEL_3, args.LABEL_4, args.ALPHA1, args.ALPHA2, args.ALPHA3, target, sess, content, y1, y2, y3, y4, alpha1, alpha2, alpha3)
            img_return = Image.fromarray(np.uint8(img_return[0, :, :, :]))

            if j == 0:
                width, height = img_return.size
                img_5 = Image.new(img_return.mode,(width*5, height))
            img_5.paste(img_return, (width*j, 0, width*(j+1), height))
            j = j+1

        if i ==0:
            img_25 = Image.new(img_return.mode, (width * 5, height*5))
        img_25.paste(img_5, (0, height*i, width*5, height*(i+1)))
        i = i+1

    img_25.save(args.PATH_RESULTS + 'result_25'+'.jpg')

#主程序
def main():
    test()


#主程序入口
if __name__ == '__main__':
    main()
