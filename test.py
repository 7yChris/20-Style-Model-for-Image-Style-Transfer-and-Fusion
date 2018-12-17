# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
from forward import forward
import argparse

# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument("--C_NUMS", type=int, default=10)
parser.add_argument("--PATH_MODEL", type=str, default="./save_para/")
parser.add_argument("--PATH_RESULTS", type=str, default="./results/")
# 选择测试图像
parser.add_argument("--PATH_IMG", type=str, default="./imgs/5.jpg")
# 选择两种风格1
parser.add_argument("--LABEL_1", type=int, default=9)
# 选择风格2
parser.add_argument("--LABEL_2", type=int, default=4)
# Alpha，默认为0.5
parser.add_argument("--ALPHA", type=float, default=0.5)
args = parser.parse_args()


def Init(c_nums=10, model_path=args.PATH_MODEL):
    content = tf.placeholder(tf.float32, [1, None, None, 3])
    y1 = tf.placeholder(tf.float32, [1, c_nums])
    y2 = tf.placeholder(tf.float32, [1, c_nums])
    alpha = tf.placeholder(tf.float32)
    target = forward(content, y1, y2, alpha)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        # 从检查点中恢复模型
        saver.restore(sess, ckpt.model_checkpoint_path)
        # 从检查点的路径名中分离出训练轮数
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    return target, sess, content, y1, y2, alpha


def stylize(img_path, result_path, label1, label2, alpha, target, sess, content_ph, y1_ph, y2_ph, alpha_ph):
    img = np.array(Image.open(img_path))
    Y1 = np.zeros([1, 10])
    Y2 = np.zeros([1, 10])
    Y1[0, label1] = 1
    Y2[0, label2] = 1
    img = sess.run(target, feed_dict={content_ph: img[np.newaxis, :, :, :], y1_ph: Y1, y2_ph: Y2, alpha_ph: alpha})
    print(alpha)
    Image.fromarray(np.uint8(img[0, :, :, :])).save(result_path + "test" + str(alpha) + ".jpg")


def test():
    target, sess, content, y1, y2, alpha = Init(args.C_NUMS, args.PATH_MODEL)
    for a in range(11):
        # 2.7需要换成浮点数运行，否则若结果小于1，则为0；python3则不需要，直接除即可
        args.ALPHA = float(a) / 10.0
        # 生成融合图像
        stylize(args.PATH_IMG, args.PATH_RESULTS, args.LABEL_1, args.LABEL_2, args.ALPHA, target, sess, content, y1, y2, alpha)


def main():
    test()


if __name__ == '__main__':
    main()
