# -*- coding: UTF-8 -*-
import os
import random
import tensorflow as tf  # 导入tensorflow模块
import numpy as np  # 导入numpy模块
from PIL import Image  # 导入PIL模块
from PIL import ImageOps
from forward import forward  # 导入前向网络模块
import argparse  # 导入参数选择模块
from generateds import center_crop_img

# 设置参数
parser = argparse.ArgumentParser()  # 定义一个参数设置器
parser.add_argument("--C_NUMS", type=int, default=20)  # 参数：图片数量，默认值为10
parser.add_argument("--PATH_MODEL", type=str, default="./save_para/")  # 参数：模型存储路径
parser.add_argument("--PATH_RESULTS", type=str, default="./results/")  # 参数：结果存储路径
parser.add_argument("--PATH_IMG", type=str, default="./imgs/shanghai1.jpg")  # 参数：选择测试图像
parser.add_argument("--PATH_STYLE", type=str, default="./style_imgs/")
parser.add_argument("--LABEL_1", type=int, default=2)  # 参数：风格1
parser.add_argument("--LABEL_2", type=int, default=8)  # 参数：风格2
parser.add_argument("--LABEL_3", type=int, default=10)  # 参数：风格3
parser.add_argument("--LABEL_4", type=int, default=19)  # 参数：风格4
parser.add_argument("--ALPHA1", type=float, default=0.25)  # 参数：Alpha1，风格权重，默认为0.25
parser.add_argument("--ALPHA2", type=float, default=0.25)  # 参数：Alpha2，风格权重，默认为0.25
parser.add_argument("--ALPHA3", type=float, default=0.25)  # 参数：Alpha3，风格权重，默认为0.25

args = parser.parse_args()  # 定义参数集合args


class Stylizer(object):

    def __init__(self, stylizer_arg):
        self.stylizer_arg = stylizer_arg
        self.content = tf.placeholder(tf.float32, [1, None, None, 3])  # 图片输入定义
        self.weight = tf.placeholder(tf.float32, [1, stylizer_arg.C_NUMS])  # 模型风格参数权重向量
        self.target = forward(self.content, self.weight)  # 定义将要生成的图片
        self.sess = tf.Session()  # 定义一个sess
        self.sess.run(tf.global_variables_initializer())  # 模型初始化
        self.img = None
        self.img_path = None
        self.label_list = None
        self.input_weight = None
        self.global_step = 0
        saver = tf.train.Saver()  # 模型存储器定义
        ckpt = tf.train.get_checkpoint_state(stylizer_arg.PATH_MODEL)  # 从模型存储路径中获取模型
        if ckpt and ckpt.model_checkpoint_path:  # 从检查点中恢复模型
            # 从检查点的路径名中分离出训练轮数
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.global_step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]  # 获取训练步数

    def __del__(self):
        self.sess.close()

    def set_image(self, img_path):
        img = Image.open(img_path)

        while img.width * img.height > 500000:
            img = img.resize((int(img.width / 1.5), int(img.height / 1.5)))

        self.img = np.array(img)

        self.img_path = img_path

        return self.img

    def set_style(self, *label_list):
        self.label_list = label_list

    def set_weight(self, weight_dict):
        self.label_list = []
        self.input_weight = np.zeros([1, self.stylizer_arg.C_NUMS])
        for k, v in weight_dict.items():
            self.input_weight[0, k] = v

    def stylize(self, alpha_list=None):
        # print(self.sess.run( tf.global_variables() ))
        # 开始预测！
        # 生成指定风格图片
        if alpha_list is not None:
            weight_dict = dict(zip(self.label_list, alpha_list))
            self.set_weight(weight_dict)

        img = self.sess.run(self.target,
                            feed_dict={self.content: self.img[np.newaxis, :, :, :], self.weight: self.input_weight})

        return img

    def save_25result(self):
        size = 5
        i = 0
        # 按行生成
        while i < size:
            # 1、2风格权重之和
            x_sum = 100 - i * 25.0
            # 3、4风格权重之和
            y_sum = i * 25
            # 1、2风格之和进行五等分，计算权重step值
            x_step = x_sum / 4.0
            # 3、4风格之和进行五等分，计算权重step值
            y_step = y_sum / 4.0

            # 按列生成
            j = 0
            while j < size:
                # 计算1、2风格的权重
                ap1 = x_sum - j * x_step
                ap2 = j * x_step
                # 计算3风格权重
                ap3 = y_sum - j * y_step
                # 归一化后存到args中
                alphas = (float('%.2f' % (ap1 / 100.0)),
                          float('%.2f' % (ap2 / 100.0)), float('%.2f' % (ap3 / 100.0)))
                # 返回融合后图像
                # img_return = stylize(args.PATH_IMG, args.PATH_RESULTS, args.LABEL_1, args.LABEL_2, args.LABEL_3,
                #                      args.LABEL_4, args.ALPHA1, args.ALPHA2, args.ALPHA3, target, sess, content, y1, y2, y3,
                #                      y4, alpha1, alpha2, alpha3)
                img_return = self.stylize(alphas)
                # array转IMG
                img_return = Image.fromarray(np.uint8(img_return[0, :, :, :]))
                # 将5个图像按行拼接
                if j == 0:
                    width, height = img_return.size
                    img_5 = Image.new(img_return.mode, (width * 5, height))
                img_5.paste(img_return, (width * j, 0, width * (j + 1), height))
                j = j + 1

            # 将多个行拼接图像，拼接成5*5矩阵
            if i == 0:
                img_25 = Image.new(img_return.mode, (width * 5, height * 5))
            img_25.paste(img_5, (0, height * i, width * 5, height * (i + 1)))

            i = i + 1

        # 将5*5矩阵图像的4个角加上4个风格图像，以作对比
        img_25_4 = Image.new(img_return.mode, (width * 7, height * 5))
        img_25_4 = ImageOps.invert(img_25_4)
        img_25_4.paste(
            center_crop_img(Image.open(
                self.stylizer_arg.PATH_STYLE + str(self.label_list[0] + 1) + '.png')).resize((width, height)),
            (0, 0, width, height))
        img_25_4.paste(
            center_crop_img(Image.open(
                self.stylizer_arg.PATH_STYLE + str(self.label_list[1] + 1) + '.png')).resize((width, height)),
            (width * 6, 0, width * 7, height))
        img_25_4.paste(
            center_crop_img(Image.open(
                self.stylizer_arg.PATH_STYLE + str(self.label_list[2] + 1) + '.png')).resize((width, height)),
            (0, height * 4, width, height * 5))
        img_25_4.paste(
            center_crop_img(Image.open(
                self.stylizer_arg.PATH_STYLE + str(self.label_list[3] + 1) + '.png')).resize((width, height)),
            (width * 6, height * 4, width * 7, height * 5))
        img_25_4.paste(img_25, [width, 0, width * 6, height * 5])

        # 存储5*5图像矩阵
        img_25.save(self.stylizer_arg.PATH_RESULTS + self.img_path.split('/')
        [-1].split('.')[0] + str(self.label_list) + '_result_25' + '.jpg')
        # 存储5*5+4风格图像矩阵

        print(self.stylizer_arg.PATH_RESULTS + self.img_path.split('/')
        [-1].split('.')[0] + str(self.label_list) + '_result_25_4' + '.jpg' + " saved!")


def diao():
    stylizer0 = Stylizer(args)
    stylizer0.set_image(args.PATH_IMG)
    stylizer0.set_style(args.LABEL_1, args.LABEL_2, args.LABEL_3, args.LABEL_4)
    stylizer0.save_25result()
    del stylizer0
    tf.reset_default_graph()


# def diao():
#     stylizer0 = Stylizer(args)
#     stylizer0.setImage(s[4])
#     stylizer0.setstyle(int(s[0]), int(s[1]), int(s[2]), int(s[3]))
#     stylizer0.save_25result()
#     del stylizer0
#     tf.reset_default_graph()


def walk():
    stylizer0 = Stylizer(args)
    file_dir = r"./imgs/"
    for root, dirs, files in os.walk(file_dir):
        for image_file in files:
            path_dic = os.path.join(file_dir, image_file)
            stylizer0.set_image(path_dic)
            for i in range(20):
                stylizer0.set_style(i, 0, 0, 0)
                img = stylizer0.stylize((1.0, 0.0, 0.0, 0.0))
                img = Image.fromarray(np.uint8(img[0, :, :, :]))
                # img.save(args.PATH_RESULTS + image_file + "( " + str(i) + " - style).jpg")
                img.save("../static/results/data" + str(random.uniform(1, 10)) + ".jpg")
                print(image_file + "( " + str(i) + " - style).jpg saved")


def test3():
    stylizer0 = Stylizer(args)

    file_dir = r"C:\Users\YRP\Desktop\tf\imgs"
    stylizer0.set_style(0, 1, 4, 5)

    for root, dirs, files in os.walk(file_dir):
        for image_file in files:
            path_dic = os.path.join(file_dir, image_file)
            stylizer0.set_image(path_dic)
            stylizer0.save_25result()


def test2():
    images = ['imgs/1.jpg', 'imgs/2.jpg']
    style_path = 'styles/var'
    styles = os.listdir(style_path)
    target, sess, content, y1, y2, y3, y4, alpha1, alpha2, alpha3 = Init(args.C_NUMS, args.PATH_MODEL)
    x_len = args.C_NUMS + 1
    y_len = len(images) + 1
    img_all = Image.new("RGB", (x_len * 512, y_len * 512))
    for i, style_name in enumerate(styles):
        path = os.path.join(style_path, style_name)
        with Image.open(path) as img:
            img_resized = center_crop_img(img).resize((512, 512))
            img_all.paste(img_resized, get_point(0, i + 1, 512))
    for content_n in range(y_len - 1):
        with Image.open(images[content_n]) as img:
            img_resized = center_crop_img(img).resize((512, 512))
            img_all.paste(img_resized, get_point(content_n + 1, 0, 512))
            for style_n in range(x_len - 1):
                img_arr = np.array(img)
                y = np.zeros([1, 10])
                y[0, style_n] = 1
                img_stylized_arr = sess.run(target,
                                            feed_dict={content: img_arr[np.newaxis, :, :, :], y1: y, y2: None, y3: None,
                                                       y4: None, alpha1: 1, alpha2: None, alpha3: None})
                img_stylized = Image.fromarray(np.uint8(img_stylized_arr[0, :, :, :]))
                img_stylized.save('{}_{}'.format(style_n, os.path.split(images[content_n])[1]))
                img_stylized = center_crop_img(img_stylized).resize((512, 512))
                img_all.paste(img_stylized, get_point(content_n + 1, style_n + 1, 512))


def get_point(row_n, col_n, cap):
    point_s = (col_n * cap, row_n * cap)
    return point_s[0], point_s[1], point_s[0] + cap, point_s[1] + cap


# 主程序
def main():
    diao()


# 主程序入口
if __name__ == '__main__':
    s = Stylizer(args)
    s.set_weight({1: 1})
    s.set_image('./MSCOCO/COCO_train2014_000000000009.jpg')
    img = s.stylize()
    img = Image.fromarray(np.uint8(img[0, :, :, :]))
    img.show()
