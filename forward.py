# -*- coding: UTF-8 -*-
import numpy as np    #导入numpy模块
from ops import *     #导入ops所有模块
#卷积
def conv_(inputs, w, b):
    return tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME") + b      #卷积函数，inputs：输入，卷积核，步长，前后形状相同
#池化
def max_pooling(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")   #池化函数：输入，池化窗口大小，滑动步长，前后形状
#vgg16提取卷积网络特征
def vggnet(inputs, vgg_path):
    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])     #处理输入图片
    para = np.load(vgg_path+"vgg16.npy", encoding="latin1").item()               #从vgg模型路径获得vgg模型
    F = {}                                                                       #定义返回值为字典F
    inputs = relu(conv_(inputs, para["conv1_1"][0], para["conv1_1"][1]))         #使用vgg中的conv1_1,conv1_2完成卷积
    inputs = relu(conv_(inputs, para["conv1_2"][0], para["conv1_2"][1]))         #激活函数 relu，即将矩阵中每行的非最大值置0。
    F["conv1_2"] = inputs
    inputs = max_pooling(inputs)                                                 #池化
    inputs = relu(conv_(inputs, para["conv2_1"][0], para["conv2_1"][1]))         #第二层卷积提取特征
    inputs = relu(conv_(inputs, para["conv2_2"][0], para["conv2_2"][1]))
    F["conv2_2"] = inputs
    inputs = max_pooling(inputs)                                                 #池化
    inputs = relu(conv_(inputs, para["conv3_1"][0], para["conv3_1"][1]))         #第三层卷积提取特征
    inputs = relu(conv_(inputs, para["conv3_2"][0], para["conv3_2"][1]))
    inputs = relu(conv_(inputs, para["conv3_3"][0], para["conv3_3"][1]))
    F["conv3_3"] = inputs
    inputs = max_pooling(inputs)                                                 #池化+第四层卷积提取特征
    inputs = relu(conv_(inputs, para["conv4_1"][0], para["conv4_1"][1]))
    inputs = relu(conv_(inputs, para["conv4_2"][0], para["conv4_2"][1]))
    inputs = relu(conv_(inputs, para["conv4_3"][0], para["conv4_3"][1]))
    F["conv4_3"] = inputs
    return F

def forward(inputs, y1, y2, alpha):
    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])   #处理输入图片
    # 3层卷积
    inputs = relu(conditional_instance_norm(conv("conv1", inputs, 9, 3, 32, 1), "cin1", y1, y2, alpha))
    inputs = relu(conditional_instance_norm(conv("conv2", inputs, 3, 32, 64, 2), "cin2", y1, y2, alpha))
    inputs = relu(conditional_instance_norm(conv("conv3", inputs, 3, 64, 128, 2), "cin3", y1, y2, alpha))
    # 仿照ResNet定义一些跳过连接
    inputs = ResBlock("res1", inputs, 3, 128, 128, y1, y2, alpha)
    inputs = ResBlock("res2", inputs, 3, 128, 128, y1, y2, alpha)
    inputs = ResBlock("res3", inputs, 3, 128, 128, y1, y2, alpha)
    inputs = ResBlock("res4", inputs, 3, 128, 128, y1, y2, alpha)
    inputs = ResBlock("res5", inputs, 3, 128, 128, y1, y2, alpha)
    inputs = upsampling("up1", inputs, 128, 64, y1, y2, alpha)
    inputs = upsampling("up2", inputs, 64, 32, y1, y2, alpha)
    inputs = sigmoid(conditional_instance_norm(conv("last", inputs, 9, 32, 3, 1), "cinout", y1, y2, alpha)) * 255
    return inputs


