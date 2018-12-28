# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
"""
                                                                  ^style loss
                                                                  |
                                    
        +----------+                            |*****************|*******|
        |  style   +---------------------------->-----------------+------->--->
        +----------+    +********************+  |*  discrimitor   |      *|
        +----------+    |* generator        *|  |* ( vgg-16 )     |      *|
        |  target  +----> (using  conditional+-->----+------------+------->--->
        +---+------+    |* normalization)   *|  |*   |                   *|
            |           +********************+  |*   |                   *|
            +----------------------------------->----+-------------------->--->
                                                |****|********************|
                                                     |
                                                     v content loss
"""
# ***********************************图像生成网络***********************************
"""
前向传播网络，也就是graph generator。
网络结构由Johnson提出
详见论文 Perceptual Losses for Real-Time Style Transfer and Super-Resolution  arxiv:1603.08155

input： target图片
y...   风格的 one-hot （tensor)
alpha...  测试时对应风格的权重 （并保证alpha之和为1）
return ： 风格融合后的图片
"""
def forward(inputs, y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain):
    # 处理输入图片
    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])
    # 3层卷积+归一化+rule
    inputs = conv("conv1", inputs, 9, 3, 32, 1)
    inputs = conditional_instance_norm(inputs, "cin1", y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    inputs = tf.nn.relu(inputs)
    inputs = conv("conv2", inputs, 3, 32, 64, 2)
    inputs = conditional_instance_norm(inputs, "cin2", y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    inputs = tf.nn.relu(inputs)
    inputs = conv("conv3", inputs, 3, 64, 128, 2)
    inputs = conditional_instance_norm(inputs, "cin3", y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    inputs = tf.nn.relu(inputs)
    # 5层resBlock
    inputs = ResBlock("res1", inputs, 3, 128, 128, y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    inputs = ResBlock("res2", inputs, 3, 128, 128, y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    inputs = ResBlock("res3", inputs, 3, 128, 128, y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    inputs = ResBlock("res4", inputs, 3, 128, 128, y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    inputs = ResBlock("res5", inputs, 3, 128, 128, y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    # 2层upsampling
    inputs = upsampling("up1", inputs, 128, 64, y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    inputs = upsampling("up2", inputs, 64, 32, y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    # 卷积+归一化+sigmoid
    inputs = conv("last", inputs, 9, 32, 3, 1)
    inputs = conditional_instance_norm(inputs, "cinout", y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    inputs = tf.nn.sigmoid(inputs) * 255
    return inputs


"""
一个封装过的卷积层：
name：名称
k_size 核大小
nums_in 上一层的channel
nums_out 本层核的个数

return tonsor
"""
def conv(name, inputs, k_size, nums_in, nums_out, strides):
    #使用正态分布初始化核
    kernel = tf.get_variable(name + "W", [k_size, k_size, nums_in, nums_out],
                             initializer=tf.truncated_normal_initializer(stddev=0.01))
    #0初始化bias
    bias = tf.get_variable(name + "B", [nums_out], initializer=tf.constant_initializer(0.))

    # 使用 valid方式进行卷积避免 输出图像边缘发黑。
    # 为了保证输入输出图像大小一致，对原始输入按"REFLECT"方式填充。
    pad_size = k_size // 2
    inputs = tf.pad(inputs, [[0, 0], [pad_size, pad_size], [
                    pad_size, pad_size], [0, 0]], mode="REFLECT")
    input = tf.nn.conv2d(
        inputs, kernel, [1, strides, strides, 1], "VALID") + bias

    return input


"""
条件归一化层：
详见论文： 
A LEARNED REPRESENTATION FOR ARTISTIC STYLE
及论文
The Missing Ingredient for Fast Stylization

这就是 gan的精髓，它将z的分布作为variable，训练之
并将其作为 vector以实现风格融合

return : the normalized, scaled, offset tensor.
"""
def conditional_instance_norm(x, scope_bn, y1=None, y2=None, y3=None, y4=None, alpha1=1, alpha2=0, alpha3=0,
                              istrain=True):
    
    if y1 == None:
        beta = tf.get_variable(name=scope_bn + 'beta', shape=[x.shape[-1]], initializer=tf.constant_initializer([0.]),
                               trainable=True)  # label_nums x C
        gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[x.shape[-1]], initializer=tf.constant_initializer([1.]),
                                trainable=True)  # label_nums x C
    else:
        if istrain:
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[y1.shape[-1], x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[y1.shape[-1], x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
            beta = tf.matmul(y1, beta)
            gamma = tf.matmul(y1, gamma)
        else:
            #此时进行风格融合，y2... alpha...有效
            beta = tf.get_variable(name=scope_bn + 'beta', shape=[y1.shape[-1], x.shape[-1]],
                                   initializer=tf.constant_initializer([0.]), trainable=True)  # label_nums x C
            gamma = tf.get_variable(name=scope_bn + 'gamma', shape=[y1.shape[-1], x.shape[-1]],
                                    initializer=tf.constant_initializer([1.]), trainable=True)  # label_nums x C
            #one-hot直接点乘beta/gamma矩阵即得到对应的 beta/gamma
            beta1 = tf.matmul(y1, beta)
            gamma1 = tf.matmul(y1, gamma)
            beta2 = tf.matmul(y2, beta)
            gamma2 = tf.matmul(y2, gamma)
            beta3 = tf.matmul(y3, beta)
            gamma3 = tf.matmul(y3, gamma)
            beta4 = tf.matmul(y4, beta)
            gamma4 = tf.matmul(y4, gamma)
            #对beta和gamma进行仿射变换。
            beta = alpha1 * beta1 + alpha2 * beta2 + alpha3 * beta3 + (1.0 - alpha1 - alpha2 - alpha3) * beta4
            gamma = alpha1 * gamma1 + alpha2 * gamma2 + alpha3 * gamma3 + (1.0 - alpha1 - alpha2 - alpha3) * gamma4

    mean, var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-10)
    return x


def upsampling(name, inputs, nums_in, nums_out, y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain):
    inputs = tf.image.resize_nearest_neighbor(inputs, [tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])
    return conditional_instance_norm(conv(name, inputs, 3, nums_in, nums_out, 1), "cin" + name, y1, y2, y3, y4, alpha1,
                                     alpha2, alpha3, istrain)


"""
一层残差块
详见：http://torch.ch/blog/2016/02/04/resnets.html
以及论文：  He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” arXiv preprint arXiv:1512.03385 (2015).

name
inputs tensor
k_size 核大小
nums_in = input.shape[3]
nums_out

返回 tonsor
"""
def ResBlock(name, inputs, k_size, nums_in, nums_out, y1, y2, y3, y4, alpha1, alpha2, alpha3, istrain):
    temp = inputs * 1.0
    inputs = conditional_instance_norm(conv("conv1_" + name, inputs, k_size, nums_in, nums_out, 1), "cin1" + name, y1,
                                       y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    inputs = tf.nn.relu(inputs)
    inputs = conditional_instance_norm(conv("conv2_" + name, inputs, k_size, nums_in, nums_out, 1), "cin2" + name, y1,
                                       y2, y3, y4, alpha1, alpha2, alpha3, istrain)
    return inputs + temp


# ***********************************特征提取网络(VGG16)***********************************
"""
vgg16:
return 包含指定层特征的字典，以供计算损失

"""
def vggnet(inputs, vgg_path):
    inputs = tf.reverse(inputs, [-1]) - np.array([103.939, 116.779, 123.68])  # 处理输入图片
    para = np.load(vgg_path + "vgg16.npy", encoding="latin1").item()  # 从vgg模型路径获得vgg模型
    F = {}  # 定义返回值为字典F
    inputs = tf.nn.relu(conv_(inputs, para["conv1_1"][0], para["conv1_1"][1]))  # 使用vgg中的conv1_1,conv1_2完成卷积
    inputs = tf.nn.relu(conv_(inputs, para["conv1_2"][0], para["conv1_2"][1]))  # 激活函数 relu，即将矩阵中每行的非最大值置0。
    F["conv1_2"] = inputs
    inputs = max_pooling(inputs)  # 池化
    inputs = tf.nn.relu(conv_(inputs, para["conv2_1"][0], para["conv2_1"][1]))  # 第二层卷积提取特征
    inputs = tf.nn.relu(conv_(inputs, para["conv2_2"][0], para["conv2_2"][1]))
    F["conv2_2"] = inputs
    inputs = max_pooling(inputs)  # 池化
    inputs = tf.nn.relu(conv_(inputs, para["conv3_1"][0], para["conv3_1"][1]))  # 第三层卷积提取特征
    inputs = tf.nn.relu(conv_(inputs, para["conv3_2"][0], para["conv3_2"][1]))
    inputs = tf.nn.relu(conv_(inputs, para["conv3_3"][0], para["conv3_3"][1]))
    F["conv3_3"] = inputs
    inputs = max_pooling(inputs)  # 池化
    inputs = tf.nn.relu(conv_(inputs, para["conv4_1"][0], para["conv4_1"][1]))  # 第四层卷积提取特征
    inputs = tf.nn.relu(conv_(inputs, para["conv4_2"][0], para["conv4_2"][1]))
    inputs = tf.nn.relu(conv_(inputs, para["conv4_3"][0], para["conv4_3"][1]))
    F["conv4_3"] = inputs
    return F


# VGG16卷积操作
def conv_(inputs, w, b):
    return tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME") + b  # 卷积函数，inputs：输入，卷积核，步长，前后形状相同


# VGG16池化操作
def max_pooling(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")  # 池化函数：输入，池化窗口大小，滑动步长，前后形状


# ***********************************损失函数***********************************

"""
以L2距离计算内容损失
"""
def content_loss(phi_content, phi_target):
    return tf.nn.l2_loss(phi_content["conv2_2"] - phi_target["conv2_2"]) * 2 / tf.cast(tf.size(phi_content["conv2_2"]),
                                                                                       dtype=tf.float32)


"""
计算风格损失

实际上是计算vgg中间输出gram矩阵 的l2距离
详见 Gatys`s A Neural Algorithm of Artistic Style arxiv:1508.06576v2

phi_style:  
phi_target:
一个 tonsor的字典，包含两个图经过vgg后的中间输出
"""
def style_loss(phi_style, phi_target):
    layers = ["conv1_2", "conv2_2", "conv3_3", "conv4_3"]
    loss = 0
    for layer in layers:
        style_output = phi_style[layer]
        style_gram = get_gram_matrix(style_output)
        target_output = phi_target[layer]
        target_gram = get_gram_matrix(target_output)
        loss += tf.nn.l2_loss(style_gram - target_gram) * 2 / tf.cast(tf.size(target_gram), dtype=tf.float32)
    return loss


"""
计算gram矩阵(tensor)

返回 ：tensor
"""
def get_gram_matrix(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)
    return grams
