# AI-Final-Project
图片风格迁移实现，输入内容图片，将多种风格迁移到内容图片上，并可实现两种风格不同层次的融合

Simple implementation of the paper "A Learned Representation for Artistic Style"

[A Learned Representation for Artistic Style.](http://cn.arxiv.org/pdf/1610.07629.pdf)

## Prepare
##### 下载coco数据集
##### 下载vgg16预训练模型
##### 制作coco数据集tfRecord
```
$ python generated.py \
      --path_data=/path/to/coco_train.tfrecords \
      --path_style=/path/to/style_folder \
      --path_content=/path/to/MSCOCO \
      --img_h=256 --img_w=256 --img_c=3\
      --style_h=512 --style_w=512 
```
## Training Model

## Eval Model