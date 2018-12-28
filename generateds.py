import os
import tensorflow as tf
import numpy as np
from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('path_data', './data', 'tfRecord save path.')
flags.DEFINE_string('path_style', "./style_imgs", 'Row style images path')
flags.DEFINE_string('path_content', "./MSCOCO", 'Row style images path')

flags.DEFINE_string('record_style_name', 'styles.tfrecords', 'Style tfrecord name')
flags.DEFINE_string('record_dataset_name', 'coco_train.tfrecords', 'Data set tfrecord name')

flags.DEFINE_integer('img_h', 256, 'Train images\' height')
flags.DEFINE_integer('img_w', 256, 'Train images\' width')
flags.DEFINE_integer('img_c', 3, 'Train images\' channels num')
flags.DEFINE_integer('style_h', 512, 'Style images\' height')
flags.DEFINE_integer('style_w', 512, 'Style images\' width')

FLAGS = flags.FLAGS


def generate_content_tfRecord():
    if not os.path.exists(FLAGS.data_path):
        os.makedirs(FLAGS.data_path)
        print("the directory was created successful")
    else:
        print("directory already exists")
    write_content_tfRecord()


def write_content_tfRecord():
    writer = tf.python_io.TFRecordWriter(FLAGS.record_dataset_name)
    num_pic = 0
    example_list = list()
    file_path_list = []
    for root, _, files in os.walk(FLAGS.path_content):
        for file in files:
            if os.path.splitext(file)[1] not in ['jpg', 'png', 'jpeg']:
                continue
            file_path = os.path.join(root, file)
            file_path_list.append(file_path)
    np.random.shuffle(file_path_list)
    for file_path in file_path_list:
        with Image.open(file_path) as img:
            img = center_crop_img(img)
            img = img.resize((FLAGS.img_w, FLAGS.img_h))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Feature(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            example_list.append(example)
            num_pic += 1
            print('the number of picture:', num_pic)
    for example in example_list:
        writer.write(example.SerializerToString())
    print('write tfrecord successful')


def center_crop_img(img : Image):
    width = img.size[0]
    height = img.size[1]
    offset = (width if width < height else height) / 2
    img = img.crop((
        width / 2 - offset,
        height /2 - offset,
        width / 2 + offset,
        height / 2 + offset
    ))
    return img

def main():
    generate_content_tfRecord()

if __name__ == '__main__':
    tf.app.run(main)

