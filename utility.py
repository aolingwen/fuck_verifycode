#encoding=utf8

import random
import os
from captcha.image import ImageCaptcha
import cv2
import numpy as np
import tensorflow as tf

TRAIN_SIZE = 50000
VALID_SIZE = 20000
CHAR_SET = '123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'
CHAR_NUM = 5
IMG_HEIGHT = 60
IMG_WIDTH = 160
FONT_SIZES = [40]
TRAIN_IMG_PATH = './train_data'
VALID_IMG_PATH = './valid_data'
TRAIN_RECORDS_NAME = 'train_data.tfrecords'
TRAIN_VALIDATE_NAME = 'validate_data.tfrecords'
LOG_DIR = './log/'
MODEL_DIR = './model/'
BATCH_SIZE = 128


#生成不落地的验证码图片
def gen_a_verifycode():
    image = ImageCaptcha(width=IMG_WIDTH, height=IMG_HEIGHT, font_sizes=FONT_SIZES)
    label = ''.join(random.sample(CHAR_SET, CHAR_NUM))
    img = image.generate_image(label)
    return np.asarray(img), label


#生成验证码图片
def gen_verifycode_img(gen_dir, total_size, chars_set, chars_num, img_height, img_width, font_sizes):
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    image = ImageCaptcha(width=img_width, height=img_height, font_sizes=font_sizes)
    for i in range(total_size):
        label = ''.join(random.sample(chars_set, chars_num))
        image.write(label, os.path.join(gen_dir, label+'_num'+str(i)+'.png'))


#独热码转文本
def one_hot_to_texts(one_hot_code):
    texts = []
    for i in range(one_hot_code.shape[0]):
        index = one_hot_code[i]
        texts.append(''.join([CHAR_SET[i] for i in index]))
    return texts

#文本转独热码
def label_to_one_hot(label, chars_num=CHAR_NUM, char_set=CHAR_SET):
    one_hot_label = np.zeros([chars_num, len(char_set)])
    offset = []
    index = []
    for i, c in enumerate(label):
        offset.append(i)
        index.append(char_set.index(c))
    one_hot_index = [offset, index]
    one_hot_label[one_hot_index] = 1.0
    return one_hot_label.astype(np.float32)



#转换成records
def conver_to_tfrecords(data_set, label_set, name):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    print('正在转换成tfrecords', name)
    writer = tf.python_io.TFRecordWriter(name)
    num_examples = len(data_set)
    for index in range(num_examples):
        image = data_set[index]
        height = image.shape[0]
        width = image.shape[1]
        image_raw = image.tostring()
        label = label_set[index]
        label_raw = label_to_one_hot(label).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'label_raw': _bytes_feature(label_raw),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    print('转换完毕!')

#生成名字列表和标签列表
def create_data_list(image_dir):
    if not os.path.exists(image_dir):
        return None, None
    images = []
    labels = []
    for file_name in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, file_name), 0)
        input_img = np.array(image, dtype='float32')
        label_name = os.path.basename(file_name).split('_')[0]
        images.append(input_img)
        labels.append(label_name)
    return images, labels


def read_and_decode(filename_queue, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, chars_num=CHAR_NUM, classes_num=len(CHAR_SET)):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label_raw': tf.FixedLenFeature([], tf.string),
      })
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image.set_shape([img_height * img_width])
    image = tf.cast(image, tf.float32) * (1.0 / 255)-0.5
    reshape_image = tf.reshape(image, [img_height, img_width, 1])
    label = tf.decode_raw(features['label_raw'], tf.float32)
    label.set_shape([chars_num * classes_num])
    reshape_label = tf.reshape(label, [chars_num, classes_num])
    return tf.cast(reshape_image, tf.float32), tf.cast(reshape_label, tf.float32)


def inputs(train, batch_size, epoch):
    filename = os.path.join('./', TRAIN_RECORDS_NAME if train else TRAIN_VALIDATE_NAME)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=epoch)
        image, label = read_and_decode(filename_queue)
        if train:
            images, sparse_labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=6, capacity=2000 + 3 * batch_size, min_after_dequeue=2000)
        else:
            images, sparse_labels = tf.train.batch([image, label], batch_size=batch_size, num_threads=6, capacity=2000 + 3 * batch_size)

    return images, sparse_labels



if __name__ == '__main__':
    print('在%s生成%d个验证码' % (TRAIN_IMG_PATH, TRAIN_SIZE))
    gen_verifycode_img(TRAIN_IMG_PATH, TRAIN_SIZE, CHAR_SET, CHAR_NUM, IMG_HEIGHT, IMG_WIDTH, FONT_SIZES)
    print('在%s生成%d个验证码' % (VALID_IMG_PATH, VALID_SIZE))
    gen_verifycode_img(VALID_IMG_PATH, VALID_SIZE, CHAR_SET, CHAR_NUM, IMG_HEIGHT, IMG_WIDTH, FONT_SIZES)
    print('生成完毕')
    #开始生成record文件
    #把训练图转成tfrecords
    training_data, training_label = create_data_list(TRAIN_IMG_PATH)
    conver_to_tfrecords(training_data, training_label, TRAIN_RECORDS_NAME)
    # 把验证图转成tfrecords
    validation_data, validation_label = create_data_list(VALID_IMG_PATH)
    conver_to_tfrecords(validation_data, validation_label, TRAIN_VALIDATE_NAME)