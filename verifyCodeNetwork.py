#encoding=utf8

import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import utility
import math
import os
import cv2

class verify_code_network(object):
    def __init__(self, is_training=True):
        #一堆常量
        self.__IMAGE_HEIGHT = utility.IMG_HEIGHT
        self.__IMAGE_WIDTH = utility.IMG_WIDTH
        self.__CHAR_SETS = utility.CHAR_SET
        self.__CLASSES_NUM = len(self.__CHAR_SETS)
        self.__CHARS_NUM = utility.CHAR_NUM
        self.__TRAIN_IMG_DIR = utility.TRAIN_IMG_PATH
        self.__VALID_DIR = utility.VALID_IMG_PATH
        self.__BATCH_SIZE = utility.BATCH_SIZE
        if is_training == False:
            self.__image = tf.placeholder(tf.float32, shape=[1, utility.IMG_HEIGHT, utility.IMG_WIDTH, 1])
            self.__logits = self.__inference(self.__image, keep_prob=1)
            self.__result = self.output(self.__logits)
            self.__sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(self.__sess, tf.train.latest_checkpoint('./model'))



    '''
    一堆在定义model的时候要用到的小工具函数
    '''
    def __conv2d(self, value, weight):
        return tf.nn.conv2d(value, weight, strides=[1, 1, 1, 1], padding='SAME')


    def __max_pool_2x2(self, value, name):
        return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


    def __weight_variable(self, name, shape):
        initializer = tf.truncated_normal_initializer(stddev=0.1)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        return var


    def __bias_variable(self, name, shape):
        initializer = tf.constant_initializer(0.1)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        return var


    #推理
    def __inference(self, images, keep_prob):
        images = tf.reshape(images, (-1, utility.IMG_HEIGHT, utility.IMG_WIDTH, 1))
        # 用于tensorboard中可视化原图
        tf.summary.image('src_img', images, 5)

        with tf.variable_scope('conv1') as scope:
            kernel = self.__weight_variable('weights_1', shape=[5, 5, 1, 64])
            biases = self.__bias_variable('biases_1', [64])
            pre_activation = tf.nn.bias_add(self.__conv2d(images, kernel), biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            tf.summary.histogram('conv1/weights_1', kernel)
            tf.summary.histogram('conv1/biases_1', biases)

            kernel_2 = self.__weight_variable('weights_2', shape=[5, 5, 64, 64])
            biases_2 = self.__bias_variable('biases_2', [64])
            pre_activation = tf.nn.bias_add(self.__conv2d(conv1, kernel_2), biases_2)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            tf.summary.histogram('conv1/weights_2', kernel_2)
            tf.summary.histogram('conv1/biases_2', biases_2)


            # 用于可视化第一层卷积后的图像
            conv1_for_show1 = tf.reshape(conv1[:, :, :, 1], (-1, 60, 160, 1))
            conv1_for_show2 = tf.reshape(conv1[:, :, :, 2], (-1, 60, 160, 1))
            conv1_for_show3 = tf.reshape(conv1[:, :, :, 3], (-1, 60, 160, 1))
            tf.summary.image('conv1_for_show1', conv1_for_show1, 5)
            tf.summary.image('conv1_for_show2', conv1_for_show2, 5)
            tf.summary.image('conv1_for_show3', conv1_for_show3, 5)

            # max pooling
            pool1 = self.__max_pool_2x2(conv1, name='pool1')



        with tf.variable_scope('conv2') as scope:
            kernel = self.__weight_variable('weights_1', shape=[5, 5, 64, 64])
            biases = self.__bias_variable('biases_1', [64])
            pre_activation = tf.nn.bias_add(self.__conv2d(pool1, kernel), biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            tf.summary.histogram('conv2/weights_1', kernel)
            tf.summary.histogram('conv2/biases_1', biases)

            kernel_2 = self.__weight_variable('weights_2', shape=[5, 5, 64, 64])
            biases_2 = self.__bias_variable('biases_2', [64])
            pre_activation = tf.nn.bias_add(self.__conv2d(conv2, kernel_2), biases_2)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            tf.summary.histogram('conv2/weights_2', kernel_2)
            tf.summary.histogram('conv2/biases_2', biases_2)


            # 用于可视化第二层卷积后的图像
            conv2_for_show1 = tf.reshape(conv2[:, :, :, 1], (-1, 30, 80, 1))
            conv2_for_show2 = tf.reshape(conv2[:, :, :, 2], (-1, 30, 80, 1))
            conv2_for_show3 = tf.reshape(conv2[:, :, :, 3], (-1, 30, 80, 1))
            tf.summary.image('conv2_for_show1', conv2_for_show1, 5)
            tf.summary.image('conv2_for_show2', conv2_for_show2, 5)
            tf.summary.image('conv2_for_show3', conv2_for_show3, 5)

            # max pooling
            pool2 = self.__max_pool_2x2(conv2, name='pool2')



        with tf.variable_scope('conv3') as scope:
            kernel = self.__weight_variable('weights', shape=[3, 3, 64, 64])
            biases = self.__bias_variable('biases', [64])
            pre_activation = tf.nn.bias_add(self.__conv2d(pool2, kernel), biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            tf.summary.histogram('conv3/weights', kernel)
            tf.summary.histogram('conv3/biases', biases)

            kernel_2 = self.__weight_variable('weights_2', shape=[3, 3, 64, 64])
            biases_2 = self.__bias_variable('biases_2', [64])
            pre_activation = tf.nn.bias_add(self.__conv2d(conv3, kernel_2), biases_2)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            tf.summary.histogram('conv3/weights_2', kernel_2)
            tf.summary.histogram('conv3/biases_2', biases_2)

            conv3_for_show1 = tf.reshape(conv3[:, :, :, 1], (-1, 15, 40, 1))
            conv3_for_show2 = tf.reshape(conv3[:, :, :, 2], (-1, 15, 40, 1))
            conv3_for_show3 = tf.reshape(conv3[:, :, :, 3], (-1, 15, 40, 1))
            tf.summary.image('conv3_for_show1', conv3_for_show1, 5)
            tf.summary.image('conv3_for_show2', conv3_for_show2, 5)
            tf.summary.image('conv3_for_show3', conv3_for_show3, 5)

            pool3 = self.__max_pool_2x2(conv3, name='pool3')


        with tf.variable_scope('conv4') as scope:
            kernel = self.__weight_variable('weights', shape=[3, 3, 64, 64])
            biases = self.__bias_variable('biases', [64])
            pre_activation = tf.nn.bias_add(self.__conv2d(pool3, kernel), biases)
            conv4 = tf.nn.relu(pre_activation, name=scope.name)
            tf.summary.histogram('conv4/weights', kernel)
            tf.summary.histogram('conv4/biases', biases)

            conv4_for_show1 = tf.reshape(conv4[:, :, :, 1], (-1, 8, 20, 1))
            conv4_for_show2 = tf.reshape(conv4[:, :, :, 2], (-1, 8, 20, 1))
            conv4_for_show3 = tf.reshape(conv4[:, :, :, 3], (-1, 8, 20, 1))
            tf.summary.image('conv4_for_show1', conv4_for_show1, 5)
            tf.summary.image('conv4_for_show2', conv4_for_show2, 5)
            tf.summary.image('conv4_for_show3', conv4_for_show3, 5)

            pool4 = self.__max_pool_2x2(conv4, name='pool4')



        #全连接层
        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(pool4, [images.get_shape()[0].value, -1])
            weights = self.__weight_variable('weights', shape=[4*10*64, 1024])
            biases = self.__bias_variable('biases', [1024])
            local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            tf.summary.histogram('local1/weights', kernel)
            tf.summary.histogram('local1/biases', biases)

            local1_drop = tf.nn.dropout(local1, keep_prob)
            tf.summary.tensor_summary('local1/dropout', local1_drop)

        #输出层
        with tf.variable_scope('softmax_linear') as scope:
            weights = self.__weight_variable('weights', shape=[1024, self.__CHARS_NUM * self.__CLASSES_NUM])
            biases = self.__bias_variable('biases', [self.__CHARS_NUM * self.__CLASSES_NUM])
            result = tf.add(tf.matmul(local1_drop, weights), biases, name=scope.name)

        reshaped_result = tf.reshape(result, [-1, self.__CHARS_NUM, self.__CLASSES_NUM])
        return reshaped_result

    #计算cost
    def __loss(self, logits, labels):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='corss_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        tf.summary.scalar('loss', total_loss)
        return total_loss


    #训练
    def __training(self, loss):
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        return optimizer

    #评估正确度
    def __evaluation(self, logits, labels):
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(logits, 2), tf.argmax(labels, 2))
            correct_batch = tf.reduce_mean(tf.cast(correct_prediction, tf.int32), 1)
            accuracy = tf.reduce_sum(tf.cast(correct_batch, tf.int32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def train(self):
        if not os.path.exists(utility.LOG_DIR):
            os.mkdir(utility.LOG_DIR)
        if not os.path.exists(utility.MODEL_DIR):
            os.mkdir(utility.MODEL_DIR)

        step = 0
        images, labels = utility.inputs(train=True, batch_size=utility.BATCH_SIZE, epoch=90)
        logits = self.__inference(images, 0.5)
        loss = self.__loss(logits, labels)
        train_op = self.__training(loss)
        accuracy = self.__evaluation(logits, labels)
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            writer = tf.summary.FileWriter(utility.LOG_DIR, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                step = 0
                while not coord.should_stop():
                    start_time = time.time()
                    _, loss_value, performance, summaries = sess.run([train_op, loss, accuracy, summary_op])
                    duration = time.time() - start_time
                    if step % 10 == 0:
                        print('>> 已训练%d个批次: loss = %.2f (%.3f sec), 该批正确数量 = %d' % (step, loss_value, duration, performance))
                    if step % 100 == 0:
                        writer.add_summary(summaries, step)
                        saver.save(sess, utility.MODEL_DIR, global_step=step)
                    step += 1
            except tf.errors.OutOfRangeError:
                print('训练结束')
                saver.save(sess, utility.MODEL_DIR, global_step=step)
                coord.request_stop()
            finally:
                coord.request_stop()
            coord.join(threads)


    def valid(self):
        images, labels = utility.inputs(train=False, batch_size=100, epoch=None)
        logits = self.__inference(images, keep_prob=1)
        eval_correct = self.__evaluation(logits, labels)
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(utility.MODEL_DIR))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            num_iter = utility.VALID_SIZE/100
            true_count = 0
            total_true_count = 0
            total_sample_count = utility.VALID_SIZE
            step = 0
            while step < num_iter and not coord.should_stop():
                true_count = sess.run(eval_correct)
                total_true_count += true_count
                step += 1
            precision = total_true_count / total_sample_count
            print('正确数量/总数: %d/%d 正确率 = %.3f' % (total_true_count, total_sample_count, precision))
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape((utility.IMG_HEIGHT, utility.IMG_WIDTH, 1))
        input_img = np.array(image, dtype='float32')
        input_img = input_img/255.0-0.5
        predict_result = self.__sess.run(self.__result, feed_dict={self.__image : [input_img]})
        text = utility.one_hot_to_texts(predict_result)
        return text


    def output(self, logits):
        return tf.argmax(logits, 2)