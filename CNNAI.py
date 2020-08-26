# coding:utf-8
from BasicAI import BasicAI
import State
from State import CHANNEL
import numpy as np
import tensorflow as tf
import random
import copy
import sys
import h5py
from sklearn.utils import shuffle

FILTERS = 16
LAYER_NUM = 15
LEARNING_RATE = 1e-2
EPSILON = 1e-7
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.5  # 最大値の50%まで
    )
)


class CNNAI(BasicAI):
    def __init__(self, color, search_nodes=1, C_puct=2, tau=1, all_parameter_zero=False, v_is_dist=False, p_is_almost_flat=False):
        super(CNNAI, self).__init__(color, search_nodes, C_puct, tau)

        #self.features = 135
        self.input_channels = CHANNEL
        self.action_num = 137
        self.batch_size = 16
        self.all_parameter_zero = all_parameter_zero
        self.v_is_dist = v_is_dist
        self.p_is_almost_flat = p_is_almost_flat

        self.init_tensorflow()

    def init_tensorflow(self):
        # tensorflowの準備
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, 9, 9, self.input_channels], name="x")

        self.W_convs = [self.weight_variable([3, 3, self.input_channels, FILTERS], name="W_conv1")]
        self.b_convs = [self.bias_variable([FILTERS], name="b_conv1")]
        h_conv = tf.nn.relu(self.conv2d(self.x, self.W_convs[0]) + self.b_convs[0])
        old_h_conv = h_conv

        for i in range(1, LAYER_NUM):
            self.W_convs.append(self.weight_variable([3, 3, FILTERS, FILTERS], name="W_conv{}".format(i)))
            self.b_convs.append(self.bias_variable([FILTERS], name="b_conv{}".format(i)))
            if i % 2 == 0:
                h_conv = tf.nn.relu(self.conv2d(h_conv, self.W_convs[i]) + self.b_convs[i]) + old_h_conv
                old_h_conv = h_conv
            else:
                h_conv = tf.nn.relu(self.conv2d(h_conv, self.W_convs[i]) + self.b_convs[i])

        h_conv_flat = tf.reshape(h_conv, [-1, 9 * 9 * FILTERS])

        self.W_fc = self.weight_variable([9 * 9 * FILTERS, 128], name="W_fc")
        self.b_fc = self.weight_variable([128], name="b_fc")
        self.W1 = self.weight_variable([128, 1], name="W1")
        self.b1 = self.bias_variable([1], name="b1")
        self.y = tf.tanh(tf.matmul(h_conv_flat, self.W_fc) + self.b_fc)
        self.y = tf.tanh(tf.matmul(self.y, self.W1) + self.b1)

        #self.x_part = self.x[:, 0, 0, :]
        #self.W_part = self.weight_variable([CHANNEL, 1], name="W_fc_part")
        #self.b_part = self.bias_variable([1], name="b1")
        #self.y = tf.tanh(tf.matmul(self.x_part, self.W_part) + self.b_part)

        self.W2 = self.weight_variable([9 * 9 * FILTERS, self.action_num], name="W2")
        self.b2 = self.bias_variable([self.action_num], name="b2")
        parameters = self.W_convs + self.b_convs + [self.W1, self.b1, self.W2, self.b2, self.W_fc, self.b_fc]
        #parameters = self.W_convs + self.b_convs + [self.W_part, self.b_part, self.W2, self.b2]
        self.p_tf = tf.nn.softmax(tf.matmul(h_conv_flat, self.W2) + self.b2)

        self.y_ = tf.placeholder(tf.float32, [None, 1])
        self.pi = tf.placeholder(tf.float32, [None, self.action_num])
        self.loss = tf.reduce_mean(tf.square(self.y - self.y_) - tf.reduce_sum(self.pi * tf.log(self.p_tf + EPSILON), 1))
        for parameter in parameters:
            self.loss += 0.001 * tf.nn.l2_loss(parameter)

        self.train_step = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(self.loss)

        init = tf.initialize_all_variables()
        self.sess = tf.Session(config=config)
        self.sess.run(init)

    def weight_variable(self, shape, name):
        if self.all_parameter_zero:
            initial = tf.zeros(shape, name=name)
        else:
            initial = tf.truncated_normal(shape, stddev=0.001, name=name)
        return tf.Variable(initial)

    def bias_variable(self, shape, name):
        initial = tf.constant(0., shape=shape, name=name)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def v(self, s):
        return self.v_array([s])[0]

    def v_array(self, states, random_flip=False):
        if self.v_is_dist:
            y = np.zeros((len(states),))
            for i, s in enumerate(states):
                B_dist, W_dist = s.get_player_dist_from_goal()
                y[i] = np.tanh((W_dist - B_dist) * 0.15)
        else:
            feature = np.zeros((len(states), 9, 9, self.input_channels))
            for i, s in enumerate(states):
                xflip = False
                if random_flip and random.random() < 0.5:
                    xflip = True
                feature[i, :] = s.feature_CNN(xflip=xflip)
            y = self.sess.run(self.y, feed_dict={self.x:feature})

        for i, s in enumerate(states):
            _, y1 = s.color_p(0)
            _, y2 = s.color_p(1)
            if y1 == 0:
                y[i] = 1.
            elif y2 == State.BOARD_LEN - 1:
                y[i] = -1.
        if self.color == 0:
            return y
        else:
            return -y

    def p(self, s):
        return self.p_array([s])[0]

    def p_array(self, states, random_flip=False):
        mask = np.zeros((len(states), self.action_num))
        if self.p_is_almost_flat:
            for i, s in enumerate(states):
                r, c = s.placable_array(s.turn % 2)
                x, y = s.color_p(s.turn % 2)
                mask[i, :] = np.concatenate([r.flatten(), c.flatten(), s.movable_array(x, y, shortest_only=True).flatten()])
                if not np.any(mask[i, :]):  # 相手がゴールにいるせいで距離を縮められない場合などに起こる
                    mask[i, :] = np.concatenate(
                        [r.flatten(), c.flatten(), s.movable_array(x, y, shortest_only=False).flatten()])
            p = np.ones((len(states), self.action_num)) + np.random.rand(len(states), self.action_num) / 1000  # 1000は適当
            p = p / np.sum(p, axis=1).reshape((-1, 1))
        else:
            feature = np.zeros((len(states), 9, 9, self.input_channels))
            for i, s in enumerate(states):
                r, c = s.placable_array(s.turn % 2)
                x, y = s.color_p(s.turn % 2)
                mask[i, :] = np.concatenate([r.flatten(), c.flatten(), s.movable_array(x, y, shortest_only=True).flatten()])
                if not np.any(mask[i, :]):  # 相手がゴールにいるせいで距離を縮められない場合などに起こる
                    mask[i, :] = np.concatenate(
                        [r.flatten(), c.flatten(), s.movable_array(x, y, shortest_only=False).flatten()])
                feature[i, :] = s.feature_CNN()
                #if s.terminate:
                #    mask[i, :] = np.zeros((self.action_num,))
            p = self.sess.run(self.p_tf, feed_dict={self.x:feature})
        p = p + 1e-7  # maskした後pが0にならないように対策
        p = p * mask
        try:
            p = p / np.sum(p, axis=1).reshape((-1, 1))
        except:
            np.set_printoptions(threshold=np.inf)
            print(p)
            print(mask)
            print(feature)
            for state in states:
                state.display_cui()
            exit()
        return p

    def learn(self, epoch, h5_num, step_num, array_size):
        features = np.zeros((array_size, 9, 9, CHANNEL))
        pis = np.zeros((array_size, 137))
        rewards = np.zeros((array_size,))

        ema = 5.
        for i in range(step_num):
            size_per_file = array_size // h5_num
            for j in range(h5_num):
                h5file = h5py.File("data/{}.h5".format(epoch + 1 - h5_num + j), "r")
                size = h5file["feature"].shape[0]
                perm = np.sort(np.random.permutation(size)[:size_per_file])
                temp = h5file["feature"][:, :, :, :]
                features[j * size_per_file:(j + 1) * size_per_file] = temp[perm, :, :, :]
                temp = h5file["pi"][:, :]
                pis[j * size_per_file:(j + 1) * size_per_file] = temp[perm, :]
                temp = h5file["reward"][:]
                rewards[j * size_per_file:(j + 1) * size_per_file] = temp[perm]
            features, pis, rewards = shuffle(features, pis, rewards)

            for j in range(array_size // self.batch_size):
                self.sess.run(self.train_step, feed_dict={
                    self.x:features[j * self.batch_size:(j + 1) * self.batch_size],
                    self.pi:pis[j * self.batch_size:(j + 1) * self.batch_size],
                    self.y_:rewards[j * self.batch_size:(j + 1) * self.batch_size].reshape(-1, 1)})
                loss = self.sess.run(self.loss, feed_dict={
                    self.x:features[j * self.batch_size:(j + 1) * self.batch_size],
                    self.pi:pis[j * self.batch_size:(j + 1) * self.batch_size],
                    self.y_:rewards[j * self.batch_size:(j + 1) * self.batch_size].reshape(-1, 1)})
                ema = ema * 0.999 + loss * 0.001
                sys.stderr.write('\r\033[K' + str(j * self.batch_size) + "/" + str(features.shape[0]) + " " + str(ema))
                sys.stderr.flush()
            print("")

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        self.init_tensorflow()
        saver = tf.train.Saver()
        saver.restore(self.sess, path)









