# coding:utf-8
from BasicAI import BasicAI
import State
import numpy as np
import tensorflow as tf
import random
import copy
import sys


class LinearAI(BasicAI):
    def __init__(self, color, search_nodes=1, C_puct=1, tau=1):
        super().__init__(color, search_nodes, C_puct, tau)

        self.features = 135
        self.action_num = 137
        self.batch_size = 16

        self.init_tensorflow()

    def init_tensorflow(self):
        # tensorflowの準備
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, self.features], name="x")

        self.W1 = self.weight_variable([self.features, 1], name="W1")
        self.b1 = self.bias_variable([1], name="b1")
        self.y = tf.matmul(self.x, self.W1) + self.b1

        self.W2 = self.weight_variable([self.features, self.action_num], name="W2")
        self.b2 = self.bias_variable([self.action_num], name="b2")
        parameters = [self.W1, self.b1, self.W2, self.b2]
        self.p_tf = tf.nn.softmax(tf.matmul(self.x, self.W2) + self.b2)

        self.y_ = tf.placeholder(tf.float32, [None, 1])
        self.pi = tf.placeholder(tf.float32, [None, self.action_num])
        self.loss = tf.reduce_mean(tf.square(self.y - self.y_) - tf.reduce_sum(self.pi * tf.log(self.p_tf), axis=1))
        for parameter in parameters:
            self.loss += 0.001 * tf.nn.l2_loss(parameter)

        self.train_step = tf.train.AdagradOptimizer(0.01).minimize(self.loss)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def weight_variable(self, shape, name):
        initial = tf.zeros(shape, name=name)
        #initial = tf.truncated_normal(shape, stddev=0.1, name=name)
        return tf.Variable(initial)

    def bias_variable(self, shape, name):
        initial = tf.constant(0., shape=shape, name=name)
        return tf.Variable(initial)

    def v(self, s):
        return self.v_array([s])[0]

    def v_array(self, states):
        feature = np.zeros((len(states), self.features))
        for i, s in enumerate(states):
            feature[i, :] = s.feature()
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

    def p_array(self, states):
        mask = np.zeros((len(states), self.action_num))
        feature = np.zeros((len(states), self.features))
        for i, s in enumerate(states):
            r, c = s.placable_array(s.turn % 2)
            x, y = s.color_p(s.turn % 2)
            mask[i, :] = np.concatenate([r.flatten(), c.flatten(), s.movable_array(x, y).flatten()])
            feature[i, :] = s.feature()
            #if s.terminate:
            #    mask[i, :] = np.zeros((self.action_num,))
        p = self.sess.run(self.p_tf, feed_dict={self.x:feature})
        p = p * mask
        p = p / np.sum(p, axis=1).reshape((-1, 1))
        return p

    def learn(self, data, epoch_num):
        data = copy.deepcopy(data)
        random.shuffle(data)
        count = 0
        features = []
        pis = []
        rs = []
        ema = 100.
        for i in range(epoch_num):
            for feature, pi, r in data:
                #print(feature)
                #print(pi)
                #print(r)
                features.append(feature)
                pis.append(pi)
                rs.append(r)
                count += 1

                if count % self.batch_size == 0:
                    self.sess.run(self.train_step, feed_dict={
                        self.x:np.array(features).reshape((self.batch_size, self.features)),
                        self.pi:np.array(pis).reshape((self.batch_size, self.action_num)),
                        self.y_:np.array(rs).reshape((self.batch_size, 1))})
                    loss = self.sess.run(self.loss, feed_dict={
                        self.x:np.array(features).reshape((self.batch_size, self.features)),
                        self.pi:np.array(pis).reshape((self.batch_size, self.action_num)),
                        self.y_:np.array(rs).reshape((self.batch_size, 1))})
                    ema = ema * 0.95 + loss * 0.05
                    #print(loss)
                    sys.stderr.write('\r\033[K' + str(count) + "/" + str(len(data)) + " " + str(ema))
                    sys.stderr.flush()
                    features = []
                    pis = []
                    rs = []
            print("")
        print(self.sess.run(self.W1))
        #print(self.sess.run(self.W2))

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def load(self, path):
        self.init_tensorflow()
        saver = tf.train.Saver()
        saver.restore(self.sess, path)









