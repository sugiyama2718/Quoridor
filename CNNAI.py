# coding:utf-8
#from memory_profiler import profile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # warning抑制
import time
from BasicAI import BasicAI
import State
from State import CHANNEL, color_p, movable_array
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import copy
import sys
import h5py
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from config import *
from tqdm import tqdm
import math
import json
from pprint import pprint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# def layernorm(x, beta, gamma):
#     LAYERNORM_EPSILON = 1e-3
#     mean = tf.reduce_mean(x, axis=range(1, len(x.shape)), keepdims=True)
#     variance = tf.math.reduce_variance(x, axis=range(1, len(x.shape)), keepdims=True)
#     #std = tf.math.reduce_std(x, axis=range(1, len(x.shape)), keepdims=True)
#     #return gamma * (x - mean) + beta
#     return gamma * (x - mean) / tf.math.sqrt(variance + LAYERNORM_EPSILON) + beta
#     #return gamma * (x - mean) / (std + LAYERNORM_EPSILON) + beta


def layernorm(x, beta, gamma, filters):
    LAYERNORM_EPSILON = 1e-3
    position_num = x.shape[1] * x.shape[2]
    mean = tf.reshape(tf.reduce_mean(tf.reshape(x, [-1, position_num * filters]), axis=1), [-1, 1, 1, 1])
    variance = tf.square(x - mean)
    return gamma * (x - mean) / tf.sqrt(variance + LAYERNORM_EPSILON) + beta

class CNNAI(BasicAI):
    def __init__(self, color, search_nodes=1, C_puct=2, tau=1, all_parameter_zero=False, v_is_dist=False, p_is_almost_flat=False, 
    seed=0, use_estimated_V=True, V_ema_w=0.01, shortest_only=False, per_process_gpu_memory_fraction=PER_PROCESS_GPU_MEMORY_FRACTION, use_average_Q=False, random_playouts=False,
    filters=DEFAULT_FILTERS, layer_num=DEFAULT_LAYER_NUM, use_global_pooling=USE_GLOBAL_POOLING, use_self_attention=USE_SELF_ATTENTION, use_slim_head=USE_SLIM_HEAD, opponent_AI=None,
    is_mimic_AI=False, force_opening=None, use_mix_precision=True):
        super(CNNAI, self).__init__(color, search_nodes, C_puct, tau, use_estimated_V=use_estimated_V, V_ema_w=V_ema_w, shortest_only=shortest_only, use_average_Q=use_average_Q, random_playouts=random_playouts, is_mimic_AI=is_mimic_AI, force_opening=force_opening)

        np.random.seed(seed)
        random.seed(seed)

        #self.features = 135
        self.input_channels = CHANNEL
        self.action_num = 137
        self.batch_size = BATCH_SIZE
        self.all_parameter_zero = all_parameter_zero
        self.v_is_dist = v_is_dist
        self.p_is_almost_flat = p_is_almost_flat
        self.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        self.filters = filters
        self.layer_num = layer_num
        self.use_global_pooling = use_global_pooling
        self.use_self_attention = use_self_attention
        self.use_slim_head = use_slim_head
        self.opponent_AI = opponent_AI  # tensorflowを自己対戦の２AIで共有するための変数。TODO: 非合法手がまだ出るので修正は必要。ただしGPUメモリが節約できなかったので着手していない。
        self.use_mix_precision = use_mix_precision

        if self.opponent_AI is None:
            self.init_tensorflow()

    def init_tensorflow(self):
        # tensorflowの準備
        tf.reset_default_graph()
        self.x = tf.placeholder(tf.float32, [None, 9, 9, self.input_channels], name="x")
        self.searched_node = tf.placeholder(tf.float32, [None], name="searched_node")
        self.warmup_coef = tf.placeholder(tf.float32, name="warmup_coef")

        LAST_CONV_FILTER_NUM = self.filters // 4
        #LAST_CONV_FILTER_NUM = 1

        # ---CNNの本体を定義する---
        LAYER_NORM_LEN = 1
        self.W_convs = [self.weight_variable([3, 3, self.input_channels, self.filters], name="W_conv1")]
        self.W_globals = [self.weight_variable([2 * self.filters, self.filters], name="W_global1")]
        self.b_globals = [self.bias_variable([self.filters], name="b_global1")]
        self.betas = [self.bias_variable([LAYER_NORM_LEN, LAYER_NORM_LEN, self.filters], name="beta1")]
        self.gammas = [self.weight_variable([LAYER_NORM_LEN, LAYER_NORM_LEN, self.filters], name="gamma1")]
        # if self.use_self_attention:
        #     self.b_convs = [self.bias_variable([self.filters], name="b_conv1")]
        #h_conv = tf.nn.relu(self.conv2d(self.x, self.W_convs[0]) + self.b_convs[0])

        #x = self.gamma * (x - mean) / tf.math.sqrt(variance + LAYERNORM_EPSILON) + self.beta
        h_conv = tf.nn.relu(layernorm(self.conv2d(self.x, self.W_convs[0]), self.betas[0], self.gammas[0], self.filters))

        # self attention
        head_num = self.filters // ATTENTION_VEC_LEN
        if self.use_self_attention:
            self.WQs = [self.weight_variable([self.input_channels, self.filters], name="WQ0")]
            self.WKs = [self.weight_variable([self.input_channels, self.filters], name="WK0")]
            self.WVs = [self.weight_variable([self.input_channels, self.filters], name="WV0")]

            position_arr = np.zeros((9, 9, self.filters), dtype=np.float32)
            for x in range(9):
                for y in range(9):
                    position_arr[x, y, 0] = (x - 4) / 4
                    position_arr[x, y, 1] = (y - 4) / 4
                    position_arr[x, y, 2] = x % 3 - 1
                    position_arr[x, y, 3] = y % 3 - 1
                    position_arr[x, y, 4] = x // 3 - 1
                    position_arr[x, y, 5] = y // 3 - 1
            h_conv = h_conv + tf.constant(position_arr)
            # for i in range(6):
            #     print(position_arr[:, :, i])

        # Headの分割
        def split_heads(x, num_heads):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]
            depth = tf.shape(x)[2] // num_heads
            x = tf.reshape(x, [batch_size, length, num_heads, depth])
            return tf.transpose(x, [0, 2, 1, 3])  # [batch_size, num_heads, length, depth]

        # 元の形に戻す
        def combine_heads(x):
            batch_size = tf.shape(x)[0]
            num_heads = tf.shape(x)[1]
            length = tf.shape(x)[2]
            depth = tf.shape(x)[3]
            x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, length, num_heads, depth]
            x = tf.reshape(x, [batch_size, length, num_heads * depth])
            return x

        old_h_conv = h_conv

        for i in range(1, self.layer_num):
            if self.use_self_attention:
                if i <= BROAD_CONV_LAYER_NUM:
                    self.W_convs.append(self.weight_variable([3, 3, self.filters, self.filters], name="W_conv{}".format(i)))
                else:
                    self.W_convs.append(self.weight_variable([1, 1, self.filters, self.filters], name="W_conv{}".format(i)))
                #self.b_convs.append(self.bias_variable([self.filters], name=f"b_conv{i}"))
                self.betas.append(self.bias_variable([LAYER_NORM_LEN, LAYER_NORM_LEN, self.filters], name=f"beta{i}"))
                self.gammas.append(self.weight_variable([LAYER_NORM_LEN, LAYER_NORM_LEN, self.filters], name=f"gamma{i}"))
                
                self.WQs.append(self.weight_variable([self.filters, self.filters], name=f"WQ{i}"))
                self.WKs.append(self.weight_variable([self.filters, self.filters], name=f"WK{i}"))
                self.WVs.append(self.weight_variable([self.filters, self.filters], name=f"WV{i}"))
                
                # if i % ATTENTION_CYCLE >= 2:
                #     h_conv = tf.nn.relu(self.conv2d(h_conv, self.W_convs[i]) + self.b_convs[i])  # layernormは重いから節約
                # elif i % ATTENTION_CYCLE == 1:
                #     h_conv = tf.nn.relu(layernorm(self.conv2d(h_conv, self.W_convs[i]), self.betas[i], self.gammas[i], self.filters))
                if i % ATTENTION_CYCLE >= 1:
                    h_conv = tf.nn.relu(layernorm(self.conv2d(h_conv, self.W_convs[i]), self.betas[i], self.gammas[i], self.filters))
                else:
                    h_conv = layernorm(h_conv, self.betas[i], self.gammas[i], self.filters)
                    h_shape1 = h_conv.shape[1]
                    h_shape2 = h_conv.shape[2]
                    position_num = h_shape1 * h_shape2
                    h_conv = tf.reshape(h_conv, [-1, position_num, self.filters])

                    h_query = tf.matmul(h_conv, self.WQs[i])
                    h_key = tf.matmul(h_conv, self.WKs[i])
                    h_value = tf.matmul(h_conv, self.WVs[i])

                    h_query = split_heads(h_query, head_num)
                    h_key = split_heads(h_key, head_num)
                    h_value = split_heads(h_value, head_num)

                    # Attention Scoresの計算
                    scores = tf.matmul(h_query, h_key, transpose_b=True)
                    scores = scores / tf.sqrt(float(ATTENTION_VEC_LEN))
                    attention_weights = tf.nn.softmax(scores, axis=-1)

                    # Weighted sum of values
                    h_conv = tf.matmul(attention_weights, h_value)  # [batch_size, num_heads, length, depth]

                    h_conv = combine_heads(h_conv)
                    h_conv = tf.reshape(h_conv, [-1, h_shape1, h_shape2, self.filters])

                h_conv += old_h_conv
                old_h_conv = h_conv
            else:
                self.W_convs.append(self.weight_variable([3, 3, self.filters, self.filters], name="W_conv{}".format(i)))
                self.betas.append(self.bias_variable([LAYER_NORM_LEN, LAYER_NORM_LEN, self.filters], name=f"beta{i}"))
                self.gammas.append(self.weight_variable([LAYER_NORM_LEN, LAYER_NORM_LEN, self.filters], name=f"gamma{i}"))
                self.W_globals.append(self.weight_variable([2 * self.filters, self.filters], name="W_global{}".format(i)))
                self.b_globals.append(self.bias_variable([self.filters], name=f"b_global{i}"))

                if i % 2 == 0:
                    if self.use_global_pooling:
                        global_features = tf.concat([
                            tf.reduce_mean(old_h_conv, axis=[1, 2]), 
                            tf.reduce_max(old_h_conv, axis=[1, 2])], axis=1)  # filterごとに盤面平均・最大を取る
                        global_features = tf.nn.relu(tf.matmul(global_features, self.W_globals[i]) + self.b_globals[i])
                        global_features = tf.reshape(global_features, [-1, 1, 1, self.filters])
                    else:
                        global_features = 0
                    h_conv = tf.nn.relu(layernorm(self.conv2d(h_conv, self.W_convs[i]), self.betas[i], self.gammas[i], self.filters)) + old_h_conv + global_features
                    old_h_conv = h_conv
                else:
                    h_conv = tf.nn.relu(self.conv2d(h_conv, self.W_convs[i]) + self.betas[i])  # layernormは計算コストが高いので２層に一回にする
                    #h_conv = tf.nn.relu(layernorm(self.conv2d(h_conv, self.W_convs[i]), self.betas[i], self.gammas[i], self.filters))

        # ---CNNのy, pのheadを定義する---
        position_num = h_conv.shape[1] * h_conv.shape[2]
        if self.use_slim_head:
            self.W_y_head_conv = self.weight_variable([3, 3, self.filters, LAST_CONV_FILTER_NUM], name="W_y_head_conv")
            self.b_y_head_conv = self.bias_variable([LAST_CONV_FILTER_NUM], name="b_y_head_conv")
            # self.W_y_head_conv2 = self.weight_variable([1, 1, LAST_CONV_FILTER_NUM, LAST_CONV_FILTER_NUM], name="W_y_head_conv2")
            # self.b_y_head_conv2 = self.bias_variable([LAST_CONV_FILTER_NUM], name="b_y_head_conv2")
            # self.head_beta = self.bias_variable([LAYER_NORM_LEN, LAYER_NORM_LEN, LAST_CONV_FILTER_NUM], name="head_beta")
            # self.head_gamma = self.weight_variable([LAYER_NORM_LEN, LAYER_NORM_LEN, LAST_CONV_FILTER_NUM], name="head_gamma")

            y_head_conv_pre = self.conv2d(h_conv, self.W_y_head_conv) + self.b_y_head_conv
            #y_head_conv_pre = tf.nn.relu(layernorm(self.conv2d(h_conv, self.W_y_head_conv), self.head_beta, self.head_gamma, LAST_CONV_FILTER_NUM))  # global average poolingのためreluはしない
            y_head_flat = tf.reshape(y_head_conv_pre, [-1, position_num * LAST_CONV_FILTER_NUM]) 

            # y_head_conv = self.conv2d(y_head_conv_pre, self.W_y_head_conv2) + self.b_y_head_conv2

            self.y = tf.reshape(tf.tanh(tf.reduce_mean(y_head_conv_pre, axis=[1, 2, 3])), [-1, 1])  # global average pooling

            self.W_p_head_conv = self.weight_variable([3, 3, self.filters, LAST_CONV_FILTER_NUM], name="W_p_head_conv")
            self.b_p_head_conv = self.bias_variable([LAST_CONV_FILTER_NUM], name="b_p_head_conv")

            self.W2 = self.weight_variable([position_num * LAST_CONV_FILTER_NUM, self.action_num], name="W2")
            self.b2 = self.bias_variable([self.action_num], name="b2")

            p_head_conv = tf.nn.relu(self.conv2d(h_conv, self.W_p_head_conv) + self.b_p_head_conv)
            p_head_flat = tf.reshape(p_head_conv, [-1, position_num * LAST_CONV_FILTER_NUM])
            self.p_tf = tf.nn.softmax(tf.matmul(p_head_flat, self.W2) + self.b2)
            
            # self.W_p_head_conv2 = self.weight_variable([1, 1, LAST_CONV_FILTER_NUM, self.action_num], name="W_p_head_conv2")
            # self.b_p_head_conv2 = self.bias_variable([self.action_num], name="b_p_head_conv2")
            # p_head_conv2 = self.conv2d(p_head_conv, self.W_p_head_conv2) + self.b_p_head_conv2
            # self.p_tf = tf.nn.softmax(tf.reduce_mean(p_head_conv2, axis=[1, 2]))

            head_params = [self.W_y_head_conv, self.b_y_head_conv,
                           self.W_p_head_conv, self.b_p_head_conv, self.W2, self.b2]
            # head_params = [self.W_y_head_conv, self.W_y_head_conv2, self.b_y_head_conv2, self.head_beta, self.head_gamma,
            #                self.W_p_head_conv, self.b_p_head_conv, self.W2, self.b2]

            # ----policyの過学習を抑える試み----

            # self.W_y_head_conv = self.weight_variable([3, 3, self.filters, LAST_CONV_FILTER_NUM], name="W_y_head_conv")
            # self.b_y_head_conv = self.bias_variable([LAST_CONV_FILTER_NUM], name="b_y_head_conv")

            # y_head_conv = tf.nn.relu(self.conv2d(h_conv, self.W_y_head_conv) + self.b_y_head_conv)
            # y_head_flat = tf.reshape(y_head_conv, [-1, 9 * 9 * LAST_CONV_FILTER_NUM])

            # self.W1 = self.weight_variable([9 * 9 * LAST_CONV_FILTER_NUM, 1], name="W1")
            # self.b1 = self.bias_variable([1], name="b1")
            # self.y = tf.tanh(tf.matmul(y_head_flat, self.W1) + self.b1)

            # self.W_p_head_conv = self.weight_variable([2, 2, self.filters, LAST_CONV_FILTER_NUM], name="W_p_head_conv")  # 8*8に落とすためにフィルタサイズは2*2である必要あり
            # self.b_p_head_conv = self.bias_variable([LAST_CONV_FILTER_NUM], name="b_p_head_conv")
            # # self.W_p_head_wall = self.weight_variable([8, 8, LAST_CONV_FILTER_NUM, 2], name="W_p_head_wall")  # matmulするとshapeが なぜか 合わずエラー
            # # self.b_p_head_wall = self.bias_variable([8, 8, 2], name="b_p_head_wall")
            # self.W_p_head_wall = self.weight_variable([1, 1, LAST_CONV_FILTER_NUM, 2], name="W_p_head_wall")
            # self.b_p_head_wall = self.bias_variable([2], name="b_p_head_wall")
            # self.W_p_head_move = self.weight_variable([8 * 8 * LAST_CONV_FILTER_NUM, 9], name="W_p_head_move")
            # self.b_p_head_move = self.bias_variable([9], name="b_p_head_move")

            # p_head_conv = tf.nn.relu(self.conv2d(h_conv, self.W_p_head_conv, padding="VALID") + self.b_p_head_conv)  # paddingはVALIDにすることで8*8に落とす
            # #p_head_conv2 = tf.reshape(p_head_conv, [-1, 1, 8, 8, LAST_CONV_FILTER_NUM])
            # p_head_flat = tf.reshape(p_head_conv, [-1, 8 * 8 * LAST_CONV_FILTER_NUM])

            # p_wall = self.conv2d(p_head_conv, self.W_p_head_wall) + self.b_p_head_wall
            # p_wall_flat = tf.reshape(p_wall, [-1, 8 * 8 * 2])
            # p_move = tf.matmul(p_head_flat, self.W_p_head_move) + self.b_p_head_move

            # self.p_tf = tf.nn.softmax(tf.concat([p_wall_flat, p_move], axis=1))

            # head_params = [self.W_y_head_conv, self.b_y_head_conv, self.W1, self.b1, 
            #                self.W_p_head_conv, self.b_p_head_conv, self.W_p_head_wall, self.b_p_head_wall, self.W_p_head_move, self.b_p_head_move]
        else:
            self.W_y_head_conv = self.weight_variable([3, 3, self.filters, LAST_CONV_FILTER_NUM], name="W_y_head_conv")
            self.b_y_head_conv = self.bias_variable([LAST_CONV_FILTER_NUM], name="b_y_head_conv")
            self.W_p_head_conv = self.weight_variable([3, 3, self.filters, LAST_CONV_FILTER_NUM], name="W_p_head_conv")
            self.b_p_head_conv = self.bias_variable([LAST_CONV_FILTER_NUM], name="b_p_head_conv")

            y_head_conv = tf.nn.relu(self.conv2d(h_conv, self.W_y_head_conv) + self.b_y_head_conv)
            y_head_flat = tf.reshape(y_head_conv, [-1, position_num * LAST_CONV_FILTER_NUM])
            p_head_conv = tf.nn.relu(self.conv2d(h_conv, self.W_p_head_conv) + self.b_p_head_conv)
            p_head_flat = tf.reshape(p_head_conv, [-1, position_num * LAST_CONV_FILTER_NUM])

            self.W1 = self.weight_variable([position_num * LAST_CONV_FILTER_NUM, 1], name="W1")
            self.b1 = self.bias_variable([1], name="b1")
            self.y = tf.tanh(tf.matmul(y_head_flat, self.W1) + self.b1)

            self.W2 = self.weight_variable([position_num * LAST_CONV_FILTER_NUM, self.action_num], name="W2")
            self.b2 = self.bias_variable([self.action_num], name="b2")
            
            self.p_tf = tf.nn.softmax(tf.matmul(p_head_flat, self.W2) + self.b2)

            head_params = [self.W_y_head_conv, self.b_y_head_conv, self.W1, self.b1, 
                           self.W_p_head_conv, self.b_p_head_conv, self.W2, self.b2]

        self.y_ = tf.placeholder(tf.float32, [None, 1])
        self.pi = tf.placeholder(tf.float32, [None, self.action_num])
        #self.v_loss = tf.reduce_mean(tf.square(self.y - self.y_))
        y_normed = self.y / 2 + 0.5
        y_label_normed = self.y_ / 2 + 0.5
        self.v_loss = tf.reduce_mean(-tf.reduce_sum(y_label_normed * tf.log(y_normed + EPSILON) + (1 - y_label_normed) * tf.log((1 - y_normed) + EPSILON), 1))
        #self.v_loss = tf.reduce_mean(tf.pow(tf.square(self.y - self.y_), 3))
        #self.p_loss = tf.reduce_mean(-tf.reduce_sum(self.pi * tf.log(self.p_tf + EPSILON), 1))
        #print(self.searched_node.shape, self.warmup_coef.shape)
        #p_coef = tf.pow(self.searched_node / SELFPLAY_SEARCHNODES_MAX, self.warmup_coef)  # searched_nodeを使う前はvalueの学習ができていて、使うと失敗するのでwarmup_coefで補間
        p_coef = self.searched_node / tf.math.reduce_max(self.searched_node)
        #print(p_coef.shape)
        self.p_loss = tf.reduce_mean(-tf.reduce_sum(self.pi * tf.log(self.p_tf + EPSILON), 1) * p_coef)
        self.y_regularizer = tf.reduce_mean(tf.square(self.y))
        self.p_regularizer = tf.reduce_mean(-tf.log(1 / self.action_num + EPSILON)
        + tf.reduce_sum(self.p_tf * tf.log(self.p_tf + EPSILON), 1))  # 負のエントロピー->最小化するとエントロピーが最大化つまり一様分布に近づく

        # ---補助的な目的関数を定義するための、目的関数定義直前までの共通する部分の記述---
        aux_ph_list = ["dist_diff", "black_walls", "white_walls", "remaining_turn_num", "remaining_black_moves", "remaining_white_moves", 
                "row_wall", "column_wall", "dist_array1", "dist_array2", "B_traversed_arr", "W_traversed_arr", "next_pi"]
        self.aux_ph_dict = {}
        self.W_aux_head_conv_dict = {}
        self.b_aux_head_conv_dict = {}
        self.W_aux_head_dict = {}
        self.b_aux_head_dict = {}
        aux_head_conv_dict = {}
        aux_head_flat_dict = {}
        self.aux_loss_dict = {}
        aux_dict = {}
        self.aux_loss = 0.0
        for ph_name in aux_ph_list:
            self.W_aux_head_conv_dict[ph_name] = self.weight_variable([3, 3, self.filters, LAST_CONV_FILTER_NUM], name=f"W_{ph_name}_head_conv")
            self.b_aux_head_conv_dict[ph_name] = self.bias_variable([LAST_CONV_FILTER_NUM], name=f"b_{ph_name}_head_conv")
            aux_head_conv_dict[ph_name] = tf.nn.relu(self.conv2d(h_conv, self.W_aux_head_conv_dict[ph_name]) + self.b_aux_head_conv_dict[ph_name])
            aux_head_flat_dict[ph_name] = tf.reshape(aux_head_conv_dict[ph_name], [-1, position_num * LAST_CONV_FILTER_NUM])

        # ---最終的な距離差等の、補助的な回帰部分の定義---
        reg_ph_list = list(REG_SCALE_DICT.keys())
        for ph_name in reg_ph_list:
            self.aux_ph_dict[ph_name] = tf.placeholder(tf.float32, [None], name=ph_name)
            self.W_aux_head_dict[ph_name] = self.weight_variable([position_num * LAST_CONV_FILTER_NUM, 1], name=f"W_{ph_name}")
            self.b_aux_head_dict[ph_name] = self.bias_variable([1], name=f"b_{ph_name}")
            aux_dict[ph_name] = tf.matmul(y_head_flat, self.W_aux_head_dict[ph_name]) + self.b_aux_head_dict[ph_name]
            self.aux_loss_dict[ph_name] = tf.reduce_mean(tf.square((aux_dict[ph_name] - tf.reshape(self.aux_ph_dict[ph_name], [-1,  1])) / REG_SCALE_DICT[ph_name]))
            self.aux_loss += self.aux_loss_dict[ph_name]

        for ph_name in ["row_wall", "column_wall"]:
            self.aux_ph_dict[ph_name] = tf.placeholder(tf.float32, [None, 8, 8], name=ph_name)
            self.W_aux_head_dict[ph_name] = self.weight_variable([position_num * LAST_CONV_FILTER_NUM, 64], name=f"W_{ph_name}")
            self.b_aux_head_dict[ph_name] = self.bias_variable([64], name=f"b_{ph_name}")
            aux_dict[ph_name] = tf.math.sigmoid(tf.matmul(y_head_flat, self.W_aux_head_dict[ph_name]) + self.b_aux_head_dict[ph_name])
            wall_label = tf.reshape(self.aux_ph_dict[ph_name], [-1, 64])
            self.aux_loss_dict[ph_name] = tf.reduce_mean(-tf.reduce_mean(wall_label * tf.log(aux_dict[ph_name] + EPSILON) + (1 - wall_label) * tf.log((1 - aux_dict[ph_name]) + EPSILON), 1))
            self.aux_loss += self.aux_loss_dict[ph_name]

        for ph_name in ["dist_array1", "dist_array2"]:
            self.aux_ph_dict[ph_name] = tf.placeholder(tf.float32, [None, 9, 9], name=ph_name)
            self.W_aux_head_dict[ph_name] = self.weight_variable([position_num * LAST_CONV_FILTER_NUM, 81], name=f"W_{ph_name}")
            self.b_aux_head_dict[ph_name] = self.bias_variable([81], name=f"b_{ph_name}")
            aux_dict[ph_name] = tf.matmul(y_head_flat, self.W_aux_head_dict[ph_name]) + self.b_aux_head_dict[ph_name]
            dist_label = tf.reshape(self.aux_ph_dict[ph_name], [-1, 81])
            self.aux_loss_dict[ph_name] = tf.reduce_mean(tf.square((aux_dict[ph_name] - dist_label) / DIST_ARRAY_SCALE))
            self.aux_loss += self.aux_loss_dict[ph_name]

        for ph_name in ["B_traversed_arr", "W_traversed_arr"]:
            self.aux_ph_dict[ph_name] = tf.placeholder(tf.float32, [None, 9, 9], name=ph_name)
            self.W_aux_head_dict[ph_name] = self.weight_variable([position_num * LAST_CONV_FILTER_NUM, 81], name=f"W_{ph_name}")
            self.b_aux_head_dict[ph_name] = self.bias_variable([81], name=f"b_{ph_name}")
            aux_dict[ph_name] = tf.math.sigmoid(tf.matmul(y_head_flat, self.W_aux_head_dict[ph_name]) + self.b_aux_head_dict[ph_name])
            traverse_label = tf.reshape(self.aux_ph_dict[ph_name], [-1, 81])
            self.aux_loss_dict[ph_name] = tf.reduce_mean(-tf.reduce_mean(traverse_label * tf.log(aux_dict[ph_name] + EPSILON) + (1 - traverse_label) * tf.log((1 - aux_dict[ph_name]) + EPSILON), 1))
            self.aux_loss += self.aux_loss_dict[ph_name]

        for ph_name in ["next_pi"]:
            self.aux_ph_dict[ph_name] = tf.placeholder(tf.float32, [None, self.action_num], name=ph_name)
            self.W_aux_head_dict[ph_name] = self.weight_variable([position_num * LAST_CONV_FILTER_NUM, self.action_num], name=f"W_{ph_name}")
            self.b_aux_head_dict[ph_name] = self.bias_variable([self.action_num], name=f"b_{ph_name}")
            aux_dict[ph_name] = tf.nn.softmax(tf.matmul(y_head_flat, self.W_aux_head_dict[ph_name]) + self.b_aux_head_dict[ph_name])
            self.aux_loss_dict[ph_name] = NEXT_PI_IMPORTANCE * tf.reduce_mean(-tf.reduce_sum(self.aux_ph_dict[ph_name] * tf.log(aux_dict[ph_name] + EPSILON), 1) * p_coef)
            self.aux_loss += self.aux_loss_dict[ph_name]

        # 正則化用にパラメータすべてをリストでつなげる
        if self.use_self_attention:
            trunk_params = self.W_convs + self.gammas + self.betas + self.W_globals + self.b_globals + self.WQs + self.WKs + self.WVs
        else:
            trunk_params = self.W_convs + self.gammas + self.betas + self.W_globals + self.b_globals
        parameters = trunk_params + head_params
        for aux_ph in aux_ph_list:
            parameters.extend([self.W_aux_head_conv_dict[aux_ph], self.b_aux_head_conv_dict[aux_ph], self.W_aux_head_dict[aux_ph], self.b_aux_head_dict[aux_ph]])

        self.loss = self.v_loss + self.p_loss + V_REGULARIZER * self.y_regularizer + P_REGULARIZER * self.p_regularizer + AUX_IMPORTANCE * self.aux_loss
        self.loss_without_regularizer = self.v_loss + self.p_loss
        for parameter in parameters:
            self.loss += WEIGHT_DECAY * tf.nn.l2_loss(parameter)
        self.loss = self.warmup_coef * self.loss

        if self.use_mix_precision:
            self.train_step = tf.train.AdagradOptimizer(LEARNING_RATE)  # 学習の関数で定義しようとするとエラー出る
            self.train_step = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.train_step)
            self.train_step = self.train_step.minimize(self.loss)
        else:
            self.train_step = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(self.loss)

        init = tf.initialize_all_variables()
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                allow_growth=True
                #per_process_gpu_memory_fraction=self.per_process_gpu_memory_fraction
            )
        )
        self.sess = tf.Session(config=config)
        self.sess.run(init)

    def weight_variable(self, shape, name, stddev=0.001):
        if self.all_parameter_zero:
            initial = tf.zeros(shape, name=name)
        else:
            initial = tf.truncated_normal(shape, stddev=stddev, name=name)
        return tf.Variable(initial)

    def bias_variable(self, shape, name):
        initial = tf.constant(0., shape=shape, name=name)
        return tf.Variable(initial)

    def conv2d(self, x, W, padding="SAME"):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    def v(self, s):
        return self.v_array([s])[0]

    def v_array(self, states, random_flip=False):
        if self.opponent_AI is not None:
            return self.opponent_AI.v_array(states, random_flip)
        
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
            if s.pseudo_terminate:
                y[i] = s.pseudo_reward
        if self.color == 0:
            return y
        else:
            return -y

    def p(self, s, leaf_movable_arrs=None):
        return self.p_array([s], leaf_movable_arrs=leaf_movable_arrs)[0]

    def p_array(self, states, random_flip=False, leaf_movable_arrs=None):
        if self.opponent_AI is not None:
            return self.opponent_AI.p_array(states, random_flip, leaf_movable_arrs)

        mask = np.zeros((len(states), self.action_num))
        if self.p_is_almost_flat:
            for i, s, movable_arr in zip(range(len(states)), states, leaf_movable_arrs):
                r, c = s.placable_array(s.turn % 2)
                x, y = color_p(s, s.turn % 2)
                mask[i, :] = np.concatenate([r.flatten(), c.flatten(), movable_arr])
                if not np.any(mask[i, :]):  # 相手がゴールにいるせいで距離を縮められない場合などに起こる
                    mask[i, :] = np.concatenate(
                        [r.flatten(), c.flatten(), movable_array(s, x, y, shortest_only=False).flatten()])
            p = np.ones((len(states), self.action_num)) + np.random.rand(len(states), self.action_num) / 1000  # 1000は適当
            p = p / np.sum(p, axis=1).reshape((-1, 1))
        else:
            feature = np.zeros((len(states), 9, 9, self.input_channels))
            for i, s, movable_arr in zip(range(len(states)), states, leaf_movable_arrs):
                r, c = s.placable_array(s.turn % 2)
                x, y = color_p(s, s.turn % 2)
                mask[i, :] = np.concatenate([r.flatten(), c.flatten(), movable_arr])
                if not np.any(mask[i, :]):  # 相手がゴールにいるせいで距離を縮められない場合などに起こる
                    mask[i, :] = np.concatenate(
                        [r.flatten(), c.flatten(), movable_array(s, x, y, shortest_only=False).flatten()])
                feature[i, :] = s.feature_CNN()
                #if s.terminate:
                #    mask[i, :] = np.zeros((self.action_num,))
            p = self.sess.run(self.p_tf, feed_dict={self.x:feature})

        p = np.asarray(p, dtype=np.float32)
        p = p + 1e-7  # maskした後pが0にならないように対策
        p = p * mask

        # 距離を縮める方向に事前確率を高める
        for i, s in enumerate(states):
            x, y = color_p(s, s.turn % 2)
            shortest_move = movable_array(s, x, y, shortest_only=True).flatten()
            p_move = p[i, 128:]
            if np.sum(shortest_move) > 0:  # 相手がゴールにいるせいで距離を縮められない場合などにsumが0になる
                p_assisted = (1 - SHORTEST_P_RATIO) * p_move + SHORTEST_P_RATIO * np.sum(p_move) * shortest_move / np.sum(shortest_move)
                p[i, 128:] = p_assisted

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
    
    def pv_array(self, states, leaf_movable_arrs=None):
        if self.v_is_dist:
            assert False, "not implemented"

        if self.opponent_AI is not None:
            return self.opponent_AI.pv_array(states, leaf_movable_arrs)

        mask = np.zeros((len(states), self.action_num))
        if self.p_is_almost_flat:
            feature = np.zeros((len(states), 9, 9, self.input_channels))
            for i, s, movable_arr in zip(range(len(states)), states, leaf_movable_arrs):
                r, c = s.placable_array(s.turn % 2)
                x, y = color_p(s, s.turn % 2)
                mask[i, :] = np.concatenate([r.flatten(), c.flatten(), movable_arr])
                if not np.any(mask[i, :]):  # 相手がゴールにいるせいで距離を縮められない場合などに起こる
                    mask[i, :] = np.concatenate(
                        [r.flatten(), c.flatten(), movable_array(s, x, y, shortest_only=False).flatten()])
                feature[i, :] = s.feature_CNN()
            p = np.ones((len(states), self.action_num)) + np.random.rand(len(states), self.action_num) / 1000  # 1000は適当
            p = p / np.sum(p, axis=1).reshape((-1, 1))
            y_pred = self.sess.run(self.y, feed_dict={self.x:feature})
        else:
            feature = np.zeros((len(states), 9, 9, self.input_channels))
            for i, s, movable_arr in zip(range(len(states)), states, leaf_movable_arrs):
                r, c = s.placable_array(s.turn % 2)
                x, y = color_p(s, s.turn % 2)
                mask[i, :] = np.concatenate([r.flatten(), c.flatten(), movable_arr])
                if not np.any(mask[i, :]):  # 相手がゴールにいるせいで距離を縮められない場合などに起こる
                    mask[i, :] = np.concatenate(
                        [r.flatten(), c.flatten(), movable_array(s, x, y, shortest_only=False).flatten()])
                feature[i, :] = s.feature_CNN()
                #if s.terminate:
                #    mask[i, :] = np.zeros((self.action_num,))
            p, y_pred = self.sess.run([self.p_tf, self.y], feed_dict={self.x:feature})

        for i, s in enumerate(states):
            if s.pseudo_terminate:
                y_pred[i] = s.pseudo_reward

        if self.color == 1:
            y_pred = -y_pred

        p = np.asarray(p, dtype=np.float32)
        p = p + 1e-7  # maskした後pが0にならないように対策
        p = p * mask

        # 距離を縮める方向に事前確率を高める
        for i, s in enumerate(states):
            x, y = color_p(s, s.turn % 2)
            shortest_move = movable_array(s, x, y, shortest_only=True).flatten()
            p_move = p[i, 128:]
            if np.sum(shortest_move) > 0:  # 相手がゴールにいるせいで距離を縮められない場合などにsumが0になる
                p_assisted = (1 - SHORTEST_P_RATIO) * p_move + SHORTEST_P_RATIO * np.sum(p_move) * shortest_move / np.sum(shortest_move)
                p[i, 128:] = p_assisted

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

        return p, y_pred

    def learn(self, epoch, step_num, init_loss_dict=None, init_valid_loss=5.0, long_warmup_epoch_num=POOL_EPOCH_NUM, data_dir=DATA_DIR, learning_rate=LEARNING_RATE):
        print("epoch={}, step_num={}, POOL_EPOCH_NUM={}, layer_num={}, filters={}, lr={:.4f}".format(
            epoch, step_num, POOL_EPOCH_NUM, self.layer_num, self.filters, learning_rate))

        if init_loss_dict is None:
            init_loss_dict = {"train_loss": 5.0, "v_loss": 1.0, "v_reg_loss": 1.0, "p_loss": 5.0, "p_reg_loss": 5.0}
            for name in AUX_NAME_LIST:
                init_loss_dict[name] = 5.0

        valid_ema = init_valid_loss

        loss_dict = copy.copy(init_loss_dict)
        loss_tensor_dict = {"train_loss": self.loss_without_regularizer, "v_loss": self.v_loss, "v_reg_loss": self.y_regularizer, "p_loss": self.p_loss, "p_reg_loss": self.p_regularizer}
        for name in AUX_NAME_LIST:
            loss_tensor_dict[name] = self.aux_loss_dict[name]

        loss_names = list(loss_tensor_dict.keys())

        train_file_num_per_epoch = math.ceil(TRAIN_FILE_NUM_PER_EPOCH)
        valid_game_num = math.ceil(VALID_GAME_NUM)

        if epoch <= long_warmup_epoch_num:
            warmup_steps = step_num  # 最初の学習では特にwarmupをしっかりやらないとvが0に収束してしまうことが多い
        else:
            warmup_steps = WARMUP_STEPS

        def get_temp_list(h5file):
            ret = []
            for x_name in FEATURE_NAME_LIST:
                shape_dim = len(h5file[x_name].shape)
                if shape_dim == 4:
                    x_temp = h5file[x_name][:, :, :, :]
                elif shape_dim == 3:
                    x_temp = h5file[x_name][:, :, :]
                elif shape_dim == 2:
                    x_temp = h5file[x_name][:, :]
                elif shape_dim == 1:
                    x_temp = h5file[x_name][:]
                ret.append(x_temp)
            return ret

        # パラメータ更新
        count = 0
        for step in range(step_num):
            train_h5_filies = []
            
            for i in range(epoch - POOL_EPOCH_NUM, epoch):
                train_files_each = random.sample([os.path.join(data_dir, str(i), f"{j}.h5") for j in range(i * EPOCH_H5_NUM, i * EPOCH_H5_NUM + EPOCH_H5_NUM - valid_game_num)], train_file_num_per_epoch)
                train_h5_filies.extend(train_files_each)

            x_list_dict = {}
            for x_name in FEATURE_NAME_LIST:
                x_list_dict[x_name] = []

            for h5_filename in train_h5_filies:
                h5file = h5py.File(h5_filename, "r")

                temp_list = get_temp_list(h5file)
                temp_list = shuffle(*temp_list)

                each_size = int(len(temp_list[0]) * min(1, TRAIN_FILE_NUM_PER_EPOCH))

                for i, x_name in enumerate(FEATURE_NAME_LIST):
                    x_list_dict[x_name].append(np.copy(temp_list[i][:each_size]))

                for x_temp in temp_list:
                    del x_temp

            x_arr_list = []
            for x_name in FEATURE_NAME_LIST:
                x_arr_list.append(np.concatenate(x_list_dict[x_name]))

            x_arr_list = shuffle(*x_arr_list)
            x_arr_dict = {}
            for i, x_name in enumerate(FEATURE_NAME_LIST):
                x_arr_dict[x_name] = x_arr_list[i]

            train_size = len(x_arr_dict["feature"])
            train_step_num = (train_size - 1) // self.batch_size + 1
            for j in range(train_step_num):
                if step < warmup_steps:
                    coef = min(1.0, count / train_step_num / warmup_steps)
                else:
                    coef = 1.0
                coef *= learning_rate / LEARNING_RATE  # self.train_stepの学習率を変えるのが難しいからcoefで実質的に学習率を設定する

                train_feed_dict = {
                    self.x:x_arr_dict["feature"][j * self.batch_size:(j + 1) * self.batch_size],
                    self.pi:x_arr_dict["pi"][j * self.batch_size:(j + 1) * self.batch_size],
                    self.y_:x_arr_dict["reward"][j * self.batch_size:(j + 1) * self.batch_size].reshape(-1, 1),
                    self.warmup_coef: coef, self.searched_node: x_arr_dict["searched_node_num"][j * self.batch_size:(j + 1) * self.batch_size]
                }

                for aux_ph in AUX_NAME_LIST:
                    train_feed_dict[self.aux_ph_dict[aux_ph]] = x_arr_dict[aux_ph][j * self.batch_size:(j + 1) * self.batch_size]

                self.sess.run(self.train_step, feed_dict=train_feed_dict)

                train_feed_dict[self.warmup_coef] = 1.0  # loss算出時はwarmupの影響を消す

                loss_list = self.sess.run([loss_tensor_dict[loss_name] for loss_name in loss_names], feed_dict=train_feed_dict)
                for loss_i, loss_name in enumerate(loss_names):
                    #loss = self.sess.run(loss_tensor_dict[loss_name], feed_dict=train_feed_dict)
                    loss_dict[loss_name] = loss_dict[loss_name] * (1 - EMA_DECAY) + loss_list[loss_i] * EMA_DECAY

                print(f"\r{step}:" + str(j * self.batch_size) + "/" + str(x_arr_dict["feature"].shape[0]) + " warmup coef = {:.4f}".format(coef), end="")
                for loss_name in ["train_loss", "v_loss"]:
                    print(", {} = {:.4f}".format(loss_name, loss_dict[loss_name]), end="")

                count += 1
        print()

        # evaluation
        if USE_VALID:
            valid_h5_files = []
            for i in range(epoch - POOL_EPOCH_NUM, epoch):
                valid_h5_files.extend([os.path.join(data_dir, str(i), f"{j}.h5") for j in range(i * EPOCH_H5_NUM + EPOCH_H5_NUM - valid_game_num, i * EPOCH_H5_NUM + EPOCH_H5_NUM)])

            x_list_dict = {}
            for x_name in FEATURE_NAME_LIST:
                x_list_dict[x_name] = []

            for h5_filename in valid_h5_files:
                h5file = h5py.File(h5_filename, "r")

                temp_list = get_temp_list(h5file)
                temp_list = shuffle(*temp_list)

                each_size = int(len(temp_list[0]) * min(1, TRAIN_FILE_NUM_PER_EPOCH))

                for i, x_name in enumerate(FEATURE_NAME_LIST):
                    x_list_dict[x_name].append(np.copy(temp_list[i][:each_size]))

                for x_temp in temp_list:
                    del x_temp

            x_arr_list = []
            for x_name in FEATURE_NAME_LIST:
                x_arr_list.append(np.concatenate(x_list_dict[x_name]))

            x_arr_list = shuffle(*x_arr_list)
            x_arr_dict = {}
            for i, x_name in enumerate(FEATURE_NAME_LIST):
                x_arr_dict[x_name] = x_arr_list[i]

            valid_size = len(x_arr_dict["feature"])
            valid_step_num = (valid_size - 1) // self.batch_size + 1
            #print(valid_size, valid_step_num)

            for j in range(valid_step_num):
                loss = self.sess.run(self.loss_without_regularizer, feed_dict={
                    self.x:x_arr_dict["feature"][j * self.batch_size:(j + 1) * self.batch_size],
                    self.pi:x_arr_dict["pi"][j * self.batch_size:(j + 1) * self.batch_size],
                    self.y_:x_arr_dict["reward"][j * self.batch_size:(j + 1) * self.batch_size].reshape(-1, 1),
                    self.warmup_coef: 1.0, self.searched_node: x_arr_dict["searched_node_num"][j * self.batch_size:(j + 1) * self.batch_size]})
                valid_ema = valid_ema * (1 - VALID_EMA_DECAY) + loss * VALID_EMA_DECAY
                sys.stderr.write('\r\033[K' + str(step) + " "  + str(j * self.batch_size) + "/" + str(x_arr_dict["feature"].shape[0]) + " valid loss = {:.4f}".format(valid_ema) + f" warmup coef = {coef}")
                sys.stderr.flush()
            print()
        #pprint(loss_dict)
        return loss_dict, valid_ema

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

        dir = os.path.dirname(path)
        name = os.path.basename(path).split(".")[0]
        params = {
            "filters": self.filters, 
            "layer_num": self.layer_num, 
            "use_global_pooling": self.use_global_pooling,
            "use_self_attention": self.use_self_attention,
            "use_slim_head": self.use_slim_head,
            "use_mix_precision": self.use_mix_precision
        }
        with open(os.path.join(dir, f"{name}.json"), "w") as fout:
            fout.write(json.dumps(params))

    def load(self, path):
        if self.opponent_AI is not None:
            return

        dir = os.path.dirname(path)
        name = os.path.basename(path).split(".")[0]
        with open(os.path.join(dir, f"{name}.json"), "r") as fin:
            params = json.load(fin)
        # print("parameter loaded:")
        # pprint(params)

        self.filters = params["filters"]
        self.layer_num = params["layer_num"]

        if "use_global_pooling" in params.keys():
            self.use_global_pooling = params["use_global_pooling"]
        else:
            self.use_global_pooling = True  # パラメータ追加前のデフォルト値（≠いまのデフォルト値）

        if "use_self_attention" in params.keys():
            self.use_self_attention = params["use_self_attention"]
        else:
            self.use_self_attention = False  # パラメータ追加前のデフォルト値（≠いまのデフォルト値）

        if "use_slim_head" in params.keys():
            self.use_slim_head = params["use_slim_head"]
        else:
            self.use_slim_head = False

        if "use_mix_precision" in params.keys():
            self.use_mix_precision = params["use_mix_precision"]
        else:
            self.use_mix_precision = False

        self.init_tensorflow()
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        # print("restore ok")








