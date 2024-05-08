import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from config import *

AI_num = 50
SIGMA = 1.0
EPSILON = 1e-10
MAX_STEP = 10000
GRAD_TH = 1e-4  # 勾配の大きさがこれを下回ったら停止
LR = 3e-4


def estimate_multi_rate(n_arr, N_arr, AI_num, r_init=None):
    n_arr = np.triu(np.copy(n_arr), k=1)
    N_arr = np.triu(np.copy(N_arr), k=1)

    @tf.function
    def loss_f(r):
        r_mat2 = tf.broadcast_to(r, [AI_num, AI_num])
        r_mat1 = tf.transpose(r_mat2)
        p = 1 / (1 + tf.math.exp(r_mat2 - r_mat1))
        loss = tf.reduce_sum(r**2) / (2 * SIGMA ** 2) - tf.reduce_sum(n_arr * tf.math.log(p + EPSILON) + (N_arr - n_arr) * tf.math.log(1 - p + EPSILON))
        return loss

    opt = tf.keras.optimizers.experimental.SGD(learning_rate=LR, momentum=0.9)
    #opt = tf.keras.optimizers.Adam(learning_rate=LR)

    if r_init is None:
        r = tf.Variable(np.zeros(n_arr.shape[0]))
    else:
        r = tf.Variable(r_init)

    for i in range(MAX_STEP):
        with tf.GradientTape() as tape:
            loss = loss_f(r)
        grad = tape.gradient(loss, r)
        grad_size = float(tf.math.sqrt(tf.reduce_mean(grad ** 2)))
        if grad_size < GRAD_TH:
            break
        opt.apply_gradients([(grad, r)])
        print("\r{}".format(i), end="")
        #print("\r{}:{}".format(i, r.numpy()), end="")
    print()

    ret = r.numpy()
    ret = ret - np.min(ret)
    return ret


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    # 固定試合のリーグ戦でレート推定してみる。正解のレートを準備しておいて、推定と正解を比較する。
    N = 2

    r_arr = np.log(np.arange(AI_num) + 1) + np.random.rand(AI_num)
    r_arr -= np.min(r_arr)
    print(r_arr)

    n_arr = np.zeros((AI_num, AI_num))
    for i in range(AI_num):
        for j in range(i + 1, AI_num):
            n_arr[i, j] = np.random.binomial(N, 1 / (1 + tf.math.exp(r_arr[j] - r_arr[i])))

    N_arr = N * np.ones((AI_num, AI_num))
    for i in range(AI_num):
        for j in range(i + 1):
            N_arr[i, j] = 0

    print("games = {}".format(np.sum(N_arr)))

    estimated_r_arr = estimate_multi_rate(n_arr, N_arr, AI_num)
    print(estimated_r_arr)

    save_dir = os.path.join(TRAIN_LOG_DIR, "detail")
    os.makedirs(save_dir, exist_ok=True)

    plt.clf()
    plt.plot(r_arr, label="true")
    plt.plot(estimated_r_arr, label="est")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "r_arr_sample.png"))

