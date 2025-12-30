from __future__ import print_function

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()   # 在 TF2 中启用 TF1 风格

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

# ---------------- 超参数 ----------------
GAME = 'bird'  # 日志文件夹名字
ACTIONS = 2    # 动作数（不 flap / flap）

GAMMA = 0.99          # 折扣因子
OBSERVE = 50000.0     # 先收集多少步再开始训练
EXPLORE = 1000000.0   # epsilon 从 1.0 衰减到 0.1 的区间长度
INITIAL_EPSILON = 1.0 # 起始探索率
FINAL_EPSILON = 0.1   # 最终探索率
REPLAY_MEMORY = 50000 # replay buffer 大小
BATCH = 32            # minibatch 大小
FRAME_PER_ACTION = 1  # 每步都决策一次

MAX_TIMESTEPS = 5000000  # 总环境步数上限

LOG_DIR = "logs_" + GAME
CHECKPOINT_DIR = "saved_networks"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# ----------------------------------------


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1


def trainNetwork(s, readout, h_fc1, sess):
    # cost
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # 游戏环境
    game_state = game.GameState()

    # replay buffer
    D = deque()

    # 日志文件
    progress_file = open(os.path.join(LOG_DIR, "progress.txt"), 'a')
    episode_file = open(os.path.join(LOG_DIR, "episode.txt"), 'a')

    # 初始状态：do nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    episode = 0
    episode_reward = 0.0

    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)

            ckpt_path = checkpoint.model_checkpoint_path  # e.g. saved_networks/bird-dqn-1200000
            try:
                t = int(ckpt_path.split('-')[-1])
            except ValueError:
                t = 0
            print("Continue training from TIMESTEP =", t)
        except tf.errors.NotFoundError:
            print("Checkpoint incompatible, training from scratch.")
            sess.run(tf.global_variables_initializer())
            t = 0
    else:
        print("Could not find old network weights")
        t = 0

    # -------- main training loop --------
    while t < MAX_TIMESTEPS:
        # 根据当前 t 计算 epsilon（而不是累计减）
        if t <= OBSERVE:
            epsilon = INITIAL_EPSILON
        elif t <= OBSERVE + EXPLORE:
            frac = (t - OBSERVE) / EXPLORE  # 0 ~ 1
            epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * frac
        else:
            epsilon = FINAL_EPSILON

        # 选动作（epsilon-greedy）
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                action_index = int(np.argmax(readout_t))
                a_t[action_index] = 1
        else:
            action_index = 0
            a_t[0] = 1

        # 与环境交互
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # 存入 replay
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 只有 buffer 够大才 train
        if t > OBSERVE and len(D) >= BATCH:
            minibatch = random.sample(D, BATCH)

            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
            for i in range(len(minibatch)):
                terminal_mb = minibatch[i][4]
                if terminal_mb:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch
            })

        # 更新状态与计数
        s_t = s_t1
        t += 1
        episode_reward += r_t

        # 每 10000 步保存一次 checkpoint & 写入 progress
        if t % 10000 == 0:
            saver.save(sess, os.path.join(CHECKPOINT_DIR, GAME + '-dqn'), global_step=t)
            progress_file.write("TIMESTEP {}\n".format(t))
            progress_file.flush()

        # episode 结束就写一行 episode 日志
        if terminal:
            episode += 1
            episode_file.write("{},{},{:.4f},{:.4f}\n".format(
                episode, t, episode_reward, epsilon))
            episode_file.flush()
            episode_reward = 0.0

        # 打印 info（PyCharm terminal 能看到）
        if t <= OBSERVE:
            state = "observe"
        elif t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t))

    print("Train Complete : TIMESTEPS =", t)

    progress_file.close()
    episode_file.close()


def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)


def main():
    playGame()


if __name__ == "__main__":
    main()
