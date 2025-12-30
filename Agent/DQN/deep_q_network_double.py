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
GAME = 'bird'
ACTIONS = 2

GAMMA = 0.99
OBSERVE = 50000.0
EXPLORE = 1000000.0
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1

MAX_TIMESTEPS = 5000000

# Double DQN 特有：target 网络更新频率
TARGET_UPDATE_FREQ = 10000   # 每 10000 步把 online 参数复制给 target

LOG_DIR = "logs_" + GAME + "_double"
CHECKPOINT_DIR = "saved_networks_double"
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


def createNetwork(scope):
    """
    在给定 scope 下创建一个 Q 网络（online 或 target）
    返回：state placeholder, readout(Q值), hidden layer, 以及该 scope 下的变量列表
    """
    with tf.variable_scope(scope):
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

        s = tf.placeholder("float", [None, 80, 80, 4])

        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    vars_in_scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    return s, readout, h_fc1, vars_in_scope


def trainNetwork(sess):
    # 构建 online 和 target 两个网络
    s_online, readout_online, _, vars_online = createNetwork("online")
    s_target, readout_target, _, vars_target = createNetwork("target")

    # 定义 online 网络的损失与优化（只训练 online）
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout_online, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost, var_list=vars_online)

    # 定义 target 更新操作：online → target
    update_target_ops = []
    for v_t, v_o in zip(vars_target, vars_online):
        update_target_ops.append(v_t.assign(v_o))
    update_target = tf.group(*update_target_ops)

    # 游戏环境
    game_state = game.GameState()

    # replay buffer
    D = deque()

    # 日志文件
    progress_file = open(os.path.join(LOG_DIR, "progress.txt"), 'a')

    episode_path = os.path.join(LOG_DIR, "episode.txt")
    need_header = (not os.path.exists(episode_path)) or (os.path.getsize(episode_path) == 0)
    episode_file = open(episode_path, 'a')
    if need_header:
        episode_file.write("episode_id,timestep,episode_reward,epsilon\n")

    # 初始状态：do nothing
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # 初始化 & 加载 checkpoint
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    episode = 0
    episode_reward = 0.0

    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            ckpt_path = checkpoint.model_checkpoint_path
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

    # 初始化一开始的 target 网络（online → target）
    sess.run(update_target)

    # -------- main training loop --------
    while t < MAX_TIMESTEPS:
        # 根据当前 t 计算 epsilon
        if t <= OBSERVE:
            epsilon = INITIAL_EPSILON
        elif t <= OBSERVE + EXPLORE:
            frac = (t - OBSERVE) / EXPLORE
            epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * frac
        else:
            epsilon = FINAL_EPSILON

        # 选动作（epsilon-greedy），用 online 网络
        readout_t = readout_online.eval(feed_dict={s_online: [s_t]})[0]
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

            # ------- Double DQN target 计算 -------
            # 用 online 网络在 s' 上选动作
            q_next_online = readout_online.eval(feed_dict={s_online: s_j1_batch})  # (B,2)
            best_actions = np.argmax(q_next_online, axis=1)  # (B,)

            # 用 target 网络估计这些动作的 Q 值
            q_next_target = readout_target.eval(feed_dict={s_target: s_j1_batch})  # (B,2)

            y_batch = []
            for i in range(len(minibatch)):
                terminal_mb = minibatch[i][4]
                if terminal_mb:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * q_next_target[i, best_actions[i]])
            # ------------------------------------

            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s_online: s_j_batch
            })

        # 更新状态与计数
        s_t = s_t1
        t += 1
        episode_reward += r_t

        # 定期更新 target 网络
        if t % TARGET_UPDATE_FREQ == 0:
            sess.run(update_target)
            print("Target network updated at TIMESTEP", t)

        # 每 10000 步保存一次 checkpoint & 写入 progress
        if t % 10000 == 0:
            saver.save(sess, os.path.join(CHECKPOINT_DIR, GAME + '-double-dqn'), global_step=t)
            progress_file.write("TIMESTEP {}\n".format(t))
            progress_file.flush()

        # episode 结束就写一行 episode 日志
        if terminal:
            episode += 1
            episode_file.write(
                "EPISODE_ID: {}, TIMESTEP: {}, EPISODE_REWARD: {:.4f}, EPSILON: {:.4f}\n"
                .format(episode, t, episode_reward, epsilon)
            )
            episode_file.flush()
            episode_reward = 0.0

        # 打印 info
        if t <= OBSERVE:
            state = "observe"
        elif t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
              "/ Q_MAX %e" % np.max(readout_t))

    print("Train Complete (Double DQN) : TIMESTEPS =", t)

    progress_file.close()
    episode_file.close()


def playGame():
    sess = tf.InteractiveSession()
    trainNetwork(sess)


def main():
    playGame()


if __name__ == "__main__":
    main()
