import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game

ACTIONS = 2
CHECKPOINT_PATH = "saved_networks_d3qn/bird-d3qn-5000000"  # 指定 5M 步模型

# ---------------- 网络构建 ----------------
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def createNetwork(scope):
    with tf.variable_scope(scope):
        W_conv1 = weight_variable([8, 8, 4, 32])
        b_conv1 = bias_variable([32])
        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])
        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])
        W_fc1 = weight_variable([1600, 512])
        b_fc1 = bias_variable([512])
        W_value = weight_variable([512,1])
        b_value = bias_variable([1])
        W_adv = weight_variable([512,ACTIONS])
        b_adv = bias_variable([ACTIONS])

        s = tf.placeholder("float", [None, 80,80,4])
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1,1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        value = tf.matmul(h_fc1, W_value) + b_value
        adv = tf.matmul(h_fc1, W_adv) + b_adv
        adv_centered = adv - tf.reduce_mean(adv, axis=1, keepdims=True)
        readout = value + adv_centered
    return s, readout

# ---------------- 主函数 ----------------
def play():
    sess = tf.InteractiveSession()

    # 创建网络
    s, readout = createNetwork("online")
    sess.run(tf.global_variables_initializer())

    # 载入指定 checkpoint
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_PATH)
    print("Loaded model from 5M steps:", CHECKPOINT_PATH)

    game_state = game.GameState()

    # 初始化状态
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, _, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80,80)), cv2.COLOR_BGR2GRAY)
    _, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t,x_t,x_t,x_t), axis=2)

    total_score = 0
    while True:
        # 选择动作
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        action_index = np.argmax(readout_t)
        a_t = np.zeros(ACTIONS)
        a_t[action_index] = 1

        # 与游戏交互
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        total_score += r_t

        # 处理新状态
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80,80)), cv2.COLOR_BGR2GRAY)
        _, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80,80,1))
        s_t1 = np.append(x_t1, s_t[:,:,:3], axis=2)
        s_t = s_t1

        # 游戏结束，重置
        if terminal:
            print("Game over! Score:", total_score)
            total_score = 0
            do_nothing = np.zeros(ACTIONS)
            do_nothing[0] = 1
            x_t, _, terminal = game_state.frame_step(do_nothing)
            x_t = cv2.cvtColor(cv2.resize(x_t, (80,80)), cv2.COLOR_BGR2GRAY)
            _, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
            s_t = np.stack((x_t,x_t,x_t,x_t), axis=2)

if __name__ == "__main__":
    play()
