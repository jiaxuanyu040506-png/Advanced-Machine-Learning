#!/usr/bin/env python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import numpy as np
import os
import argparse
from collections import deque
import matplotlib.pyplot as plt

sys.path.append("game/")
import wrapped_flappy_bird as game

# ================== PARAMETERS ==================
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
LEARNING_RATE = 7e-4
ENTROPY_BETA = 0.01
N_STEPS = 5               # number of steps to collect before an update (mini-batch)
SAVE_EVERY = 10000
EVAL_EVERY = 50000
EVAL_EPISODES = 5
MAX_STEP = 5000000
CHECKPOINT_DIR = "saved_networks/a2c_1/"  # <- 新的子文件夹
# =================================================

# ------------------ NETWORK HELPERS ------------------
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

# ------------------ ACTOR-CRITIC NETWORK ------------------
def createA2CNetwork():
    W_conv1 = weight_variable([8,8,4,32]); b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([4,4,32,64]); b_conv2 = bias_variable([64])
    W_conv3 = weight_variable([3,3,64,64]); b_conv3 = bias_variable([64])
    W_fc1 = weight_variable([1600, 512]); b_fc1 = bias_variable([512])

    s = tf.placeholder(tf.float32, [None, 80, 80, 4], name='state')
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # Actor
    W_actor = weight_variable([512, ACTIONS]); b_actor = bias_variable([ACTIONS])
    logits = tf.matmul(h_fc1, W_actor) + b_actor
    policy = tf.nn.softmax(logits)

    # Critic
    W_critic = weight_variable([512, 1]); b_critic = bias_variable([1])
    value = tf.matmul(h_fc1, W_critic) + b_critic

    return s, logits, policy, value

# ------------------ UTIL ------------------
def preprocess(img):
    img = cv2.cvtColor(cv2.resize(img, (80,80)), cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    return img

# ------------------ EVALUATION FUNCTION ------------------
def evaluateA2C(s, logits, policy, sess, episodes=5, render=False):
    total_rewards = []
    for ep in range(episodes):
        game_state = game.GameState()
        do_nothing = np.zeros(ACTIONS); do_nothing[0]=1
        x_t, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = preprocess(x_t)
        s_t = np.stack((x_t,x_t,x_t,x_t), axis=2)

        done = False
        ep_reward = 0
        while not done:
            pi = sess.run(policy, feed_dict={s:[s_t]})[0]
            action_index = np.argmax(pi)
            a_t = np.zeros([ACTIONS]); a_t[action_index]=1

            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            if render:
                cv2.imshow('A2C', cv2.resize(x_t1_colored,(400,400)))
                cv2.waitKey(1)

            x_t1 = preprocess(x_t1_colored)
            x_t1 = np.reshape(x_t1,(80,80,1))
            s_t1 = np.append(x_t1, s_t[:,:, :3], axis=2)

            s_t = s_t1
            ep_reward += r_t
            done = terminal

        total_rewards.append(ep_reward)
    if render:
        cv2.destroyAllWindows()
    avg_reward = np.mean(total_rewards)
    max_reward = np.max(total_rewards)
    return avg_reward, max_reward

# ------------------ TRAIN FUNCTION ------------------
def trainA2C(s, logits, policy, value, sess):
    actions_ph = tf.placeholder(tf.int32, [None], name='actions')
    returns_ph = tf.placeholder(tf.float32, [None], name='returns')
    advantages_ph = tf.placeholder(tf.float32, [None], name='advantages')

    pi_safe = policy + 1e-10
    log_pi = tf.log(pi_safe)
    indices = tf.range(tf.shape(actions_ph)[0])
    action_probs = tf.gather_nd(log_pi, tf.stack([indices, actions_ph], axis=1))

    actor_loss = -tf.reduce_mean(action_probs * advantages_ph)
    value_loss = tf.reduce_mean(tf.square(returns_ph - tf.reshape(value, [-1])))
    entropy = -tf.reduce_mean(tf.reduce_sum(pi_safe * log_pi, axis=1))

    total_loss = actor_loss + 0.5 * value_loss - ENTROPY_BETA * entropy

    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=0.99, epsilon=1e-5)
    gradients, variables = zip(*optimizer.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 40.0)
    train_step = optimizer.apply_gradients(zip(gradients, variables))

    # --------- Saver & folder ---------
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    start_step = 0
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Loaded previous A2C checkpoint:", checkpoint.model_checkpoint_path)
            start_step = int(checkpoint.model_checkpoint_path.split('-')[-1])
            print(f"Resuming from step {start_step}")
        except Exception as e:
            print("Checkpoint exists but failed to restore. Starting fresh A2C training.", e)

    game_state = game.GameState()
    do_nothing = np.zeros(ACTIONS); do_nothing[0]=1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = preprocess(x_t)
    s_t = np.stack((x_t,x_t,x_t,x_t), axis=2)

    t = start_step
    reward_window = deque(maxlen=1000)
    eval_steps = []
    eval_avg_rewards = []

    states_buf = []
    actions_buf = []
    rewards_buf = []
    values_buf = []
    dones_buf = []

    log_file = open("A2C_training_log.txt", "a")

    while t < MAX_STEP:
        for _ in range(N_STEPS):
            pi = sess.run(policy, feed_dict={s:[s_t]})[0]
            action_index = np.random.choice(range(ACTIONS), p=pi)
            a_t = np.zeros([ACTIONS]); a_t[action_index]=1

            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = preprocess(x_t1_colored)
            x_t1 = np.reshape(x_t1,(80,80,1))
            s_t1 = np.append(x_t1, s_t[:,:, :3], axis=2)

            v_curr = sess.run(value, feed_dict={s:[s_t]})[0][0]

            states_buf.append(s_t)
            actions_buf.append(action_index)
            rewards_buf.append(r_t)
            values_buf.append(v_curr)
            dones_buf.append(terminal)

            s_t = s_t1
            t += 1
            reward_window.append(r_t)

            if terminal:
                game_state = game.GameState()
                x_t, r_0, terminal = game_state.frame_step(do_nothing)
                x_t = preprocess(x_t)
                s_t = np.stack((x_t,x_t,x_t,x_t), axis=2)

            if t % SAVE_EVERY == 0:
                saver.save(sess, CHECKPOINT_DIR + GAME + '-a2c', global_step=t)
                print("Saved model at step", t)

            if t % EVAL_EVERY == 0:
                avg_reward, max_reward = evaluateA2C(s, logits, policy, sess, episodes=EVAL_EPISODES, render=False)
                log_file.write(f"Step {t}: Eval Avg Reward = {avg_reward}, Max Reward = {max_reward}\n")
                log_file.flush()
                eval_steps.append(t)
                eval_avg_rewards.append(avg_reward)
                print(f"Step {t}: Eval Avg Reward = {avg_reward}, Max Reward = {max_reward}")

        if dones_buf[-1]:
            R = 0.0
        else:
            R = sess.run(value, feed_dict={s:[s_t]})[0][0]

        returns = []
        for reward, done in zip(reversed(rewards_buf), reversed(dones_buf)):
            R = reward + GAMMA * R * (1.0 - float(done))
            returns.insert(0, R)

        values = values_buf
        advantages = [ret - val for ret, val in zip(returns, values)]

        states_np = np.stack(states_buf, axis=0)
        actions_np = np.array(actions_buf, dtype=np.int32)
        returns_np = np.array(returns, dtype=np.float32)
        advantages_np = np.array(advantages, dtype=np.float32)

        sess.run(train_step, feed_dict={
            s: states_np,
            actions_ph: actions_np,
            returns_ph: returns_np,
            advantages_ph: advantages_np
        })

        states_buf = []
        actions_buf = []
        rewards_buf = []
        values_buf = []
        dones_buf = []

        if t % 1000 == 0:
            avg_window = sum(reward_window)/len(reward_window) if len(reward_window)>0 else 0.0
            print(f"Step: {t}, Sliding Avg Reward: {avg_window:.2f}")

# ------------------ MAIN ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='Play mode: use trained model to play')
    args = parser.parse_args()

    sess = tf.InteractiveSession()
    s_var, logits, policy, value = createA2CNetwork()

    if args.play:
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        saver = tf.train.Saver()
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Loaded checkpoint for play:", checkpoint.model_checkpoint_path)
        else:
            print("No checkpoint found. Cannot play.")
            return

        avg, mx = evaluateA2C(s_var, logits, policy, sess, episodes=5, render=True)
        print(f"Eval (play) Avg: {avg}, Max: {mx}")
    else:
        trainA2C(s_var, logits, policy, value, sess)

if __name__ == "__main__":
    main()
