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
import time

sys.path.append("game/")
import wrapped_flappy_bird as game

# ================== HYPERPARAMETERS ==================
GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99

# Learning rate schedule
LR_INIT = 2.5e-4
LR_FINAL = 2.5e-6

# Logging / saving
SLIDING_WINDOW = 32          # sliding avg window for quick logs
QUICK_LOG_EVERY = 50         # quick log frequency (steps)
EVAL_QUICK_EVERY = 10000     # quick eval frequency (steps) & checkpoint interval
EVAL_FULL_EVERY = 50000      # full eval frequency (steps) — write to training_log.txt
EVAL_EPISODES = 5
SAVE_EVERY = 10000           # save checkpoint every SAVE_EVERY steps
CHECKPOINT_DIR = "saved_networks/ppo"
LOG_FILE = "PPO_training_log.txt"

# Training
MAX_STEP = 5_000_000
PPO_BATCH_SIZE = 32          # collect this many transitions before doing an update (CPU-friendly)
CLIP_RATIO = 0.2
ENTROPY_COEF = 0.01
NUM_EPOCHS = 4               # number of passes over the batch
MINIBATCH_SIZE = 16          # minibatch size for each epoch
GAE_LAMBDA = 0.95
# =====================================================

# ------------------ NETWORK HELPERS ------------------
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool_2x2(x):
    # compatibility for TF1.x
    try:
        return tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    except AttributeError:
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# ------------------ NETWORK ------------------
def createPPONetwork():
    W_conv1 = weight_variable([8,8,4,32], name="W_conv1"); b_conv1 = bias_variable([32], name="b_conv1")
    W_conv2 = weight_variable([4,4,32,64], name="W_conv2"); b_conv2 = bias_variable([64], name="b_conv2")
    W_conv3 = weight_variable([3,3,64,64], name="W_conv3"); b_conv3 = bias_variable([64], name="b_conv3")
    W_fc1 = weight_variable([1600, 512], name="W_fc1"); b_fc1 = bias_variable([512], name="b_fc1")

    s = tf.placeholder(tf.float32, [None, 80, 80, 4], name="state")
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # Actor
    W_actor = weight_variable([512, ACTIONS], name="W_actor"); b_actor = bias_variable([ACTIONS], name="b_actor")
    logits = tf.matmul(h_fc1, W_actor) + b_actor
    policy = tf.nn.softmax(logits, name="policy")

    # Critic
    W_critic = weight_variable([512, 1], name="W_critic"); b_critic = bias_variable([1], name="b_critic")
    value = tf.matmul(h_fc1, W_critic) + b_critic

    return s, policy, value, logits

# ------------------ EVALUATION ------------------
def evaluatePPO(s_ph, policy, sess, episodes=5, render=False):
    total_rewards = []
    for ep in range(episodes):
        game_state = game.GameState()
        do_nothing = np.zeros(ACTIONS); do_nothing[0] = 1
        x_t, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80,80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        done = False
        ep_reward = 0
        while not done:
            pi = sess.run(policy, feed_dict={s_ph: [s_t]})[0]
            action_index = np.random.choice(range(ACTIONS), p=pi)
            a_t = np.zeros([ACTIONS]); a_t[action_index] = 1

            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            if render:
                cv2.imshow('Flappy Bird', cv2.resize(x_t1_colored, (400,400)))
                cv2.waitKey(1)

            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80,80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80,80,1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            s_t = s_t1
            ep_reward += r_t
            done = terminal

        total_rewards.append(ep_reward)
    if render:
        cv2.destroyAllWindows()
    avg_reward = float(np.mean(total_rewards))
    max_reward = float(np.max(total_rewards))
    return avg_reward, max_reward

# ------------------ PLAY ------------------
def playPPO(s_ph, policy, sess, episodes=5, render=True):
    ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if ckpt and "bird-ppo" in ckpt:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        print("Loaded PPO checkpoint for play:", ckpt)
    else:
        print("No PPO checkpoint found. Cannot play.")
        return

    for ep in range(episodes):
        game_state = game.GameState()
        do_nothing = np.zeros(ACTIONS); do_nothing[0] = 1
        x_t, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80,80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

        done = False
        ep_reward = 0
        while not done:
            pi = sess.run(policy, feed_dict={s_ph: [s_t]})[0]
            action_index = np.random.choice(range(ACTIONS), p=pi)
            a_t = np.zeros([ACTIONS]); a_t[action_index] = 1
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            if render:
                cv2.imshow('Flappy Bird', cv2.resize(x_t1_colored, (400,400)))
                cv2.waitKey(1)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80,80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80,80,1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
            s_t = s_t1
            ep_reward += r_t
            done = terminal
        print(f"Episode {ep+1}: Total Reward = {ep_reward}")
    cv2.destroyAllWindows()

# ------------------ LR schedule ------------------
def linear_schedule(lr_init, lr_final, current_step, max_steps):
    frac = float(current_step) / float(max_steps)
    if frac > 1.0:
        frac = 1.0
    return lr_init + frac * (lr_final - lr_init)

# ------------------ GAE ------------------
def compute_gae_py(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    # values length = len(rewards) + 1 (bootstrap)
    advs = np.zeros(len(rewards), dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advs[t] = lastgaelam
    return advs

# ------------------ TRAIN ------------------
def trainPPO(s_ph, policy, value, logits, sess):
    # placeholders for training
    actions_ph = tf.placeholder(tf.int32, [None], name="actions_ph")
    returns_ph = tf.placeholder(tf.float32, [None], name="returns_ph")
    advantages_ph = tf.placeholder(tf.float32, [None], name="advantages_ph")
    old_probs_ph = tf.placeholder(tf.float32, [None], name="old_probs_ph")
    lr_ph = tf.placeholder(tf.float32, shape=[], name="lr_ph")

    # action probs
    indices = tf.range(tf.shape(actions_ph)[0])
    new_probs = tf.gather_nd(policy, tf.stack([indices, actions_ph], axis=1))
    ratio = new_probs / (old_probs_ph + 1e-10)
    clipped = tf.clip_by_value(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO)
    surrogate = tf.minimum(ratio * advantages_ph, clipped * advantages_ph)
    actor_loss = -tf.reduce_mean(surrogate)

    # entropy
    entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.log(policy + 1e-10), axis=1))
    # critic
    critic_loss = tf.reduce_mean(tf.square(returns_ph - tf.reshape(value, [-1])))
    total_loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

    optimizer = tf.train.AdamOptimizer(lr_ph)
    train_op = optimizer.minimize(total_loss)

    # saver and checkpoint dir
    saver = tf.train.Saver(max_to_keep=50)
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    # Try restore latest checkpoint safely
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    start_step = 0
    if latest and "bird-ppo" in latest:
        try:
            saver.restore(sess, latest)
            print("Restored PPO model from:", latest)
            try:
                start_step = int(os.path.basename(latest).split('-')[-1])
            except:
                start_step = 0
            print("Continuing from step:", start_step)
        except Exception as e:
            print("Failed to restore checkpoint (will init fresh):", e)
            sess.run(tf.global_variables_initializer())
            start_step = 0
    else:
        print("No valid checkpoint found — starting fresh training.")
        sess.run(tf.global_variables_initializer())
        start_step = 0

    # environment init
    game_state = game.GameState()
    do_nothing = np.zeros(ACTIONS); do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80,80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    t = start_step
    reward_window = deque(maxlen=SLIDING_WINDOW)

    # buffers for batch collection
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_old_probs = []
    batch_next_values_placeholder = []  # not used directly; we'll query values

    log_file = open(LOG_FILE, "a")

    try:
        while t < MAX_STEP:
            # get policy and sample action
            pi = sess.run(policy, feed_dict={s_ph: [s_t]})[0]
            action_index = np.random.choice(range(ACTIONS), p=pi)
            a_t = np.zeros([ACTIONS]); a_t[action_index] = 1

            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80,80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80,80,1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            # store transition
            batch_states.append(s_t)
            batch_actions.append(action_index)
            batch_rewards.append(r_t)
            batch_old_probs.append(pi[action_index])

            # advance
            s_t = s_t1
            t += 1
            reward_window.append(r_t)

            # quick logs
            if t % QUICK_LOG_EVERY == 0:
                avg_window = float(sum(reward_window)) / len(reward_window) if len(reward_window) > 0 else 0.0
                max_window = float(max(reward_window)) if len(reward_window) > 0 else 0.0
                print(f"Step {t}: Sliding Avg Reward = {avg_window:.3f}, Max Reward = {max_window}")

            # When collected enough transitions -> do PPO update
            if len(batch_states) >= PPO_BATCH_SIZE:
                # compute values for states and bootstrap value for last next state
                states_np = np.array(batch_states, dtype=np.float32)
                values_np = sess.run(value, feed_dict={s_ph: states_np})[:, 0]  # shape (N,)
                # compute value for next state bootstrap (call value on last s_t)
                next_value = sess.run(value, feed_dict={s_ph: [s_t]})[0][0] if not terminal else 0.0
                # values array for GAE should be len = N + 1
                values_for_gae = np.append(values_np, next_value)

                rewards_np = np.array(batch_rewards, dtype=np.float32)
                advantages_np = compute_gae_py(rewards_np, values_for_gae, gamma=GAMMA, lam=GAE_LAMBDA)
                returns_np = advantages_np + values_np

                # normalize advantages
                advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)

                actions_np = np.array(batch_actions, dtype=np.int32)
                old_probs_np = np.array(batch_old_probs, dtype=np.float32)

                current_lr = linear_schedule(LR_INIT, LR_FINAL, t, MAX_STEP)

                data_size = states_np.shape[0]
                idxs = np.arange(data_size)

                # multiple epochs with random minibatches
                for ep in range(NUM_EPOCHS):
                    np.random.shuffle(idxs)
                    for start in range(0, data_size, MINIBATCH_SIZE):
                        end = start + MINIBATCH_SIZE
                        mb_idx = idxs[start:end]

                        feed = {
                            s_ph: states_np[mb_idx],
                            actions_ph: actions_np[mb_idx],
                            returns_ph: returns_np[mb_idx],
                            advantages_ph: advantages_np[mb_idx],
                            old_probs_ph: old_probs_np[mb_idx],
                            lr_ph: current_lr
                        }
                        sess.run(train_op, feed_dict=feed)

                # clear batch
                batch_states, batch_actions, batch_rewards, batch_old_probs = [], [], [], []

            # QUICK-EVAL + SAVE checkpoint every EVAL_QUICK_EVERY steps (e.g. 10000)
            if t % EVAL_QUICK_EVERY == 0:
                # quick eval 1 episode for fast signal
                avg_r, max_r = evaluatePPO(s_ph, policy, sess, episodes=1, render=False)
                print(f"[Quick Eval] Step {t}: Avg Reward = {avg_r}, Max Reward = {max_r}")
                # save checkpoint
                saver.save(sess, os.path.join(CHECKPOINT_DIR, GAME + '-ppo'), global_step=t)
                print(f"Saved checkpoint at step {t}")

            # FULL EVAL every EVAL_FULL_EVERY steps (e.g. 50000) and log
            if t % EVAL_FULL_EVERY == 0:
                avg_reward, max_reward = evaluatePPO(s_ph, policy, sess, episodes=EVAL_EPISODES, render=False)
                log_file.write(f"Step {t}: Eval Avg Reward = {avg_reward}, Max Reward = {max_reward}\n")
                log_file.flush()
                print(f"[Full Eval] Step {t}: Avg Reward = {avg_reward}, Max Reward = {max_reward}")

            # If episode ended, reset environment state
            if terminal:
                do_nothing = np.zeros(ACTIONS); do_nothing[0] = 1
                x_t, r_0, terminal = game_state.frame_step(do_nothing)
                x_t = cv2.cvtColor(cv2.resize(x_t, (80,80)), cv2.COLOR_BGR2GRAY)
                ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
                s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        saver.save(sess, os.path.join(CHECKPOINT_DIR, GAME + '-ppo'), global_step=t)
        print("Saved checkpoint at step", t)
    finally:
        log_file.close()
        print("Training loop ended.")

# ------------------ MAIN ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='Play mode: use trained model to play')
    args = parser.parse_args()

    sess = tf.InteractiveSession()
    s_ph, policy, value, logits = createPPONetwork()

    if args.play:
        playPPO(s_ph, policy, sess, episodes=5, render=True)
    else:
        trainPPO(s_ph, policy, value, logits, sess)

if __name__ == "__main__":
    main()
