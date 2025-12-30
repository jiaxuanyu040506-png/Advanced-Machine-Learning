#!/usr/bin/env python3
"""
REINFORCE with baseline (Monte-Carlo policy gradient + value baseline)
TF1 style (no Keras layers), train for fixed steps to compare with A2C.
Checkpoints: saved_networks/reinforce/
"""
import os
import argparse
import numpy as np
import cv2
from collections import deque
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Add your game path
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game

# ---------------- Hyperparams ----------------
GAMMA = 0.99
LEARNING_RATE = 7e-4
ENTROPY_BETA = 0.01
BATCH_EPISODES = 5
MAX_STEP = 5_000_000           # train steps instead of episodes
CHECKPOINT_DIR = "saved_networks/reinforce/"
SAVE_EVERY_STEPS = 10000
EVAL_EVERY_STEPS = 50000
EVAL_EPISODES = 5

INPUT_SHAPE = (80, 80)
STACK_SIZE = 4
ACTIONS = 2

# ---------------- Utils ----------------
def preprocess(img):
    img = cv2.cvtColor(cv2.resize(img, INPUT_SHAPE), cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    return img

def discount_returns(rewards, gamma=GAMMA):
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return returns

# ---------------- Network Helpers ----------------
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# ---------------- Network ----------------
def create_network():
    s_ph = tf.placeholder(tf.float32, [None, INPUT_SHAPE[0], INPUT_SHAPE[1], STACK_SIZE], name="state")
    actions_ph = tf.placeholder(tf.int32, [None], name="actions")
    returns_ph = tf.placeholder(tf.float32, [None], name="returns")

    # Convs
    W_conv1 = weight_variable([8,8,STACK_SIZE,32]); b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([4,4,32,64]); b_conv2 = bias_variable([64])
    W_conv3 = weight_variable([3,3,64,64]); b_conv3 = bias_variable([64])
    W_fc1 = weight_variable([5*5*64, 512]); b_fc1 = bias_variable([512])

    h_conv1 = tf.nn.relu(conv2d(s_ph, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_flat = tf.reshape(h_conv3, [-1, 5*5*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

    # Policy head
    W_actor = weight_variable([512, ACTIONS]); b_actor = bias_variable([ACTIONS])
    logits = tf.matmul(h_fc1, W_actor) + b_actor
    probs = tf.nn.softmax(logits)

    # Value head (baseline)
    W_critic = weight_variable([512,1]); b_critic = bias_variable([1])
    value = tf.reshape(tf.matmul(h_fc1, W_critic) + b_critic, [-1])

    # Loss
    action_oh = tf.one_hot(actions_ph, ACTIONS)
    safe_logits = logits + 1e-10
    log_probs = tf.reduce_sum(action_oh * tf.nn.log_softmax(safe_logits), axis=1)
    advantage = returns_ph - value
    adv_mean, adv_var = tf.nn.moments(advantage, axes=[0])
    advantage_norm = (advantage - adv_mean) / (tf.sqrt(adv_var) + 1e-8)
    policy_loss = -tf.reduce_mean(log_probs * advantage_norm)
    value_loss = 0.5 * tf.reduce_mean(tf.square(returns_ph - value))
    entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.nn.log_softmax(safe_logits), axis=1))
    loss = policy_loss + value_loss - ENTROPY_BETA * entropy

    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=0.99, epsilon=1e-5)
    grads_and_vars = optimizer.compute_gradients(loss)
    grads, vars_ = zip(*grads_and_vars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, 40.0)
    train_op = optimizer.apply_gradients(zip(clipped_grads, vars_))

    # Sampling & greedy action
    action_sample = tf.squeeze(tf.random.categorical(tf.math.log(probs + 1e-10), 1), axis=1)
    action_greedy = tf.argmax(probs, axis=1)

    return {
        's_ph': s_ph,
        'actions_ph': actions_ph,
        'returns_ph': returns_ph,
        'probs': probs,
        'value': value,
        'train_op': train_op,
        'action_sample': action_sample,
        'action_greedy': action_greedy
    }

# ---------------- Episode runner ----------------
def run_episode(env, sess, s_ph, sample_op):
    states, actions, rewards = [], [], []

    do_nothing = np.zeros(ACTIONS); do_nothing[0] = 1
    x, r, terminal = env.frame_step(do_nothing)
    s = preprocess(x)
    state = np.stack([s]*STACK_SIZE, axis=2)

    while True:
        states.append(state)
        a = sess.run(sample_op, feed_dict={s_ph: [state]})[0]
        act_vec = np.zeros(ACTIONS); act_vec[a]=1
        x, r, terminal = env.frame_step(act_vec)
        next_s = preprocess(x)
        state = np.append(state[:,:,1:], np.expand_dims(next_s,2), axis=2)
        actions.append(int(a))
        rewards.append(float(r))
        if terminal:
            break
    return states, actions, rewards, sum(rewards)

# ---------------- Evaluation ----------------
def evaluate_policy(env, sess, s_ph, greedy_op, episodes=5, render=False):
    scores = []
    for _ in range(episodes):
        do_nothing = np.zeros(ACTIONS); do_nothing[0] = 1
        x, r, terminal = env.frame_step(do_nothing)
        s = preprocess(x)
        state = np.stack([s]*STACK_SIZE, axis=2)
        ep_reward = 0
        while True:
            a = sess.run(greedy_op, feed_dict={s_ph: [state]})[0]
            act_vec = np.zeros(ACTIONS); act_vec[a]=1
            x, r, terminal = env.frame_step(act_vec)
            s_new = preprocess(x)
            state = np.append(state[:,:,1:], np.expand_dims(s_new,2), axis=2)
            ep_reward += r
            if render:
                cv2.imshow("Flappy", cv2.resize(x,(400,400)))
                cv2.waitKey(1)
            if terminal:
                if render:
                    cv2.destroyAllWindows()
                break
        scores.append(ep_reward)
    return np.mean(scores), np.max(scores)

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true", help="Run greedy policy")
    args = parser.parse_args()

    tf.reset_default_graph()
    net = create_network()
    saver = tf.train.Saver(max_to_keep=50)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if ckpt:
        try:
            saver.restore(sess, ckpt)
            print("Restored checkpoint:", ckpt)
        except Exception as e:
            print("Failed to restore checkpoint:", e)

    env = game.GameState()

    if args.play:
        print("Play mode (greedy).")
        avg, mx = evaluate_policy(env, sess, net['s_ph'], net['action_greedy'], episodes=5, render=True)
        print(f"Eval: avg={avg}, max={mx}")
        return

    # ---------- Training ----------
    total_steps = 0
    reward_history = deque(maxlen=1000)
    while total_steps < MAX_STEP:
        batch_states, batch_actions, batch_returns = [], [], []

        for _ in range(BATCH_EPISODES):
            states, actions, rewards, score = run_episode(env, sess, net['s_ph'], net['action_sample'])
            returns = discount_returns(rewards)
            batch_states.extend(states)
            batch_actions.extend(actions)
            batch_returns.extend(returns)

            total_steps += len(states)
            reward_history.append(score)
            print(f"Total Steps {total_steps} | score={score} | avg(last100)={np.mean(list(reward_history)):.2f}")

            # periodic evaluation
            if total_steps >= EVAL_EVERY_STEPS and (total_steps - len(states)) < EVAL_EVERY_STEPS:
                avg_eval, max_eval = evaluate_policy(env, sess, net['s_ph'], net['action_greedy'],
                                                     episodes=EVAL_EPISODES)
                print(f"[EVAL] Total Steps {total_steps}: eval_avg={avg_eval}, eval_max={max_eval}")

            # periodic saving
            if total_steps >= SAVE_EVERY_STEPS and (total_steps - len(states)) < SAVE_EVERY_STEPS:
                saver.save(sess, os.path.join(CHECKPOINT_DIR, "model"), global_step=total_steps)
                print(f"Saved checkpoint at total steps {total_steps}")

        # Train
        states_np = np.array(batch_states, dtype=np.float32)
        actions_np = np.array(batch_actions, dtype=np.int32)
        returns_np = np.array(batch_returns, dtype=np.float32)
        returns_np = (returns_np - returns_np.mean()) / (returns_np.std() + 1e-8)
        sess.run(net['train_op'], feed_dict={
            net['s_ph']: states_np,
            net['actions_ph']: actions_np,
            net['returns_ph']: returns_np
        })

    # ---------- Ensure final checkpoint ----------
    saver.save(sess, os.path.join(CHECKPOINT_DIR, "model"), global_step=total_steps)
    print(f"Saved final checkpoint at total steps {total_steps}")
    print("Training finished.")

if __name__ == "__main__":
    main()
